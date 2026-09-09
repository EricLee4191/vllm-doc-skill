# 投机解码与高级推理特性

> 基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（v1 架构）源码分析。路径均相对仓库根 `/Users/baofeng/baofeng/github/vllm`。

本文覆盖 vLLM 的六类高级特性：**投机解码 (speculative decoding)**、**采样 (sampling)**、**结构化输出 (structured output)**、**LoRA 多适配器**、**多模态 (multimodal)**、**reasoning/思考模型支持**。重点讲架构与部署/调优旋钮。

---

## 1. 投机解码 (Speculative Decoding)
<!-- tags: speculative-decoding, eagle, mtp, ngram, draft-model, dflash, medusa, 投机解码 -->

### 1.1 原理：draft + verify
<!-- tags: speculative-decoding, draft, verify, rejection, 原理 -->

投机解码用一个廉价的 **drafter** 先猜 K 个 token，再用目标模型 (target) 一次 forward 并行验证这 K 个 token + 1 个 bonus 位置，从而把 K+1 个 token 的生成压缩成一次 target forward。核心循环：

1. **Draft**：`drafter.propose(...)` 产出 `draft_token_ids`（每请求最多 K 个）。
2. **Verify**：target 模型对 `[原输入 + draft tokens]` 做一次 forward，得到 `logits`（形状 `[num_tokens + batch_size, vocab_size]`，其中每个 draft token 位置一行 + 每请求一行 bonus）。
3. **Rejection sampling**：`RejectionSampler` 逐位置比较 draft 分布与 target 分布，接受 (accepted) / 恢复采样 (recovered) / 追加 bonus token，产出最终 `output tokens = accepted + recovered + bonus`。

关键数据结构 `SpecDecodeMetadata`（`vllm/v1/spec_decode/metadata.py`）：

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor          # [num_tokens]
    num_draft_tokens: list[int]            # [batch_size]
    cu_num_draft_tokens: torch.Tensor      # [batch_size]
    cu_num_sampled_tokens: torch.Tensor    # [batch_size]
    target_logits_indices: torch.Tensor    # [num_tokens]
    bonus_logits_indices: torch.Tensor     # [batch_size]
    logits_indices: torch.Tensor           # [num_tokens + batch_size]
```

执行入口在 `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py`）：
- drafter 在 `__init__` 里按 `speculative_config.method` 分派（`gpu_model_runner.py:634-709`）。
- `sample_tokens()`（`gpu_model_runner.py:4667`）里：先 `apply_grammar_bitmask`（若有结构化输出）→ `_sample()`（`gpu_model_runner.py:3759`，spec 时走 `rejection_sampler`）→ `propose_draft_token_ids()`（`gpu_model_runner.py:5120`）为下一步准备 draft。
- EAGLE/DraftModel 类 drafter 可以直接消费 GPU 上的 sampled tokens（`use_gpu_toks` 分支，`gpu_model_runner.py:4759`），不必等 bookkeeping；ngram/suffix 类消费 CPU token，在 bookkeeping 之后运行（`draft_after_bookkeeping`，`gpu_model_runner.py:4848`）。

### 1.2 支持的 draft 方式
<!-- tags: draft, method, ngram, eagle, mtp -->

`SpeculativeConfig.method` 的取值（`vllm/config/speculative.py:69-79`）：

| method | 说明 | 是否需要 draft 模型权重 |
|---|---|---|
| `ngram` | CPU 上基于 prompt 内 n-gram 匹配（numba 加速） | 否 |
| `ngram_gpu` | GPU 版 n-gram（`NgramProposerGPU`，triton kernel） | 否 |
| `suffix` | Suffix Decoding（依赖 `arctic_inference`，全局+prompt 后缀树） | 否 |
| `medusa` | Medusa 多头，对 target hidden states 直接 argmax | 是（medusa head） |
| `mlp_speculator` | MLP speculator | 是 |
| `draft_model` | 独立小型 draft model（自回归） | 是 |
| `eagle` / `eagle3` | EAGLE / EAGLE3（用 target hidden states 做 drafter） | 是（eagle head） |
| `mtp` | Multi-Token Prediction（DeepSeek/Qwen3-Next/GLM4 等自带 MTP 层） | 是（模型自带） |
| `dflash` | DFlash 并行 draft（Qwen3 等，非因果注意力） | 是 |
| `dspark` | DSpark 块验证 draft | 是 |
| `extract_hidden_states` | 抽取 target 中间层 hidden states 做 draft | 是 |
| `custom_class` | 用户自定义 proposer（`model` 传 `pkg.MyProposer` 点分路径） | 自定义 |

分派逻辑（`gpu_model_runner.py:647-706`）：`custom_class` → `create_custom_proposer`；`ngram` → `NgramProposer`；`uses_draft_model()` → `DraftModelProposer`；`use_ngram_gpu()` → `NgramProposerGPU`；`use_gemma4_mtp()` → `Gemma4Proposer`；`use_step3p5_mtp()` → `Step3p5MTPProposer`；`use_dflash()` → `DFlashProposer`；`suffix` → `SuffixDecodingProposer`；`use_eagle()` → `EagleProposer`（eagle3 时 `pass_hidden_states_to_model=True`）；`medusa` → `MedusaProposer`；`extract_hidden_states` → `ExtractHiddenStatesProposer`。

各 proposer 大多继承 `SpecDecodeBaseProposer`（`vllm/v1/spec_decode/llm_base_proposer.py`），其 `propose()`（`llm_base_proposer.py:510`）实现多步自回归 draft：先跑一次 target 对齐的 forward，再循环 `num_speculative_tokens-1` 次采样并推进 positions/slot_mapping。

- **EAGLE**（`eagle.py`）：`EagleProposer(SpecDecodeBaseProposer)`，`pass_hidden_states_to_model=True`，把 target 的 hidden states 喂给 draft 模型。
- **MTP**：`hf_config_override`（`speculative.py:342`）把 DeepSeek-V3/V4、Qwen3-Next、GLM4-MoE、Kimi-K3、MiniMax-M3 等 checkpoint 的 `num_nextn_predict_layers` 归一化成 `n_predict`，`method` 统一为 `mtp`。`num_speculative_tokens` 必须能被 `n_predict` 整除（`speculative.py:1083-1088`）。
- **ngram**（`ngram_proposer.py`）：`prompt_lookup_min`/`prompt_lookup_max` 定义匹配窗口（默认 5/5）；用 KMP 类算法在已生成 token 序列里找最长匹配 n-gram 并取其后 k 个 token。
- **suffix**（`suffix_decoding.py`）：`SuffixDecodingProposer` 包装 `arctic_inference.suffix_decoding.SuffixDecodingCache`，动态决定每请求每步的投机长度（`max_spec_tokens = min(num_speculative_tokens, max_spec_factor * prefix_match_length)`）。

### 1.3 acceptance 与验证流程
<!-- tags: rejection-sampler, acceptance, 验证, 接受率, draft-sampling -->

`RejectionSampler`（`vllm/v1/sample/rejection_sampler.py`）严格实现 arXiv:2211.17192：

- `forward()`（`rejection_sampler.py:92`）：
  1. 从 `bonus_logits` 用普通 `Sampler` 采 bonus token（可带 top_p/top_k）。
  2. 对 `target_logits` 应用 logits processors（penalties / bad words / min_tokens / thinking budget）与 `apply_sampling_constraints`（temperature + top_k/top_p）。
  3. `rejection_sample()`（`rejection_sampler.py:394`）：greedy 请求走 `rejection_greedy_sample_kernel`；随机请求先 `sample_recovered_tokens` 再 `rejection_random_sample_kernel`。
- 输出 `output_token_ids` 形状 `[batch_size, max_spec_len+1]`，被拒位置填 `PLACEHOLDER_TOKEN_ID=-1`，`parse_output()` 过滤。
- `draft_probs` 可为 None（ngram 无概率分布，退化为确定性接受）。
- `MAX_SPEC_LEN = 128`（`rejection_sampler.py:35`）是单步每请求 draft token 上限。

**rejection_sample_method**（`speculative.py:219`）：
- `standard`：概率式拒绝采样（默认）。配合 `draft_sample_method`：
  - `greedy`（默认）：draft 取 argmax，概率视为 one-hot。
  - `probabilistic`：draft 从分布随机采样，用完整 draft logits 做概率比检验（更准但更耗显存）。
- `synthetic`：按 `synthetic_acceptance_rates`（或 `synthetic_acceptance_length`）的衰减概率接受 draft，用于合成/测试。
- `block`：块验证（Sun et al.），把 draft token 作为整体联合验证。

**draft 采样**（`llm_base_proposer.py:436-505`）：`_greedy_sample` 支持 `use_local_argmax_reduction`（vocab-parallel 局部 argmax，把通信从 O(vocab) 降到 O(2*tp)）；`use_heterogeneous_vocab` 时用 `VocabMapping`（`vocab_mapping.py`，TLI 算法）在 draft/target 词表交集上约束 logits 并双向映射 token id。

### 1.4 如何配置
<!-- tags: speculative-config, 配置, cli, 字段, 示例 -->

`SpeculativeConfig`（`vllm/config/speculative.py:85`）主要字段：

- `num_speculative_tokens`：投机 token 数 K。若未给，默认取 draft 模型 config 的 `n_predict`。
- `model`：draft 模型 / eagle head / MTP 权重路径；ngram 时传 `"ngram"`。
- `method`：见上表；给 `model` 时可自动探测（`eagle-`/`eagle3`/`medusa`/`mlp_speculator`/MTP 类型/`dflash`/`dspark`），否则默认 `draft_model`。
- `draft_tensor_parallel_size`：draft 的 TP（须为 1 或与 target 相同，`draft_model.py:71` 强制相同）。
- `quantization` / `kv_cache_dtype` / `moe_backend` / `attention_backend` / `max_model_len`：draft 模型独立配置。
- `prompt_lookup_min` / `prompt_lookup_max`：ngram 窗口。
- `parallel_drafting`：并行 draft（EAGLE/draft_model，dflash/dspark 强制开启）。
- `disable_padded_drafter_batch`：禁用 padded drafter batch（仅 EAGLE 系）。
- `use_local_argmax_reduction` / `use_heterogeneous_vocab`：见上。
- `suffix_decoding_max_tree_depth`(24) / `suffix_decoding_max_cached_requests`(10000) / `suffix_decoding_max_spec_factor`(1.0) / `suffix_decoding_min_token_prob`(0.1)。
- `rejection_sample_method` / `synthetic_acceptance_rates` / `synthetic_acceptance_length` / `draft_sample_method` / `dspark_draft_topk` / `enable_adaptive_verification`(仅 dspark)。
- `num_speculative_tokens_per_batch_size`：**动态投机解码**，`[(range_start, range_end, K), ...]`，按运行时 batch size 查表选 K（`vllm/v1/spec_decode/dynamic/utils.py`，scheduler 展开成 dense lookup，`scheduler.py:1267-1269`）。

CLI（`vllm/engine/arg_utils.py`）：
- `--speculative-config` / `-sc`（JSON dict，最灵活）
- 快捷 flag：`--spec-method`、`--spec-model`、`--spec-tokens`（`arg_utils.py:1637-1640`）

示例：
```bash
# EAGLE
--speculative-config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 5}'
# ngram
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5}'
# DeepSeek MTP（权重在 target checkpoint 内）
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

`VllmConfig.num_speculative_tokens`（`config/vllm.py:608`）与 `num_lookahead_tokens`（`config/vllm.py:622`）供 scheduler 预留 KV slot（drafter 会写 target query 范围之外的 KV，scheduler 必须多预留 `num_lookahead_tokens` 个 slot）。

### 1.5 调度器侧
<!-- tags: scheduler, 调度, spec-tokens, draft-tokens, grammar -->

- scheduler 每步把 draft token 放进 `SchedulerOutput.scheduled_spec_decode_tokens`（`scheduler.py:724`），`update_draft_token_ids`（`scheduler.py:2264`）在下一步前把新 draft 写回 `request.spec_token_ids`；结构化输出请求会用 `grammar.validate_tokens` 过滤非法 draft（`scheduler.py:2283`）。

### 1.6 指标
<!-- tags: metrics, 指标, acceptance-rate, 接受率, per-request -->

`SpecDecodingStats` / `SpecDecodingLogging`（`vllm/v1/spec_decode/metrics.py`）：记录 `num_drafts`、`num_draft_tokens`、`num_accepted_tokens`、per-position 接受率，日志输出 `draft_acceptance_rate` 与 `mean_acceptance_length = 1 + accepted/drafts`。

**per-request 接受率指标（v0.28 新增，#48915）**：除全局日志外，每个请求的接受率统计（`num_draft_tokens`/`num_accepted_tokens`/per-position 接受率）现在会随 OpenAI API 响应返回（`docs/features/speculative_decoding/acceptance_metrics.md`），便于按请求观测投机解码收益；Rust frontend 的 `EngineCoreOutput` 协议也带上了这些字段。

---

## 2. 采样 (Sampling)
<!-- tags: sampling, sampler, sampling-params, 采样 -->

### 2.1 Sampler 工作流
<!-- tags: sampler, 采样, workflow, topk-topp, logits-processor -->

`Sampler`（`vllm/v1/sample/sampler.py:21`）`forward()` 步骤（见 docstring）：
1. 若请求 logprobs：`raw_logprobs` 模式先算 `log_softmax`；`raw_logits` 模式 clone logits。
2. logits 转 float32。
3. `apply_logits_processors`：allowed_token_ids 白名单 → bad words → 非 argmax-invariant 处理器（min_tokens、logit_bias）→ penalties（repetition/frequency/presence）→ thinking budget。
4. `sample()`：
   - `all_greedy` 直接 argmax 返回。
   - 否则 `apply_temperature` → argmax-invariant 处理器（min_p）→ `TopKTopPSampler`（top_k/top_p + 加权随机采样）→ 按 temperature 是否 < 1e-5 决定取 greedy 还是 random。
5. `gather_logprobs` 取 top-k + sampled token 的 logprob/rank。

`TopKTopPSampler`（`vllm/v1/sample/ops/topk_topp_sampler.py:77`）按平台选实现：CUDA 优先 FlashInfer（`forward_cuda`），否则 native；CPU/XPU 各有 kernel。

**Logits processors**（`vllm/v1/sample/logits_processor/`）：`MinPLogitsProcessor`、`LogitBiasLogitsProcessor`、`MinTokensLogitsProcessor`（`builtin.py`），通过 `is_argmax_invariant()` 区分是否影响 greedy。`SamplingMetadata`（`vllm/v1/sample/metadata.py`）携带每 batch 的 temperature/top_p/top_k/penalties/generators/`spec_token_ids`/`thinking_budget_state_holder`。

### 2.2 SamplingParams 字段
<!-- tags: sampling-params, 字段, 采样参数, beam-search -->

`SamplingParams`（`vllm/sampling_params.py:215`）关键字段：
- `n`(1)、`presence_penalty`(0)、`frequency_penalty`(0)、`repetition_penalty`(1.0)、`temperature`(1.0)、`top_p`(1.0)、`top_k`(0=禁用)、`min_p`(0.0)、`seed`、`stop`/`stop_token_ids`、`ignore_eos`、`max_tokens`(16)、`min_tokens`(0)。
- logprobs：`logprobs`、`prompt_logprobs`、`logprob_token_ids`（generative_scoring 用，只取指定 token 的 logprob）、`flat_logprobs`。
- `structured_outputs`（`StructuredOutputsParams`）、`logit_bias`、`allowed_token_ids`、`bad_words`、`thinking_token_budget`、`repetition_detection`。

**beam search**：`SamplingParams` 仍保留 `beam_width` 字段（`sampling_params.py:1254`，"not supported by OpenAI"），但 v1 引擎没有 beam search 实现（`vllm/v1/` 下无 `beam_search` 代码）——beam search 属于已废弃的 V0 路径，v1 不支持。

---

## 3. 结构化输出 (Structured Output / Guided Decoding)
<!-- tags: structured-output, guided-decoding, json-schema, xgrammar, outlines, 结构化输出 -->

### 3.1 架构
<!-- tags: structured-output, 架构, xgrammar, bitmask, fsm -->

`StructuredOutputManager`（`vllm/v1/structured_output/__init__.py:36`）是 engine 级单例，管理一个 backend（V1 不支持 per-request 切换 backend）。

- **backend 选择**：`StructuredOutputsConfig.backend`（`config/structured_outputs.py:21`）默认 `"auto"`。`auto` 模式在 `sampling_params.py:1151-1190` 里按优先级尝试：先 `xgrammar`，失败则 `guidance`（Mistral 非 tekken tokenizer 或 schema 含 guidance 不支持特性时退到 `outlines`）。
- **backend 实现**：
  - `XgrammarBackend`（`backend_xgrammar.py:37`）：默认。`xgr.GrammarCompiler` + `GrammarMatcher`，`compile_grammar` 支持 `JSON`/`JSON_OBJECT`/`GRAMMAR`/`REGEX`/`STRUCTURAL_TAG`。
  - `GuidanceBackend`（`backend_guidance.py`）、`OutlinesBackend`（`backend_outlines.py`，SQLite 磁盘缓存 `OUTLINES_CACHE_DIR`）、`LMFormatEnforcerBackend`（`backend_lm_format_enforcer.py`）。
- **约束机制 = bitmask**：每个请求的 grammar 是一个 FSM。每步 `grammar_bitmask()`（`__init__.py:220`）为每个待采样位置（含每个投机位置 + bonus）填一个 token bitmask（`grammar.fill_bitmask`），大 batch 时用线程池并行填（`fill_bitmask_parallel_threshold=128`）。GPU 侧 `apply_grammar_bitmask`（`vllm/v1/structured_output/utils.py:87`）把 bitmask 重排到与 batch 对齐，再 `xgr.apply_token_bitmask_inplace` 把非法 token 的 logit 置 -inf。
- **FSM 推进**：`XgrammarGrammar.accept_tokens`（`backend_xgrammar.py:157`）逐 token 推进并检测终止；`validate_tokens`（`backend_xgrammar.py:181`）预校验 draft token（不推进，用于投机解码过滤）；`rollback` 支持回滚（`max_rollback_tokens=num_speculative_tokens`）。

### 3.2 请求侧
<!-- tags: structured-output, 请求, request, key, 异步编译 -->

`StructuredOutputRequest`（`vllm/v1/structured_output/request.py:22`）由 `SamplingParams.structured_outputs` 构造，`structured_output_key` 区分类型（`get_structured_output_key`，`request.py:82`：JSON / JSON_OBJECT / REGEX / CHOICE / GRAMMAR / STRUCTURAL_TAG）。grammar 编译是异步的（`Future`），`is_grammar_ready` 轮询。

### 3.3 与投机解码 / reasoning 的交互
<!-- tags: structured-output, 投机解码, reasoning, 交互, bitmask -->

- 投机解码时 bitmask 要为每个投机位置 + bonus 各生成一行（`__init__.py:230-351`），并模拟 reasoning-end 检测。
- reasoning 模型：`should_fill_bitmask`/`should_advance`（`__init__.py:365/389`）在思考阶段不约束，检测到 reasoning 结束才启用 bitmask；`enable_in_reasoning=True` 时思考阶段也约束。

### 3.4 配置/调优旋钮
<!-- tags: structured-output, 配置, 旋钮, xgrammar-cache, reasoning -->

- `--structured-outputs-config`（JSON，含 `backend`/`disable_any_whitespace`/`disable_additional_properties`/`reasoning_parser`/`reasoning_parser_plugin`/`enable_in_reasoning`）。
- `--reasoning-parser`、`--reasoning-parser-plugin`（`arg_utils.py:1004-1009`）。
- 环境变量 `VLLM_XGRAMMAR_CACHE_MB`（默认 512，`envs.py:1626`）控制 xgrammar 编译缓存；`OUTLINES_CACHE_DIR` 控制 outlines 磁盘缓存。
- 请求级：`response_format`（OpenAI）/ `structured_outputs`（`json`/`regex`/`choice`/`grammar`/`json_object`/`structural_tag`）。

---

## 4. LoRA / 多适配器
<!-- tags: lora, adapter, 适配器 -->

### 4.1 架构
<!-- tags: lora, 架构, punica, worker-manager, 多模态 -->

- **`WorkerLoRAManager`**（`vllm/lora/worker_manager.py:27`）：worker 侧管理。每个请求的 LoRA 按需加载（`_load_adapter`），其余卸载。
- **`LoRAModelManager`**（`vllm/lora/model_manager.py:71`）：核心。持有 `PunicaWrapper`（`punica_wrapper/`，基于 Punica 论文 arXiv:2310.18547 的 triton kernel `lora_shrink`/`lora_expand`），维护 `lora_slots`/`adapter_slots`，`activate_adapter`/`_deactivate_adapter` 做 slot 分配，`AdapterLRUCache` 做 LRU 卸载。`create_lora_manager` 把 base model 的各 linear/embedding 层包装成 `BaseLayerWithLoRA`（`vllm/lora/layers/`：`column_parallel_linear`、`row_parallel_linear`、`replicated_linear`、`vocal_parallel_embedding`、`fused_moe` 等）。
- **`PunicaWrapperGPU`**（`vllm/lora/punica_wrapper/punica_gpu.py:34`）：维护 `LoRAKernelMeta`（token→lora 映射），`update_metadata` 每步更新。投机解码时 `max_num_samples = max_batches*(num_spec_tokens+1)`。
- **`LoRARequest`**（`vllm/lora/request.py:8`）：`lora_name`/`lora_int_id`/`lora_path`/`load_inplace`/`is_3d_lora_weight`。按 `lora_name` 判等/哈希。
- **多模态 LoRA**：`LoRAConfig.default_mm_loras`（modality→path 映射）、`enable_tower_connector_lora`（对 vision tower/connector 加 LoRA，实验性，仅部分 Qwen-VL）。

### 4.2 配置/调优旋钮
<!-- tags: lora, 配置, 旋钮, max-loras, max-rank -->

`LoRAConfig`（`vllm/config/lora.py:32`）/ CLI（`arg_utils.py:1424-1454`）：
- `--enable-lora`：总开关。
- `--max-loras`（默认 1）：单 batch 最多同时激活的 LoRA 数。
- `--max-lora-rank`（默认 16）：最大 rank（支持 1/8/16/32/64/128/256）。
- `--max-cpu-loras`：CPU 内存中缓存的 LoRA 数（须 ≥ max_loras），用于 LRU 换入换出。
- `--lora-dtype`（默认 auto）。
- `--fully-sharded-loras`：TP 下全分片 LoRA 计算（长序列/高 rank 更快）。
- `--lora-target-modules`：限制 LoRA 作用模块（如 `["o_proj","qkv_proj"]`）。
- `--enable-tower-connector-lora`、`--specialize-active-lora`（按激活 LoRA 数捕获多份 cuda graph）、`--enable-mixed-moe-lora-format`、`--enable-moe-shared-loras`（MoE 专用）。

部署要点：`max_loras` 决定 batch 内可并发服务的适配器数；`max_cpu_loras` 决定可缓存的适配器池大小（LRU 淘汰）。

---

## 5. 多模态 (Multimodal)
<!-- tags: multimodal, vision, vlm, 多模态 -->

### 5.1 架构
<!-- tags: multimodal, 架构, registry, processor, encoder -->

- **`MultiModalRegistry`**（`vllm/multimodal/registry.py:98`）：注册每模型的 processor（`register_processor` 装饰器）、`ProcessingInfo`、`DummyInputsBuilder`。`supports_multimodal_inputs` 判断模型是否多模态。
- **`BaseMultiModalProcessor`**（`vllm/multimodal/processing/processor.py:1178`）：`apply()` 调 HF processor（`_call_hf_processor`）→ `_get_prompt_updates`（把 `<image>` 等占位符替换成 N 个 embed token）→ `_find_mm_placeholders` 生成 `PlaceholderRange`（`vllm/multimodal/inputs.py:122`，记录每个 mm item 在 prompt 中的 offset/length/embeds 区间）。
- **`MultiModalBudget`**（`vllm/multimodal/encoder_budget.py:45`）：计算 encoder 计算预算与缓存大小（`get_encoder_budget = min(compute_budget, cache_size)`），以及每 prompt/每 batch 的 mm item 上限（`mm_max_items_per_prompt`/`mm_max_items_per_batch`）。区分 tower modality（过 encoder）与 embed-only modality（`enable_mm_embeds`，直接传预计算 embedding）。
- **encoder 与 LLM 衔接**（`gpu_model_runner.py`）：
  - scheduler 输出 `scheduled_encoder_inputs`（哪些 mm item 本步要跑 encoder）。
  - `_execute_mm_encoder`（`gpu_model_runner.py:3077`）批量跑 vision encoder，输出存入 `self.encoder_cache[mm_hash]`（`gpu_model_runner.py:623`，按 mm hash 去重，`_cache_encoder_output`）。
  - `_gather_mm_embeddings`（`gpu_model_runner.py:3300`）在 target forward 前，按 `PlaceholderRange` 把 encoder 输出 embed 拼进 `inputs_embeds` 的对应位置（`is_mm_embed` mask）。
  - `reset_encoder_cache`（`gpu_model_runner.py:1023`）清理。
- **媒体处理**：`vllm/multimodal/` 下 `image.py`/`video.py`/`audio.py`（输入解析）、`media/`（IO）、`video_decoders/`、`video_prune/`（视频抽帧/剪枝）、`hasher.py`（mm hash 去重）、`cache.py`（processor 缓存）。
- **多模态 + 投机解码**：`SpecDecodeBaseProposer` 支持 mm 输入（`supports_mm_inputs`），text-only draft 模型会告警并退回纯文本 draft（`llm_base_proposer.py:1366`）。

### 5.2 配置/调优旋钮
<!-- tags: multimodal, 配置, 旋钮, limit-mm, embeds -->

`MultiModalConfig`（`vllm/config/multimodal.py:98`）/ CLI：
- `--limit-mm-per-prompt`：每 prompt 各模态 item 上限，支持 `{"image": 16, "video": {"count":1,"num_frames":32,"width":512,"height":512}}`。
- `--enable-mm-embeds`：允许直接传预计算 embedding（`*_embeds`），可省 encoder 显存。
- `--mm-processor-cache-gb`（默认 4）/ `--mm-processor-cache-type`：processor 缓存（每 API/engine 进程各一份）。
- `--media-io-kwargs`：如 `{"video": {"num_frames": 40}}`。
- `--mm-processor-kwargs`：透传给 HF processor（如 `{"num_crops": 4}`）。
- `language_model_only`：禁用所有 mm 输入。

---

## 6. Reasoning / 思考模型支持
<!-- tags: reasoning, thinking, reasoning-parser, 思考模型 -->

### 6.1 架构
<!-- tags: reasoning, 架构, parser, manager, thinking-budget -->

- **`ReasoningParser`**（`vllm/reasoning/abs_reasoning_parsers.py:26`）：抽象基类，负责识别思考段（`reasoning_start_str`/`reasoning_end_str`）、`is_reasoning_end`/`is_reasoning_end_streaming`（流式检测思考结束）、`extract_reasoning`/`extract_reasoning_streaming`（把输出拆成 reasoning + content，供 OpenAI API 的 `reasoning` 字段）。
- **`ReasoningParserManager`**（`abs_reasoning_parsers.py:213`）：注册表，`register_lazy_module` 懒加载。`vllm/reasoning/__init__.py` 的 `_REASONING_PARSERS_TO_REGISTER` 列出内置 parser：`deepseek_r1`、`deepseek_v3/v4`、`qwen3`、`kimi_k2/k3`、`glm45/glm47`、`minimax_m2/m3`、`mistral`、`gemma4`、`step3/step3p5`、`nemotron_v3`、`olmo3`、`granite`、`hunyuan_a13b`、`cohere_command3/4`、`openai_gptoss`、`inkling`、`muse_glimmer` 等（30+ 个，`__init__.py:22-147`）。
- **与结构化输出集成**：`StructuredOutputManager` 持有 `reasoner_cls`（`structured_output/__init__.py:44`），在思考阶段跳过 grammar 约束，思考结束后启用（见 §3.3）。
- **thinking token budget**：`ThinkingBudgetStateHolder`（`vllm/v1/sample/thinking_budget_state.py:34`）在采样时跟踪思考 token 数，超 `thinking_token_budget` 时强制插入 `reasoning_end` token。由 `ReasoningConfig`（`vllm/config/reasoning.py:13`，`reasoning_start_str`/`reasoning_end_str`，token id 自动推导）驱动，请求侧 `SamplingParams.thinking_token_budget`（OpenAI chat 协议 `thinking_token_budget`，`chat_completion/protocol.py:258`）。

### 6.2 配置/调优旋钮
<!-- tags: reasoning, 配置, 旋钮, parser, budget -->

- `--reasoning-parser`：选 parser 名（如 `deepseek_r1`、`qwen3`）。
- `--reasoning-parser-plugin`：动态加载自定义 parser 插件。
- `--reasoning-config`（`arg_utils.py:1662`）：`reasoning_start_str`/`reasoning_end_str`（强制结束思考的字符串）。
- `--structured-outputs-config` 里的 `enable_in_reasoning`：思考阶段是否也施加结构化约束。
- 请求级 `thinking_token_budget`：限制思考 token 数。

---

## 7. 关键文件清单
<!-- tags: files -->

**投机解码**
- `vllm/config/speculative.py` — `SpeculativeConfig`（method 探测、MTP hf_config 归一化、字段校验）
- `vllm/v1/spec_decode/llm_base_proposer.py` — `SpecDecodeBaseProposer`（多步 draft 主循环、embedding/lm_head 共享）
- `vllm/v1/spec_decode/{eagle,draft_model,medusa,dflash,step3p5,gemma4,extract_hidden_states}.py` — 各 proposer
- `vllm/v1/spec_decode/{ngram_proposer,ngram_proposer_gpu,suffix_decoding,custom_class_proposer}.py` — 非模型 draft
- `vllm/v1/spec_decode/metadata.py` — `SpecDecodeMetadata`
- `vllm/v1/spec_decode/vocab_mapping.py` — 异构词表 TLI 映射
- `vllm/v1/spec_decode/metrics.py` — 接受率指标
- `vllm/v1/spec_decode/dynamic/utils.py` — 动态 K 查表
- `vllm/v1/sample/rejection_sampler.py` — `RejectionSampler` + triton kernel
- `vllm/v1/worker/gpu_model_runner.py` — drafter 分派、`sample_tokens`、`propose_draft_token_ids`
- `vllm/v1/core/sched/scheduler.py` — `scheduled_spec_decode_tokens`、`update_draft_token_ids`

**采样**
- `vllm/v1/sample/sampler.py` — `Sampler`
- `vllm/v1/sample/ops/{topk_topp_sampler,penalties,bad_words,logprobs}.py` — 采样 kernel
- `vllm/v1/sample/logits_processor/{builtin,interface,state}.py` — logits processors
- `vllm/v1/sample/metadata.py` — `SamplingMetadata`
- `vllm/v1/sample/thinking_budget_state.py` — thinking budget
- `vllm/sampling_params.py` — `SamplingParams` / `StructuredOutputsParams`

**结构化输出**
- `vllm/v1/structured_output/__init__.py` — `StructuredOutputManager`
- `vllm/v1/structured_output/backend_{xgrammar,guidance,outlines,lm_format_enforcer,types}.py`
- `vllm/v1/structured_output/{request,utils}.py` — `StructuredOutputRequest`、`apply_grammar_bitmask`
- `vllm/config/structured_outputs.py` — `StructuredOutputsConfig`

**LoRA**
- `vllm/lora/{worker_manager,model_manager,request}.py`
- `vllm/lora/punica_wrapper/{punica_gpu,punica_base}.py` — Punica kernel 元数据
- `vllm/lora/layers/` — 各 LoRA 层包装
- `vllm/config/lora.py` — `LoRAConfig`

**多模态**
- `vllm/multimodal/registry.py` — `MultiModalRegistry`
- `vllm/multimodal/processing/processor.py` — `BaseMultiModalProcessor`
- `vllm/multimodal/encoder_budget.py` — `MultiModalBudget`
- `vllm/multimodal/inputs.py` — `PlaceholderRange` / `MultiModalFeatureSpec`
- `vllm/config/multimodal.py` — `MultiModalConfig`
- `vllm/v1/worker/gpu_model_runner.py` — `_execute_mm_encoder` / `_gather_mm_embeddings` / `encoder_cache`

**Reasoning**
- `vllm/reasoning/abs_reasoning_parsers.py` — `ReasoningParser` / `ReasoningParserManager`
- `vllm/reasoning/*.py` — 各模型 parser
- `vllm/config/reasoning.py` — `ReasoningConfig`

---

## 8. 部署/调优速查
<!-- tags: deployment, tuning, 速查 -->

| 目标 | 旋钮 |
|---|---|
| 开启投机解码 | `--speculative-config '{"method":..., "model":..., "num_speculative_tokens":K}'` |
| 无权重投机（最省） | `method: ngram`（配 `prompt_lookup_max`）或 `suffix`（需 `arctic-inference`） |
| 提高接受率 | 用 EAGLE/MTP/draft_model；`draft_sample_method: probabilistic`；调 `num_speculative_tokens`（过大接受率下降） |
| 大 batch 降低投机开销 | `num_speculative_tokens_per_batch_size` 动态 K |
| 异构词表 draft | `use_heterogeneous_vocab: true`（仅 draft_model） |
| 结构化输出 | `--structured-outputs-config '{"backend":"xgrammar"}'`（默认 auto）；`VLLM_XGRAMMAR_CACHE_MB` |
| 思考模型 | `--reasoning-parser <name>`；`thinking_token_budget`（请求级）；`enable_in_reasoning` |
| 多 LoRA 并发 | `--enable-lora --max-loras N --max-cpu-loras M --max-lora-rank R` |
| 多模态吞吐 | `--limit-mm-per-prompt`、`--mm-processor-cache-gb`、`--enable-mm-embeds`（省 encoder 显存） |

**注意事项**
- v1 不支持 beam search（仅 V0 遗留字段）。
- 投机解码 + 结构化输出 + reasoning 三者有复杂交互（bitmask 需覆盖每个投机位置、reasoning 结束检测在 draft 窗口内模拟），是 vLLM 较易出 bug 的交叉区域（源码多处注释引用 issue #42452/#43388/#44006）。
- `draft_tensor_parallel_size` 必须等于 target TP（draft_model 路径强制，`draft_model.py:71`）。
- MTP 的 `num_speculative_tokens` 须整除 checkpoint 的 `n_predict`。
