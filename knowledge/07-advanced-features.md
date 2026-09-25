# 投机解码与高级推理特性

> 基于 vLLM main（`afea5c20c7`，2026-09-25），最新 tag **v0.30.1rc0**（`153242a314`，2026-09-23，release candidate，HEAD 领先 147 commits；上一正式 release 为 v0.30.0，`9ed533eb4a`，2026-09-20）（v1 架构）源码分析。路径均相对仓库根 `/Users/baofeng/baofeng/github/vllm`。

本文覆盖 vLLM 的高级特性：**投机解码 (speculative decoding)**、**采样 (sampling)**、**结构化输出 (structured output)**、**LoRA 多适配器**、**多模态 (multimodal)**、**reasoning/思考模型支持**，以及 v0.29 新增的**文本水印 (watermarking)** 与 **Engram/PLE（n-gram 嵌入存储）**。重点讲架构与部署/调优旋钮。

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
- drafter 在 `__init__` 里按 `speculative_config.method` 分派（`gpu_model_runner.py:606-645`）。
- `sample_tokens()`（`gpu_model_runner.py:4532`）里：先 `apply_grammar_bitmask`（若有结构化输出）→ `_sample()`（`gpu_model_runner.py:3638`，spec 时走 `rejection_sampler`）→ `propose_draft_token_ids()`（`gpu_model_runner.py:4959`）为下一步准备 draft。
- EAGLE/DraftModel 类 drafter 可以直接消费 GPU 上的 sampled tokens（`use_gpu_toks` 分支，`gpu_model_runner.py:4620`），不必等 bookkeeping；ngram/suffix 类消费 CPU token，在 bookkeeping 之后运行（`draft_after_bookkeeping`，`gpu_model_runner.py:4713`）。

### 1.2 支持的 draft 方式
<!-- tags: draft, method, ngram, eagle, mtp -->

`SpeculativeConfig.method` 的取值（`vllm/config/speculative.py:72-82`，`SpeculativeMethod` Literal）：

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
| `dflash` | DFlash 并行 draft（Qwen3 等，非因果注意力）。**v0.29**：draft 架构为 `DFlash2DraftModel` 时自动切到 DFlash2 speculator（`gpu/spec_decode/dflash2/speculator.py`，selector-walk kernel），无需单独 method。**v0.30**：启用 async scheduling（#58065） | 是 |
| `dspark` | DSpark 块验证 draft | 是 |
| `extract_hidden_states` | 抽取 target 中间层 hidden states 做 draft | 是 |
| `custom_class` | 用户自定义 proposer（`model` 传 `pkg.MyProposer` 点分路径） | 自定义 |

> **MTP 模型类型（v0.29 大幅扩展）**：`MTPModelTypes`（`config/speculative.py:37`）从少数几种扩到 **27 种**，含 `qwen4_exp_mtp`、`qwen3_5_mtp`、`hy_v4_mtp`、`glm5_next_mtp`、`gemma4_mtp`、`kimi_k3_mtp`、`longcat_flash_mtp`、`bailing_hybrid_v3_mtp`、`minimax_m3_mtp`、`inkling_mtp` 等。这些是 `method="mtp"` 下按 checkpoint 架构自动识别的子类型（`hf_config_override` 归一化 `n_predict`）。

分派逻辑（`gpu_model_runner.py:606-645`）：`custom_class` → `create_custom_proposer`；`ngram` → `NgramProposer`；`uses_draft_model()` → `DraftModelProposer`；`use_ngram_gpu()` → `NgramProposerGPU`；`use_gemma4_mtp()` → `Gemma4Proposer`；`use_step3p5_mtp()` → `Step3p5MTPProposer`；`use_dflash()` → `DFlashProposer`；`suffix` → `SuffixDecodingProposer`；`use_eagle()` → `EagleProposer`（eagle3 时 `pass_hidden_states_to_model=True`）；`medusa` → `MedusaProposer`；`extract_hidden_states` → `ExtractHiddenStatesProposer`。

各 proposer 大多继承 `SpecDecodeBaseProposer`（`vllm/v1/spec_decode/llm_base_proposer.py`），其 `propose()`（`llm_base_proposer.py:515`）实现多步自回归 draft：先跑一次 target 对齐的 forward，再循环 `num_speculative_tokens-1` 次采样并推进 positions/slot_mapping。

- **EAGLE**（`vllm/v1/spec_decode/eagle.py:10`）：`EagleProposer(SpecDecodeBaseProposer)`，`pass_hidden_states_to_model=True`，把 target 的 hidden states 喂给 draft 模型。
- **MTP**：`hf_config_override`（`speculative.py:647`）把 DeepSeek-V3/V4、Qwen3-Next、GLM4-MoE、Kimi-K3、MiniMax-M3 等 checkpoint 的 `num_nextn_predict_layers` 归一化成 `n_predict`，`method` 统一为 `mtp`。`num_speculative_tokens` 必须能被 `n_predict` 整除（`speculative.py:1478-1486`）。
- **ngram**（`ngram_proposer.py`）：`prompt_lookup_min`/`prompt_lookup_max` 定义匹配窗口（默认 5/5）；用 KMP 类算法在已生成 token 序列里找最长匹配 n-gram 并取其后 k 个 token。
- **suffix**（`suffix_decoding.py`）：`SuffixDecodingProposer` 包装 `arctic_inference.suffix_decoding.SuffixDecodingCache`，动态决定每请求每步的投机长度（`max_spec_tokens = min(num_speculative_tokens, max_spec_factor * prefix_match_length)`）。

### 1.3 acceptance 与验证流程
<!-- tags: rejection-sampler, acceptance, 验证, 接受率, draft-sampling -->

`RejectionSampler`（`vllm/v1/sample/rejection_sampler.py`）严格实现 arXiv:2211.17192：

- `forward()`（`rejection_sampler.py:106`）：
  1. 从 `bonus_logits` 用普通 `Sampler` 采 bonus token（可带 top_p/top_k）。
  2. 对 `target_logits` 应用 logits processors（penalties / bad words / min_tokens / thinking budget）与 `apply_sampling_constraints`（temperature + top_k/top_p）。
  3. `rejection_sample()`（`vllm/v1/sample/rejection_sampler.py:410`）：greedy 请求走 `rejection_greedy_sample_kernel`；随机请求先 `sample_recovered_tokens` 再 `rejection_random_sample_kernel`。
- 输出 `output_token_ids` 形状 `[batch_size, max_spec_len+1]`，被拒位置填 `PLACEHOLDER_TOKEN_ID=-1`，`parse_output()` 过滤。
- `draft_probs` 可为 None（ngram 无概率分布，退化为确定性接受）。
- `MAX_SPEC_LEN = 128`（`rejection_sampler.py:42`）是单步每请求 draft token 上限。
- **v0.30.1rc0 区间**：异构 vocab spec decode 去掉 CPU-GPU 同步（#57396）——`vocab_mapping.py` 的 `draft_ids[draft_ids == -1] = self.draft_unk_token_id` 替代原 `.any()` 同步检查，消除每步一次 device→host 往返。Kimi-K3 变长 decode（#52988）：MLA/KDA metadata builder 支持 `max_query_len`（见 04 §4.2）。GLM MTP head 延迟加载（#55442）：`deepseek_mtp.py` 新增 `defer_lm_head` 参数，`glm5next/common/mtp.py` 传 `defer_lm_head=True`，MTP head 权重推迟到首次需要时加载，缩短启动时间。

**rejection_sample_method**（`speculative.py:517`）：
- `standard`：概率式拒绝采样（默认）。配合 `draft_sample_method`：
  - `greedy`（默认）：draft 取 argmax，概率视为 one-hot。
  - `probabilistic`：draft 从分布随机采样，用完整 draft logits 做概率比检验（更准但更耗显存）。
- `synthetic`：按 `synthetic_acceptance_rates`（或 `synthetic_acceptance_length`）的衰减概率接受 draft，用于合成/测试。
- `block`：块验证（Sun et al.），把 draft token 作为整体联合验证。

**自适应验证（adaptive verification，v0.30 新增，#52228）**：`enable_adaptive_verification`（`speculative.py:539`，默认 False）开启后，MRV2 用 `OnlineAcceptanceEstimator`（`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:313`，501 行）在线拟合每请求的接受率（log-odds 特征 + 线性模型，Triton kernel 做 accumulate/refit/predict，冷启动系数来自 DeepSeek-V4-Flash 等 5 个模型的离线拟合），据此动态调整每步实际验证的 draft 长度，把算力从低接受率请求上省下来。

**draft 采样**（`llm_base_proposer.py:441-513`，`_greedy_sample`/`_sample_from_logits`/`_sample_draft_tokens`）：`_greedy_sample` 支持 `use_local_argmax_reduction`（vocab-parallel 局部 argmax，把通信从 O(vocab) 降到 O(2*tp)）；`use_heterogeneous_vocab` 时用 `VocabMapping`（`vocab_mapping.py`，TLI 算法）在 draft/target 词表交集上约束 logits 并双向映射 token id。

### 1.4 如何配置
<!-- tags: speculative-config, 配置, cli, 字段, 示例 -->

`SpeculativeConfig`（`vllm/config/speculative.py:375`）主要字段：

- `num_speculative_tokens`：投机 token 数 K。若未给，默认取 draft 模型 config 的 `n_predict`。
- `model`：draft 模型 / eagle head / MTP 权重路径；ngram 时传 `"ngram"`。
- `method`：见上表；给 `model` 时可自动探测（`eagle-`/`eagle3`/`medusa`/`mlp_speculator`/MTP 类型/`dflash`/`dspark`），否则默认 `draft_model`。
- `draft_tensor_parallel_size`：draft 的 TP（须为 1 或与 target 相同，`draft_model.py:71` 强制相同）。
- `quantization` / `kv_cache_dtype` / `moe_backend` / `attention_backend` / `max_model_len`：draft 模型独立配置。**v0.30**：`kv_cache_dtype`/`moe_backend`/`attention_backend` 的覆盖统一收敛到 `SpeculativeConfig.apply_draft_overrides`（`config/speculative.py`，`_DRAFT_VLLM_CONFIG_OVERRIDES` 表）——**仅非 None 时覆盖** target 的 `kernel_config.moe_backend`/`cache_config.cache_dtype`/`attention_config.backend`（否则 draft 继承 target 的 `--moe-backend` 等，draft 未量化时该 backend 可能不可用）；EAGLE/DFlash 等 speculator 的 `load_model` 统一经此 + `get_draft_load_config`（见 03 §2.5）。
- `prompt_lookup_min` / `prompt_lookup_max`：ngram 窗口。
- `parallel_drafting`：并行 draft（EAGLE/draft_model，dflash/dspark 强制开启）。
- `disable_padded_drafter_batch`：禁用 padded drafter batch（仅 EAGLE 系）。
- `use_local_argmax_reduction` / `use_heterogeneous_vocab`：见上。
- `suffix_decoding_max_tree_depth`(24) / `suffix_decoding_max_cached_requests`(10000) / `suffix_decoding_max_spec_factor`(1.0) / `suffix_decoding_min_token_prob`(0.1)。
- `rejection_sample_method` / `synthetic_acceptance_rates` / `synthetic_acceptance_length` / `draft_sample_method` / `dspark_draft_topk` / `enable_adaptive_verification`(仅 dspark)。
- `num_speculative_tokens_per_batch_size`：**动态投机解码**，`[(range_start, range_end, K), ...]`，按运行时 batch size 查表选 K（`vllm/v1/spec_decode/dynamic/utils.py`，scheduler 在 `__init__` 展开成 dense lookup `dynamic_sd_lookup`，`scheduler.py:287-294`，调度时查表 `scheduler.py:1444-1447`）。

CLI（`vllm/engine/arg_utils.py`）：
- `--speculative-config` / `-sc`（JSON dict，最灵活）
- 快捷 flag：`--spec-method`、`--spec-model`、`--spec-tokens`（`arg_utils.py:1725` 附近）

示例：
```bash
# EAGLE
--speculative-config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 5}'
# ngram
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5}'
# DeepSeek MTP（权重在 target checkpoint 内）
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

`VllmConfig.num_speculative_tokens`（`config/vllm.py:613`）与 `num_lookahead_tokens`（`config/vllm.py:627`）供 scheduler 预留 KV slot（drafter 会写 target query 范围之外的 KV，scheduler 必须多预留 `num_lookahead_tokens` 个 slot）。

### 1.5 调度器侧
<!-- tags: scheduler, 调度, spec-tokens, draft-tokens, grammar -->

- scheduler 每步把 draft token 放进 `SchedulerOutput.scheduled_spec_decode_tokens`（`scheduler.py:832`），`update_draft_token_ids`（`scheduler.py:2481`）在下一步前把新 draft 写回 `request.spec_token_ids`；结构化输出请求会用 `grammar.validate_tokens` 过滤非法 draft（`scheduler.py:2526`）。

### 1.6 指标
<!-- tags: metrics, 指标, acceptance-rate, 接受率, per-request -->

`SpecDecodingStats` / `SpecDecodingLogging`（`vllm/v1/spec_decode/metrics.py`）：记录 `num_drafts`、`num_draft_tokens`、`num_accepted_tokens`、per-position 接受率，日志输出 `draft_acceptance_rate` 与 `mean_acceptance_length = 1 + accepted/drafts`。

**per-request 接受率指标（v0.28 新增，#48915）**：除全局日志外，每个请求的接受率统计（`num_draft_tokens`/`num_accepted_tokens`/per-position 接受率）现在会随 OpenAI API 响应返回（`docs/features/speculative_decoding/acceptance_metrics.md`），便于按请求观测投机解码收益；Rust frontend 的 `EngineCoreOutput` 协议也带上了这些字段。

**generate API 暴露 per-request 投机解码指标（v0.30 新增，#43310）**：Rust frontend 的 `/inference/v1/generate`（`GenerateResponse`/`GenerateStreamResponse`，`rust/src/server/src/routes/inference/generate/`）新增 `metrics.speculative_decoding` 字段，由 `RequestSpecDecodeMetrics` 推导：`mean_acceptance_length`（=1+accepted/spec_steps）、`draft_acceptance_rate`、`acceptance_histogram`、`num_spec_steps`/`num_accepted_draft_tokens`/`num_draft_tokens`/`num_spec_tokens`，以及可选的 `per_step_accepted`/`per_step_drafted`（仅 detailed 模式）。流式响应（`StreamingSpeculativeDecodingMetrics`）在请求 summary 指标时省略 detailed 字段。Python `GenerateResponse` 与 scale-out（`token_in_token_out`/`derender`）端点同步透传。

**derender 流式解析文档化（v0.30 新增，#57922）**：`docs/serving/online_serving/derenderer.md` 补齐 scale-out derender 的流式语义——`stream: true` 时客户端携带 `stream_state`（**无状态协议**：server 不存流状态，state 由客户端回传），每次收一个 `GenerateStreamResponse` delta 返回 `{chunk, stream_state}`；chat 端点流式路径支持 reasoning + tool call 解析，产出与 generate 流式路径相同的 `reasoning`/`content`/`tool_calls` delta。配套 `entrypoints/scale_out/token_in_token_out/protocol.py` 的流式 parity 测试。

**v0.30.1rc0 区间**：
- **Granite 流式 tool-call 解析**（#49648）：`vllm/parser/granite.py`（新文件）为 Granite 3.0/3.1 的 JSON-array 格式 tool call 提供流式 Parser Engine 支持（`tool_call_body_array` 模式），`streaming_parser_engine.py` 新增 `_feed_array_text`/`_reset_array_state`；旧的 `tool_parsers/granite_tool_parser.py`（257 行）删除。
- **length finish_reason 流式 tool call 修复**（#46303）：`chat_completion/serving.py` 仅在 `output.finish_reason == "stop"` 时才把 finish_reason 报为 `"tool_calls"`，避免流式中途因达到 `max_tokens` 误报 tool_calls。
- **FIM completion 渲染**（#44229）：`BaseRenderer.render_completion_suffix(prompt, suffix)`（默认返回 None），`DeepseekV4Renderer` 实现为 `<｜fim▁begin｜>{prompt}<｜fim▁hole｜>{suffix}<｜fim▁end｜>`；online_renderer 在非 echo/prompt_embeds/truncate_prompt_tokens 时接受 suffix。

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

`TopKTopPSampler`（`vllm/v1/sample/ops/topk_topp_sampler.py:192`）按平台选实现：CUDA 优先 FlashInfer（`forward_cuda`），否则 native；CPU/XPU 各有 kernel。**v0.30 新增 XPU fused sampler kernel**（#57277）：`xpu_sampler_supported()`（`:129`）+ `xpu_sample()`（`:151`），`VLLM_XPU_USE_SAMPLER_KERNEL=1`（默认开）时 XPU 走 fused kernel（不排序 vocab、不物化概率张量，恒随机采样；per-request seed/greedy 请求不可用），MRV2 sampler 同步接入（见 06 §XPU）。

**Logits processors**（`vllm/v1/sample/logits_processor/`）：`MinPLogitsProcessor`、`LogitBiasLogitsProcessor`、`MinTokensLogitsProcessor`（`builtin.py`），通过 `is_argmax_invariant()` 区分是否影响 greedy。`SamplingMetadata`（`vllm/v1/sample/metadata.py`）携带每 batch 的 temperature/top_p/top_k/penalties/generators/`spec_token_ids`/`thinking_budget_state_holder`。

**自定义 logits processors（v0.30 起 V2 也支持，#56497）**：用户自定义 processor 经 `model_config.logits_processors`（类路径列表）或 `vllm.logits_processors` entry-point 插件注册。V1 走 `vllm/v1/sample/logits_processor/`；**Model Runner V2** 走新增的 `vllm/v1/worker/gpu/sample/logits_processor/`（`interface.py` 定义 `LogitsProcessor` 协议 + `is_argmax_invariant`，`loader.py` 负责按 runner 加载类并构建 per-request 参数校验器）。参数校验从 `SamplingParams._validate_logits_processors` 上移到 `InputProcessor`（`vllm/v1/engine/input_processor.py`，准入时按 `use_v2_model_runner` 选 validator），`_get_v2_model_runner_unsupported_features` 不再把 "custom logits processors" 列为 V2 不支持项。

### 2.2 SamplingParams 字段
<!-- tags: sampling-params, 字段, 采样参数, beam-search -->

`SamplingParams`（`vllm/sampling_params.py:211`）关键字段：
- `n`(1)、`presence_penalty`(0)、`frequency_penalty`(0)、`repetition_penalty`(1.0)、`temperature`(1.0)、`top_p`(1.0)、`top_k`(0=禁用)、`min_p`(0.0)、`seed`、`stop`/`stop_token_ids`、`ignore_eos`、`max_tokens`(16)、`min_tokens`(0)。**v0.30 安全校验（#57731）**：`max_tokens` 未设置时按 `max_model_len - prompt_len` 填充，`InputProcessor` 随后校验 `min_tokens <= max_tokens`（此前 `min_tokens` 在 `max_tokens` 未设时不检查，可超过实际可生成上限）。
- logprobs：`logprobs`、`prompt_logprobs`、`logprob_token_ids`（generative_scoring 用，只取指定 token 的 logprob）、`flat_logprobs`。
- `structured_outputs`（`StructuredOutputsParams`）、`logit_bias`、`allowed_token_ids`、`bad_words`、`thinking_token_budget`、`repetition_detection`。

**beam search**：`SamplingParams` 仍保留 `beam_width` 字段（`sampling_params.py:1319`，"not supported by OpenAI"），但 v1 引擎没有 beam search 实现（`vllm/v1/` 下无 `beam_search` 代码）——beam search 属于已废弃的 V0 路径，v1 不支持。

---

## 3. 结构化输出 (Structured Output / Guided Decoding)
<!-- tags: structured-output, guided-decoding, json-schema, xgrammar, outlines, 结构化输出 -->

### 3.1 架构
<!-- tags: structured-output, 架构, xgrammar, bitmask, fsm -->

`StructuredOutputManager`（`vllm/v1/structured_output/__init__.py:36`）是 engine 级单例，管理一个 backend（V1 不支持 per-request 切换 backend）。

- **backend 选择**：`StructuredOutputsConfig.backend`（`config/structured_outputs.py:21`）默认 `"auto"`。`auto` 模式在 `sampling_params.py:1201-1246`（`_validate_structured_outputs` 的 auto 分支）里按优先级尝试：先 `xgrammar`，失败则 `guidance`（Mistral 非 tekken tokenizer 或 schema 含 guidance 不支持特性时退到 `outlines`）。
- **backend 实现**：
  - `XgrammarBackend`（`backend_xgrammar.py:36`）：默认。`xgr.GrammarCompiler` + `GrammarMatcher`，`compile_grammar`（`:78`）支持 `JSON`/`JSON_OBJECT`/`GRAMMAR`/`REGEX`/`STRUCTURAL_TAG`；2026-09-25 窗口起 **Lark grammar 原生解析**（`1f0bc49ee6` #58321，不再经 EBNF 转换）。
  - `GuidanceBackend`（`backend_guidance.py`）、`OutlinesBackend`（`backend_outlines.py`，SQLite 磁盘缓存 `OUTLINES_CACHE_DIR`）、`LMFormatEnforcerBackend`（`backend_lm_format_enforcer.py`）。
- **约束机制 = bitmask**：每个请求的 grammar 是一个 FSM。每步 `grammar_bitmask()`（`__init__.py:314`）为每个待采样位置（含每个投机位置 + bonus）填一个 token bitmask（`grammar.fill_bitmask`），大 batch 时用线程池并行填（`fill_bitmask_parallel_threshold=128`）。GPU 侧 `apply_grammar_bitmask`（`vllm/v1/structured_output/utils.py:101`）把 bitmask 重排到与 batch 对齐，再 `xgr.apply_token_bitmask_inplace` 把非法 token 的 logit 置 -inf。
- **FSM 推进**：`XgrammarGrammar.accept_tokens`（`backend_xgrammar.py:159`）逐 token 推进并检测终止；`validate_tokens`（`backend_xgrammar.py:183`）预校验 draft token（不推进，用于投机解码过滤）；`rollback` 支持回滚（`max_rollback_tokens=num_speculative_tokens`）。
- **v0.30 增量**：xgrammar JSON Schema feature gating 修复（#48416，`backend_xgrammar.py`）：`_schema_types()`（`:249`）把 **list-valued `"type"`**（如 `"type": ["string", "null"]`）归一化成 `set[str]`，`has_xgrammar_unsupported_json_features`（`:259`）据此判断——此前 list 型 type 被误判/漏判导致 schema 支持性检查出错（issue #57550 的相关修复，FIXME 标注该方案待重设计）。

### 3.2 请求侧
<!-- tags: structured-output, 请求, request, key, 异步编译 -->

`StructuredOutputRequest`（`vllm/v1/structured_output/request.py:22`）由 `SamplingParams.structured_outputs` 构造，`structured_output_key` 区分类型（`get_structured_output_key`，`request.py:76`：JSON / JSON_OBJECT / REGEX / CHOICE / GRAMMAR / STRUCTURAL_TAG）。grammar 编译是异步的（`Future`），`is_grammar_ready` 轮询。**v0.30**：`_check_grammar_completion` 的 poll 改**非阻塞**（#55931）——`Future.done()` 检查替代 `result(timeout=0.0001)`（100µs 阻塞轮询在调度热路径上累积）。请求侧校验（`sampling_params.py`）：**拒绝空 `structural_tag`**（#47450，strip 后为空即 `VLLMValidationError`）；空 regex 仍合法（可编译成有效 grammar）。**v0.30.1rc0 区间**：DiffusionGemma 结构化生成（#57250，Jev-like）——扩散式 LM 也支持 grammar 约束：`vllm/utils/diffusion.py` 新增 `validate_diffusion_sampling_params`（canvas_length/vocab_size/async_scheduling 校验），`InputProcessor` 构造时加载 `diffusion_config` 并在准入时做 `is_diffusion` 检查；`diffusion_config` + `async_scheduling` + 未显式指定 `scheduler_cls` 时自动选 `DiffusionAsyncScheduler`（`vllm/v1/core/sched/diffusion_scheduler.py`，继承 `AsyncScheduler`，新增 `diffusion_canvas_width()`/`_read_in_flight()`）。

### 3.3 与投机解码 / reasoning 的交互
<!-- tags: structured-output, 投机解码, reasoning, 交互, bitmask -->

- 投机解码时 bitmask 要为每个投机位置 + bonus 各生成一行（`__init__.py:338-427`），并模拟 reasoning-end 检测。
- reasoning 模型：约束起点由 `_get_constraint_start`（`__init__.py:220`）决定——`enable_in_reasoning=True` 时从第 0 个 token 就约束；否则思考阶段不约束，检测到 reasoning 结束（`reasoning_ended`）才启用 bitmask；`validate_tokens`（`__init__.py:294`）对 draft 做同样处理（返回最长未约束/合法前缀）。

### 3.4 配置/调优旋钮
<!-- tags: structured-output, 配置, 旋钮, xgrammar-cache, reasoning -->

- `--structured-outputs-config`（JSON，含 `backend`/`disable_any_whitespace`/`disable_additional_properties`/`reasoning_parser`/`reasoning_parser_plugin`/`enable_in_reasoning`）。
- `--reasoning-parser`、`--reasoning-parser-plugin`（`arg_utils.py:1061` 附近）。
- 环境变量 `VLLM_XGRAMMAR_CACHE_MB`（默认 512，`envs.py:1644`）控制 xgrammar 编译缓存；`OUTLINES_CACHE_DIR` 控制 outlines 磁盘缓存。
- 请求级：`response_format`（OpenAI）/ `structured_outputs`（`json`/`regex`/`choice`/`grammar`/`json_object`/`structural_tag`）。

---

## 4. LoRA / 多适配器
<!-- tags: lora, adapter, 适配器 -->

### 4.1 架构
<!-- tags: lora, 架构, punica, worker-manager, 多模态 -->

- **`WorkerLoRAManager`**（`vllm/lora/worker_manager.py:27`）：worker 侧管理。每个请求的 LoRA 按需加载（`_load_adapter`），其余卸载。
- **`LoRAModelManager`**（`vllm/lora/model_manager.py:77`）：核心。持有 `PunicaWrapper`（`punica_wrapper/`，基于 Punica 论文 arXiv:2310.18547 的 triton kernel `lora_shrink`/`lora_expand`），维护 `lora_slots`/`adapter_slots`，`activate_adapter`/`_deactivate_adapter` 做 slot 分配，`AdapterLRUCache` 做 LRU 卸载。`create_lora_manager` 把 base model 的各 linear/embedding 层包装成 `BaseLayerWithLoRA`（`vllm/lora/layers/`：`column_parallel_linear`、`row_parallel_linear`、`replicated_linear`、`vocal_parallel_embedding`、`fused_moe` 等）。
- **`PunicaWrapperGPU`**（`vllm/lora/punica_wrapper/punica_gpu.py:33`）：维护 `LoRAKernelMeta`（token→lora 映射），`update_metadata` 每步更新。投机解码时 `max_num_samples = max_batches*(num_spec_tokens+1)`。
- **`LoRARequest`**（`vllm/lora/request.py:8`）：`lora_name`/`lora_int_id`/`lora_path`/`load_inplace`/`is_3d_lora_weight`。按 `lora_name` 判等/哈希。
- **多模态 LoRA**：`LoRAConfig.default_mm_loras`（modality→path 映射）、`enable_tower_connector_lora`（对 vision tower/connector 加 LoRA，实验性，仅部分 Qwen-VL）。

### 4.2 配置/调优旋钮
<!-- tags: lora, 配置, 旋钮, max-loras, max-rank -->

`LoRAConfig`（`vllm/config/lora.py:32`）/ CLI（`arg_utils.py:1514` 附近）：
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

- **`MultiModalRegistry`**（`vllm/multimodal/registry.py:89`）：注册每模型的 processor（`register_processor` 装饰器）、`ProcessingInfo`、`DummyInputsBuilder`。**v0.30 瘦身（#57913/#57967）**：`supports_multimodal_inputs` 与 receiver cache 工厂从 registry 迁出——前者移到 `ModelConfig.supports_multimodal_inputs` **缓存属性**（`config/model.py`，按 `is_multimodal_model`/`runner_type`/mm limits 缓存；multimodal draft 模型如 Qwen3_5MTP 声明 `SupportsMultiModal` 但无 processor，`runner_type="draft"` 时静默退回 text-only），后者迁到 `vllm/multimodal/cache/factories.py`（`worker_receiver_cache_from_config` 等）。
- **`BaseMultiModalProcessor`**（`vllm/multimodal/processing/processor.py:1031`）：`apply()` 调 HF processor（`_call_hf_processor`）→ `_get_prompt_updates`（把 `<image>` 等占位符替换成 N 个 embed token）→ `_find_mm_placeholders` 生成 `PlaceholderRange`（`vllm/multimodal/inputs.py:122`，记录每个 mm item 在 prompt 中的 offset/length/embeds 区间）。
- **`MultiModalBudget`**（`vllm/multimodal/encoder_budget.py:40`）：计算 encoder 计算预算与缓存大小（`get_encoder_budget = min(compute_budget, cache_size)`），以及每 prompt/每 batch 的 mm item 上限（`mm_max_items_per_prompt`/`mm_max_items_per_batch`）。区分 tower modality（过 encoder）与 embed-only modality（`enable_mm_embeds`，直接传预计算 embedding）。
- **encoder 与 LLM 衔接**（`gpu_model_runner.py`）：
  - scheduler 输出 `scheduled_encoder_inputs`（哪些 mm item 本步要跑 encoder）。
  - `_execute_mm_encoder`（`gpu_model_runner.py:2960`）批量跑 vision encoder，输出存入 `self.encoder_cache[mm_hash]`（`gpu_model_runner.py:595`，按 mm hash 去重，`_cache_encoder_output`）。
  - `_gather_mm_embeddings`（`gpu_model_runner.py:3183`）在 target forward 前，按 `PlaceholderRange` 把 encoder 输出 embed 拼进 `inputs_embeds` 的对应位置（`is_mm_embed` mask）。
  - `reset_encoder_cache`（`gpu_model_runner.py:994`）清理。
- **媒体处理**：`vllm/multimodal/` 下 `image.py`/`video.py`/`audio.py`（输入解析）、`media/`（IO）、`video_decoders/`、`video_prune/`（视频抽帧/剪枝）、`hasher.py`（mm hash 去重）、`cache/`（**v0.30 重构**：processor/receiver cache 从单文件 `cache.py` 拆成包——`base.py`（`BaseMultiModalProcessorCache`/`BaseMultiModalReceiverCache` 抽象）、`lru.py`（`LruKeyReplicated{Sender,Receiver}Cache`）、`shm.py`（`ShmObjectStore{Sender,Receiver}Cache`）、`factories.py`（按 config 选实现的工厂），`WorkerWrapperBase` 经 `worker_receiver_cache_from_config` 构造，`shared_worker_lock` 缺失不再告警）。
- **v0.30 增量（多模态 bugfix/安全）**：
  - **Receiver cache 安全语义**（#57833，`multimodal/cache.py`，`MultiModalReceiverCache.update_receiver_cache_item` ~`:710-734`）：P0（API server 侧）miss 时可能以**同一 mm_hash 重发不同 payload**（独立 LRU 驱逐会让 P1/EngineCore 侧残留旧 tensor）。此前会把旧 tensor 顶替新 payload，导致 placeholder 与错误 item 配对、EngineCore 崩溃。现在**优先采用新 payload** 并保留供后续命中（`get_and_update_item`/`get_and_update_features` 语义更新）。
  - **模型预处理容错**：Molmo2 容忍畸形 EXIF（#57234，`models/molmo2.py`）；Whisper 30s 音频上限处理（`models/whisper.py`）；Mistral3 图像 grid 修复（`models/mistral3.py`）；DiffusionGemma 多模态修复（`models/diffusion_gemma.py`）。
- **多模态 + 投机解码**：`SpecDecodeBaseProposer` 支持 mm 输入（`supports_mm_inputs`），text-only draft 模型会告警并退回纯文本 draft（`llm_base_proposer.py:1346-1353`）。

### 5.2 配置/调优旋钮
<!-- tags: multimodal, 配置, 旋钮, limit-mm, embeds -->

`MultiModalConfig`（`vllm/config/multimodal.py:121`）/ CLI：
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

- **`ReasoningParser`**（`vllm/reasoning/abs_reasoning_parsers.py:25`）：抽象基类，负责识别思考段（`reasoning_start_str`/`reasoning_end_str`）、`is_reasoning_end`/`is_reasoning_end_streaming`（流式检测思考结束）、`extract_reasoning`/`extract_reasoning_streaming`（把输出拆成 reasoning + content，供 OpenAI API 的 `reasoning` 字段）。
- **`ReasoningParserManager`**（`abs_reasoning_parsers.py:216`）：注册表，`register_lazy_module` 懒加载。`vllm/reasoning/__init__.py` 的 `_REASONING_PARSERS_TO_REGISTER` 列出内置 parser：`deepseek_r1`、`deepseek_v3/v4`、`qwen3`、`kimi_k2/k3`、`glm45/glm47`、`minimax_m2/m3`、`mistral`、`gemma4`、`step3/step3p5`、`nemotron_v3`、`olmo3`、`granite`、`hunyuan_a13b`、`cohere_command3/4`、`openai_gptoss`、`inkling`、`muse_glimmer` 等（30+ 个，`__init__.py:22-159`）。
- **与结构化输出集成**：`StructuredOutputManager` 持有 `reasoner_cls`（`structured_output/__init__.py:44`），在思考阶段跳过 grammar 约束，思考结束后启用（见 §3.3）。
- **thinking token budget**：`ThinkingBudgetStateHolder`（`vllm/v1/sample/thinking_budget_state.py:34`）在采样时跟踪思考 token 数，超 `thinking_token_budget` 时强制插入 `reasoning_end` token。由 `ReasoningConfig`（`vllm/config/reasoning.py:13`，`reasoning_start_str`/`reasoning_end_str`，token id 自动推导）驱动，请求侧 `SamplingParams.thinking_token_budget`（OpenAI chat 协议 `thinking_token_budget`，`chat_completion/protocol.py:258`）。

### 6.2 配置/调优旋钮
<!-- tags: reasoning, 配置, 旋钮, parser, budget -->

- `--reasoning-parser`：选 parser 名（如 `deepseek_r1`、`qwen3`）。
- `--reasoning-parser-plugin`：动态加载自定义 parser 插件。
- `--reasoning-config`（`arg_utils.py:1753`）：`reasoning_start_str`/`reasoning_end_str`（强制结束思考的字符串）。
- `--structured-outputs-config` 里的 `enable_in_reasoning`：思考阶段是否也施加结构化约束。
- 请求级 `thinking_token_budget`：限制思考 token 数。

---

## 7. 文本水印与 Engram（v0.29 新增）
<!-- tags: watermarking, engram, gumbel, philox, prf, 水印, n-gram, 嵌入 -->

### 7.1 文本水印（Watermarking）
<!-- tags: watermarking, gumbel, philox, prf, 水印, 生成水印 -->

`vllm/v1/watermarking/`：在**采样阶段**给生成文本打可检测水印（Gumbel-max 算法 + Philox 伪随机函数 PRF）。

- **Watermarker**（`watermarker.py`）：采样时对每个 token 的 logits 按 PRF 生成的 per-context 偏置做 Gumbel-max 扰动，使"绿名单"token 更易被选中。`GumbelWatermarker`（`gumbel.py`）+ `PhiloxPRF`（`prfs/philox.py`，64-bit key，`context_width` 个前序 token 作 context）。
- **WatermarkDetector**（`detector.py`）：对已生成 token 序列做统计检测，产出 `WatermarkDetection`（`score`/`p_value`/`is_watermarked`），供下游验证文本是否带水印。
- **配置**：`WatermarkConfig`（`vllm/config/watermarking.py:28`，`key`/`algorithm`/`alpha=0.1`/`context_width=4`/`deduplicate_contexts="single_turn"`/`prf="philox"`/`allow_target_only_watermarking=False`），CLI `--watermark-config`（`arg_utils.py:1731`，`create_watermark_config` :2041）。`VllmConfig.watermark_config`（`config/vllm.py:392`）。
- **两种算法**（`WatermarkingAlgorithm`，`config/watermarking.py:15`）：
  - `gumbel`（默认）：单密钥 Gumbel-max，**不支持投机解码**（`supports_speculative_decoding` 属性为 `False`）。
  - `dual_key_gumbel`（v0.30 新增）：**双密钥** Gumbel-max，target 与 draft 各用一个密钥角色，`supports_speculative_decoding=True`；`alpha` 控制选 key B 的概率，检测用加权 early fusion（`gumbel.py:208`）。
- **投机解码支持（v0.30 新增，#56122）**：`spec_decode.py` 提供 `create_speculative_target_watermarker`/`create_speculative_draft_watermarker`，把 target/draft 拆成两个 watermarker 角色；`allow_target_only_watermarking=True` 时允许 draft token 不打水印。`_check_watermarking_unsupported`（`config/vllm.py:1219`）现在**允许**投机解码，但要求：`draft_sample_method='probabilistic'`、`rejection_sample_method='standard'`、method ∈ {`dspark`,`eagle`,`eagle3`,`mtp`}（自回归模型类），且非 `dspark` 时不允许 parallel drafting；否则（`gumbel` 算法且未开 `allow_target_only_watermarking`）仍报错。也不支持 beam search。**强制 Model Runner V2**。
- 采样侧实现在 `vllm/v1/worker/gpu/sample/watermark.py`（V2 runner 的 sample 子模块）；`vllm/v1/watermarking/gpu_sampler.py` 提供 GPU 采样器封装。

### 7.2 Engram / PLE（n-gram 嵌入存储与分片）
<!-- tags: engram, ple, n-gram, embedding, 嵌入, etp, 分片 -->

`EngramConfig`（`vllm/config/engram.py:41`，CLI `--engram-config`）：为带 n-gram 层的模型（DeepSeek-V4.1、Qwen4-Exp 的 PLE 层）配置 **n-gram 嵌入表的存储与分片**。

- `cpu_offload`（默认 `True`；**v0.30**：legacy `VLLM_PLE_CPU_OFFLOAD` 环境变量已移除，固定默认开启）：嵌入表是否 offload 到 CPU（经 UVA 在独立 CUDA stream 上按需取行）。
- 分片：默认每个 DP rank 持独立 TP 分片副本；`embedding_across_dp=True` 时跨 TP+DP rank 共享一张嵌入表，用 **ETP 进程组**（`get_etp_group()`，见 05 §3.3）把单张表切到多 rank；`dp_shared_memory=True`（需 `cpu_offload=True`）让同节点 DP 副本共享 CPU 侧嵌入表（每节点每 TP shard 只存一份，省 host 内存，需足够 `/dev/shm`）。
- **v0.30 新增**：offloaded engram lookup 的**异步 prefetch** + engram **DP sharding**（#56512，DSv4.1），lookup 与 forward 重叠，降低 CPU offload 延迟。
- 架构映射（`_NGRAM_LAYER_FIELDS`）：`DeepseekV41ForCausalLM→engram_layer_ids`、`Qwen4ExpForCausalLM/ForConditionalGeneration→ple_layer_ids`。
- `VllmConfig.engram_config`（`config/vllm.py:382`）。
- **v0.30 增量**：Engram **DP shared memory 默认开启**（#57651，`config/engram.py`）：`dp_shared_memory` 字段（`:52`）+ `resolve_dp_shared_memory`（`:86`）——同机 co-located 的多个 DP replica 默认共享 n-gram 嵌入表的 host 内存表（`models/deepseek_v41/nvidia/engram.py` 配套），避免每 replica 各占一份大表；ROCm 上 Engram 表留 host 内存（#57491，上一轮已记录）。
- **v0.30.1rc0 区间**：`/dev/shm` 回退（#57914）——`models/deepseek_v41/nvidia/engram.py` 在共享表前检查 `os.path.isdir(SHM_PATH)`，`/dev/shm` 不可用时回退 per-rank 独立表，不再 crash。

---

## 8. 关键文件清单
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
- `vllm/v1/sample/logits_processor/{builtin,interface,state}.py` — logits processors（V1）
- `vllm/v1/worker/gpu/sample/logits_processor/{interface,loader}.py` — 自定义 logits processors（Model Runner V2，v0.30 新增 #56497）
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
- `vllm/multimodal/registry.py` — `MultiModalRegistry`（v0.30 瘦身：`supports_multimodal_inputs` 与 receiver cache 工厂迁出）
- `vllm/multimodal/cache/` — **v0.30 重构**：`base.py`/`lru.py`/`shm.py`/`factories.py`（processor/receiver cache 包）
- `vllm/multimodal/processing/processor.py` — `BaseMultiModalProcessor`
- `vllm/multimodal/encoder_budget.py` — `MultiModalBudget`
- `vllm/multimodal/inputs.py` — `PlaceholderRange` / `MultiModalFeatureSpec`
- `vllm/config/multimodal.py` — `MultiModalConfig`
- `vllm/v1/worker/gpu_model_runner.py` — `_execute_mm_encoder` / `_gather_mm_embeddings` / `encoder_cache`

**Reasoning**
- `vllm/reasoning/abs_reasoning_parsers.py` — `ReasoningParser` / `ReasoningParserManager`
- `vllm/reasoning/*.py` — 各模型 parser
- `vllm/config/reasoning.py` — `ReasoningConfig`

**水印 / Engram（v0.29）**
- `vllm/v1/watermarking/{watermarker,gumbel,detector,factory}.py` + `prfs/philox.py` — 水印生成/检测
- `vllm/config/watermarking.py` — `WatermarkConfig`
- `vllm/v1/worker/gpu/sample/watermark.py` — V2 runner 采样侧水印
- `vllm/config/engram.py` — `EngramConfig`（n-gram 嵌入存储/分片）

---

## 9. 部署/调优速查
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
