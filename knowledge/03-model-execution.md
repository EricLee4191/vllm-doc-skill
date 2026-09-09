# 模型执行与编译优化

> 版本：基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（v1 引擎为默认 active engine）。本文聚焦**架构**与**部署/调优**，不逐行注释。
> 路径均相对仓库根 `/Users/baofeng/baofeng/github/vllm`。

vLLM 的"模型执行层"由三层组成：
1. **ModelRunner**（`vllm/v1/worker/gpu_model_runner.py`）——把调度器产出的 `SchedulerOutput` 组织成 forward 输入、执行模型 forward、产出 logits/sampled tokens。
2. **model_executor**（`vllm/model_executor/`）——模型定义、算子层、权重加载、kernel。
2. **compilation**（`vllm/compilation/`）——torch.compile 集成、piecewise 编译 + CUDA Graph。
3. **Worker**（`vllm/v1/worker/gpu_worker.py`）——设备初始化、显存 profiling、warmup/capture 的编排入口。

数据流：`SchedulerOutput → GPUModelRunner.execute_model() → _prepare_inputs() → _model_forward() → compute_logits() → sample_tokens()`。

---

## 1. ModelRunner 的职责（GPUModelRunner）
<!-- tags: modelrunner, gpu-model-runner, execute-model, step -->

核心类 `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py:501`，约 8000 行）。它持有：
- `self.model`（`nn.Module`，由 `load_model` 加载并可能包上 CUDA Graph wrapper）
- `self.input_batch`（`InputBatch`，`vllm/v1/worker/gpu_input_batch.py:92`）——**持久化**的批状态（token ids、positions、block table、num_computed_tokens 等），跨 step 复用，避免每步重建
- `self.sampler`、`self.drafter`（spec decode）、`self.kv_caches`、`self.attn_groups`

### 1.1 一次 step 的主流程（`execute_model`，:4288）
<!-- tags: execute-model, step, prepare-inputs, forward-context, 主流程 -->

1. `_update_states(scheduler_output)`（:1246）：把本步新增/被抢占/被移除的 request 同步进 `input_batch`（增删行、更新 token ids、num_computed_tokens）。
2. `_prepare_inputs(scheduler_output, num_scheduled_tokens)`（:2019）——**核心**，把一批 sequence 摊平成 forward 输入：
   - `block_table.commit_block_num_reqs` 先异步拷贝 block table（与后续 CPU 计算重叠）。
   - `req_indices = np.repeat(arange(num_reqs)` 按每请求的 scheduled token 数重复 → 把"请求级"展开成"token 级"（如 `[2,5,3] → [0,0,1,1,1,1,1,2,2,2]`）。
   - `positions = num_computed_tokens[req_indices] + query_pos`（每个 token 的绝对位置）。
   - `token_indices = positions + req_indices * max_model_len`，用 `torch.index_select` 从 `input_batch.token_ids_cpu_tensor` 取出 `input_ids`（:2079）。
   - 构建 `query_start_loc`（各请求 query 的累计起点，供 FlashAttention 用，:2133）。
   - 在 GPU 上重算 `positions`、`seq_lens`（:2248-2255）。
   - `block_table.compute_slot_mapping(...)`（:2257）算出每个 token 写入 KV cache 的 `slot_mapping`。
   - 返回 `logits_indices`（无 spec decode 时 = `query_start_loc[1:]-1`，即每请求最后一个 token；:2309）。
3. `_build_attention_metadata(...)`（:2355）构建 attention metadata（含 block table、slot mapping、seq lens、cascade attn 等）。
3.5 `_determine_batch_execution_and_padding`（:4054）：判定本步的 **CUDA Graph 运行模式**（FULL/PIECEWISE/NONE）、padding 后的 `num_tokens`、是否 microbatch（DBO）、DP 协调。
4. `_preprocess`（:3612）把 `input_ids/inputs_embeds/positions` 等整理成 padded 张量。
5. `set_forward_context(attn_metadata, ..., cudagraph_runtime_mode, batch_descriptor, slot_mapping)`（:4546）——**关键**：把 attention metadata、slot mapping、cudagraph 模式塞进 thread-local 的 `ForwardContext`（`vllm/forward_context.py:132`），模型内部的 `Attention` 层通过 `get_forward_context()` 读取。
6. `_model_forward(input_ids, positions, ...)`（:3956）→ `self.model(...)` 得到 `hidden_states`。
7. `sample_hidden_states = hidden_states[logits_indices]`；`logits = self.model.compute_logits(sample_hidden_states)`（:4598-4599）。
8. 结果存入 `self.execute_model_state`，`execute_model` 返回 `None`；随后 `sample_tokens()`（:4667）真正采样。

> 设计要点：`execute_model` 与 `sample_tokens` 分离，是为了让 forward 的 GPU kernel 先入队后、CPU 可继续准备下一步（async scheduling）。

### 1.2 中间状态 / 持久 buffer
<!-- tags: buffers, 持久buffer, cpugpu-buffer, batchdescriptor -->

`GPUModelRunner` 预分配大量 pinned CPU + GPU buffer（`CpuGpuBuffer`，`_make_buffer` :1100）：`input_ids`、`positions`、`seq_lens`、`query_start_loc`、`num_scheduled_tokens`、`mrope_positions` 等。每步只覆写前 N 个元素，避免反复分配。`BatchDescriptor`（`forward_context.py:30`）是 cudagraph 的 key：`num_tokens / num_reqs / uniform / has_lora / num_active_loras`。

---

## 2. 模型加载（model_loader）
<!-- tags: loading, load-format, weights, 权重加载, sharded -->

目录 `vllm/model_executor/model_loader/`。入口 `get_model()`（`__init__.py:127`）→ `get_model_loader(load_config)` 按 `load_format` 选 loader。

### 2.1 load_format 与 loader 映射（`__init__.py:48`）
<!-- tags: load-format, loader, 加载, safetensors, dummy -->

| load_format | loader | 说明 |
|---|---|---|
| `auto`/`hf`/`safetensors`/`fastsafetensors`/`instanttensor`/`mistral`/`npcache`/`pt` | `DefaultModelLoader` |
| `dummy` | `DummyModelLoader` | 随机权重（profiling / 测试） |
| `modelexpress` | `ModelExpressModelLoader` |
| `runai_streamer` / `runai_streamer_sharded` / `sharded_state` | `RunaiModelStreamerLoader` / `ShardedStateLoader` |
| `tensorizer` | `TensorizerLoader` |

`LoadFormats`（`__init__.py:32`）不含 gguf；GGUF 走 `llm-compressor`/`mistral` 等外部量化路径。`register_model_loader`（:66）允许注册自定义 loader。

### 2.2 加载流程（`BaseModelLoader.load_model`，`base_loader.py:43`）
<!-- tags: loading, load-model, weights, 加载流程, process-weights -->

1. `initialize_model(vllm_config, ...)`（`model_loader/utils.py:38`）：按 `model_config.architectures` 找到模型类（`get_model_architecture`），在 `set_current_vllm_config` 上下文里实例化——**此时所有并行层（ColumnParallelLinear 等）据此读取 TP rank 创建**，权重先建在目标 device 上。
2. `self.load_weights(model, model_config)`：`DefaultModelLoader.load_weights`（`default_loader.py:415`）调用 `model.load_weights(self.get_all_weights(...))`。
   - `get_all_weights`（:321）产出 `(name, tensor)` 迭代器。`_get_weights_iterator`（:244）按格式选迭代器：`safetensors_weights_iterator` / `fastsafetensors_weights_iterator` / `multi_thread_safetensors_weights_iterator` / `pt_weights_iterator` 等（`weight_utils.py`）。
   - `DefaultModelLoader.Source`（:49）支持多来源（主权重 + `secondary_weights`，如 MoE 专家分片）。
   - `_init_ep_weight_filter`（:351）：EP 场景下预计算 `local_expert_ids`，**读盘前就跳过非本 rank 的专家**，加速 MoE 加载。
3. `process_weights_after_loading(model, ...)`（`utils.py:97`）：遍历所有带 `quant_method` 的 module，调用其 `process_weights_after_loading`（repack / 在线量化 / 融合 scale），并处理 attention 延迟权重。

### 2.3 权重如何切分到 TP 各 rank
<!-- tags: tp, weight-loader, 切分, column-parallel, row-parallel -->

切分**不在 loader 里做**，而在**每个并行层的 `weight_loader`** 里做。模型 `load_weights`（如 `LlamaModel.load_weights`，`models/llama.py:441`）用 `AutoWeightsLoader` 把 checkpoint 的 tensor 按名字分发给各层，各层 `weight_loader` 只 `narrow` 出本 rank 的分片：
- `ColumnParallelLinear.weight_loader`（`layers/linear.py:548`）：按 `output_dim` 切。
- `RowParallelLinear.weight_loader`（:1614）：按 `input_dim` 切，`start_idx = tp_rank * shard_size`。
- `QKVParallelLinear` / `MergedColumnParallelLinear`：按 head / 子模块切，`packed_modules_mapping`（`llama.py:457`）把 checkpoint 的 `q_proj/k_proj/v_proj` 映射到融合的 `qkv_proj`。
- 量化权重：`weight_loader_v2` + `BasevLLMParameter`（`WEIGHT_LOADER_V2_SUPPORTED`）。

> 即：**每个 rank 只加载自己那份分片**，因此 TP>1 时单卡权重 = 全量 / tp_size。

### 2.4 量化加载
<!-- tags: quantization, 量化, quant-method, 加载 -->

`quant_config`（`vllm/config/quantization.py` + `layers/quantization/`（`fp8.py`、`awq_triton.py`、`auto_gptq.py`、`mxfp4.py`、`compressed_tensors/`、`online/` 等）。量化方法通过 `quant_method.create_weights` / `apply` 介入权重创建与 forward。

### 2.5 配置/调优旋钮（加载）
<!-- tags: load-config, 旋钮, load-format, quantization -->

- `--load-format`（`arg_utils.py:448`）：`auto/hf/safetensors/fastsafetensors/instanttensor/mistral/tensorizer/...`。
- `LoadConfig`（`vllm/config/load.py:27`）：`download_dir`、`safetensors_load_strategy`（`lazy` 等）、`safetensors_prefetch_num_threads`、`safetensors_prefetch_block_size`、`model_loader_extra_config`（`enable_multithread_load`、`num_threads`）、`ignore_patterns`。
- `--quantization`：选择量化后端。

---

## 3. 核心算子层（layers）
<!-- tags: layers, operators, attention-layer, linear, 算子 -->

目录 `vllm/model_executor/layers/`。绝大多数算子继承 `CustomOp`（`vllm/model_executor/custom_op.py:103`）：`forward` 通过 `dispatch_forward` 分发到 `forward_cuda / forward_hip / forward_xpu / forward_cpu / forward_native`，并支持 OOT（out-of-tree）平台覆盖（`op_registry_oot`）。`forward_native` 是纯 PyTorch 实现，供 torch.compile 融合或测试。

### 3.1 Attention（`layers/attention/attention.py:218`）
<!-- tags: attention, 算子, kv-cache-update, piecewise, forward -->

`Attention.forward(query, key, value, ...)`（:478）：
- reshape q/k/v → `[num_tokens, heads, head_size]`。
- 若 backend 不在 forward 内做 KV 更新，先调 `unified_kv_cache_update`（custom op `vllm::unified_kv_cache_update` 写 KV cache。
- 调 `torch.ops.vllm.unified_attention_with_output(...)`（:562）——**这是 piecewise 编译的切分点**（见 §5）。
- attention metadata 从 `get_forward_context().attn_metadata` 读取（模型层不直接持有）。
- 变体：`mla_attention.py`（MLA）、`cross_attention.py`、`encoder_only_attention.py`、`sparse_mla_attention.py`、`rswa_attention.py`（sliding window）等。
- KV cache 由 `Attention` 持有（`self.kv_cache`），dtype 由 `cache_config.cache_dtype` 决定（支持 fp8 KV）。

### 3.2 Linear / 并行层（`layers/linear.py`）
<!-- tags: linear, 并行层, column-parallel, row-parallel, tp -->

- `LinearBase`（:221）：持有 `quant_method`，`forward` 调 `quant_method.apply`。
- `ReplicatedLinear`（:302）：不切分。
- `ColumnParallelLinear`（:407）：权重按**输出维**切 `output_size/tp_size`；`gather_output=True` 时 all-gather。
- `MergedColumnParallelLinear`（:645）：多个输出子矩阵沿输出维拼接（如 gate+up）。
- `QKVParallelLinear`（:971）：按 **head 维**切 Q/K/V，KV head 数 < tp 时复制。
- `RowParallelLinear`（:1510）：权重按**输入维**切；`reduce_results=True` 时 all-reduce（bias 只在 rank0 加，避免重复）。
- `DCPGroupColumnParallelLinear`（:604）：Decode Context Parallelism 下按 DCP group 切。

**TP 配合**：一个 decoder 层典型是 `QKVParallelLinear → Attention → RowParallelLinear`（attention 部分无通信），`gate_up_proj(MergedColumnParallelLinear) → act → down_proj(RowParallelLinear)`（MLP 尾部 all-reduce）。`LlamaAttention`（`models/llama.py:122`）与 `LlamaMLP`（:115）展示了标准组合。

### 3.3 其他算子
<!-- tags: operators, rmsnorm, activation, rotary, moe -->

- **Layernorm**（`layernorm.py`）：`RMSNorm`（:37，支持 fused `forward_cuda` 且可带 residual）、`GemmaRMSNorm`、`RMSNormGated`、`LayerNorm`。
- **Activation**（`activation.py`）：`SiluAndMul`（:112，SwiGLU 的 gate*up 融合）、`GeluAndMul`、`GELU/GELUTanh/NewGELU/FastGELU/QuickGELU` 等。
- **Rotary embedding**（`rotary_embedding/`）：`RotaryEmbeddingBase`/`RotaryEmbedding`（`base.py`），`forward_cuda` 走 fused kernel；变体 `yarn_scaling_rope.py`、`ntk_scaling_rope.py`、`llama3_rope.py`、`mrope.py`（多模态）、`dual_chunk_rope.py` 等。
- **Embedding / LMHead**（`vocab_parallel_embedding.py`）：`VocabParallelEmbedding`（:198，按 vocab 切）、`ParallelLMHead`（:521）。
- **Logits**（`logits_processor.py:23`）：`LogitsProcessor` 做 scale/soft-cap/processor。
- **MoE**（`fused_moe/`）：`FusedMoE`（`layer.py`）+ `modular_kernel.py`，含 `router/`、`experts/`、`prepare_finalize/`、`oracle/`；EP 用 all2all（`all2all_utils.py`）。
- **MLA**（`mla.py`）、**Mamba/线性注意力**（`mamba/`、`lightning_attn.py`）。

---

## 4. 编译与 torch.compile 集成
<!-- tags: compile, torch-compile, inductor, 编译 -->

配置类 `CompilationConfig`（`vllm/config/compilation.py:398`）。

### 4.1 CompilationMode（`compilation.py:37`）
<!-- tags: compilation-mode, 编译模式, vllm-compile, eager -->

- `NONE=0`：纯 eager。
- `STOCK_TORCH_COMPILE=1`：标准 `torch.compile` 全图。
- `DYNAMO_TRACE_ONCE=2`：单次 Dynamo trace，去 guard 避免重编译。
- `VLLM_COMPILE=3`：**v1 默认**，vLLM 自定义 Inductor 后端 + 缓存 + piecewise 编译 + shape 特化 + 自定义 pass。

### 4.2 编译入口与后端
<!-- tags: compile, backend, split-graph, piecewise, inductor-pass -->

- 模型类用 `@support_torch_compile` 装饰（`compilation/decorators.py:118`），声明 `dynamic_arg_dims`（如 `{"input_ids": {0: "b"}, "positions": {0: "b"}}`，`models/llama.py:337`）标记动态维。
- `TorchCompileWithNoGuardsWrapper`（`compilation/wrapper.py:47`）：对非 STOCK 模式**丢弃所有 guard**（`skip_all_guards_unsafe`），保证只编译一次。
- `VllmBackend`（`compilation/backends.py:805`）：`__call__`（:1020）拿到 Dynamo 的 FX 图后：
  1. 计算 cache key（env_hash + config_hash + code_hash + compiler_hash），生成 `cache_dir`（`torch_compile_cache/<hash>/rank_i_j/`）。
  2. `split_graph(graph, splitting_ops)`（:553）按 `splitting_ops`（默认 attention ops）把图切成 **piecewise 子图**。
  3. `PiecewiseCompileInterpreter`（:687）把每个子图替换成 `PiecewiseBackend` 实例并编译。
  4. `generate_execution_code` + `compile_execution_fn`（`codegen.py`）生成把子图串起来的执行函数。
- `PiecewiseBackend`（`compilation/piecewise_backend.py:86`）：对每个子图，先按**通用 shape**（symbolic）编译一次，再对 `compile_sizes`/`compile_ranges` 里的具体 shape 各编译一份，运行时按 shape 分发。
- `CompilerManager`（`backends.py:124`）：缓存 `(Range, graph_index, backend) → 编译产物；`make_compiler`（:96）选 `InductorAdaptor` / `InductorStandaloneAdaptor`（`VLLM_USE_STANDALONE_COMPILE=1` 默认）/ `EagerAdaptor`。
- 自定义 Inductor pass 在 `compilation/passes/`：`fusion/`（`rms_quant_fusion.py`、`act_quant_fusion.py`、`attn_quant_fusion.py`、`allreduce_rms_fusion.py`、`sequence_parallelism.py`、`qk_norm_rope_fusion.py` 等）、`ir/`、`utility/`（`noop_elimination.py` 等），由 `PassConfig`（`compilation.py:107`）开关。

### 4.3 编译缓存
<!-- tags: compile-cache, 编译缓存, cache-hash, 落盘 -->

编译产物落盘到 `VLLM_CACHE_ROOT/torch_compile_cache/<hash>/rank_i_j/`（`vllm_compile_cache.py`、`computation_graph.py`、`transformed_code.py`）。hash 因子：`env_hash/config_hash/code_hash/compiler_hash`（`backends.py:1055-1079`）。`VLLM_DISABLE_COMPILE_CACHE=1` 关闭；`compile_cache_save_format`（`binary`/`unpacked`）控制格式。

---

## 5. CUDA Graph：capture / dispatch
<!-- tags: cudagraph, cuda-graph, capture, 图捕获 -->

### 5.1 CUDAGraphMode（`compilation.py:53`）
<!-- tags: cudagraph-mode, 模式, full, piecewise, decode-only -->

- `NONE`：不 capture。
- `PIECEWISE`：只对 piecewise 子图 capture，attention 等不兼容 op 留在图外。
- `FULL`：整模型 capture。
- `FULL_DECODE_ONLY = (FULL, NONE)`：decode 用 FULL，mixed prefill-decode 不 capture。
- `FULL_AND_PIECEWISE = (FULL, PIECEWISE)`：**v1 默认**，decode 用 FULL，prefill/mixed 用 PIECEWISE。

`separate_routine()` 的 mode 用 `decode_mode()` / `mixed_mode()` 取两个分量。

### 5.2 捕获哪些 batch size（capture sizes）
<!-- tags: cudagraph, capture-sizes, 批大小, 捕获, spec-decode -->

`VllmConfig.post_init`（`config/vllm.py:1937`）计算 `cudagraph_capture_sizes`：
- `max_cudagraph_capture_size` 默认 = `min(max_num_seqs * decode_query_len * 2, 512)`（Blackwell 数据中心卡为 1024），再 `min(max_num_batched_tokens)`。
- 默认 sizes = `[1,2,4] + range(8,256,8) + range(256,max,16)`；`performance_mode="interactivity"` 时用 `range(1, min(max,32)+1)` 细粒度。
- spec decode 时 `adjust_cudagraph_sizes_for_spec_decode`（`compilation.py:1519`）把 sizes 向上取整到 `uniform_decode_query_len`（=1+num_spec_tokens）的倍数。

### 5.3 CudagraphDispatcher（`vllm/v1/cudagraph_dispatcher.py:15`）
<!-- tags: cudagraph-dispatcher, dispatch, 分发, batch-descriptor, padding -->

- 持有两套 key：`cudagraph_keys[PIECEWISE]`、`cudagraph_keys[FULL]`，是**运行时可 dispatch 的合法 cudagraph 的唯一真源**。
- `initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)`（:166）：在 attention backend 初始化后调用，按 `cudagraph_capture_sizes × lora_cases` 生成 `BatchDescriptor`。PIECEWISE 的 key 放宽 `num_reqs=None`（可容纳任意请求数）；FULL 需要精确 `num_reqs`（FA3 scheduler metadata 依赖）。
- `dispatch(num_tokens, uniform_decode, has_lora, ...)`（:235）：把实际 batch 映射到 padding 后的 `num_tokens`（`_bs_to_padded_graph_size`，向上取到最近的 capture size），返回 `(runtime_mode, BatchDescriptor)`。优先 FULL，其次 PIECEWISE，否则 NONE。`num_tokens > max_size` 或 mode 不匹配 → NONE（eager）。
- `get_capture_descs()`（:326）：按 PIECEWISE→FULL、size 从大到小排序返回待 capture 的 desc（大 shape 先 capture 以复用内存池）。

### 5.4 CUDAGraphWrapper（`compilation/cuda_graph.py:145）
<!-- tags: cudagraph-wrapper, capture, replay, 包装, 图捕获 -->

- 包装 runnable，`__call__`（:233）读 `ForwardContext` 的 `cudagraph_runtime_mode` + `batch_descriptor`：
  - mode 不匹配或 NONE → 直接调 runnable（passthrough）。
  - 匹配且该 desc 未 capture → `torch.cuda.graph(...)` 捕获（:313），存 `CUDAGraphEntry`；capture 时禁用 gc、`set_graph_pool_id` 共享全局 pool、offloader sync。
  - 已 capture → `entry.cudagraph.replay()`（:360）。
- 输入 buffer 由外部（model runner 的持久 buffer）保证地址稳定；wrapper 不复制输入（`cudagraph_copy_inputs` 例外，见下）。

### 5.5 capture 编排（`gpu_model_runner.py`）
<!-- tags: capture, 编排, warmup, breakable, decode -->

- `capture_model()`（:6944）：`set_cudagraph_capturing_enabled(True)`，遍历 `cudagraph_dispatcher.get_capture_descs()`，对每组调 `_capture_cudagraphs`。
- `_warmup_and_capture(desc, mode)`（:7050）：先 `cudagraph_num_of_warmups` 次 `_dummy_run(mode=NONE)` 预热，再一次 `_dummy_run(mode=目标mode, is_graph_capturing=True)` 触发 capture。
- decode 用 FULL cudagraph 的**原理与收益**：decode 时每个请求 query_len=1（或 spec 的 1+k），batch 形状规整、attention 是 memory-bound，把整图 capture 后 replay 消除 Python/launch 开销，kernel dispatch 开销，显著降低小 batch decode 延迟。
- **BreakableCUDAGraphWrapper**（`compilation/breakable_cudagraph.py`，`VLLM_USE_BREAKABLE_CUDAGRAPH=1` 开启）：替代 FX 切分，用单次 stream-capture + 在 attention/kv-cache op 处 `eager_break_during_capture` 打断，eager 段跑 op、graph 段 replay，产物是 zero-arg callable 列表。

### 5.6 模型如何被包上 wrapper（`gpu_model_runner.load_model`，:5413）
<!-- tags: wrapper, load-model, 包装, ubatch, breakable -->

- `STOCK_TORCH_COMPILE`：直接 `self.model.compile(fullgraph=True, backend=...)`（:5544）。
- 否则：`is_breakable_cudagraph_enabled()` → `BreakableCUDAGraphWrapper`；或 `cudagraph_mode.has_full_cudagraphs()` → `CUDAGraphWrapper(runtime_mode=FULL)`（:5567）；`use_ubatching` → `UBatchWrapper`。
- piecewise 的 wrapper 在编译期由 `wrap_with_cudagraph_if_needed`（`backends.py:633`）加到每个子图上。

---

## 6. Warmup 与显存 profiling
<!-- tags: warmup, memory-profiling, oom, 显存 -->

入口 `Worker.compile_or_warm_up_model()`（`gpu_worker.py:694`）与 `Worker.determine_available_memory()`（:475）。

### 6.1 显存 profiling（`determine_available_memory`）
<!-- tags: memory-profiling, 显存, profile-run, cudagraph-estimate, available-memory -->

1. 若设了 `kv_cache_memory_bytes`：跳过 profiling，直接返回该值。
2. 否则 `memory_profiling(...)`（`utils/mem_utils.py:234`）上下文里跑 `model_runner.profile_run()`（`gpu_model_runner.py:6553`）：
   - `profile_run` 用 `max_num_batched_tokens` 的 dummy batch 跑一次 forward（含 MM encoder profiling），触发编译 + 峰值激活。
   - `memory_profiling` 用 `torch.accelerator.get_memory_info` 与 peak stats 区分三类内存：非 vLLM / torch / 非 torch，算出 `non_kv_cache_memory`、`transient_peak_headroom`、`total_consumed`。
2.5 若启用 cudagraph：`profile_cudagraph_memory()`（:6775）用临时 pool 真实 capture 前 2 个最大 shape，估算 `first_capture + (N-1)*per_graph`，得 `cudagraph_memory_estimate`（受 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 控制，默认开）。
3. `available_kv_cache_memory = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate`（:659）。`requested_memory = gpu_memory_utilization * total`。
4. `initialize_from_config`（:664）按此分配 KV cache 块数。

> 公式核心：**KV 显存 = 请求显存 − 权重 − 峰值激活 − cudagraph 估算`。

### 6.2 warmup / capture 顺序（`compile_or_warm_up_model`）
<!-- tags: warmup, capture, 顺序, kernel-warmup, compile -->

1. `VLLM_COMPILE` 模式下，对 `compile_sizes` 中不在 capture sizes 里的 size（如 `max_num_batched_tokens`）逐个 `_dummy_run` 编译（:721）。
2. `kernel_warmup(worker)`（`model_executor/warmup/kernel_warmup.py:98`）：预热/autotune 推理期 kernel（Triton JIT、flashinfer 等），避免首请求 JIT 卡顿。
3. `if not enforce_eager: capture_model()`（:732）——真正 capture 所有 cudagraph，返回实际占用的 `cuda_graph_memory_bytes`，与估算对比打日志。
4. 打印建议的 `--kv-cache-memory`（:794），`maybe_save_startup_plan`（:806）。
5. 末尾 `trigger_inductor_lazy_init`、`activate_jit_monitor`、`freeze_gc_heap`、`set_torch_threads_for_runtime`。

### 6.3 Startup plan（启动计划持久化，`vllm/v1/worker/startup_plan.py`）
<!-- tags: startup-plan, 启动计划, 持久化, fingerprint, 跳过profiling -->

显存 profiling 结果（可复现的 `--kv-cache-memory` 值）在 `VLLM_ENABLE_STARTUP_PLAN=1` 时持久化到 `{VLLM_CACHE_ROOT}/startup_plan/`，按 (model, config, hardware, library) 指纹为 key；后续启动指纹匹配且显存不小于上次时直接套用，**跳过 memory-profiling 测量与 CUDA graph 显存估算**，缩短启动时间。与 torch.compile cache 同属"可再生的派生状态"。

### 6.4 Encoder CUDA Graph（`vllm/v1/worker/encoder_cudagraph.py`）
<!-- tags: encoder-cudagraph, 多模态, vision, capture, 启动开销 -->

多模态 vision encoder 的 budget-batch 执行也支持 CUDA graph capture（`encoder_cudagraph_defs.py` 定义 capture 参数），减少 MM 负载下 encoder forward 的 launch 开销。

---

### 6.5 采样拆分与容错（v0.27+ 新结构）
<!-- tags: sampling, fault-tolerance, sentinel, 采样, 容错 -->

- **`vllm/v1/sample/`**：采样逻辑独立成目录——`sampler.py`（`Sampler.forward`，:73）、`rejection_sampler.py`（spec decode 的 rejection sampling）、`logits_processor/`（bad words / logit bias 等）、`thinking_budget_state.py`（reasoning budget 跟踪）。`GPUModelRunner.sample_tokens`（:4667）仍是入口，但具体采样实现委托此目录。
- **`vllm/v1/fault_tolerance/`**：`EngineCoreSentinel`（`engine_core_sentinel.py`）——EngineCore 进程的哨兵/容错包装，通过 `FT_STATUS_CALL_ID` utility 调用上报 `EngineStatusType`，支持 engine core 故障检测与进程组无状态重建（`stateless_init/destroy_torch_distributed_process_group`），为弹性恢复（Elastic EP 等）提供基础。
- **Model Runner V2**（`vllm/v1/worker/gpu/`）：实验性重构版 runner（`model_runner.py` + `input_batch/`、`sample/`、`spec_decode/`、`warmup.py` 等子模块），按功能拆分 `gpu_model_runner.py` 的巨石结构，仍在活跃开发中（见其 README），默认路径仍走 `gpu_model_runner.py`。

---

## 7. 关键文件清单
<!-- tags: files -->

| 文件 | 作用 |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | ModelRunner 主逻辑（prepare inputs / execute / capture / profile） |
| `vllm/v1/worker/gpu_worker.py` | Worker：设备初始化、`determine_available_memory`、`compile_or_warm_up_model` |
| `vllm/v1/worker/gpu_input_batch.py` | `InputBatch` 持久批状态、`block_table` |
| `vllm/v1/cudagraph_dispatcher.py` | cudagraph 运行时 dispatch + capture desc |
| `vllm/compilation/cuda_graph.py` | `CUDAGraphWrapper`（capture/replay） |
| `vllm/compilation/breakable_cudagraph.py` | Breakable cudagraph（stream-capture 打断） |
| `vllm/compilation/backends.py` | `VllmBackend`、`split_graph`、`CompilerManager` |
| `vllm/compilation/piecewise_backend.py` | `PiecewiseBackend`（per-shape 编译/分发） |
| `vllm/compilation/wrapper.py` | `TorchCompileWithNoGuardsWrapper` |
| `vllm/compilation/decorators.py` | `@support_torch_compile` |
| `vllm/compilation/passes/` | 自定义 Inductor pass（fusion/ir/utility） |
| `vllm/config/compilation.py` | `CompilationConfig` / `CUDAGraphMode` / `PassConfig` |
| `vllm/forward_context.py` | `ForwardContext` / `BatchDescriptor`（attn metadata、cudagraph 模式载体） |
| `vllm/model_executor/model_loader/` | 权重加载（`default_loader.py`、`weight_utils.py`、`base_loader.py`） |
| `vllm/model_executor/layers/` | 算子层（`attention/`、`linear.py`、`layernorm.py`、`activation.py`、`rotary_embedding/`、`fused_moe/`、`quantization/`） |
| `vllm/model_executor/models/` | 290+ 模型定义（`llama.py` 为参考实现） |
| `vllm/model_executor/warmup/` | kernel warmup / JIT warmup |
| `vllm/utils/mem_utils.py` | `memory_profiling` |
| `vllm/v1/worker/gpu/` | **实验性 Model Runner V2**（重构中，见其 README） |

---

## 8. 配置 / 调优旋钮（编译 & 性能）
<!-- tags: tuning, knobs, compile, cudagraph, enforce-eager -->

### CLI / 顶层
<!-- tags: cli, 旋钮, enforce-eager, compilation-config, flags -->
- `--enforce-eager`（`ModelConfig.enforce_eager`，`config/model.py:241`）：禁用 CUDA Graph + 编译，全 eager（调试/排障）。
- `--compilation-config` / `-cc`（`arg_utils.py:681`）：JSON 覆盖 `CompilationConfig`，如 `{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}`。
- `--gpu-memory-utilization` / `--kv-cache-memory-bytes`（`arg_utils.py:537/538`）：控制 KV 显存预算。
- `--max-num-batched-tokens` / `--max-num-seqs`（:539/542）：决定 `max_cudagraph_capture_size` 上界与 capture sizes。

### CompilationConfig 关键字段
<!-- tags: compilation-config, 字段, cudagraph-mode, splitting-ops, pass-config -->
- `mode`（0-3，默认 v1 默认 3）、`backend`（`""`→inductor / `eager` / 自定义 qualname）。
- `cudagraph_mode`（`NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE`）。
- `cudagraph_capture_sizes` / `max_cudagraph_capture_size`：手动指定 capture 的 batch size 集合。
- `compile_sizes` / `compile_ranges_endpoints`：Inductor 额外编译的具体 size / range。
- `splitting_ops`：piecewise 切分点（默认 attention ops；`[]` 表示不切分→FULL）。
- `use_inductor_graph_partition`：改为在 Inductor codegen 期按 `cudagraph_unsafe` tag 切分。
- `cudagraph_num_of_warmups`、`cudagraph_copy_inputs`、`cudagraph_specialize_lora`。
- `custom_ops`（`all`/`none`/`+op`/`-op`）：细粒度开关 custom op（Inductor 下默认禁用 custom op，改由 Inductor 生成 Triton）。
- `pass_config`（`PassConfig`）：`fuse_norm_quant`、`fuse_act_quant`、`fuse_attn_quant`、`enable_sp`（sequence parallelism）、`fuse_gemm_comms`（async TP）、`fuse_allreduce_rms`、`enable_qk_norm_rope_fusion` 等。
- `dynamic_shapes_config`（`backed`/`unbacked`/`backed_size_oblivious`）。
- `cache_dir` / `compile_cache_save_format`。

### 环境变量
<!-- tags: env-vars, 环境变量, breakable, standalone-compile, aot -->
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`（默认 1）：是否把 cudagraph 显存计入 KV 预算。
- `VLLM_USE_BREAKABLE_CUDAGRAPH`（默认 0）：启用 breakable cudagraph。
- `VLLM_USE_STANDALONE_COMPILE`（默认 1）：用 Inductor standalone compile。
- `VLLM_USE_AOT_COMPILE`（默认 0）、`VLLM_USE_BYTECODE_HOOK`（默认 1）、`VLLM_DISABLE_COMPILE_CACHE`（默认 0）。
- `VLLM_ENABLE_CUDAGRAPH_GC`：capture 期间是否禁用 gc。

### 部署建议（速查）
<!-- tags: deployment, 部署建议, 速查, 延迟, 显存 -->
- **延迟敏感 decode**：保持默认 `FULL_AND_PIECEWISE`；小模型/小 prompt 可试 `FULL`；P/D 分离的 decode 实例用 `FULL_DECODE_ONLY` 省显存。
- **显存紧张**：调低 `--gpu-memory-utilization` 或显式 `--kv-cache-memory-bytes`；`max_cudagraph_capture_size` 上限 512/1024 已限制大 graph 的显存/启动开销。
- **MoE + EP**：`--all2all-backend` 用 `deepep_low_latency`（`deepep_high_throughput` 与 cudagraph 不兼容，会自动降级 `cudagraph_mode=NONE`，见 `compilation.py:1235`）。
- **首次启动慢**：编译 + capture 通常 5~20s+；用编译缓存（`torch_compile_cache`）与 `kernel_warmup` 缓解。
- **排障**：`--enforce-eager` 关闭图/编译；`VLLM_LOGGING_LEVEL=DEBUG` 会校验 cudagraph 输入地址一致性。
