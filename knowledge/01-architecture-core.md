# 核心架构与请求生命周期

> 基于 vLLM main（`f32b17b6d6`，2026-08-21）源码，最新 release tag **v0.28.0rc1**（v1 引擎为默认且唯一活跃引擎）。技术标识符保留英文。

## 1. v1 与旧版 (v0) 引擎
<!-- tags: v1, engine, 引擎 -->

v1 是 vLLM 重写的引擎，当前版本中 **v0 引擎已完全移除**，`vllm/engine/` 下的旧类只是 v1 的别名：

- `vllm/engine/llm_engine.py`：`LLMEngine = V1LLMEngine`（`from vllm.v1.engine.llm_engine import LLMEngine`）。
- `vllm/engine/async_llm_engine.py`：`AsyncLLMEngine = AsyncLLM`（`from vllm.v1.engine.async_llm import AsyncLLM`）。
- `vllm/engine/arg_utils.py`：`EngineArgs` / `AsyncEngineArgs`（配置解析，v1 沿用）。
- `vllm/engine/protocol.py`：`EngineClient` 抽象基类（`AsyncLLM` 实现它）、`StreamingInput`。

v1 相对 v0 的关键架构变化：

1. **前后端进程解耦**：调度器 (Scheduler) 与模型执行 (Model Runner) 运行在独立的 **EngineCore 进程** 中，前端 (API server / LLM 客户端) 通过 **ZMQ** 与之通信。v0 是单进程内 `step()` 轮询。
2. **连续批处理 (continuous batching) 原生内建**：不再有显式的 "prefill phase / decode phase"，调度器每步只关心让每个请求的 `num_computed_tokens` 追上 `num_tokens_with_spec`（见 `vllm/v1/core/sched/scheduler.py:484` 的注释）。
3. **KV cache 用 block pool + 前缀缓存**：`KVCacheManager` / `BlockPool`（`vllm/v1/core/`），请求级 `block_hashes` 支持 prefix caching。
4. **异步调度 (async scheduling)**：`SchedulerConfig.async_scheduling`（默认在 `VllmConfig.__post_init__` 中按 executor 能力自动开启），让 GPU 前向与下一步调度重叠，`max_concurrent_batches` 因此为 2（`vllm/config/vllm.py:584`）。
5. **`vllm/sequence.py` 已不再是请求载体**：v1 用 `vllm/v1/request.py` 的 `Request` 对象 + `EngineCoreRequest`/`EngineCoreOutput`（msgspec.Struct）作为进程间消息。`vllm/sequence.py` 现在只剩 `IntermediateTensors`（PP 中间张量容器）。

## 2. 进程与线程模型
<!-- tags: process, threading, zmq, 进程模型, multiprocess -->

一次 `vllm serve` 启动的典型进程拓扑（`vllm/entrypoints/cli/serve.py`）：

```
┌─────────────────────────────────────────────────────────────────┐
│  主进程 (launcher)                                                │
│   - 解析 CLI → AsyncEngineArgs → VllmConfig                      │
│   - launch_core_engines(): 分配 ZMQ 地址、拉起 EngineCore 进程     │
│   - APIServerProcessManager: spawn N 个 API server 子进程         │
└─────────────────────────────────────────────────────────────────┘
        │ ZMQ (input ROUTER / output PULL)          ▲
        ▼                                            │
┌──────────────────────┐   ZMQ   ┌────────────────────────────────┐
│ API server 进程 (×N)  │ ──────▶ │ EngineCore 进程 (×DP)           │
│  AsyncLLM 前端         │ ◀────── │  Scheduler + ModelExecutor     │
│  InputProcessor        │  msgpack│   └─ Executor (mp/ray/uni)     │
│  OutputProcessor       │         │        └─ Worker 进程 (×TP×PP) │
└──────────────────────┘         └────────────────────────────────┘
```

- **API server 进程**：`vllm/entrypoints/launchers/api_server/entry.py` 的 `run_server` / `run_server_worker`。单进程模式直接 `uvloop.run(run_server(args))`；多进程模式 `run_multi_api_server` 用 `APIServerProcessManager`（`vllm/v1/utils.py:166`）spawn 子进程，共享一个 `reuse_port` socket。每个 API server 进程内持有一个 `AsyncLLM` 前端实例。
- **EngineCore 进程**：`vllm/v1/engine/core.py` 的 `EngineCoreProc`（继承 `EngineCore`）。每个 DP rank 一个。进程标题设为 `EngineCore` / `EngineCore_DP{rank}`（`set_process_title`）。
- **Worker 进程**：由 `Executor` 拉起。`MultiprocExecutor`（`vllm/v1/executor/multiproc_executor.py:111`）为每个 TP×PP rank 起一个 `WorkerProc`，通过 `MessageQueue` 广播 `SchedulerOutput` 并回收 `ModelRunnerOutput`。

### EngineCore 内部线程
<!-- tags: enginecore, threads, zmq, 线程, busy-loop -->

`EngineCoreProc.__init__`（`core.py:1007`）在启动时创建两个 daemon 线程，把 ZMQ socket IO 与 GPU 前向解耦（socket 操作释放 GIL，可与模型前向重叠）：

- `process_input_sockets`（`core.py:1674`）：ZMQ `DEALER` 收 `EngineCoreRequest`，反序列化后 `preprocess_add_request` 转成 `Request`，压入 `input_queue`。
- `process_output_sockets`（`core.py:1777`）：ZMQ `PUSH` 发 `EngineCoreOutputs`，复用发送 buffer 做零拷贝。

主线程跑 `run_busy_loop`（`core.py:1391`）：`_process_input_queue` → `_process_engine_step`（调 `step_fn`）循环。

### ZMQ 通信细节
<!-- tags: zmq, ipc, 通信, msgpack, handshake -->

- 地址由 `get_engine_zmq_addresses`（`vllm/v1/engine/utils.py:1039`）分配：本地 (同机) 用 `ipc://`，跨节点用 `tcp://host:0`（bind 后回填真实端口）。
- 消息类型 `EngineCoreRequestType`（`vllm/v1/engine/__init__.py:284`）：`ADD`/`ABORT`/`START_DP_WAVE`/`UTILITY`/`EXECUTOR_FAILED`/`WAKEUP`，用单字节 hex 编码。
- 序列化用 `msgspec.msgpack`（`MsgpackEncoder`/`MsgpackDecoder`，`vllm/v1/serial_utils.py`），多模态张量可走 out-of-band tensor IPC（`tensor_ipc.py`）。
- 启动握手：EngineCore 发 `HELLO`，前端回 `EngineHandshakeMetadata`（含 ZMQ 地址），EngineCore 再回 `EngineCoreReadyResponse`（`core.py:1632`，含 `num_gpu_blocks`、`block_size`、`max_model_len` 等）。客户端等待超时由 `VLLM_ENGINE_READY_TIMEOUT_S`（默认 600s）控制。

### Rust frontend（实验性，v0.27 起）
<!-- tags: rust-frontend, axum, 实验性, crate, zmq -->

`rust/` 目录（Cargo workspace）是 Python 前端的 **drop-in 替代**：用 Rust 重建 northbound serving 层，仍通过 ZMQ + msgpack 走现有 engine 边界对接 Python EngineCore 进程。定位是**实验性、功能未齐**（见 `rust/README.md`）。

- **crate 分层**（自底向上）：`vllm-engine-core-client`（ZMQ 传输 + msgpack 协议）→ `vllm-llm`（token-in/token-out facade）→ `vllm-text`（tokenizer + 增量 detokenizer）→ `vllm-chat`（chat 模板渲染、reasoning/tool 解析）→ `vllm-server`（axum OpenAI 兼容 HTTP API）→ `vllm-cmd`/`vllm-rs`（CLI 入口）。
- **进程模型**：Python 仍是 launcher，负责进程启动，把 Rust API server 作为受管子进程拉起，继承监听 socket 并传入 ZMQ 地址（`vllm/entrypoints/cli/serve.py` 中 `rust_frontend_path` 分支）。
- **启用方式**：`VLLM_USE_RUST_FRONTEND=1`（`vllm/envs.py:578` 的 `_resolve_rust_cli_path()` 据此解析二进制路径，结果存 `VLLM_RUST_FRONTEND_PATH`）；需 setuptools-rust 构建或显式指定路径。`vllm serve` 检测到 rust frontend 时走不同的多 API server 编排（`serve.py:108/125/145`）。
- 另有 `VLLM_USE_RUST_BENCH`（Rust 版 bench 工具开关）。

## 3. 核心类：EngineCore / EngineCoreClient
<!-- tags: enginecore, enginecoreclient, classes -->

### EngineCore（引擎内环）
<!-- tags: enginecore, step, 内环, dp-enginecore, ray-actor -->

`vllm/v1/engine/core.py`

- **`EngineCore`**（`core.py:104`）：引擎"内环"。`__init__` 里依次：建 `model_executor`（Executor）→ `_initialize_kv_caches`（profile 显存、算 KV cache 配置、`compile_or_warm_up_model`）→ 建 `Scheduler` → 建 `batch_queue`（PP/async 用）。
  - `step()`（`core.py:583`）：`scheduler.schedule()` → `model_executor.execute_model(non_block=True)` → `scheduler.get_grammar_bitmask()` → `model_executor.sample_tokens()` → `scheduler.update_from_output()`。返回 `dict[client_index, EngineCoreOutputs]`。
  - `step_with_batch_queue()`（`core.py:624`）：async scheduling / PP 下的流水线版本，用 `batch_queue` 让"调度下一批"与"取上一批结果"重叠。
  - `preprocess_add_request`（`core.py:968`）：`EngineCoreRequest` → `Request`（在 input 线程里跑，可与 GPU 前向并行）。
- **`EngineCoreProc`**（`core.py:1007`）：ZMQ 包装，后台进程版。`run_engine_core`（`core.py:1271`）是进程入口，注册 SIGTERM/SIGINT handler 后跑 `run_busy_loop`。
- **`DPEngineCoreProc`**（`core.py:1986`）：MoE + DP 场景，额外维护 DP 进程组、wave 同步、dummy batch、Elastic EP 扩缩容。
- **`DPMoEEngineCoreActor` / `EngineCoreActor`**（`core.py:2519/2542`）：Ray actor 版（`data_parallel_backend=ray` 时用）。

### EngineCoreClient（前端侧）
<!-- tags: enginecoreclient, client, 前端, syncmp, asyncmp -->

`vllm/v1/engine/core_client.py`，`EngineCoreClient.make_client(multiprocess_mode, asyncio_mode, ...)` 分派：

| 子类 | 场景 | 说明 |
|------|------|------|
| `InprocClient` | 同进程（V0 兼容 / 调试） | 直接调 `EngineCore.step_fn()`，无 busy loop |
| `SyncMPClient` | `LLM`（同步离线） | ZMQ + 后台输出线程，`queue.Queue` 拉结果 |
| `AsyncMPClient` | `AsyncLLM`（在线服务） | ZMQ + asyncio task 拉结果 |
| `DPAsyncMPClient` / `DPLBAsyncMPClient` | DP 外部/内部负载均衡 | 多 engine，按负载选 engine |

`MPClient`（`core_client.py:503`）基类负责 ZMQ 建连、`launch_core_engines`、等 ready、监控 engine 存活（`start_engine_core_monitor`）。`SyncMPClient.get_output()` 阻塞取 `EngineCoreOutputs`；`AsyncMPClient.get_output_async()` 从 `asyncio.Queue` 取。

## 4. 请求完整数据流
<!-- tags: request, lifecycle, dataflow, 请求生命周期 -->

以 OpenAI `/v1/chat/completions` 为例（同步 `LLM.generate` 路径类似，只是输出端不同）：

```
HTTP 请求
  │
  ▼
OpenAIServingChat.create_chat_completion   (vllm/entrypoints/openai/chat_completion/serving.py:226)
  │  渲染 chat → EngineInput (OnlineRenderer)
  │  构造 SamplingParams
  ▼
engine_client.generate(engine_input, sampling_params, request_id, ...)   (serving.py:352)
  │  = AsyncLLM.generate()   (vllm/v1/engine/async_llm.py:550)
  ▼
AsyncLLM.add_request()   (async_llm.py:283)
  │  InputProcessor.process_inputs_async()  → EngineCoreRequest  (tokenize / 多模态)
  │  OutputProcessor.add_request()          → RequestState + RequestOutputCollector(队列)
  │  n>1 时用 ParentRequest 扇出子请求
  ▼
engine_core.add_request_async(request)   (AsyncMPClient, core_client.py:1149)
  │  ZMQ ROUTER 发送 (EngineCoreRequestType.ADD, EngineCoreRequest)
  ▼
[EngineCore 进程] process_input_sockets 线程
  │  反序列化 → preprocess_add_request → Request
  ▼
input_queue → run_busy_loop → step_fn
  │
  ├─ Scheduler.schedule()          (vllm/v1/core/sched/scheduler.py:484)
  │    分配 KV blocks、算 num_scheduled_tokens → SchedulerOutput
  ├─ ModelExecutor.execute_model(scheduler_output, non_block=True)
  │    └─ [Worker 进程] GPUModelRunner.execute_model  (vllm/v1/worker/gpu_model_runner.py:4288)
  │         _update_states → _prepare_inputs → 模型前向 → 返回 logits (不采样)
  ├─ Scheduler.get_grammar_bitmask(scheduler_output)   (结构化输出)
  ├─ ModelExecutor.sample_tokens(grammar_output)
  │    └─ [Worker 进程] GPUModelRunner.sample_tokens  (gpu_model_runner.py:4667)
  │         apply_grammar_bitmask → Sampler.forward  (vllm/v1/sample/sampler.py:73)
  │         → ModelRunnerOutput (sampled_token_ids, logprobs, ...)
  └─ Scheduler.update_from_output(scheduler_output, model_runner_output)
       (scheduler.py:1737)  更新请求状态、检测 finish、释放/保留 blocks
       → dict[client_index, EngineCoreOutputs]
  ▼
output_queue → process_output_sockets 线程 → ZMQ PUSH
  ▼
[API server 进程] AsyncMPClient 输出 task → outputs_queue
  ▼
AsyncLLM._run_output_handler 的 output_handler 协程  (async_llm.py:665)
  │  OutputProcessor.process_outputs()  (output_processor.py:598)
  │    detokenize (IncrementalDetokenizer) → stop 检查 → 构造 RequestOutput
  │    推入每个请求的 RequestOutputCollector 队列
  ▼
AsyncLLM.generate() 的 async generator 从队列取 RequestOutput → yield
  ▼
OpenAIServingChat 流式/非流式打包 → SSE / JSON 响应
```

**同步 `LLM.generate` 路径**（`vllm/entrypoints/llm.py:418`）：
`LLM.generate` → `_run_completion`（`vllm/entrypoints/offline_utils.py:326`）→ `_add_completion_requests`（逐条 `llm_engine.add_request`）→ `_run_engine`（`offline_utils.py:573`）循环 `while llm_engine.has_unfinished_requests(): llm_engine.step()`。`LLMEngine.step()`（`vllm/v1/engine/llm_engine.py:298`）= `engine_core.get_output()` + `output_processor.process_outputs()`（无队列，直接返回 `list[RequestOutput]`）+ abort + 记 stats。

**关键数据结构**（`vllm/v1/engine/__init__.py`）：
- `EngineCoreRequest`（`:107`）：`request_id, prompt_token_ids, mm_features, sampling_params, pooling_params, arrival_time, lora_request, cache_salt, data_parallel_rank, client_index, priority, ...`。
- `EngineCoreOutput`（`:196`）：`request_id, new_token_ids, new_logprobs, finish_reason, stop_reason, ...`。
- `EngineCoreOutputs`（`:253`）：`outputs: list[EngineCoreOutput], scheduler_stats, timestamp, utility_output, finished_requests, wave_complete`。
- `FinishReason`（`:47`）：`STOP/LENGTH/ABORT/ERROR/REPETITION`。

## 5. 核心配置体系
<!-- tags: config, vllmconfig, 配置 -->

### VllmConfig（`vllm/config/vllm.py:357`）
<!-- tags: vllmconfig, dataclass, 子配置, 字段 -->

聚合所有子配置的 dataclass。主要字段：

| 字段 | 类型 | 文件 |
|------|------|------|
| `model_config` | `ModelConfig` | `config/model.py` |
| `cache_config` | `CacheConfig` | `config/cache.py` |
| `parallel_config` | `ParallelConfig` | `config/parallel.py` |
| `scheduler_config` | `SchedulerConfig` | `config/scheduler.py` |
| `device_config` | `DeviceConfig` | `config/device.py` |
| `load_config` | `LoadConfig` | `config/load.py` |
| `offload_config` | `OffloadConfig` | `config/offload.py` |
| `attention_config` | `AttentionConfig` | `config/attention.py` |
| `kernel_config` | `KernelConfig` | `config/kernel.py` |
| `lora_config` | `LoRAConfig \| None` | `config/lora.py` |
| `speculative_config` | `SpeculativeConfig \| None` | `config/speculative.py` |
| `structured_outputs_config` | `StructuredOutputsConfig` | `config/structured_outputs.py` |
| `observability_config` | `ObservabilityConfig` | `config/observability.py` |
| `compilation_config` | `CompilationConfig` | `config/compilation.py` |
| `kv_transfer_config` | `KVTransferConfig \| None` | `config/kv_transfer.py` |
| `quant_config` | `QuantizationConfig \| None` | `config/quantization.py` |
| `optimization_level` | `OptimizationLevel`（默认 O2） | `config/vllm.py:130` |
| `performance_mode` | `"balanced"/"interactivity"/"throughput"` | `config/vllm.py` |

> 注意：本版本**没有独立的 `DecodingConfig`**（vLLM 早期有，现已并入 `ModelConfig`/`SamplingParams`）。

### 配置解析链
<!-- tags: config, 解析链, engineargs, post-init, 推导 -->

1. **CLI / 代码参数 → `EngineArgs`**（`vllm/engine/arg_utils.py:424`）：`EngineArgs` 是一个扁平 dataclass，字段名与 CLI flag 一一对应（如 `tensor_parallel_size`、`gpu_memory_utilization`、`max_num_batched_tokens`）。`AsyncEngineArgs(EngineArgs)`（`:2870`）加 `enable_log_requests` 等。
2. **`EngineArgs.create_engine_config(usage_context)`**（`arg_utils.py:1956`）：按序构造各子 config —— `create_model_config()` → `CacheConfig` → `ParallelConfig` → `create_speculative_config()` → `SchedulerConfig` → `LoRAConfig` → `AttentionConfig` → ... → 组装 `VllmConfig`。期间做大量默认值推导与合法性校验（如 DP 模式互斥、`max_num_batched_tokens`/`max_num_seqs` 默认值由 `_set_default_max_num_seqs_and_batched_tokens_args` 定）。
3. **`VllmConfig.__post_init__`**（`vllm.py:1101`）：做跨 config 的最终推导，例如按 executor 能力决定 `async_scheduling`（`vllm.py:1220-1307`）、设 `distributed_executor_backend`、算 `max_concurrent_batches` 等。
4. **环境变量**：`vllm/envs.py` 集中定义（`VLLM_ENABLE_V1_MULTIPROCESSING`、`VLLM_ENGINE_READY_TIMEOUT_S`、`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`、`VLLM_LOG_STATS_INTERVAL` 等），`EngineArgs` 字段默认值多取自对应 config 的默认值，CLI 未显式给时回落到 env / config 默认。
5. **JSON / dict**：`compilation_config`、`speculative_config`、`kv_transfer_config`、`attention_config` 等支持传 dict，在 `LLM.__init__`（`entrypoints/llm.py:260` 的 `_make_config`）或 `create_engine_config` 里转成对应 config 实例。

### 关键子配置字段速查
<!-- tags: config, 字段, modelconfig, cacheconfig, schedulerconfig -->

- **`ModelConfig`**（`config/model.py:124`）：`model`、`runner`（`auto/generate/pooling`）、`dtype`、`max_model_len`、`quantization`、`enforce_eager`、`seed`、`trust_remote_code`、`tokenizer_mode`、`enable_sleep_mode`、`pooler_config`。
- **`CacheConfig`**（`config/cache.py:56`）：`gpu_memory_utilization`（默认 0.92）、`kv_cache_memory_bytes`、`block_size`、`enable_prefix_caching`（默认 True）、`kv_cache_dtype`、`num_gpu_blocks`（运行时 profile 后填）。
- **`SchedulerConfig`**（`config/scheduler.py:26`）：`max_num_batched_tokens`（默认 2048）、`max_num_seqs`（默认 128）、`enable_chunked_prefill`、`policy`（`scheduling_policy`）、`async_scheduling`、`stream_interval`（默认 1）、`scheduler_cls`、`watermark`、`prefill_schedule_interval`。
- **`ParallelConfig`**（`config/parallel.py:119`）：`tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size`、`distributed_executor_backend`（`mp`/`ray`/`uni`/`external_launcher`）、`enable_expert_parallel`、`data_parallel_external_lb`/`hybrid_lb`、`enable_elastic_ep`、`numa_bind`。
- **`LoadConfig`**（`config/load.py:27`）：`load_format`（默认 `auto`）、`download_dir`、`device`、`max_parallel_loading_workers`。
- **`ObservabilityConfig`**（`config/observability.py:18`）：`otlp_traces_endpoint`、`collect_detailed_traces`、`kv_cache_metrics`、`show_hidden_metrics_for_version`。

## 6. 同步 LLM vs 异步 AsyncLLM
<!-- tags: llm, asyncllm, offline, online -->

| 维度 | `LLM`（同步，离线） | `AsyncLLM`（异步，在线） |
|------|--------------------|--------------------------|
| 入口类 | `vllm/entrypoints/llm.py:67` | `vllm/v1/engine/async_llm.py:72` |
| 引擎 | `LLMEngine`（`v1/engine/llm_engine.py:48`） | 自身即 `EngineClient` |
| EngineCoreClient | `SyncMPClient`（或 `InprocClient`） | `AsyncMPClient`（或 DP 变体） |
| 驱动方式 | 用户循环调 `llm_engine.step()`（`offline_utils._run_engine`） | 后台 `output_handler` asyncio task 自动拉取 |
| 输出 | `step()` 返回 `list[RequestOutput]`，攒到全部完成 | `generate()` 返回 `AsyncGenerator[RequestOutput]`，逐 token 流式 yield |
| 典型用途 | 离线批量推理、评测、脚本 | OpenAI 兼容 API server、在线服务 |
| 多模态预处理 | 同步 `process_inputs` | 异步 `process_inputs_async`（不阻塞 event loop） |

- `LLM` 构造时 `disable_log_stats=True`（`llm.py:228`），并禁止单进程 `data_parallel_size>1`（`llm.py:283`，会挂起）。
- `AsyncLLM` 支持 streaming input（`AsyncGenerator[StreamingInput]`，`async_llm.py:441`）、`data_parallel_rank` 路由、reasoning parser 等在线特性。
- 两者都通过 `InputProcessor`（`v1/engine/input_processor.py:38`）把 prompt 转 `EngineCoreRequest`，通过 `OutputProcessor`（`v1/engine/output_processor.py:438`）把 `EngineCoreOutput` 转 `RequestOutput`。区别只在 `OutputProcessor` 是否带 per-request 队列（`RequestOutputCollector`）。

## 7. 关键文件
<!-- tags: files -->

| 路径 | 作用 |
|------|------|
| `vllm/v1/engine/core.py` | `EngineCore` / `EngineCoreProc` / `DPEngineCoreProc`，引擎内环 + ZMQ 包装 |
| `vllm/v1/engine/core_client.py` | `EngineCoreClient` 及 `Inproc/SyncMP/AsyncMP/DP*` 客户端 |
| `vllm/v1/engine/llm_engine.py` | `LLMEngine`（同步前端） |
| `vllm/v1/engine/async_llm.py` | `AsyncLLM`（异步前端，OpenAI server 用） |
| `vllm/v1/engine/input_processor.py` | prompt → `EngineCoreRequest` |
| `vllm/v1/engine/output_processor.py` | `EngineCoreOutput` → `RequestOutput`，detokenize |
| `vllm/v1/engine/__init__.py` | `EngineCoreRequest/Output(s)`、`FinishReason`、`EngineCoreRequestType` |
| `vllm/v1/engine/utils.py` | ZMQ 地址分配、`launch_core_engines`、`CoreEngineProcManager` |
| `vllm/v1/engine/coordinator.py` | `DPCoordinator`（DP 内部负载均衡协调） |
| `vllm/v1/request.py` | `Request`、`RequestStatus` |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` / `update_from_output()` |
| `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler`（async scheduling） |
| `vllm/v1/core/kv_cache_manager.py` / `block_pool.py` | KV cache 块管理、前缀缓存 |
| `vllm/v1/executor/abstract.py` | `Executor.get_class()` 分派 mp/ray/uni |
| `vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor`（默认多进程 worker） |
| `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` / `sample_tokens` |
| `vllm/v1/sample/sampler.py` | `Sampler.forward`（实际采样） |
| `vllm/engine/arg_utils.py` | `EngineArgs` / `AsyncEngineArgs` / `create_engine_config` |
| `vllm/config/vllm.py` | `VllmConfig` 聚合 + `__post_init__` 推导 |
| `vllm/entrypoints/llm.py` | 同步 `LLM` 入口 |
| `vllm/entrypoints/offline_utils.py` | `OfflineInferenceMixin._run_engine`（同步 step 循环） |
| `vllm/entrypoints/launchers/api_server/entry.py` | API server 启动、`build_async_engine_client` |
| `vllm/entrypoints/cli/serve.py` | `vllm serve` 子命令、`run_multi_api_server`、Rust frontend 分支 |
| `vllm/entrypoints/openai/chat_completion/serving.py` | OpenAI chat 服务，调 `engine_client.generate` |
| `vllm/renderers/` | chat 渲染层（`OnlineRenderer`、`RendererRegistry`，按模型族分实现；原 `vllm/inputs/preprocess.py` 已删除，逻辑迁入此处） |
| `vllm/parser/` | 模型族 tool-call parser（qwen3/deepseek/kimi…，`parser_manager.py` 统一注册） |
| `vllm/reasoning/` | reasoning parser（thinking 块解析，按模型族分实现） |
| `rust/` | 实验性 Rust frontend（axum HTTP + ZMQ 对接 Python engine） |

## 8. 配置 / 调优旋钮（架构相关）
<!-- tags: tuning, knobs, flags -->

- **进程/并行**：`--tensor-parallel-size`、`--pipeline-parallel-size`、`--data-parallel-size`、`--distributed-executor-backend`（`mp`/`ray`/`uni`）、`--data-parallel-external-lb` / `--data-parallel-hybrid-lb`、`--api-server-count`。
- **批处理/调度**：`--max-num-batched-tokens`（默认 2048）、`--max-num-seqs`（默认 128）、`--enable-chunked-prefill`、`--scheduling-policy`、`--async-scheduling`（默认自动）、`--stream-interval`。
- **KV cache / 显存**：`--gpu-memory-utilization`（默认 0.92）、`--kv-cache-memory-bytes`、`--block-size`、`--enable-prefix-caching`（默认开）、`--kv-cache-dtype`。
- **性能模式**：`optimization_level`（O0–O3，默认 O2）、`performance_mode`（`balanced`/`interactivity`/`throughput`）。
- **关键环境变量**：`VLLM_ENABLE_V1_MULTIPROCESSING`（默认 1，控制 LLM 是否用多进程 EngineCore）、`VLLM_ENGINE_READY_TIMEOUT_S`（默认 600）、`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`（默认 128，output_handler 分块大小）、`VLLM_LOG_STATS_INTERVAL`（默认 10s）、`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`（默认 5）。
- **优雅停机**：`VllmConfig.shutdown_timeout`（默认 0=立即 abort；>0 则 drain 在途请求）。
