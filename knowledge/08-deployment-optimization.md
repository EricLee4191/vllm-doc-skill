# 部署、API 服务与性能调优

> 基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（`cd6ae1e0a0`，2026-08-20）。V1 架构为默认且唯一的活跃引擎。所有路径相对于仓库根目录 `/Users/baofeng/baofeng/github/vllm`。

## 1. 部署形态总览
<!-- tags: deployment, serve, docker, offline, api-server, 部署 -->

vLLM 的入口统一由 CLI 分发：`pyproject.toml` 中 `vllm = "vllm.entrypoints.cli.main:main"`，子命令定义在 `vllm/entrypoints/cli/`：

| 子命令 | 用途 | 源码 |
|---|---|---|
| `vllm serve` | 启动 OpenAI 兼容 HTTP API server（默认端口 8000） | `vllm/entrypoints/cli/serve.py` |
| `vllm serve --grpc` | 启动 gRPC server（需 `pip install vllm[grpc]`，依赖 `smg-grpc-servicer`） | `vllm/entrypoints/grpc_server.py` |
| `vllm chat` / `vllm complete` | 交互式客户端（走 OpenAI 协议） | `vllm/entrypoints/cli/openai.py` |
| `vllm bench serve/latency/throughput/startup/sweep` | 压测工具 | `vllm/entrypoints/cli/benchmark/` |
| `vllm run-batch` | 离线批量推理（JSONL 输入输出） | `vllm/entrypoints/cli/run_batch.py` |
| `vllm collect-env` | 收集环境诊断信息 | `vllm/entrypoints/cli/collect_env.py` |
| `vllm launch render` | 无 GPU 的 render（预处理/后处理）server | `vllm/entrypoints/cli/launch.py` |

### 1.1 离线推理（`LLM` 类）
<!-- tags: llm-class, offline, 离线, generate, sleep-mode -->

`vllm/entrypoints/llm.py` 中的 `class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin)` 是离线推理入口，构造参数直接透传 `EngineArgs`（`vllm/engine/arg_utils.py`）。核心方法：

- `LLM.generate(prompts, sampling_params)` / `LLM.chat(conversations, ...)` — 同步批量生成；
- `LLM.enqueue` / `LLM.enqueue_chat` + `LLM.wait_for_completion` — 异步队列式提交；
- `LLM.embed` / `LLM.classify` / `LLM.score` / `LLM.encode` — pooling 模型；
- `LLM.sleep(level=1|2)` / `LLM.wake_up()` — sleep mode（`enable_sleep_mode=True` 开启，权重 offload 到 CPU、释放 KV cache，可释放 90%+ 显存，见 `docs/features/sleep_mode.md`）；
- `LLM.reset_prefix_cache()`、`LLM.start_profile()/stop_profile()`、`LLM.get_metrics()`。

```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          tensor_parallel_size=2, max_model_len=8192,
          gpu_memory_utilization=0.90)
out = llm.generate(["Hello"], SamplingParams(temperature=0.7, max_tokens=128))
```

离线与在线共享同一套 `VllmConfig`（`vllm/config/vllm.py:357`）：`model_config` / `cache_config` / `parallel_config` / `scheduler_config` / `compilation_config` / `kv_transfer_config` / `speculative_config` / `observability_config` 等。

### 1.2 在线 API server（`vllm serve`）
<!-- tags: serve, api-server, 在线, 进程模型, rust-frontend -->

进程模型（V1）：`vllm serve` 启动 **API server 进程**（FastAPI + uvicorn，`vllm/entrypoints/launchers/api_server/entry.py`）+ **EngineCore 进程**（调度与执行）+ 每个 GPU 一个 worker 进程，进程间通过 ZMQ/消息队列通信（`VLLM_MQ_MAX_CHUNK_BYTES_MB` 控制大对象走 ZMQ）。`--api-server-count` 可横向扩展 API server 进程数（默认等于 `data_parallel_size`），用于输入处理成为瓶颈时扩容。

`vllm/entrypoints/openai/api_server.py` 现在只是**兼容 shim**（已发 DeprecationWarning），实际逻辑在 `vllm/entrypoints/launchers/`。路由注册入口：`vllm/entrypoints/launchers/api_server/routers.py:register_api_routers()`，按模型 `supported_tasks`（generate / pooling / transcription / realtime）动态挂载各 router。

常用启动参数（`vllm/entrypoints/openai/cli_args.py` 的 `FrontendArgs` + `AsyncEngineArgs`）：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --api-key token-abc123 \
  --served-model-name my-model \
  --config config.yaml        # 也可用 YAML 文件传全部参数
```

- `--api-key`（或 `VLLM_API_KEY`）只对 `/v1`、`/v2`、`/inference` 前缀的端点鉴权，`/invocations`、`/health` 等**不受保护**，生产环境要放反向代理后面（`docs/serving/online_serving/openai_compatible_server.md`）。
- `--headless`：只跑 engine 不跑 API server（多节点 DP 的从节点用）。
- `--grpc`：改用 gRPC 协议（`VllmEngineServicer`，proto 来自 `smg-grpc-proto`）。
- `--optimization-level`（`-O0`~`-O3`，默认 O2）：编译/cudagraph 优化档位，见 §5.4。
- `--performance-mode balanced|interactivity|throughput`（默认 balanced；throughput 会把 `max_num_batched_tokens` 和 `max_num_seqs` 默认值翻倍，`arg_utils.py:2810`）。

**Rust frontend（v0.28 新增，实验性）**：`rust/` 目录下的 `vllm-frontend-rs` 是 Python 前端的 Rust 替代实现，用 axum 重建北向 OpenAI 兼容 HTTP 层，仍通过 ZMQ + MessagePack 走既有 engine 边界与 Python engine 进程通信（`rust/README.md`）。分层 crate：`vllm-server`（axum HTTP）→ `vllm-chat`（模板渲染/reasoning/tool 解析）→ `vllm-text`（tokenizer/增量 detokenizer）→ `vllm-llm`（token-in/out facade）→ `vllm-engine-core-client`（ZMQ 传输）。Python 仍负责进程启动，把 Rust API server 作为受管 worker 拉起并传入继承的监听 socket：

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

当前**实验性、功能未对齐** Python 前端，生产环境默认仍用 Python 前端（`VLLM_USE_RUST_FRONTEND` 默认 `0`，`envs.py:165`）。

### 1.3 Docker 镜像
<!-- tags: docker, 镜像, target, ipc-host, nonroot -->

`docker/Dockerfile` 多阶段构建（ARG 默认值即版本事实源，`docker/versions.json` 自动生成）：

- 基础版本：`CUDA_VERSION=13.0.3`、`PYTHON_VERSION=3.12`、`UBUNTU_VERSION=24.04`、`NCCL_VERSION=2.30.7`；
- 关键 target：`vllm-openai`（默认，`ENTRYPOINT ["vllm", "serve"]`）、`vllm-openai-nonroot`（内置 `vllm` 用户 UID 2000）、`vllm-sagemaker`、`test`；
- `ARG INSTALL_KV_CONNECTORS=false` 可把 KV connector 依赖（NIXL/Mooncake/LMCache 等）打进镜像。

运行要点（`docs/deployment/docker.md`）：

```bash
# 官方镜像直接跑
docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \   # 持久化 torch.compile 缓存，二次启动免编译
  -p 8000:8000 \
  --ipc=host \                        # 多进程共享内存需要
  vllm/vllm-openai:latest meta-llama/Llama-3.1-8B-Instruct
```

- 非 root：`--user 2000:0`，挂载路径改到 `/home/vllm` 下；
- 多机 TP/PP：`--ipc=host` + `--net=host`（或暴露 NCCL 端口），多节点需设 `VLLM_HOST_IP`；
- 从源码构建：`docker build -f docker/Dockerfile .`（或 `docker buildx bake -f docker/docker-bake.hcl`）。

### 1.4 多机 / 大规模部署
<!-- tags: multi-node, 多机, ray, dp, pd-disaggregation -->

- 单机多卡默认 `mp`（multiprocessing）executor；跨节点用 `--distributed-executor-backend ray`（`ParallelConfig.distributed_executor_backend`，`vllm/config/parallel.py:243`）。
- 多节点 mp 后端：`--nnodes N --node-rank R --master-addr IP`。
- DP（数据并行）：`--data-parallel-size N`，三种 LB 模式（`docs/serving/data_parallel_deployment.md`）：
  - 内部 LB（单入口，默认）；
  - `--data-parallel-external-lb` / `--data-parallel-rank`：K8s 一 pod 一 rank，外部 LB；
  - `--data-parallel-hybrid-lb` / `--data-parallel-start-rank`：节点内 LB + 节点间外部 LB。
  - `--data-parallel-backend ray` 可用 Ray 统一拉起多节点 DP。
- 反向代理/多实例：`docs/deployment/nginx.md`（least_conn upstream）、`docs/deployment/k8s.md`（原生 K8s / Helm / KubeRay / llm-d / Dynamo 等）。
- P/D 分离（prefill/decode disaggregation）：`KVTransferConfig`（`vllm/config/kv_transfer.py`）+ `--kv-transfer-config`，connector 有 NIXL、Mooncake、LMCache、P2P 等（`examples/disaggregated/`）。

## 2. OpenAI 兼容 API
<!-- tags: openai-api, endpoints, anthropic, cohere, 兼容, 端点 -->

### 2.1 端点清单（从源码 router 提取）
<!-- tags: endpoints, 端点, openai, anthropic, cohere -->

| 端点 | 说明 | 源码 |
|---|---|---|
| `POST /v1/chat/completions` | Chat API（支持流式、tools、reasoning parser） | `entrypoints/openai/chat_completion/api_router.py:41` |
| `POST /v1/chat/completions/batch` | Chat 批量 | 同上 :78 |
| `POST /v1/completions` | Completions API（`suffix` 不支持） | `entrypoints/openai/completion/api_router.py:35` |
| `POST /v1/responses`、`GET /v1/responses/{id}`、`POST /v1/responses/{id}/cancel` | Responses API | `entrypoints/openai/responses/api_router.py` |
| `GET /v1/models` | 模型列表 | `entrypoints/openai/models/api_router.py:20` |
| `POST /v1/embeddings` | Embedding（pooling 模型） | `entrypoints/pooling/embed/api_router.py:29` |
| `POST /v2/embed`、`/score`、`/v1/score`、`/rerank`、`/v1/rerank`、`/v2/rerank`、`/pooling`、`/classify` | 其他 pooling 任务 | `entrypoints/pooling/*/api_router.py` |
| `POST /v1/audio/transcriptions`、`/v1/audio/translations` | ASR（Whisper 类模型） | `entrypoints/speech_to_text/*/api_router.py` |
| `POST /v1/messages`、`/v1/messages/count_tokens` | **Anthropic 兼容** | `entrypoints/anthropic/api_router.py` |
| `POST /cohere/v2/chat` | Cohere 兼容（`VLLM_ENABLE_COHERE_API=1` 开启） | `entrypoints/cohere/api_router.py` |
| `POST /inference/v1/generate`、`/abort_requests` | vLLM 原生 token-in/token-out 接口 | `entrypoints/scale_out/token_in_token_out/api_router.py` |
| `GET /health`、`/load`、`/version`、`/metrics` | 健康检查/负载/Prometheus | `entrypoints/serve/instrumentator/` |
| `POST /tokenize`、`/detokenize`、`GET /tokenizer_info` | tokenizer 服务 | `entrypoints/serve/tokenize/api_router.py` |
| `POST /sleep`、`/wake_up`、`GET /is_sleeping` | sleep mode（`VLLM_SERVER_DEV_MODE=1` 才挂载 dev 路由） | `entrypoints/serve/dev/sleep/api_router.py` |
| `POST /reset_prefix_cache`、`/reset_mm_cache`、`/reset_encoder_cache` | 缓存管理（dev 模式） | `entrypoints/serve/dev/cache/api_router.py` |

### 2.2 兼容程度
<!-- tags: compatibility, 兼容, openai-sdk, extra-body, 差异 -->

- 请求/响应协议与 OpenAI 对齐；OpenAI Python SDK 直接可用（`base_url="http://host:8000/v1"`）。
- vLLM 扩展参数通过 `extra_body` 传入，如 `top_k`、`structured_outputs`（JSON schema/regex/EBNF，backend 可选 `xgrammar`/`guidance`/`outlines`/`lm-format-enforcer`，`vllm/config/structured_outputs.py`）、`priority`（配合 `--scheduling-policy priority`）。
- 已知差异：`/v1/completions` 不支持 `suffix`；chat 的 `user` 字段被忽略；`parallel_tool_calls=false` 保证每请求至多一个 tool call。
- 默认会应用 HF 仓库里的 `generation_config.json` 覆盖采样默认值，`--generation-config vllm` 可禁用。
- `--served-model-name` 可改 `/v1/models` 返回的名字（支持列表）；`VLLM_SKIP_MODEL_NAME_VALIDATION=1` 让任意 model 名通过（代理/网关场景）。
- `--enable-request-id-headers` 支持 `X-Request-Id` 透传。

## 3. 关键环境变量（`vllm/envs.py`）
<!-- tags: env-vars, vllm-envs, 环境变量 -->

`envs.py` 用 `environment_variables` dict 注册所有 `VLLM_*` 变量（惰性求值，服务初始化后 `enable_envs_cache()` 缓存）。**注意：本版本已移除 `VLLM_USE_V1`（V1 是唯一引擎）和 `VLLM_ATTENTION_BACKEND`（改用 `--attention-config backend=...`，见 `vllm/config/attention.py` 的 `AttentionConfig.backend`）**。对部署/性能影响最大的变量：

### 3.1 进程与分布式
<!-- tags: env-vars, 进程, 分布式, multiproc, dp -->

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `1` | V1 中 LLM 类是否用多进程 engine（API server 总是多进程）。调试可设 0 单进程跑 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | worker 启动方式 `fork`/`spawn`。Jupyter/已初始化 CUDA 的进程里用 `spawn`；`--numa-bind` 会强制 spawn |
| `VLLM_HOST_IP` | 空 | 多节点时各节点必须显式设置，否则 Ray/NCCL 选错网卡 |
| `VLLM_PORT` | 空 | 分布式通信起始端口（多端口需求时递增） |
| `VLLM_RPC_BASE_PATH` | tmpdir | API server 与 engine 进程间 IPC socket 目录 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | TP>1 时 execute_model RPC 超时 |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | 单次 engine 迭代超时 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | 启动等待 engine 就绪超时 |
| `VLLM_SKIP_P2P_CHECK` | `1` | 跳过 P2P 检测；custom allreduce 挂死时可设 0 强制校验 |
| `VLLM_DISABLE_PYNCCL` | `0` | 禁用 pynccl 回退 torch.distributed |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` / `VLLM_ALLREDUCE_USE_FLASHINFER` | `1`/`1` | allreduce 后端开关 |
| `VLLM_DP_RANK` / `VLLM_DP_SIZE` / `VLLM_DP_MASTER_IP` / `VLLM_DP_MASTER_PORT` | — | 外部 DP 部署（external LB 模式）的 rank/拓扑 |
| `VLLM_RAY_PER_WORKER_GPUS` | `1.0` | Ray 下每 worker 的 GPU 数（分数可共卡） |

### 3.2 性能/编译
<!-- tags: env-vars, 性能, 编译, compile-cache, cudagraph -->

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_DISABLE_COMPILE_CACHE` | `0` | 禁用 torch.compile 缓存（缓存目录在 `VLLM_CACHE_ROOT`） |
| `VLLM_USE_AOT_COMPILE` | torch≥2.10 时 `1` | AOT 编译，warmup 阶段完成编译 |
| `VLLM_FORCE_AOT_LOAD` | `0` | 缓存未命中时直接报错而非静默重编译 |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `binary` | 编译缓存格式；`unpacked` 便于调试但多进程不安全 |
| `VLLM_ENABLE_STARTUP_PLAN` | `0` | 持久化启动内存规划（`VLLM_CACHE_ROOT/startup_plan/`），命中指纹时跳过 memory profiling |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | 内存 profiling 时估算 cudagraph 占用（v0.21+ 默认开） |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `0` | 0=GC freeze 加速 cudagraph 捕获 |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | FlashInfer top-k/top-p sampler |
| `VLLM_USE_DEEP_GEMM` / `VLLM_MOE_USE_DEEP_GEMM` | `1`/`1` | DeepGEMM（Hopper/Blackwell FP8 GEMM） |
| `VLLM_DEEP_GEMM_WARMUP` | `relax` | `skip`/`full`/`relax`：DeepGEMM JIT warmup 策略，`skip` 可显著缩短启动 |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `128` | 流式输出批处理粒度：调大降 host 开销/提吞吐，调小降 ITL 方差 |
| `VLLM_BATCH_INVARIANT` | `0` | batch 不变性（确定性输出，需 SM≥9.0） |
| `VLLM_GPU_SYNC_CHECK` | 空 | `warn`/`error`：检测 GPU 同步点，性能调试用 |
| `VLLM_LOG_STATS_INTERVAL` | `10` | 周期日志（含 preemption 计数等）秒数 |

### 3.3 服务与资源限制
<!-- tags: env-vars, 服务, 资源限制, api-key, cache-root -->

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_API_KEY` | 空 | API key（可多个，逗号分隔） |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `0` | engine 崩溃后 API server 是否存活（便于拉日志） |
| `VLLM_SERVER_DEV_MODE` | `0` | 挂载 `/reset_prefix_cache`、`/sleep` 等 dev 端点 |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `5` | HTTP keep-alive 秒数 |
| `VLLM_MAX_N_SEQUENCES` | `16384` | 单请求 `n` 参数上限（防 DoS） |
| `VLLM_MAX_COMPLETION_PROMPTS` | `1024` | 单请求 prompt 列表长度上限 |
| `VLLM_MAX_STOP_STRINGS` | `4` | stop 字符串数上限 |
| `VLLM_SKIP_MODEL_NAME_VALIDATION` | `0` | 接受任意 model 名（网关场景） |
| `VLLM_ENABLE_COHERE_API` | `0` | 开启 `/cohere/v2/chat` |
| `VLLM_ENABLE_RESPONSES_API_STORE` | `0` | Responses API 的 store 选项（内存态，有泄漏风险） |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | 编译/资产缓存根目录（容器里务必挂载） |
| `VLLM_NO_USAGE_STATS` / `VLLM_DO_NOT_TRACK` | `0` | 关闭遥测 |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `0` | 允许 `max_model_len` 超过模型 config 上限 |
| `VLLM_MODEL_REDIRECT_PATH` | 空 | 模型名→本地目录映射（json/sv 文件） |
| `VLLM_USE_MODELSCOPE` | `0` | 从 ModelScope 拉模型 |
| `VLLM_USE_FASTOKENS` | `0` | 用 Rust fastokens 替换 HF fast tokenizer（v0.23+，tokenizer 密集负载收益大） |
| `VLLM_USE_RUST_FRONTEND` | `0` | 用 Rust `vllm-frontend-rs` 替代 Python 前端（v0.28+，实验性，见 §1.2） |
| `VLLM_LOGGING_LEVEL` | `INFO` | 日志级别 |
| `VLLM_KV_CACHE_LAYOUT` | 空 | `NHD`/`HND`，KV cache 内存布局 |
| `VLLM_MM_HASHER_ALGORITHM` | `blake3` | 多模态内容哈希（FIPS 合规用 sha256/sha512） |
| `VLLM_IMAGE_FETCH_TIMEOUT` / `VLLM_MAX_IMAGE_PIXELS` | `5` / ~179M | 多模态媒体抓取/解压炸弹防护 |

MoE/EP 相关（DeepSeek 类大 MoE 常用）：`VLLM_DEEPEP_BUFFER_SIZE_MB`（1024）、`VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE`、`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL`（GB200 MNNVL）、`VLLM_MOE_SKIP_PADDING`（1）。KV 传输相关：`VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`（5600）、`VLLM_P2P_SIDE_CHANNEL_HOST/PORT`（5710）、`VLLM_MOONCAKE_BOOTSTRAP_PORT`（8998）。

## 4. 性能调优指南
<!-- tags: tuning, throughput, latency, oom, preemption, prefix-caching, 调优, 排查 -->

### 4.1 吞吐 vs 延迟：`max_num_seqs` / `max_num_batched_tokens`
<!-- tags: throughput, latency, max-num-seqs, max-num-batched-tokens, 旋钮 -->

两者定义在 `SchedulerConfig`（`vllm/config/scheduler.py`），是**最核心的两个旋钮**：

- `max_num_batched_tokens`：单个 engine 迭代最多处理的 token 数（prefill+decode 合计预算）。
- `max_num_seqs`：单迭代最多并发的序列数。

默认值按硬件自动选择（`EngineArgs.get_batch_defaults`，`vllm/engine/arg_utils.py:2574`）：

| GPU | LLM 类 | API server |
|---|---|---|
| ≥160GB（B200/B300） | batched_tokens=16384, seqs=1024 | 16384 / 1024 |
| ≥70GB 非 A100（H100/H200） | 16384 / 1024 | 8192 / 1024 |
| 其他（A100 等） | 8192 / 256 | 2048 / 256 |

调参经验（`docs/configuration/optimization.md`）：

- **小值（如 2048）** → 更好的 ITL/TPOT（decode 不被长 prefill 拖慢），适合延迟敏感；
- **大值（>8192，小模型大显存 GPU）** → 更好 TTFT 和总吞吐；
- 约束：`max_num_batched_tokens >= max_num_seqs`；关闭 chunked prefill 时必须 `>= max_model_len`（`scheduler.py:verify_max_model_len`）；
- `--performance-mode throughput` 自动把两个默认值翻倍；
- 用 `vllm bench sweep serve --serve-params params.json` 网格扫描（`docs/benchmarking/sweeps.md` 给了 `max_num_seqs` × `max_num_batched_tokens` 的 JSON 示例）。

### 4.2 显存管理
<!-- tags: memory, 显存, gpu-memory-utilization, kv-cache, 量化 -->

- `--gpu-memory-utilization`（`CacheConfig.gpu_memory_utilization`，默认 **0.92**，`vllm/config/cache.py:80`）：vLLM 实例占用的显存比例（权重+激活+KV cache）。OOM 时调低，吞吐不够时调高。
- `--kv-cache-memory-bytes`：直接指定每 GPU KV cache 字节数，**设置后忽略 gpu_memory_utilization**，更精细。启动日志会打印建议值（`vllm/v1/worker/gpu_worker.py:794`），回灌可跳过 memory profiling 加速启动（文档中写作 `--kv-cache-memory`，代码中 flag 为 `--kv-cache-memory-bytes`）。
- `--kv-cache-dtype`（`CacheConfig.cache_dtype`）：`auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2`/`nvfp4`/`turboquant_*`/`int8_per_token_head` 等。**FP8 KV cache 使 KV 显存减半、并发翻倍**，精度损失通常可接受（H100/H200/B 系列支持）。
- 量化权重：`--quantization` 支持 `awq`/`gptq`/`gptq_marlin`/`awq_marlin`/`fp8`/`modelopt`/`modelopt_fp4`/`mxfp8`/`nvfp4`/`compressed-tensors`/`torchao` 等（`vllm/model_executor/layers/quantization/__init__.py:12`）。HF 上直接下量化好的 checkpoint 即可（如 RedHatAI 的 FP8 模型）；在线量化用 `fp8_per_tensor`/`fp8_per_block` 等 shorthand。
- `--max-model-len`：限制上下文长度直接省 KV cache（长上下文是 KV 显存大头）。
- `--enforce-eager` / 缩小 `cudagraph_capture_sizes`：省 cudagraph 占用的显存（`docs/configuration/conserving_memory.md`）。
- `--cpu-offload-gb N`：权重 offload 到 CPU（牺牲速度换容量）；`--mm-processor-cache-gb`（默认 4）控制多模态预处理缓存。
- 启动后看日志两行关键指标：`GPU KV cache size: N tokens` 和 `Maximum concurrency for X tokens per request: Yx`（`docs/serving/parallelism_scaling.md`）——Y 低于需求就加卡或降 `max_model_len`。

### 4.3 Prefix Caching
<!-- tags: prefix-caching, 前缀缓存, hash, dp, 命中率 -->

- `--enable-prefix-caching`（`CacheConfig.enable_prefix_caching`，默认 **True**，`cache.py:107`）：按 block（默认 `block_size=16`）哈希前缀，命中则跳过 prefill。多轮对话、RAG 长文档、few-shot 场景收益巨大（`docs/features/automatic_prefix_caching.md`）。
- 只省 prefill 不省 decode；无共享前缀的负载无收益。
- `--prefix-caching-hash-algo`：`sha256`（默认）/`xxhash`（更快，多租户有碰撞风险）；
- 与 DP 结合：内部 LB 按各 DP rank 的 running/waiting 队列分发，可配合外部路由让同前缀请求落同一 rank 提高命中率。
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`（已迁移为 `--prefix-cache-retention-interval`）：sliding-window/Mamba 模型的 checkpoint 保留间隔。

### 4.4 并行策略选择（TP/PP/DP/EP）
<!-- tags: parallel, 并行策略, tp, pp, dp -->

`ParallelConfig`（`vllm/config/parallel.py:119`）字段：`tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size`、`prefill_context_parallel_size`、`enable_expert_parallel`、`all2all_backend` 等。

决策规则（`docs/serving/parallelism_scaling.md`）：

1. **模型单卡放得下** → 不并行，或 DP 多副本扩吞吐；
2. **单节点多卡放不下** → `--tensor-parallel-size N`（N=节点内卡数）。TP 是单节点首选；
3. **跨节点** → `TP=每节点卡数` × `PP=节点数`（`--distributed-executor-backend ray`）。PP 也用于无 NVLink 的卡（如 L40S）减少通信；
4. **MoE 大模型（DeepSeek/Qwen3-MoE 等）** → DP attention + EP/TP expert：`--data-parallel-size N --enable-expert-parallel`，expert 层按 `DP×TP` 切分；`--all2all-backend` 可选 `allgather_reducescatter`（默认）/`deepep_high_throughput`/`deepep_low_latency`/`nixl_ep`/`flashinfer_nvlink_*`（GB200 用 MNNVL 系）；
5. **扩吞吐而非扩模型** → `--data-parallel-size N`（模型完整复制 N 份，`--max-num-seqs` 是 per-rank 的）；
6. 多 socket 服务器加 `--numa-bind`（自动检测 GPU-NUMA 映射，`numactl` 绑定，容器需 `--cap-add SYS_NICE`）；
7. 多模态 encoder 可用 `mm_encoder_tp_mode="data"` 做 batch 级 DP（TP8 下吞吐/TTFT 约 +10%，`docs/configuration/optimization.md`）。

经验法则：TP 越大 allreduce 开销越大，小模型别盲目拉 TP；70B dense 典型配置 TP4/TP8；DeepSeek-V3 类 671B MoE 典型 DP8×EP（8 节点 H100）或单机 TP8+EP。

### 4.5 编译与 CUDA Graph
<!-- tags: compile, cudagraph, 编译, optimization-level, enforce-eager -->

- `--enforce-eager`：完全禁用 torch.compile 和 cudagraph。启动最快、显存最省，但 decode 性能明显下降。**只在调试/测启动时间/显存紧张时用**。
- 默认（`-O2`）：`CompilationMode.VLLM_COMPILE`（Inductor 后端 + piecewise 编译 + 自定义 pass）+ `CUDAGraphMode.FULL_AND_PIECEWISE`（`vllm/config/compilation.py:53`）。cudagraph 消除 decode 的 kernel launch 开销，小 batch 收益最大。
- `-O0`~`-O3`（`VllmConfig.optimization_level`，默认 O2，`vllm/config/vllm.py:130`）：O0 无优化最快启动；O1 Dynamo+Inductor+PIECEWISE cudagraph；O2 加 FULL_AND_PIECEWISE；O3 目前等同 O2。
- `--compilation-config`（或 `-cc.mode=3`、`-cc.cudagraph_capture_sizes=[1,2,4,8]`）：精细控制。`cudagraph_capture_sizes` 默认到 `max_num_seqs`；显存不够时截断（如 `[1,2,4,8,16]`）。
- `--performance-mode interactivity`：小 batch 细粒度 capture（1..32 每个都抓），padding 开销最小，延迟最优。
- 编译缓存：`VLLM_CACHE_ROOT/torch_compile_cache`，跨容器/机器可拷贝；任何模型/配置/相关 `VLLM_*` 环境变量/硬件变化都会使缓存失效（`envs.py:compile_factors()`）。

### 4.6 Chunked Prefill 与 Continuous Batching
<!-- tags: chunked-prefill, continuous-batching, 调度, async-scheduling, decode-priority -->

- V1 中 **chunked prefill 默认开启**（`SchedulerConfig.enable_chunked_prefill=True`，`scheduler.py:74`）：长 prompt 按 `max_num_batched_tokens` 预算切块，与 decode 请求混批。
- 调度策略：**decode 优先**——先排所有 pending decode，剩余预算再排 prefill。这同时改善了 ITL（decode 不被 prefill 阻塞）和 GPU 利用率（compute-bound prefill 与 memory-bound decode 互补）。
- 长文本/混合负载影响：
  - 长 prompt 不再独占一个迭代 → TTFT 更可预测，不会把整批 decode 卡死；
  - 纯短 prompt 高并发场景 chunked prefill 几乎无感；
  - `--long-prefill-token-threshold`：超过该长度的 prompt 视为"长"（默认 0=不限制）；
  - 关闭 chunked prefill（`--no-enable-chunked-prefill`，部分模型不支持）时 `max_num_batched_tokens` 必须 ≥ `max_model_len`。
- `--async-scheduling`（默认按 executor 能力自动开启）：调度与执行重叠，减少 GPU 空泡，改善延迟与吞吐。
- `--scheduling-policy priority`：按请求 `priority` 字段调度（默认 `fcfs`）。
- `--watermark`（`SchedulerConfig.watermark`，默认 0）：准入时预留的 KV block 比例，防频繁抢占。

### 4.7 抢占（Preemption）
<!-- tags: preemption, 抢占, recompute, 处理, metrics -->

V1 默认抢占模式是 **RECOMPUTE**（重算，比 swap 开销低）。日志出现 `Sequence group ... preempted by PreemptionMode.RECOMPUTE` 说明 KV cache 不够。处理优先级（`docs/configuration/optimization.md`）：

1. 调高 `gpu_memory_utilization`；
2. 调低 `max_num_seqs` / `max_num_batched_tokens`；
3. 调大 `tensor_parallel_size`（每卡权重更省，KV 更多，但有通信代价）；
4. 调大 `pipeline_parallel_size`（层切分，延迟代价）。

Prometheus `/metrics` 有 preemption 计数；`--disable-log-stats` 默认关着周期日志，生产建议开着（`VLLM_LOG_STATS_INTERVAL` 控制频率）。

### 4.8 常见 OOM / 性能问题排查
<!-- tags: oom, 排查, 吞吐, 延迟, 多机 -->

**启动 OOM（权重+激活装不下）**
- 加 TP/PP；上量化（FP8/AWQ/GPTQ）；`--cpu-offload-gb`；降 `max_model_len`；
- 看启动日志的 memory 分解（weights / peak activation / CUDAGraph，`gpu_worker.py` 会打印各部分 GiB）；
- 多进程 TP 下每个进程都读全量 checkpoint，磁盘 IO 慢可用 sharded checkpoint（`examples/features/sharded_state/`）。

**运行期 OOM / 频繁抢占**
- 见 §4.7；另查 `--kv-cache-dtype fp8`、`max_num_seqs` 是否过大、是否有超长请求（`max_model_len` 设太大 → 单请求 KV 占用大，`kv_cache_max_concurrency` 日志可见）。

**吞吐上不去**
- 看 `/metrics` 的 `gpu_cache_usage`、queue 长度、preemption 计数；
- 输入处理瓶颈 → `--api-server-count` 扩容、`VLLM_USE_FASTOKENS=1`；
- 小模型大卡 → 提高 `max_num_batched_tokens`（>8192）；
- MoE → 确认 EP + 合适的 `all2all_backend`（跨节点用 deepep/nixl）；
- 确认没开 `--enforce-eager`、编译缓存命中（启动日志有 "cache hit" 字样）。

**延迟高（TTFT/ITL）**
- TTFT 高：长 prompt 排队 → 提高 `max_num_batched_tokens`；prefix 复用差 → 开/查 prefix caching；
- ITL 高：batch 太大 → 降 `max_num_seqs`；cudagraph 未命中（batch 超出 capture sizes）→ 检查 `cudagraph_capture_sizes`；
- 用 `vllm bench serve` 分别看 TTFT/TPOT/ITL 的 P50/P99 定位（§6）。

**多机通信问题**
- `VLLM_HOST_IP` 每节点必设；NCCL 变量（`NCCL_SOCKET_IFNAME` 等）建议在集群创建时注入（`docs/serving/distributed_troubleshooting.md`）；
- custom allreduce 挂死 → `--disable-custom-all-reduce` 回退 NCCL，或 `VLLM_SKIP_P2P_CHECK=0` 排查 P2P。

## 5. 配置/调优旋钮速查（CLI flag → config 字段）
<!-- tags: flags, cheatsheet, cli, 速查 -->

| 类别 | 常用 flag | 默认 |
|---|---|---|
| 模型 | `--model`、`--served-model-name`、`--max-model-len`、`--dtype`、`--quantization`、`--enforce-eager`、`--trust-remote-code`、`--enable-sleep-mode` | dtype=auto |
| 显存/KV | `--gpu-memory-utilization`、`--kv-cache-memory-bytes`、`--kv-cache-dtype`、`--block-size`、`--enable-prefix-caching`、`--cpu-offload-gb` | 0.92 / auto / 16 / on |
| 调度 | `--max-num-batched-tokens`、`--max-num-seqs`、`--max-num-scheduled-tokens`、`--long-prefill-token-threshold`、`--scheduling-policy`、`--async-scheduling`、`--watermark`、`--stream-interval` | 见 §4.1 |
| 并行 | `--tensor-parallel-size`、`--pipeline-parallel-size`、`--data-parallel-size`、`--enable-expert-parallel`、`--all2all-backend`、`--distributed-executor-backend`、`--numa-bind` | 1/1/1 |
| 编译 | `--optimization-level`、`--compilation-config`（`-cc.*`）、`--performance-mode` | O2 / balanced |
| 投机解码 | `--speculative-config`（method/model/num_speculative_tokens，`vllm/config/speculative.py:85`） | 关 |
| 结构化输出 | `--structured-outputs-config`（backend: xgrammar/guidance/outlines/lm-format-enforcer） | auto |
| 可观测 | `--disable-log-stats`、`--otlp-traces-endpoint`、`--collect-detailed-traces`、`--kv-cache-metrics`、`--enable-mfu-metrics` | 关 |
| 前端 | `--host`、`--port`、`--api-key`、`--api-server-count`、`--allowed-origins`、`--ssl-certfile`、`--root-path`、`--middleware` | 8000 |
| P/D 分离 | `--kv-transfer-config`（`kv_connector`/`kv_role`/`kv_rank` 等） | 关 |

## 6. Benchmark 工具
<!-- tags: benchmark, bench, 压测, 评测 -->

`benchmarks/` 目录下的旧脚本（`benchmark_serving.py` 等）已废弃，统一用 CLI（`vllm/entrypoints/cli/benchmark/`，实现委托 `vllm/benchmarks/`）：

```bash
# 在线压测（最常用）：吞吐 + TTFT/TPOT/ITL
vllm bench serve \
  --model <served-model-name> \
  --backend openai \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt --dataset-path ShareGPT.json \
  --num-prompts 1000 \
  --request-rate inf \        # 或固定 QPS；配合 --burstiness 做泊松突发
  --max-concurrency 64        # 客户端并发上限
# 输出：Request/Output/Total token throughput、Mean/Median/P99 TTFT、TPOT、ITL

# 离线吞吐（LLM 类，无 server）
vllm bench throughput --model <hf-model> --dataset-name sharegpt --num-prompts 1000

# 单 batch 延迟（纯模型延迟，无排队）
vllm bench latency --model <hf-model> --batch-size 32 --input-len 512 --output-len 256

# 启动时间
vllm bench startup --model <hf-model>

# 参数网格扫描（自动起 server 逐组压测并可视化）
vllm bench sweep serve --serve-cmd "vllm serve M --tensor-parallel-size 4" \
  --bench-cmd "vllm bench serve --model M --num-prompts 200" \
  --serve-params sweep.json   # [{"max_num_seqs":32,"max_num_batched_tokens":1024}, ...]
```

- 数据集：`random`（合成，`--input-len/--output-len`）、`sharegpt`、`burstgpt`、`hf`（任意 HF dataset）、`sonnet`、prefix repetition 等（`docs/benchmarking/cli.md` 有完整表格与下载命令）。
- 指标口径：TTFT=首 token 时间；TPOT=除首 token 外每 token 平均时间；ITL=token 间隔。文档明确这些指标在**客户端**测量，跨工具对比要看测量点。
- 生产级压测官方推荐 [GuideLLM](https://github.com/vllm-project/guidellm)（`docs/benchmarking/cli.md`）。
- 专项脚本（`benchmarks/`）：`benchmark_prefix_caching.py`（prefix cache 命中率）、`benchmark_batch_invariance.py`、`kernels/`（kernel 微基准）、`attention_benchmarks/` 等。

## 7. 部署决策树 / 检查清单
<!-- tags: decision-tree, checklist, 决策树, 上线检查 -->

**给定：模型大小 M、GPU 数 G（单卡显存 S）、目标 T（吞吐 or 延迟）**

```
1. 权重能否单卡放下？(M×2B for bf16, 或量化后)
   ├─ 能 → TP=1。G>1 且目标是吞吐 → --data-parallel-size=G（或起 G 个实例+LB）
   └─ 不能 → 2.

2. 单节点(8卡)能放下？
   ├─ 能 → --tensor-parallel-size=8（MoE 大模型改 --data-parallel-size + --enable-expert-parallel）
   └─ 不能 → --tensor-parallel-size=8 --pipeline-parallel-size=节点数 --distributed-executor-backend ray

3. MoE 模型跨节点？
   → DP attention + EP expert：--data-parallel-size N --enable-expert-parallel
     --all2all-backend deepep_low_latency（延迟优先）/ deepep_high_throughput（吞吐优先）

4. 显存仍紧（KV cache 不够）？
   → --kv-cache-dtype fp8 → 量化权重(fp8/awq/gptq) → 降 --max-model-len
     → 仍不行：--cpu-offload-gb / 加卡

5. 目标=延迟：
   --performance-mode interactivity（细粒度 cudagraph）
   适度 --max-num-batched-tokens（2048~8192）、--max-num-seqs 适中
   开 prefix caching（默认开）、chunked prefill（默认开）
   考虑投机解码 --speculative-config（draft model / ngram / eagle）

6. 目标=吞吐：
   --performance-mode throughput（默认值×2）
   --max-num-batched-tokens ≥ 8192（小模型大卡可 16384）
   --max-num-seqs 拉高直到 KV cache 打满（看 "Maximum concurrency" 日志）
   --gpu-memory-utilization 0.92~0.95
   多副本 DP 线性扩吞吐
```

**上线前检查清单**

- [ ] 启动日志确认：`GPU KV cache size` 与 `Maximum concurrency` 满足业务并发；无 preemption 警告；
- [ ] 编译缓存已挂载持久卷（`VLLM_CACHE_ROOT`），二次启动无重编译；
- [ ] `--api-key` 只保护 `/v1` 前缀 → 生产放 nginx/网关，`/invocations`、`/health` 不暴露公网；
- [ ] 多节点：每节点 `VLLM_HOST_IP`、`--ipc=host`、NCCL 网络变量；
- [ ] 监控：`/metrics`（Prometheus）接告警——queue 长度、`gpu_cache_usage`、preemption 计数、TTFT/TPOT 分位；
- [ ] 用 `vllm bench serve` 以真实分布（sharegpt/自有流量）压出 P99 TTFT/TPOT 基线，调参后用 `vllm bench sweep` 回归；
- [ ] 弹性/成本：多模型共卡考虑 `--enable-sleep-mode` + `/sleep`、`/wake_up`；
- [ ] 长上下文/多轮：确认 prefix caching 生效（`/reset_prefix_cache` 可手动清，dev 模式）。

## 8. 关键文件
<!-- tags: files -->

| 文件 | 内容 |
|---|---|
| `vllm/entrypoints/cli/main.py`、`cli/serve.py` | CLI 分发、`vllm serve` 实现（headless/LB 模式判定） |
| `vllm/entrypoints/launchers/api_server/entry.py`、`routers.py` | API server 启动与路由注册 |
| `vllm/entrypoints/openai/cli_args.py` | `FrontendArgs`（host/port/api-key/ssl 等前端参数） |
| `vllm/entrypoints/openai/{chat_completion,completion,models,responses}/api_router.py` | OpenAI 各端点 |
| `vllm/entrypoints/pooling/`、`anthropic/`、`cohere/`、`speech_to_text/` | 其他协议端点 |
| `vllm/entrypoints/llm.py` | 离线 `LLM` 类 |
| `vllm/entrypoints/grpc_server.py` | gRPC server |
| `rust/`（`vllm-frontend-rs`） | Rust 前端（实验性）：axum HTTP + ZMQ engine client，`VLLM_USE_RUST_FRONTEND=1` 启用 |
| `vllm/engine/arg_utils.py` | `EngineArgs`/`AsyncEngineArgs`：全部引擎 CLI 参数、默认值逻辑（`get_batch_defaults`） |
| `vllm/config/vllm.py` | `VllmConfig` 聚合、`OptimizationLevel`、`PerformanceMode` |
| `vllm/config/scheduler.py` | `max_num_batched_tokens`/`max_num_seqs`/chunked prefill/policy/watermark |
| `vllm/config/cache.py` | `gpu_memory_utilization`/`kv_cache_dtype`/prefix caching/KV offloading |
| `vllm/config/parallel.py` | TP/PP/DP/EP、all2all backend、executor backend、NUMA |
| `vllm/config/compilation.py` | `CompilationMode`/`CUDAGraphMode`/capture sizes |
| `vllm/config/attention.py` | `AttentionConfig`（attention backend 选择） |
| `vllm/config/kv_transfer.py` | P/D 分离 KV connector 配置 |
| `vllm/config/observability.py` | metrics/OTLP trace 配置 |
| `vllm/envs.py` | 全部 `VLLM_*` 环境变量注册表 |
| `docker/Dockerfile`、`docker/versions.json` | 官方镜像构建 |
| `vllm/benchmarks/serve.py`、`vllm/entrypoints/cli/benchmark/` | 压测实现与 CLI |
| `docs/configuration/optimization.md`、`conserving_memory.md` | 官方调优/省显存指南 |
| `docs/serving/{data_parallel,expert_parallel,parallelism_scaling,distributed_troubleshooting}.md` | 并行与排障 |
| `docs/deployment/{docker,k8s,nginx}.md` | 部署 |
| `docs/benchmarking/{cli,sweeps}.md` | 压测手册 |
| `examples/basic/`、`examples/deployment/`、`examples/disaggregated/`、`examples/scale_out/` | 可运行示例 |
