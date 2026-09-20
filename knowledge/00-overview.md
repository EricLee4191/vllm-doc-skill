# vLLM 知识库总览

> 基于 vLLM main（`751f6807d9`，2026-09-19），最新 tag **v0.30.0rc2**（`fa6ff06066`，2026-09-18，release candidate）。v1 架构为默认且唯一的活跃引擎，v0 引擎已完全移除。本文是整个知识库的入口：先给全局地图，再导读 8 个子系统，最后给快速上手与部署优化速查。

## 1. vLLM 是什么
<!-- tags: intro, overview, 简介 -->

vLLM 是一个高性能 LLM 推理与服务引擎，核心贡献是 **PagedAttention**（KV cache 分页管理）与 **continuous batching**（每步动态组批），并在此之上构建了一套生产级 serving 栈：OpenAI 兼容 API、TP/PP/DP/EP 分布式、量化、投机解码、结构化输出、P/D 分离等。

**v1 架构的核心思想**（相对 v0 的重写）：

1. **前后端进程解耦**：调度器（Scheduler）与模型执行（ModelRunner）运行在独立的 **EngineCore 进程**，前端（API server / 离线 `LLM` 客户端）通过 **ZMQ + msgpack** 与之通信；ZMQ IO 在独立线程，可与 GPU 前向重叠。
2. **continuous batching 原生内建**：没有 "prefill phase / decode phase"，每个请求只有 `num_computed_tokens` 与 `num_tokens_with_spec` 两个计数器，调度器每步的目标是让前者追上后者——chunked prefill、prefix caching、speculative decoding 都是这一统一模型的推论。
3. **KV cache = block pool + 自动前缀缓存（APC）**：`BlockPool` 物理块池 + 链式块哈希，LRU 驱逐；抢占只有 **recompute**（无 v0 的 CPU swap）。
4. **async scheduling**：调度与执行重叠（`max_concurrent_batches=2`），默认按 executor 能力自动开启。
5. **编译与 CUDA Graph 深度集成**：`torch.compile`（`VLLM_COMPILE` 模式，piecewise 切图 + 自定义 Inductor pass）+ CUDA Graph（默认 `FULL_AND_PIECEWISE`：decode 整图 replay，prefill/mixed 走 piecewise）。
6. **Model Runner V2（v0.29 起默认）**：GPU 执行层重构为 `vllm/v1/worker/gpu/` 下的模块化 runner（`use_v2_model_runner` 默认 True），旧版 `gpu_model_runner.py` 降为 legacy 回退（`VLLM_USE_V2_MODEL_RUNNER=0`）。HiSparse、watermarking 等新特性强制要求 V2。

## 2. 架构地图
<!-- tags: architecture, map, 架构, dataflow, 数据流 -->

### 2.1 主数据流（一次 `/v1/chat/completions` 请求）
<!-- tags: dataflow, request, api-server, enginecore, worker, zmq, 数据流 -->

```mermaid
flowchart TB
    subgraph FE["API server 进程 (×N, FastAPI/uvicorn)"]
        A["OpenAI / Anthropic / gRPC 端点<br/>AsyncLLM 前端"]
        B["InputProcessor<br/>prompt+多模态 → EngineCoreRequest"]
        C["OutputProcessor<br/>detokenize → RequestOutput"]
    end

    subgraph EC["EngineCore 进程 (每 DP rank 一个)"]
        D["Scheduler.schedule()<br/>continuous batching / 分配 KV blocks"]
        E["KVCacheManager + BlockPool<br/>前缀缓存命中 / LRU 驱逐 / 抢占"]
        F["ModelExecutor (mp / ray / uni)"]
        D <--> E
        D --> F
    end

    subgraph WK["Worker 进程 (×TP×PP, 每进程 1 GPU)"]
        G["GPUModelRunner.execute_model<br/>_prepare_inputs → 模型 forward → logits"]
        H["Sampler / RejectionSampler<br/>→ sampled tokens"]
        G --> H
    end

    A --> B
    B -->|"ZMQ ROUTER (msgpack)"| D
    F -->|"collective_rpc 广播 SchedulerOutput"| G
    H --> F
    F -->|"ModelRunnerOutput"| D
    D -->|"update_from_output()<br/>ZMQ PUSH EngineCoreOutputs"| C
    C --> A
```

要点：

- **前端可选 Rust 实现（v0.28 新增，实验性）**：`rust/` 的 `vllm-frontend-rs` 用 axum 重建北向 OpenAI 兼容 HTTP 层，仍经 ZMQ + MessagePack 走既有 engine 边界，`VLLM_USE_RUST_FRONTEND=1` 启用（默认 `0`，生产仍用 Python 前端）。
- **`EngineCore.step()`**（`vllm/v1/engine/core.py:634`）是引擎内环：`scheduler.schedule()` → `executor.execute_model(non_block=True)` → `get_grammar_bitmask()` → `executor.sample_tokens()` → `scheduler.update_from_output()`。
- **`execute_model` 与 `sample_tokens` 分离**，让 forward 的 GPU kernel 入队后 CPU 可继续准备下一步（async scheduling 的基础）。
- 同步离线路径（`LLM.generate`）走 `SyncMPClient` + 用户循环 `step()`；在线路径（`AsyncLLM`）走 `AsyncMPClient` + 后台 `output_handler` 协程流式 yield。
- 进程拓扑：主进程（launcher）→ N 个 API server 子进程 + DP 个 EngineCore 进程 → 每个 EngineCore 经 `MultiprocExecutor`/`RayDistributedExecutor` 拉起 TP×PP 个 Worker 进程。

### 2.2 横切层（自底向上）
<!-- tags: layers, 分层, distributed, attention, quantization, 执行层 -->

```
┌──────────────────────────────────────────────────────────────────┐
│ 分布式层  parallel_state (TP/PP/DP/EP/PCP/DCP 进程组)             │
│   custom all-reduce / NCCL symm-mem / all2all (DeepEP/NIXL/MoRI) │
│   KV connector (P/D 分离: NIXL/LMCache/Mooncake) · EPLB · Elastic EP
├──────────────────────────────────────────────────────────────────┤
│ Attention 层  FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX /     │
│   FLASHMLA / FLASHINFER_MLA / CUTLASS_MLA …（按平台+能力自动选择）│
├──────────────────────────────────────────────────────────────────┤
│ 量化层  FP8 / AWQ / GPTQ / ModelOpt(NVFP4) / MXFP4 / 在线量化 …  │
│   QuantizationConfig → QuantizeMethod → kernel 选择器 (scaled_mm)│
├──────────────────────────────────────────────────────────────────┤
│ 模型层  model_executor: 380+ 模型架构 + 并行算子                  │
│   (ColumnParallelLinear / RowParallelLinear / FusedMoE / RMSNorm)│
├──────────────────────────────────────────────────────────────────┤
│ 执行层  torch.compile (VLLM_COMPILE, piecewise + 自定义 pass)     │
│   + CUDA Graph (FULL_AND_PIECEWISE) + 显存 profiling             │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 配置体系
<!-- tags: vllmconfig, config, 配置, engineargs, 解析链 -->

所有配置聚合在 `VllmConfig`（`vllm/config/vllm.py:354`）：`model_config` / `cache_config` / `parallel_config` / `scheduler_config` / `compilation_config` / `attention_config` / `speculative_config` / `kv_transfer_config` / `quant_config` / `lora_config` / `observability_config` …。解析链：**CLI flag → `EngineArgs`（`vllm/engine/arg_utils.py:446`，字段名与 flag 一一对应）→ `create_engine_config()` 逐个子 config → `VllmConfig.__post_init__` 跨 config 推导**（如按 executor 能力定 `async_scheduling`）。环境变量集中在 `vllm/envs.py`。

## 3. 子系统导读
<!-- tags: index, navigation, 导读 -->

| # | 子系统 | 一句话概括 | 子文档 |
|---|--------|-----------|--------|
| 01 | 核心架构与请求生命周期 | v1 进程/线程模型（API server ↔ ZMQ ↔ EngineCore ↔ Worker）、`EngineCore`/`EngineCoreClient` 类族、请求完整数据流与 `VllmConfig` 配置体系 | [01-architecture-core](./01-architecture-core.md) |
| 02 | 调度与 KV Cache | 统一 token 模型的 `Scheduler`（continuous batching、chunked prefill、recompute 抢占）、`KVCacheManager`/`BlockPool` 块池与自动前缀缓存、`num_gpu_blocks` 容量计算、KV offload | [02-scheduling-kv-cache](./02-scheduling-kv-cache.md) |
| 03 | 模型执行与编译优化 | `GPUModelRunner` 的 step 主流程（`_prepare_inputs`/`ForwardContext`）、权重加载与 TP 切分、`torch.compile` piecewise 编译、CUDA Graph capture/dispatch、显存 profiling | [03-model-execution](./03-model-execution.md) |
| 04 | Attention 后端与底层算子 | "一个 backend = 三个类"的抽象与自动选择机制、FLASH_ATTN/FLASHINFER/TRITON/MLA 各 backend 适用场景、`csrc/` CUDA kernel 与 Triton kernel 清单、切换旋钮 | [04-attention-kernels](./04-attention-kernels.md) |
| 05 | 分布式并行与 KV 传输 | TP/PP/DP/EP/PCP/DCP 的 rank 布局与进程组、all-reduce/all2all 通信后端、Executor 层、P/D 分离（KV connector）、EPLB 与 Elastic EP | [05-distributed](./05-distributed.md) |
| 06 | 量化与多硬件平台 | 量化方法全景（FP8/AWQ/GPTQ/ModelOpt/MXFP4/在线量化/KV 量化）与 kernel 选择器、`Platform` 抽象与 CUDA/ROCm/CPU/XPU/TPU 各平台部署要点 | [06-quantization-hardware](./06-quantization-hardware.md) |
| 07 | 投机解码与高级特性 | draft+verify 投机解码（ngram/EAGLE/MTP/draft_model…）与 `RejectionSampler`、`Sampler` 采样链、结构化输出（xgrammar bitmask）、LoRA 多适配器、多模态、reasoning parser | [07-advanced-features](./07-advanced-features.md) |
| 08 | 部署、API 服务与性能调优 | CLI 入口与 Docker 镜像、OpenAI 兼容端点清单、关键 `VLLM_*` 环境变量、吞吐/延迟/显存调优指南、部署决策树与压测工具 | [08-deployment-optimization](./08-deployment-optimization.md) |

## 4. 快速上手
<!-- tags: quickstart, 入门, llm-class, serve, docker -->

### 4.1 离线推理（Python `LLM` 类）
<!-- tags: llm-class, offline, 离线推理, generate, sleep-mode -->

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          tensor_parallel_size=2, max_model_len=8192,
          gpu_memory_utilization=0.90)
out = llm.generate(["Hello"], SamplingParams(temperature=0.7, max_tokens=128))
```

适合批量推理、评测、脚本；`LLM` 还支持 `chat()` / `embed()` / `classify()` / `score()`、`sleep()/wake_up()`（sleep mode）、`reset_prefix_cache()`。

### 4.2 在线服务（`vllm serve`）
<!-- tags: serve, online, api-server, openai, 在线服务 -->

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --served-model-name my-model
```

启动 OpenAI 兼容 HTTP API（`/v1/chat/completions` 等，OpenAI SDK 直接可用，`base_url="http://host:8000/v1"`）；也支持 `--grpc`、`--headless`（多节点 DP 从节点）、`--config config.yaml`。

### 4.3 Docker
<!-- tags: docker, 镜像, ipc-host, cache, nonroot -->

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  vllm/vllm-openai:latest meta-llama/Llama-3.1-8B-Instruct
```

`--ipc=host` 是多进程共享内存必需；挂载 `vllm-cache` 持久化 `torch.compile` 缓存可让二次启动免编译。非 root 用 `vllm-openai-nonroot` 镜像 + `--user 2000:0`。

## 5. 部署优化速查
<!-- tags: deployment, tuning, 速查, cheatsheet -->

浓缩自 [08-deployment-optimization](./08-deployment-optimization.md) 的调优要点：

| 目标 | 关键旋钮 | 说明 |
|------|---------|------|
| **吞吐** | `--max-num-batched-tokens`、`--max-num-seqs` | 最核心的两个旋钮（默认按硬件：H100 8192-16384/1024，A100 2048-8192/256）；`--performance-mode throughput` 自动翻倍；小模型大卡可拉到 16384 |
| **延迟** | `--performance-mode interactivity`、较小的 `max_num_batched_tokens` | 细粒度 cudagraph capture（1..32）；decode 不被长 prefill 拖慢，改善 ITL/TPOT |
| **显存** | `--gpu-memory-utilization`（默认 0.92）、`--kv-cache-dtype fp8`、`--max-model-len` | KV 显存 = 请求显存 − 权重 − 峰值激活 − cudagraph 估算；FP8 KV 使 KV 容量约翻倍；`--kv-cache-memory-bytes` 可精确指定 |
| **并行策略** | 单卡放得下→DP 多副本；单节点→`-tp N`；跨节点→`-tp 8 -pp 节点数 --distributed-executor-backend ray`；MoE→`-dp N --enable-expert-parallel --all2all-backend deepep_low_latency` | TP 越大 all-reduce 开销越大，小模型别盲目拉 TP |
| **前缀复用** | `--enable-prefix-caching`（默认开）、`--prefix-caching-hash-algo` | 多轮对话/RAG/few-shot 收益巨大；只省 prefill 不省 decode |
| **编译/启动** | 默认 `-O2`（VLLM_COMPILE + FULL_AND_PIECEWISE）；`--enforce-eager` 仅调试用；挂载 `VLLM_CACHE_ROOT` 持久卷 | 编译缓存命中可省 5~20s+ 启动时间；`--compilation-config` 精细控制 capture sizes |
| **抢占/OOM** | 调高 `gpu_memory_utilization` → 调低 `max_num_seqs` → 加 TP → 加 PP；KV 量化 | v1 抢占是 RECOMPUTE；看日志 `preempted by PreemptionMode.RECOMPUTE` 与 `/metrics` preemption 计数 |
| **投机解码** | `--speculative-config '{"method":"ngram"/"eagle"/"mtp", "num_speculative_tokens":K}'` | ngram 无需 draft 权重最省；EAGLE/MTP 接受率更高；大 batch 用 `num_speculative_tokens_per_batch_size` 动态 K |
| **P/D 分离** | `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"/"kv_consumer"}'` | 长 prefill 与 decode 分池部署，connector 可选 NIXL/LMCache/Mooncake |
| **输入瓶颈** | `--api-server-count`、`VLLM_USE_FASTOKENS=1` | API server 进程横向扩容；Rust tokenizer 加速 tokenize 密集负载 |
| **观测/压测** | `/metrics`（Prometheus）、`VLLM_LOG_STATS_INTERVAL`、`vllm bench serve` / `vllm bench sweep` | 盯 queue 长度、`gpu_cache_usage`、preemption、TTFT/TPOT/ITL 分位；上线前压出 P99 基线 |

**部署决策树（浓缩版）**：

```
权重单卡放得下？
├─ 是 → TP=1；要多吞吐 → --data-parallel-size=G（或 G 个实例 + LB）
└─ 否 → 单节点 8 卡放得下？
    ├─ 是 → -tp 8（MoE 大模型改 -dp N --enable-expert-parallel）
    └─ 否 → -tp 8 -pp 节点数 --distributed-executor-backend ray
显存仍紧 → --kv-cache-dtype fp8 → 量化权重 (fp8/awq/gptq) → 降 --max-model-len → --cpu-offload-gb
```

**上线前检查清单**：启动日志确认 `GPU KV cache size` 与 `Maximum concurrency` 满足业务并发且无 preemption；编译缓存已持久化；`--api-key` 只保护 `/v1` 前缀（`/health` 等不鉴权，生产放反向代理后）；多节点每节点设 `VLLM_HOST_IP` + `--ipc=host`；`/metrics` 接告警。

## 6. 关键文件（顶层索引）
<!-- tags: files, 源码索引 -->

| 路径 | 作用 |
|------|------|
| `vllm/v1/engine/core.py` | `EngineCore` / `EngineCoreProc`：引擎内环 + ZMQ 包装 |
| `vllm/v1/engine/core_client.py` | `EngineCoreClient`（Inproc/SyncMP/AsyncMP/DP* 前端客户端） |
| `vllm/v1/engine/async_llm.py` / `llm_engine.py` | 在线 `AsyncLLM` / 同步 `LLMEngine` 前端 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler`：调度、抢占、KV 集成 |
| `vllm/v1/core/kv_cache_manager.py` / `block_pool.py` | KV cache 块管理与前缀缓存 |
| `vllm/v1/worker/gpu/model_runner.py`（V2，默认）/ `gpu_model_runner.py`（V1 legacy）/ `gpu_worker.py` | ModelRunner 主逻辑 / Worker 初始化与显存 profiling |
| `vllm/v1/executor/multiproc_executor.py` | 默认多进程 worker 编排 |
| `vllm/v1/attention/` | attention backend 抽象、选择器与实现 |
| `vllm/compilation/` | torch.compile 集成、CUDA Graph |
| `vllm/distributed/` | 进程组、通信后端、KV transfer、EPLB、Elastic EP |
| `vllm/model_executor/` | 模型定义、算子层、权重加载、量化 |
| `vllm/config/vllm.py` | `VllmConfig` 聚合与推导 |
| `vllm/engine/arg_utils.py` | `EngineArgs`：全部 CLI 参数与默认值逻辑 |
| `vllm/entrypoints/` | CLI（`serve`/`bench`/`run-batch`）、OpenAI 端点、`LLM` 类 |
| `rust/`（`vllm-frontend-rs`） | Rust 前端（实验性，v0.28+）：axum HTTP + ZMQ engine client |
| `vllm/envs.py` | 全部 `VLLM_*` 环境变量注册表 |
| `docker/Dockerfile` | 官方镜像（`vllm-openai` 等 target） |

## 7. 增量更新记录
<!-- tags: changelog, 增量更新, baseline, 基线 -->

- **2026-09-12**：基线从 `f32b17b6d6`（2026-08-21，v0.28.0rc1）推进到 `2f59050eda`（2026-09-12，最新 tag **v0.29.0**，`98dff2a81d`）。区间 985 commits。主要变更：
  - **Model Runner V2 成为默认**（`use_v2_model_runner` 默认 True，`vllm/v1/worker/gpu/` 模块化 runner；旧 `gpu_model_runner.py` 降为 legacy 回退）。详见 03 §1。
  - **KV cache 物理布局重构**（RFC #42082）：新增 `vllm/v1/kv_cache_layout.py` 的 `KVCacheLayout` 枚举（`LBNHC/LBHNC/LHBNC/BLHNC/BLNHC/BHLNC` + 兼容 `NHD/HND`），`VLLM_KV_CACHE_LAYOUT` 扩展为 7 种取值。详见 02 §5。
  - **HiSparse**（host-resident sparse-MLA decode 热缓冲，`vllm/v1/hisparse/`，强制 V2）+ **Engram/PLE**（n-gram 嵌入存储与分片，`vllm/config/engram.py` + ETP 进程组）。详见 02/05。
  - **文本水印 watermarking**（`vllm/v1/watermarking/`，Gumbel-max + Philox PRF，`--watermark-config`，强制 V2）。详见 07。
  - **调度器队列上限**：`SchedulerConfig.max_num_queued_reqs` / `max_num_queued_tokens`（`--max-num-queued-reqs`/`--max-num-queued-tokens`）。详见 02。
  - **投机解码**：新增 `dflash2`、`gemma4_mtp`、`qwen4_exp_mtp`/`hy_v4_mtp`/`glm5_next_mtp` 等 MTP 变体；MTP 模型类型扩到 27 种。详见 07。
  - **Attention**：新增 `B12X`（SM12x paged causal）、`FLASHINFER_MLA_SPARSE_SM90` 等 backend；MLA sparse/indexer 大幅扩展。详见 04。
  - **分布式**：`weight_transfer/` 新增 `sharded_rdt`（分片 RDMA 权重传输）引擎；`parallel_state` 新增 ETP 组与 `suspend/resume_device_comms`。详见 05。
  - **部署**：新增 scale-out 端点（`/v1/chat/completions/render`、`/inference/v1/generate` 等，`VLLM_ENABLE_SCALE_OUT_ENDPOINTS`）。详见 08。
  - 模型架构数从 290+ 增至 **380+**（新增 DeepSeek-V4/V4.1、GLM-5.3-Flash、Qwen4-Exp、Kimi-K3 等）。
- **2026-09-19**：基线从 `2f59050eda`（2026-09-12，v0.29.0）推进到 `751f6807d9`（2026-09-19，最新 tag **v0.30.0rc2**，`fa6ff06066`，release candidate）。区间 419 commits。主要变更：
  - **水印支持投机解码**：新增 `dual_key_gumbel` 算法（双 key gumbel-max，`supports_speculative_decoding=True`，`alpha` 控制 key-B 概率，加权 early-fusion 检测 `vllm/v1/watermarking/gumbel.py:208`）；`spec_decode.py` 新增 `create_speculative_target_watermarker`/`create_speculative_draft_watermarker` 与 `allow_target_only_watermarking`；`_check_watermarking_unsupported`（`vllm/config/vllm.py:1162`）约束 `draft_sample_method='probabilistic'`、`rejection_sample_method='standard'`、method ∈ {dspark,eagle,eagle3,mtp}。详见 07 §7.1。
  - **调度器 RUNNING 准入上限**：`SchedulerConfig.max_num_active_seqs`（`--max-num-active-seqs`，`vllm/config/scheduler.py:70`，`vllm/v1/core/sched/scheduler.py:129`，执行点 `vllm/v1/core/sched/scheduler.py:872-874`）；队列上限计数改用 `SharedAdmissionStats`（`vllm/v1/engine/admission_control.py:13`）跨进程无锁计数。详见 02 §2.2。
  - **投机解码自适应验证**：`enable_adaptive_verification`（`vllm/config/speculative.py:539`）+ `OnlineAcceptanceEstimator`（`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:313`，501 行，log-odds 线性模型，Triton accumulate/refit/predict kernels）。详见 07 §1.3。
  - **KV offload 增强**：back-pressure（#50045，`vllm/v1/kv_offload/tiering/backpressure.py`）、KVCR（#53624，`vllm/v1/kv_offload/tiering/kvcr/`）、per-request `max_load_tokens`（#55885，`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:367`）、chunked region 注册（#51081）、cgroup 检查（#54014）、MLA compact（#56799）。详见 02 §6.1。
  - **Model Runner V2**：DBO FULL CUDA graph（#51700，`vllm/v1/worker/gpu/model_runner.py:1750`）；Fast Start 支持 nnode>1（#55468）。详见 03。
  - **分布式**：MoonEP BF16 all2all backend（#52101，`vllm/model_executor/layers/fused_moe/prepare_finalize/moonep.py`）；DeepEPv2 async finalize（#52781/#57236）；PCP+DCP on sparse-MLA（#56157）；PCP decode-only FULL CUDA graphs（#53867）；NIXL attention-HMA PP push prefill（#50494）；Elastic EP CUDA graph 复用（#54985）。详见 05。
  - **Attention**：新增 `COMPOSITE` backend（`vllm/v1/attention/backends/composite.py`，Triton/FlashInfer 或 Triton/FlashAttention 组合，用于 multimodal prefix attention `mm_prefix`，selector 在 `use_mm_prefix=True` 时自动选择）。详见 04 §4.1。
  - **量化**：Quark 原生 W4A16 INT4/UINT4（#48606，`vllm/model_executor/layers/quantization/quark/schemes/quark_w4a16_int4.py`）；CPU FP8 W8A8 linear/MoE（#49942，`csrc/cpu/sgl-kernels/gemm_fp8_w8a8.cpp` + `moe_fp8_w8a8.cpp`）。详见 06。
  - **部署**：新增 `POST /release_kv_cache_memory` 端点（#44890，`vllm/entrypoints/serve/dev/sleep/api_router.py:31`）；`--enable-scale-out` CLI flag 取代 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` 环境变量（#55176，`vllm/entrypoints/scale_out/factories.py:72`）。详见 08。
  - **结构化输出重构**：`should_fill_bitmask`/`should_advance` 移除，改用 `_get_constraint_start`（`vllm/v1/structured_output/__init__.py:220`）/`validate_tokens`（`vllm/v1/structured_output/__init__.py:294`）；调度器 grammar 验证迁移到 `structured_output_manager.validate_tokens`（`vllm/v1/core/sched/scheduler.py:2526`）。详见 07。
  - **Engram**：新增 `embedding_across_dp`/`dp_shared_memory` 字段 + 异步预取 + DP 分片（#56512）。详见 07 §7.2。
