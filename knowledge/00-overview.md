# vLLM 知识库总览

> 基于 vLLM main（`458ba2edf8`，2026-10-07），最新 tag **v0.31.1rc0**（`e37e51dd24`，2026-10-06；main 领先其 116 commits；上一正式 release 为 v0.31.0，`db9527a468`，2026-10-02，main 领先其 582 commits；再上一正式 release 为 v0.30.0，`9ed533eb4a`，2026-09-20）。v1 架构为默认且唯一的活跃引擎，v0 引擎已完全移除。本文是整个知识库的入口：先给全局地图，再导读 8 个子系统，最后给快速上手与部署优化速查。

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
- **`EngineCore.step()`**（`vllm/v1/engine/core.py:630`）是引擎内环：`scheduler.schedule()` → `executor.execute_model(non_block=True)` → `get_grammar_bitmask()` → `executor.sample_tokens()` → `scheduler.update_from_output()`。
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

所有配置聚合在 `VllmConfig`（`vllm/config/vllm.py:357`）：`model_config` / `cache_config` / `parallel_config` / `scheduler_config` / `compilation_config` / `attention_config` / `speculative_config` / `kv_transfer_config` / `quant_config` / `lora_config` / `observability_config` …。解析链：**CLI flag → `EngineArgs`（`vllm/engine/arg_utils.py:465`，字段名与 flag 一一对应）→ `create_engine_config()` 逐个子 config → `VllmConfig.__post_init__` 跨 config 推导**（如按 executor 能力定 `async_scheduling`）。环境变量集中在 `vllm/envs.py`。

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
| `vllm/entrypoints/` | CLI（`serve`/`bench`/`run-batch`/`snapshot`）、OpenAI 端点、`LLM` 类 |
| `vllm/snapshot/` | **v0.30 新增**：initialized engine snapshots（CRIU 引擎快照，`vllm snapshot create/restore`） |
| `rust/`（`vllm-frontend-rs`） | Rust 前端（实验性，v0.28+）：axum HTTP + ZMQ engine client |
| `vllm/envs.py` | 全部 `VLLM_*` 环境变量注册表 |
| `docker/Dockerfile` | 官方镜像（`vllm-openai` 等 target） |

## 7. 增量更新记录
<!-- tags: changelog, 增量更新, baseline, 基线 -->

- **2026-10-07**：基线从 `30d4032363`（2026-10-07）推进到 `458ba2edf8`（2026-10-07），最新 tag **v0.31.1rc0**（`e37e51dd24`，2026-10-06；main 领先其 116 commits）。区间共 201 commit（795 文件，+42562/−9028），主线：
  - **调度/核心**：free KV cache queue 开销削减（#60269，`v1/core/kv_cache_utils.py`）；pause(mode="wait") 排空 async KV loads（#60572）；pooling chunked prefill 用满 context（#48039）；`execute_dummy_batch` RPC 等待受 execute-model timeout 约束（#60398）；encoder cache 对重复多模态输入保留引用（#59942）；session 截断时丢弃过期 block hash（#59103）；Mamba spec 块在 prefill checkpoint 步退役（#59759）。
  - **投机解码/MRV2**：NGram GPU 投机解码实现（#40704，ModelRunner V2）；AR speculators 更名 StandaloneAR/TargetDependentAR（#60335）；dynamic K 支持 1.2~1.3x kernel 提速（#57053）；MTP fused multi-step 去掉 eager metadata rebuild（#58463）；异构词表 draft 模型留在 MRV1（#59541）；DSpark fill-in 保留 bonus KV slot（#59105）；speculative metrics 在输出合并时保留（#55708）。
  - **前端/Rust**：ZMQ 端口 TOCTOU 以继承 listener 消除（#54113）；gRPC forbidden token sequences + cache usage（#59837）；`SchemaRoot` 在参数 coercion 与 grammar 间共享（#59408）；移除 PyO3 tool-parser bridge（#59744）；MiniMax M3 tool parser 移植到 parser engine（#59743）；per-session profiling 控制（#57875）；RL entrypoints 合并（#57849）；`offline_utils.py` 迁 `entrypoints/common`（#58052）；Cohere Chat v2 logprobs（#59072）+ 4xx 客户端错误（#60309）；Anthropic count_tokens 应用 output_config/thinking（#59180）。
  - **结构化输出**：JSON schema 嵌套深度封顶防 500/API-server 卡死（#60036）；xgrammar 多分支 allOf 标记不支持（#59061）；`disable_any_whitespace` 真正禁用空白（#58067）；Guidance `disable_additional_properties` 保留字面值（#58709）；tool-call grammar 从 prompt 实际渲染的 tools 构建（#59879）；tool-parser/tokenizer 兼容性启动期校验（#59749）。
  - **Attention/KV**：NVFP4 KV cache 支持 SM8x/SM12x + FlashInfer（#46963）；UltraQuant 4-bit KV cache backend（FlyDSL D=256，#57057）；GLM 默认切 fp8 KV cache，E2E 吞吐 +2.3%~5.5%（#60140）；FlashInfer autotune 缓存按 rank 持久化修复 rank-0 死锁（#57635）；HiSparse per-request residency 缓存（#60083）+ 启动期拒绝 cudagraph_mode=FULL（#59688）；flashMLA sparse Q heads pad 到 64（#60029）；FP8 KV cache 扩到 Triton DiffKV/MiMo-V2.6-Flash（#58128）。
  - **量化**：per-token NVFP4 MoE 支持 ReLU2（#56740）；SM90 上 Humming 优先于 Marlin（#56997）；online quantization API 线性层重量化（MXFP8→FP8 PTPC，#55684）；block-FP8 DeepGEMM experts 跳过 padding 工作（#59128）；OAI Triton MXFP4 MoE 启用 SM12x（#58877）；safetensors 头读取支持其他文件名 checkpoint（#60411）。
  - **KV offload/NIXL**：KVCR 走 secondary-tier factory 恢复（#58088）+ pool-and-index API（#59899）；SimpleCPU 尊重 speculative cacheability（#60071）；NIXL bump 1.5.0（#56907）+ per-region replicate flags 按 block 数尺寸化（#60107）+ 本地失效 pull peer 元数据恢复（#55471）。
  - **平台/ROCm**：ROCm 23 commits（AITER MegaMoEV2 for DSv4 #59685、GLM-5.3-Flash BF16 splitk sparse MLA decode #58584、FlyDSL GDN prefill backend #57560、native merge_attn_states gfx942/950 #60159、Kimi-K3 AttnRes+FP8 融合 #59069）；XPU 8 commits（tuned Mamba SSU B70 #57565、Triton fused MoE 调优 #53065）；Python 3.10 EOL 移除（#60402）；Transformers bump 5.19.0（#60381）；tpu-inference v0.31.0（#60190）；vendored DeepGEMM 迁 TORCH_LIBRARY abi3（#48962）。
  - **模型**：EmbeddingGemma2 多模态 pooling 架构（#60254）+ config 解析精简（#60289）；Nemotron 3.5 ASR（#59827）；LongCat-Flash MLA norms 加载期缩放（#60080）；DSv4.1 mega-attention wq_b/wo_a 加载期置换（#60064）；Mamba2 internal prefill checkpoints（#57329）；Qwen3-VL 未知 fps 除零修复（#60501）。
  - **其他**：每 DP engine 独立全局 RNG streams（#59788）；`suppress_stdout` 改重定向 fd 1（#59336）；`LLM.score()` 不再改调用方参数（#59959）；watermark 兼容性校验（#56801）；vllm-bench Mooncake 风格 timed-traces replay（#55937）；CI 清理多项（TPU dead scripts #60591、H100 DP+EP 可选 #60540、automatic sharding 五步 #60492）。
  - 受影响文档：00/01/02/03/04/05/06/07/08 基线头全同步 `458ba2edf8`；02/03/04/05/06/07/08 新增 2026-10-07 基线新增段（01 无实质主题变更，仅基线头）。

- **2026-10-01**：基线从 `72e7874fa6`（2026-09-30，v0.30.1rc0-477）推进到 `bc21cba967`（2026-10-01，最新 tag **v0.31.0rc2**（2026-09-29），main 领先其 191 commits；上一正式 release 仍为 v0.30.0，`9ed533eb4a`，2026-09-20）。区间 46 commits / 308 文件（+11772/−1717）。主要变更：
  - **调度/核心**：encoder-only 长 prompt（>1 step）调度修复（#59029，02 §2）；LoRA 路径纳入 prefix-cache block hash（#59335，02 §3.1）；mamba prefill checkpoint block 预留与 prompt-end eviction（align mode）修复（#59175，02 §3.1）；KV offloading replicated_layout 检测扩到多 group MLA（#57652，05 §9）。
  - **投机解码**：MiniMax-M3 EAGLE3 PP>1 aux-state relay + per-stage FlashInfer autotune（#57197，03/05）；Mamba MTP 支持 FlashInfer ReplaySSM（#52928，04/07）；watermarking 投机解码 context 去重（#56807，07 §7.1）；CT 格式 unquantized ngram 支持（#59431，07 §1）；async scheduling 精度测试（#55840，CI）。
  - **HiSparse**：三项修复——host pool 不再喂 device KV cache residency 指标（#58725）、GPU prefix copy 在 hit 分配后采用（#59282）、请求完成后保留 host prefix publication（#59007，04 §2）。
  - **前端/解析**：chat_parsing 核心从 Transformers 移植（#58602，`vllm/parser/chat_parsing/` 新包，01/07）；Rust frontend Shutdown 控制 RPC（#59316，08 §1.2）；score centering via top-k processed logprobs 文档（#59361）。
  - **安全**：chat template 资源耗尽 DoS 修复（GHSA-4hhp-h66f，#50300）；共享内存多模态 cache handle 鉴权（#59357，07 §5）；pyjwt/rand 及 Dependabot 依赖升级（#59427/#59315）。
  - **可观测性**：`--custom-histogram-buckets` 覆盖 histogram bucket families（#48867，08 §1.1）。
  - **分布式/平台**：GPU/XPU worker 共享 workspace 与 model runner init（#59200，03/06）；GPU Model Runner V2 支持 PP+PCP（#59139，05 §2.2）；XPU LayerNorm 走 fused SYCL kernel（#57172，06 §XPU）；CPU aarch64 Conv1d 优化 kernel（#54093，06 §CPU）；Mistral Pixtral vision encoder 编译支持（#57168，07 §5.1）。
  - **ROCm**：Kimi-K3 MLA decode KV-cache write + Q-prep AITER 融合（#57640，04 §2）；GLM-5.3-Flash kpool top-k indices 单 Triton kernel 适配 AITER（#58008，04 §2）；MiniMax-M3 Triton indexer decode grid 重调 + SM12.0 split-K（#56151，04 §2）；compressed-tensors MoE 权重设备端转置（#59253）+ AITER MoE intermediate 分配期 padding（#55368，06 §ROCm）；The Rock dockerfile Triton 3.8 + mori 构建修复（#59287/#59372）。
  - **其他**：DeepSeek-V4 VL dummy image 取 worst-case 尺寸（#59271，07 §5.1）；CI 去抖/去重多项（#59508/#59507/#58125/#59499/#59256/#59237/#59379/#59338）；PR checklist skill（#57084）；docs build gate 恢复（#59411）；CODEOWNERS 更新（#59369/#59456）。
- **2026-09-30**：基线从 `924707f1bf`（2026-09-27，v0.30.1rc0-427）推进到 `72e7874fa6`（2026-09-30，最新 tag **v0.31.0rc2** 尚未落在 main，上一正式 release 为 v0.30.0，`9ed533eb4a`，2026-09-20）。区间 205 commits / 826 文件（+43757/−5897）。主要变更：
  - **投机解码/调度**：MRV2 支持独立 draft 模型投机解码（#43091，03 §1/07 §1.3）；acceptance estimator Triton 重编译避免（#57107，07 §1.3）；FlashInfer trtllm-gen fused multi-step draft decode（#58371，04 §2）；scheduler `skipped_waiting` 队列重构为 `kv_holding_waiting` + `deferred_waiting`（#58947，02 §2.2）；streaming continuation max_tokens 刷新（#57676）+ logprobs 保留（#57447）+ async scheduling handoff 竞态修复（#58259）；fixed-token prefill scoring（#54335，02 §2.2）；one-token prompt tail FULL decode graph（#58400，03 §1）；Mamba/GDN metadata 跨 KV cache group 复用（#58762，03 §1）；从未 propose 的 draft slot 拒绝（#58784，03 §1）。
  - **引擎/前端**：UniProc EngineCore 启动线程按 CPU 数设上限（#58946，01 §2）；per-engine utility 结果合并对齐 Rust client（#59240，01 §2）；RL weight checker（#51350，08 §1.1）；sleep-mode API 响应与操作指标对齐（#52864，08 §1.1）；HTTP 权重操作结果与并发跟踪（#55781，08 §1.1）；in-flight 请求保持同一 DP engine（#59017，05 §2.1）；`add_dp_placement_groups` 不再要求 `ray[default]`（#57648，05 §2.1）；Responses API 系列修复（#59307/#55596/#59298/#50502/#58927/#55771/#59173，08 §1.2）；Anthropic named tool calls 报告为 tool_use（#47598，08 §1.2）；LoRA adapter 名与 served model 名冲突拒绝（#59286，08 §1.2/07 §4）；batched chat completions Harmony `adjust_request` 修复（#58958）+ 非流式每 choice 独立 parser（#58939）+ 从 adjusted requests 采样（#58929）；forced named tool choice 空参数约束（#45290）；Step3p5 forced tool choice 走 XML parser（#51810）；Rust frontend roundtrip output grammar 经 XGrammar 回放（#59143，08 §1.2）+ MiMo structural-tag builder（#59148，08 §1.2）；Python Harmony 依赖切换到 oss-harmony（#55128，07 §6.2）；Harmony "Unexpected token" 抑制（#59254，07 §6.2）。
  - **Attention/MLA**：GLM-5.3-Flash SM90 sparse MLA fp8 plan dtype 修复 + indexer prefill workspace 右尺寸化（#55222，04 §2）；ROCm AITER Gluon sparse MLA kernel（#53492，04 §2）；ROCm DSv4.1 paged MXFP4 sparse indexer 走 AITER MQA-logits（#58671，04 §2）；ROCm AITER ASM round-robin decode 路由支持 DCP 多 token 验证（#56861，04 §2）；ROCm ragged sparse-MLA indices 构建丢弃 -1 哨兵（#58058，04 §2）；HiSparse MTP 验证行 union residency kernel（#59235，04 §2）+ 无 host backing 不分配 GPU 页（#59036，04 §2）；FlashInfer 升级 0.7.0.post1（#59323，04 §2）；DiffusionGemma CUDA graph replay attention mask 冻结修复（#51994，03 §5.1）；GLM-5.3-Flash KDA prefill checkpoint（#56960，03 §5.1）。
  - **分布式/KV**：WideEP DeepEPv2 默认自动选择 hybrid 模式（#57991，05 §2.4）；Mooncake hybrid/MLA KV 打包成合并传输区域（#57952，05 §9.1）+ bootstrap 端口启动期保持绑定（#58967）+ bootstrap 注册超时重试（#58919）；MoRIIO K3 DSpark hybrid READ（#57700，05 §9.1）；NIXL PP push prefill 支持 packed MLA KV 布局（#50499，05 §9.1）；Elastic EP 支持 MRV2（#53934，05 §9.3）；EPD encoder-only 异步步跳过采样（#58490，05 §9.1）；eager 模式下 FlashInfer all-reduce workspace 创建不再 sync-police/重试（#58498，05 §3.1）；stateless process-group 超时时显式传递（#58611，05 §3.1）；prefix-cache extra_keys 按来源打标签（#51899，02 §3.1）；单个 KV cache group 无法满足的 prefix_match_unit 拒绝（#58021，02 §3.1）；Mamba prompt-end prefill checkpoint 在 sparse retention 下保留（#59146，02 §3.1）。
  - **量化/硬件**：per-token NVFP4 CuTe-DSL MoE 后端（#50030，06 §1.2）；ROCm DSv4.1 AITER opt-in a4w4（FP4 激活）MoE（#58819，06 §ROCm）；MiMo-V2.6 MXFP4 支持 gfx942（#58262，06 §ROCm）；GLM-5.3-Flash stride-aware decode KDA（#57979，06 §ROCm）；DSv4/DSv4.1 ROCm sparse prefill 复用共享 prefill chunk plan（#58405/#58539，06 §ROCm）；PLE prefetch pinned buffer 延迟分配（#58797，06 §ROCm）；CT WNA16 MoE 改用规范 N-first 权重格式（#52798，06 §1.2）；sleep(level=2) 不再把 CT KV scale 清零（#57163，06 §1.2）；moe_wna16 w13 zero-point shard 拆分修复（8-bit 非对称 GPTQ MoE，#58950，06 §1.2）；CPU 向量化 Sampler kernel（#53913，06 §CPU/07 §2）；Whisper W4A16 量化走 CPU WNA16 kernel（#58268，06 §CPU）；Mamba2 量化 in_proj 权重/尺度 TP>1 加载修复（#58083，06 §CPU）；DiffusionGemma 窄 canvas 同步调度（#59107，06 §CPU）；PowerPC auto dtype 优先 bfloat16（#58528，06 §CPU）；POWER10 VSX 启用 W4A16（AWQ & GPTQ）量化（#59149，06 §CPU）；XPU EC producer 实例使用 encoder-only model runner（#59320，06 §XPU）；sm_120 batch-invariant matmul 表新增 TP=2/4/8 per-rank shapes（#58495，06 §CUDA）；batch invariance 测试模型扩到 gemma-2/SmolLM2/Qwen2.5-Coder/Llama-3.2-1B（#54441，06 §CUDA）。
  - **编译/启动**：torch.compile 日志行标注正在编译的组件（#48133，03 §4.1）；standalone torch.compile 缓存 relocation 后加载修复（#52142，03 §4.1）；functionalized split slices 规范化用于 fusion pass（#57299，03 §4.1）；MRV2 支持 stock torch.compile 模式（#59079，03 §1/§4.1）；Fast Start 支持 PP（#55477，03 §2.5）+ daemon 持有权重计入 `gpu_memory_utilization`（#57298，03 §2.5）+ weight cache daemon 新增 `/health` 端点（#58552，03 §2.5）。
  - **多模态**：flat/scoped `mm_processor_kwargs` 合并与解析修复（#56372，07 §5.1）；device-side mm normalization 扩到 GLM4V/GLM5Next（#55389，07 §5.1）与 Llama Nemotron VL Embed/Rerank（#57928，07 §5.1）；encoder 编译启用时保持 fused device input normalization（#59195，07 §5.1）；encoder cudagraph + fused input norm 避免额外 d2d（#56711，07 §5.1）；compiled ViT attention 输出布局修复（#58182，07 §5.1）；Mistral3 图像预处理优化（#57531，07 §5.1）；MiMo 声明 `embedding_fields` 使 EPD 对可服务图像（#58938，07 §5.1）。
  - **模型**：MiniMax M3 PP 下 target embedding 与 MTP 共享（#58648，03 §5.1）；K2 Horizon partial-RoPE 置换折叠进 q/k（及 norm）权重（#55335，03 §5.1）；MiMo fused fp8 qkv_proj 配对状态跨权重加载调用保持（#58142，03 §5.1）；BERT/RoBERTa embedding 类改为类属性（#59348，03 §5.1）。
  - **Engram/LoRA**：移除 Engram CUDA-alike 设备限制（#59171，07 §7.2）；共享内存解析时 THP 表保持私有（#59068，07 §7.2）；sequence classification 支持可变 `num_labels`（#57766，07 §4）。
  - **采样**：Triton sampler warmup 冗余特化削减（#58605，07 §2）。
  - **可观测性/其他**：process manager force kill 时记录子进程终止日志（#52314，08 §1.2）；sweep 压测新增 warmup 与失败恢复（#57305，08 §7）；安全依赖升级 nltk/aiohttp/pillow/datamodel-code-generator（#59249）；API 兼容性检查 skill（#58003）；dead tests 代码清理（#58916）；CI IPC weight-checker 测试非 CUDA 平台跳过（#59398）；多模态 scoped processor kwargs 优先级测试覆盖（#59399）；LoRA serving 测试 mock 类型窄化修复（#59344）；MyPy 测试组错误修复（#55939）；Zamba2/Whisper/Ultravox/Unlimited-OCR MyPy 类型修复（#58255/#58254/#58239）。
- **2026-09-25**：基线从 `9f07d023d0`（2026-09-23，v0.30.1rc0）推进到 `afea5c20c7`（2026-09-25，最新 tag 仍 **v0.30.1rc0**，`153242a314`）。区间 127 commits（721 文件，+20634/−5948）。主要变更：
  - **前端/启动**：`vllm preload` CLI（#56680，模型加载/编译前置，serve 复用预加载产物，08 §1.1）；slow tokenizer mode 移除（#58545）；`/v1/messages` 支持 Disable Thinking（#58613）；`--enable-log-requests` 请求体 debug 日志（#58163）；streaming derender detokenization offload（#57528）。
  - **投机解码/调度**：DSpark PP 支持**回退**（#56956 被 `09fe178dba` #58484 revert，03 §6.2/05 §2.2 已同步）；DFlash context K/V precompute 进 draft CUDA graph（#57632）；Mamba2 prefill SSM state save 批量化去 GPU↔CPU 同步（#49371）；`--long-prefill-token-threshold` 自适应调参（#58459）。
  - **结构化输出**：xgrammar 原生解析 Lark grammar（#58321，07 §3）；outlines 接受 grammar finish 后的 EOS + 拒绝 json_object 校验（#57743）+ rejected drafts 后 EOS/mask 修复（#58612）。
  - **分布式/平台**：**PCP+DP 组合支持**（#57075，05 §2.2）；SM100/103 low-SM multimem reduce-scatter（#55072）；XPU GRAPH 默认开（#51600）+ MRV2+PP microbatch flag（#55145）；CPU AVX10.2 按编译器支持门控（#58133）+ 预构建 triton（#58140）+ Zen CPU encoder attention 走 zentorch SDPA（#54508）；ROCm gfx950 MXFP8 GEMM native 32x32 block scales（#58510）+ skinny GEMM 去 69 次冗余 contiguous copy（#58566）+ DSv4.1 sparse decode MXFP8 + grouped FP8 GEMM（#58456）。
  - **量化/内核**：Triton kernel dispatcher（#43048）；FlashInfer 升级 0.7.0（#58069）；fp8.py 在线量化支持移除改用 online shorthands（#53585）；Quark 静默在线量化移除（#51800）；DSv4 inverse RoPE+FP8 quant 融合进 FlashInfer sparse MLA（#58621）；DSv4.1 恢复 fused query RMSNorm+MXFP8（#57679）；Engram offloaded lookup 串行化 + huge pages（#56926）；启动期 Triton kernel warmup 并行化（#58582）；VLLM_BATCH_INVARIANT 下默认 breakable CUDA graphs（#57586）。
  - **KV/多模态**：KV connector 无 forward 步 finalize saves（#57775）；增量多模态 block hashing 修复（#51694）；partial-block KV event 保留全部多模态 feature（#58288）；encoder-cache hit embedding 数不匹配拒绝（#57696）；prompt_embeds tensor 随 InputBatch slot 释放（#57988）；NIXL push 完成上报恢复（#58188）。
  - **Rust 前端**：histogram 观测 lock-free（#58574）；`--sse-keep-alive-interval`（#58306）；HF 模板自定义 chat roles（#58311）；Nemotron-H vision 预处理上下文（#57634）。
- **2026-09-24**：基线从 `d90f0eade5`（2026-09-22，v0.30.0）推进到 `9f07d023d0`（2026-09-23，最新 tag **v0.30.1rc0**，`153242a314`，2026-09-23 release candidate）。区间 69 commits。主要变更：
  - **投机解码**：DSpark 支持 PP（#56956，`pp_utils.py` 的 `PPHandler.set_disabled()` + `broadcast_drafts` 折叠进 `post_update`）；Kimi-K3 变长 decode（#52988，MLA/KDA metadata builder 支持 `max_query_len`，`FlashInferMLADecodeMetadata` 数据类化）；异构 vocab spec decode 去掉 CPU-GPU 同步（#57396）；GLM MTP head 延迟加载（#55442，`defer_lm_head`）；MRV1+PP>1+async sched+structured output 组合禁止（#56250）。详见 02/04/05/07/08。
  - **结构化生成/解析**：DiffusionGemma 结构化生成（#57250，`DiffusionAsyncScheduler` + `validate_diffusion_sampling_params`）；Granite 流式 tool-call 解析（#49648，`parser/granite.py` 新文件）；length finish_reason 流式 tool call 修复（#46303）；FIM completion 渲染（#44229，`DeepseekV4Renderer.render_completion_suffix`）。详见 01/07/08。
  - **MoE/量化**：MoE gate 统一 `GateLinear`（#58234，33 个模型文件切换）；TritonExperts EP 丢弃远端专家 top-k slot（#58051）；Humming wNaM 非对称量化（#46528，`zero_point` 支持，`WNA16_ZP_SUPPORTED_TYPES_MAP` 扩到 {2-8}）；per-token NVFP4 MoE 后端（#57176）；FP8/MLA 权重变换重构（#57732，`split_kv_b_proj()` 纯函数）；DeepGEMM arch capability 优先检查（#58073）；AllSpark INT8 W8A16 GEMM 移除（#58001）。详见 03/06。
  - **编译/启动**：`enforce_eager` 禁用 JIT warmup（#58197/#55146）；`set_torch_threads_for_runtime` 移到 `load_model` 末尾（#55891）；Fast Start DP weight cache daemon（#57386）；`with_hf_config` 子模型视图跳过 `__post_init__`（#58212）；ModelState `max_model_len` 从 model config 读取（#58149）。详见 01/03/08。
  - **分布式/平台**：AuxOutput KV connector 限制收窄为 5 个具名 PD connector（#58150）；batch_invariant NCCL>=2.31 用 `NCCL_ALGO="ring,tree;allreduce:tree"`（#58179）；XPU batch-invariant 支持（#55881）；ROCm MRV2 sampler JIT warmup（#58092）；ROCm BF16 AsyncTP 融合（#58098）；ROCm AITER static FP8 attention output 融合（#58099）；RDNA3/4 narrow KV tile（#58225）；MiniMax MXFP8 zero blocks 修复（#58089）；DSV4/DSV4.1 inverse RoPE 融合（#57451/#57435）；GLM-5.2-MXFP4 ROCm（#51915）；GLM-5.3-Flash dense MLP sequence-parallel shard（#58061）。详见 05/06。
  - **其他**：SM120 NoPE sparse MLA 修复（#55277，`concat_and_cache_ds_mla_kernel` 清零 RoPE 尾部）；encoder-only prefix caching 自动禁用（#58287）；Engram `/dev/shm` 回退（#57914）；Triton softcap NaN 修复（#56579）；EPD metadata-only audio（#57887）；dead code 清理（#58002，-585 行）。详见 02/04/07。
- **2026-09-22**：基线从 `8b98b7d0b4`（2026-09-21，v0.30.0rc2）推进到 `d90f0eade5`（2026-09-22，最新 tag **v0.30.0**，`9ed533eb4a`，2026-09-20 正式 release）。区间 56 commits。主要变更：
  - **Initialized engine snapshots**（#51360，`vllm/snapshot/` 新包 + `vllm snapshot create/restore` CLI）：CRIU + CUDA checkpoint 捕获**已初始化引擎**的进程树快照，restore 时校验环境指纹并复现记录的 token 输出，换取极快激活；实验性，限 Linux x86-64 + 单 NVIDIA GPU + TP1 明文 HTTP，详见 08 §1.1。
  - **KV hints 请求信封**（#53423，`vllm/v1/kv_hints/`）：orchestrator 可编程 KV 管理提示（`KvHintsEnvelope`/`KvHintAction`，版本化 action），经 `InputProcessor`→`Request`→`ReqContext` 贯通到 KV offload tiering（KVCR 的 router hint 改用此信封）。详见 02 §3.3。
  - **调度器**：`long_prefill_token_threshold` 软化（#57951）——batch 中只有 1 个请求时不再截断其 prefill chunk（无人可饿死）；`ParallelConfig.nnodes_within_dp` 修复 external LB 下 DP rank 数超过节点数时 floor 到 0 的问题（#53743，external LB 时 `max(...,1)`，非 external 且不可整除直接报错）。详见 02/05。
  - **投机解码**：DFlash 启用 async scheduling（#58065）；draft 模型配置覆盖统一为 `SpeculativeConfig.apply_draft_overrides`（`moe_backend`/`attention_backend`/`kv_cache_dtype` 仅非 None 时覆盖 target，`config/speculative.py`）；draft 加载统一走 `get_draft_load_config`（`model_loader/utils.py`）——Fast Start（`ipc_cache`）下 MTP draft 模型缓存到 daemon 独立 draft group（#57312）；DFlash/DSpark profiling query batch 上限（#56448）；MTP draft KV cache group 位置标注通用化（#55390，`kv_cache_utils.py` 的 trailing-layer fallback 从 DSV4 专属扩到所有 `method="mtp"`，含 DSV4.1 DSpark）。详见 02/03/07。
  - **Attention**：GLM5Next **NoPE sparse-MLA**（head_size 512、`qk_rope_head_dim=0`）接入 FA/FlashMLA（#55385，`FLASHMLA_SPARSE` 支持 512 仅限 SM90 bf16 NoPE；FA3 QV 路径用 64-wide 零 Q 占位）；sparse MLA 准备开销削减（#57458，`sparse_utils.py` index-remap kernel 重构）；ROCm DSV4 自适应验证走 flattened device query lens（#52362）；GDN stateless first-chunk 分类修复（#51565，`gdn_attn.py` 用 `seq_lens_cpu_upper_bound` 区分首 chunk prefill 与 capture batch）。详见 04。
  - **多模态**：processor/receiver cache 从 `multimodal/cache.py` 单文件重构为 `multimodal/cache/` 包（`base.py`/`lru.py`/`shm.py`/`factories.py`，`worker_receiver_cache_from_config` 等工厂从 registry 迁出）；`supports_multimodal_inputs` 从 registry 移到 `ModelConfig` 缓存属性（#57913）；Qwen2.5-VL 视频 fps 用于 temporal M-RoPE（#47736）。详见 07。
  - **量化/平台**：MXFP4 emulation 加载期反量化（#50814，`VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD`，对齐已有 MXFP8 开关）；`VLLM_KIMI_K3_GEMM_RS` 更名 `VLLM_ENABLE_GEMM_RS`（#57428，GEMM-RS 融合 kernel 从 Kimi-K3 专属扩到 DSV4.1 `wo_b`，`kernels/linear/cute_dsl/gemm_rs_ar.py` 新增 1177 行）；`VLLM_PLE_CPU_OFFLOAD` 移除（Engram `cpu_offload` 固定默认 True）；CPU 新增 `--device-memory-utilization` CLI 别名（#56547，`CacheConfig.device_memory_utilization` property）。详见 06/08。
  - **分布式**：NIXL DCP 跨 MLA cache region 的 pull 修复（#57389，`nixl/base_worker.py` 按全局 DCP 位置对齐 block + region 分组校验）；ROCm 显式拒绝 DSV4 FSE=1 + DPA+ETP 组合（#57919）。详见 05。
  - **sleep mode**：level-2 sleep 保留冻结权重（#57891，`ModelConfig.sleep_preserve_parameter_names`（CLI `--sleep-preserve-parameter-names`）按 glob 保留指定参数跨 sleep，RL 场景免重传）；`gpu_worker.py` 新增 `_save/_restore_sleep_parameters`。详见 08。
  - **安全/校验**：拒绝超过填充后 `max_tokens` 默认的 `min_tokens`（#57731，`input_processor.py`）；拒绝空 `structural_tag`（#47450，`sampling_params.py`）；grammar poll 改非阻塞（#55931，`structured_output/request.py` 用 `Future.done()` 替代 100µs timeout）；prompt embeds 的 `is_token_ids` 长度校验（#57006）。详见 01/07。
  - 其他：Rust frontend 系列（parser-owned output grammar #55269、MiMo V2.5 parser #57933、vision processor spec #58109、mm-processor benchmark #51922）；ROCm Qwen GDN 输出 norm 省 reshape（#47842）；ROCm fused shared-expert gate GEMM 走 platform dispatcher（#54185）；fused silu-mul block-quant fast path 在 swiglu clamp 时跳过（#57984）；MoE 拒绝 monolithic backend 不支持的 hash routing（#57867）；chunked long-text embedding 归一化修复（#57498）。详见 03/06/07。
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
  - **水印支持投机解码**：新增 `dual_key_gumbel` 算法（双 key gumbel-max，`supports_speculative_decoding=True`，`alpha` 控制 key-B 概率，加权 early-fusion 检测 `vllm/v1/watermarking/gumbel.py:211`）；`spec_decode.py` 新增 `create_speculative_target_watermarker`/`create_speculative_draft_watermarker` 与 `allow_target_only_watermarking`；`_check_watermarking_unsupported`（`vllm/config/vllm.py:1318`）约束 `draft_sample_method='probabilistic'`、`rejection_sample_method='standard'`、method ∈ {dspark,eagle,eagle3,mtp}。详见 07 §7.1。
  - **调度器 RUNNING 准入上限**：`SchedulerConfig.max_num_active_seqs`（`--max-num-active-seqs`，`vllm/config/scheduler.py:69`，`vllm/v1/core/sched/scheduler.py:132-135`，执行点 `vllm/v1/core/sched/scheduler.py:913-894`）；队列上限计数改用 `SharedAdmissionStats`（`vllm/v1/engine/admission_control.py:13`）跨进程无锁计数。详见 02 §2.2。
  - **投机解码自适应验证**：`enable_adaptive_verification`（`vllm/config/speculative.py:551`）+ `OnlineAcceptanceEstimator`（`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:313`，501 行，log-odds 线性模型，Triton accumulate/refit/predict kernels）。详见 07 §1.3。
  - **KV offload 增强**：back-pressure（#50045，`vllm/v1/kv_offload/tiering/backpressure.py`）、KVCR（#53624，`vllm/v1/kv_offload/tiering/kvcr/`）、per-request `max_load_tokens`（#55885，`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:366`）、chunked region 注册（#51081）、cgroup 检查（#54014）、MLA compact（#56799）。详见 02 §6.1。
  - **Model Runner V2**：DBO FULL CUDA graph（#51700，实现收敛在 `vllm/v1/worker/gpu/cudagraph_utils.py`）；Fast Start 支持 nnode>1（#55468）。详见 03。
  - **分布式**：MoonEP BF16 all2all backend（#52101，`vllm/model_executor/layers/fused_moe/prepare_finalize/moonep.py`）；DeepEPv2 async finalize（#52781/#57236）；PCP+DCP on sparse-MLA（#56157）；PCP decode-only FULL CUDA graphs（#53867）；NIXL attention-HMA PP push prefill（#50494）；Elastic EP CUDA graph 复用（#54985）。详见 05。
  - **Attention**：新增 `COMPOSITE` backend（`vllm/v1/attention/backends/composite.py`，Triton/FlashInfer 或 Triton/FlashAttention 组合，用于 multimodal prefix attention `mm_prefix`，selector 在 `use_mm_prefix=True` 时自动选择）。详见 04 §4.1。
  - **量化**：Quark 原生 W4A16 INT4/UINT4（#48606，`vllm/model_executor/layers/quantization/quark/schemes/quark_w4a16_int4.py`）；CPU FP8 W8A8 linear/MoE（#49942，`csrc/cpu/sgl-kernels/gemm_fp8_w8a8.cpp` + `moe_fp8_w8a8.cpp`）。详见 06。
  - **部署**：新增 `POST /release_kv_cache_memory` 端点（#44890，`vllm/entrypoints/serve/dev/sleep/api_router.py:37`）；`--enable-scale-out` CLI flag 取代 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` 环境变量（#55176，`vllm/entrypoints/scale_out/factories.py:65`）。详见 08。
  - **结构化输出重构**：`should_fill_bitmask`/`should_advance` 移除，改用 `_get_constraint_start`（`vllm/v1/structured_output/__init__.py:220`）/`validate_tokens`（`vllm/v1/structured_output/__init__.py:294`）；调度器 grammar 验证迁移到 `structured_output_manager.validate_tokens`（`vllm/v1/core/sched/scheduler.py:2567/2536`）。详见 07。
  - **Engram**：新增 `embedding_across_dp`/`dp_shared_memory` 字段 + 异步预取 + DP 分片（#56512）。详见 07 §7.2。
- **2026-09-20**：基线从 `751f6807d9`（2026-09-19，v0.30.0rc2）推进到 `4868312128`（2026-09-20，最新 tag 仍为 **v0.30.0rc2**，`fa6ff06066`）。区间 30 commits。主要变更：
  - **Humming 特性整合**（#56685）：`utils/humming_utils.py` 拆成 `utils/humming/` 包（`schema.py`/`activation.py`/`linear.py`/`moe.py`），新增 `mxfp6/humming.py` kernel 与 `WeightScale2Type`/`InputQuantizationMode`/`MmaType` 等 schema 类型；显式 input schema 默认禁用 fallback（`allow_fallback` 控制）；Marlin 与 Humming 共享持久 workspace（#57421，`vllm/v1/worker/workspace.py` 新增 `get_persistent_resource`/`get_persistent`）。详见 06。
  - **Model Runner V2 支持自定义 logits processors**（#56497）：`vllm/v1/worker/gpu/sample/logits_processor/`（`interface.py`/`loader.py`）新增，`_get_v2_model_runner_unsupported_features` 移除 "custom logits processors" 限制；`InputProcessor` 在准入时按 runner 选 validator。详见 03/07。
  - **`--enable-mamba-fine-grained-prefix-cache` 更名**（#57382）→ `--enable-mamba-shared-prefix-checkpoint`（`CacheConfig.enable_mamba_shared_prefix_checkpoint`，`config/cache.py:199`），语义不变（EAGLE/MTP 共享前缀 junction 处注册 Mamba align checkpoint）。详见 02。
  - **generate API 暴露 per-request 投机解码指标**（#43310）：Rust frontend `GenerateResponse`/`GenerateStreamResponse` 新增 `metrics.speculative_decoding`（`mean_acceptance_length`/`draft_acceptance_rate`/`acceptance_histogram`/`per_step_*` 等）。详见 07/08。
  - **EPD 动态注册**（#54176）：`disagg_epd_proxy.py` 支持 `--dynamic-registration`，通过 `POST/DELETE /instances`（`X-API-Key`）在线注册/摘除 encode/prefill/decode 实例，带健康探测与自动重连。详见 05。
  - 其他：DeepSeek-V4.1-flash encoder CUDA graph（#56625，`models/deepseek_v41/common/vl_cudagraph.py`）；MiMo V2 bf16 MoE router + mxfp4 MoE（#57784，`GateLinear`）；GLM-5.3-Flash kpool/sparse-indexer 系列修复与性能（#57546/#57534/#57477/#57701/#56810）；SM100 fp8_ds_mla cache scales 修复（#49435）；dead kernel code 清理（#57621，-559 行）。详见 02/04/06。
- **2026-09-21**：基线从 `4868312128`（2026-09-20，v0.30.0rc2）推进到 `86ce4d10e2`（2026-09-21，最新 tag 仍为 **v0.30.0rc2**，`fa6ff06066`）。区间 11 commits。主要变更：
  - **Profiler 统一为平台感知**（#57460）：torch profiling 逻辑从各 worker（`gpu_worker.py`/`cpu_worker.py`/`xpu_worker.py` 各删 22~41 行）收敛到 `vllm/profiler/wrapper.py` 工厂 `create_worker_profiler`（:675）；`ProfilerConfig` 新增 `torch_profiler_activities`（`config/profiler.py:55`，`CPU`/`CUDA`/`PrivateUse1`/`XPU`，缺省按平台默认）；`WorkerProfiler` 基类新增 `should_annotate` 属性（`wrapper.py:60`）。详见 08 §4.9。
  - **sleep 时 KV connector cache reset 失败上抛**（#54581）：`EngineCore` 的 `reset_prefix_cache` 返回 False 时抛 `RuntimeError`（`vllm/v1/engine/core.py:877`），`pause_generation` 的 idle callback 异常经 future 传播而非吞掉。
  - **MoRIIO KV connector 大改**（#51052，+2746 行）：READ 模式传输 hybrid mamba/KDA recurrent state，`moriio_connector.py`/`moriio_layout.py` 重写。
  - 其他：spec decode dummy draft 步不再经 stale block-table 行写 KV（#56734，`vllm/v1/worker/gpu/spec_decode/speculator.py`）；DSV4.1 mHC 小 TP batch 系数 overlap（#57603）；ROCm Engram 表留 host 内存（#57491）；HY4 full CUDA graph capture 记录 indexer completion event（#57811）；XPU communicator world_size 可见性修复（#57779）；Kthena/EPD 文档更新。详见 02/03/06/07。
- **2026-09-21（第二次）**：基线从 `86ce4d10e2`（2026-09-21，v0.30.0rc2）推进到 `8b98b7d0b4`（2026-09-21，最新 tag 仍为 **v0.30.0rc2**，`fa6ff06066`；main 已领先 `v0.30.0` release 分支 310+ commits）。区间 27 commits。主要变更：
  - **AuxOutput Connector**（#45635，`vllm/distributed/aux_output_connector/` + `config/aux_output.py`）：MoE routed-expert 输出按 **KV block hash** 为键持久化到进程内 mmap arena（LRU 驱逐、fail-closed），供外部按 prefix-cache block 读取；`AuxOutputConfig`（`enable_return_routed_experts`/`max_bytes`）挂在 `VllmConfig.aux_output_config`（`config/vllm.py:380`），兼容性校验 `_verify_aux_output_compatibility`（`config/vllm.py:1092`，要求 V2 runner + MoE + prefix caching，拒绝 PP>1/DCP/PCP>1/5 个已知 PD KV connector（Nixl/NixlPull/NixlPush/MoRIIO/Mooncake，#58150 收窄））；scheduler 钩子 `scheduler.py:394/1475/1536/2074/2564`。详见 05 §9.4、02 §2.5。
  - **DSV4.1 mHC 三连**：TP all-reduce 与 mHC 输入准备融合为 MNNVL Lamport multicast CUDA kernel（#57643，`models/deepseek_v41/nvidia/ops/mhc.py:46` `supports_mhc_all_reduce`，TP4/hidden5120/hc_mult4/CUDA_ARCH>=900）；mHC overlap 收紧到 full CUDA graph 捕获路径（#57874，`torch.cuda.is_current_stream_capturing()` 门控，`model.py:413-420`）；ROCm 上禁用 SWA bounded replay（#57906，`models/deepseek_v41/attention.py`，window clamp 只在 FlashInfer/FlashMLA prefill kernel）。详见 05 §4.3、04 §4.2。
  - **ROCm/XPU 平台**：Hy4 ROCm 路径 + backbone compile（#57526，`models/hy_v4/amd/model.py` +718 行）；MiniMax-M3 packed LBHNC AITER QK-norm 融合（#54535）；XPU fused top-k/top-p sampler kernel（#57277，`VLLM_XPU_USE_SAMPLER_KERNEL` 默认开，`topk_topp_sampler.py:129/151`，MRV2 同步接入）；`moe_align_block_size` 7-arg 回退（#57855）；DeepGEMM CUDA 12.9 构建修复（#57554，pin fork `e1f418c2`）。详见 06。
  - **Mamba/KDA prefill checkpoint 通用化**（#57783）：`mamba/checkpoint.py` 抽出 `MambaPrefillCheckpointBuilder`/`Exporter`（ABC），`kda_checkpoint.py` 的 `FlashKDAPrefillCheckpointExporter` 复用，`kimi_k3/nvidia/kda.py`/`kda_metadata.py` 瘦身。详见 03 §3。
  - **多模态 + 结构化输出 bugfix/安全**：receiver cache 优先采用新 payload（#57833，`multimodal/cache.py`，防 stale tensor 顶替新 payload 导致 EngineCore 崩溃）；Molmo2 容忍畸形 EXIF（#57234）；Whisper 30s 上限、Mistral3 grid、DiffusionGemma 修复；xgrammar list-valued `"type"` 归一化（#48416，`backend_xgrammar.py:247`）。详见 07。
  - **Engram DP shared memory 默认开启**（#57651，`config/engram.py:52/86`）：同机 co-located DP replica 默认共享 n-gram 嵌入表 host 内存。详见 07 §7.2。
  - **derender 流式解析文档化**（#57922，`docs/serving/online_serving/derenderer.md`）：scale-out derender 的无状态 `stream_state` 流式协议 + 流式 parity 测试。详见 07 §1。
  - 其他：sparse attention metadata 去冗余（#57885，`mla/indexer.py`/`sparse_swa.py`）；8 个 CI-only commits（XPU/ROCm/Intel job 调整）。

**2026-10-04 增量更新**：基线从 `df8fd42116`（2026-10-01）推进到 `bc21cba967`（2026-10-03），main 仍领先 v0.31.0rc2。区间共 143 commit（670 文件，+36701/−14613），主线：

- **EPLB contention-aware 迁移批处理**（#52641）：新 `vllm/distributed/eplb/migration_scheduler.py`（`schedule_migration_batches:20`），每 rank 每批最多与一个 peer 通信；`EPLBConfig.migration_batching_enabled`（`config/parallel.py:116`）。
- **PCP decode 分片**（#52162）：`pcp_shard_decode_requests`（`config/parallel.py:615`）——PCP-only 复制 KV，decode 请求可有单一 PCP owner。
- **ModelExpress 原生 weight transfer backend**（#58399）：`vllm/distributed/weight_transfer/modelexpress.py`（外部包 `ai-dynamo/modelexpress`）。
- **sleep 模式资源释放**：KV-init runtime state offload（#59158）+ WorkspaceManager scratch 释放（#59156，`v1/worker/workspace.py:49`）。
- **HiSparse 系列**：MTP FULL graphs 接受率坍塌修复（#59309）、KV cache 按分配 group 定容（#59450）、chunked-prefill 抢占活锁修复（#59494）、重复 prefix scan/residency 更新消除（#57930）。
- **KV connector**：NIXL host-buffer 拷贝跨 cache group 合并（#54483）+ KV 过期后的 completion 通知计数（#58875）；Mooncake 空 pull 抑制 completion（#59347）；async KV load 的 KV-fetch stage gauges（#58874）；async KV load 写入的 block 精确豁免 zeroing（#59504）。
- **Frontend**：`/inference/v1/generate` 加 `output_mode`（RFC #56851 Phase 1，#58588，`entrypoints/scale_out/token_in_token_out/protocol.py:224`）；Rust frontend gRPC port 暴露（#59659）+ `hf` response template parser（#59005）+ 结构化 tag grammar 快照（#59393）；内置 JSON log formatter（#58739，`config/logging.py:31`）；Anthropic tool_addition/tool_removal content blocks（#57693）；GLM-4.7 非严格 tool call 浅层结构 tag（#56403）。
- **模型迁移到 Transformers modeling backend**：GPT-NeoX/Phi/Seed-OSS/Jais2（#59701）、Glm/Arcee/CWM/Mellum（#59679）；删除 Transformers < 5.16.1 代码路径（#59762）；GLM-5.3/Qwen4-Exp 用上游 config/processor（#57387）。
- **性能**：GLM-5.3 fused multi-step decode（#57443，并发 1 E2E +13.3%）+ sparse MLA index 跨层复用（#59464，3.5~3.9x）；FlashInfer CuteDSL MegaMoE（#54049）+ one-sided MoE all2all fp8 combine（#57995）；GDN 纯投机行切片（#58763）；PP 跳过离开引擎请求的 sampled-token 广播（#58542）；per-token FP8 量化 value-only reduction（#59800）；flash-maxsim late-interaction Triton kernel（#40337）。
- **投机解码**：MRV2 多层 MTP per-module LM heads（#58921）+ sampling mask replay（#59359）；Kimi-K3 FlashInfer 投机 KDA backend（#54255）；Sarvam MLA EAGLE3/DSpark PP（#55902）。
- **量化/硬件**：MiniMax-M3 MSA 稀疏路径 NVFP4 KV cache（#59300）；ModelOpt 混合精度 NVFP4 检测（#56050）；ROCm Kimi-K3 gfx942 MXFP4→int4（#51274）；SM120 占用率自适应 split-K（#58482）；XPU B70 W8A8 block-FP8 GEMM 调优（#56063）。
- **安全/正确性**：structured-output 请求不得从无 mask 行采样（#54442）；tokenizer max_token_id off-by-one（#59491）；weight loader dtype 相等校验（#51792）；DeepSeek-OCR 零和像素张量接受（#59417）。

**2026-10-07 增量更新**：基线从 `d0d6e5f3a2`（2026-10-05）推进到 `30d4032363`（2026-10-07），main 领先 v0.31.0（`db9527a468`，2026-10-02）381 commits。区间共 5 commit（35 文件，+484/-478），主线：

- **GLM-5.3-Flash kpool sparse indexer DCP 支持**（`30d4032363` / #59211）：kpool sparse indexer 支持 DCP（distribute checkpoint）（9 文件 +242/-37）。详见 03 篇。
- **LoRA 移除 tensorizer**（`f9c9e8ac24` / #60024）：LoRA 加载路径移除 tensorizer 依赖（9 文件 +30/-375，净删 345 行）。详见 03 篇。
- **device-side mm normalization 扩到 Kimi K2.5/K3**（`0e468adb43` / #59278）：device 侧多模态 normalization 扩展到 Kimi K2.5/K3（11 文件 +130/-15）。详见 03 篇。
- **MLA fp8_ds_mla KV cache 支持 NoPE-512 模型（SM90）**（`ff53f32409` / #59246）：SM90 上 NoPE-512 模型的 fp8_ds_mla KV cache 支持（5 文件 +82/-44）。详见 04 篇。
- **Voxtral HF reference 测试在 Transformers v5 重新启用**（`d3547f9d03` / #59771，测试）。

**2026-10-06 增量更新**：基线从 `bc21cba967`（2026-10-03）推进到 `d0d6e5f3a2`（2026-10-05），main 领先 v0.31.0（`db9527a468`，2026-10-02）376 commits。区间共 42 commit（153 文件，+5468/−1352），主线：

- **sleep 模式释放 CUDA graph 池**（#59160）：新 `vllm/compilation/cudagraph_pool.py`（`capture_pool` contextmanager）；`ModelConfig.sleep_mode_offload_cudagraph` + CLI `--sleep-mode-offload-cudagraph`；`VllmConfig.use_cumem_cudagraph_pool` 为真时 CUDA graph 捕获分配进 cuMem allocator 的 `cudagraph` tag 池，sleep 时随权重 offload 一并释放；NCCL graph registration 会 pin 该池，故自动 `NCCL_GRAPH_REGISTER=0`。详见 08。
- **DCP**：TokenSpeed MLA 支持 block-interleaved DCP（#59462，`v1/attention/backends/mla/tokenspeed_mla.py` `supports_mtp_with_cp_non_trivial_interleave_size=True`）；空 KV shard 输出（NaN）由下游 DCP combine 掩码。详见 05。
- **prefill token scoring per-row candidate IDs**（#56984，M2 of #56860）：`SamplingParams.prompt_logprob_token_ids`（`[num_rows, num_ids]` 整数数组/嵌套 list，-1 填充得 -inf），Rust frontend `logprobs.rs`/`request.rs` 同步。详见 07。
- **安全**：untrusted media 路径限制 Pillow 图片格式（#60022，`multimodal/image.py`）。详见 06。
- **KV offload（SimpleCPU）**：MRV2 resume 时重置 eager-store placement 状态（#57816）；cache reset 释放 pending CPU lookup pins（#59862，`v1/simple_kv_offload/manager.py`）。详见 02。
- **NIXL**：heartbeat 计入 remote engine 活动（#59873，`kv_connector/v1/nixl/base_worker.py`）。详见 02。
- **LoRA**：确定性 split-K=8 shrink kernel 保 batch invariance（#59377，`lora/ops/triton_ops/lora_shrink_op.py`）；代码清理（#60017）。详见 03。
- **ROCm/RDNA3**：W4A16 split-K 精度与确定性修复（#54706，`csrc/rocm/q_gemm_rdna3*.cu`）；AITER bump 0.1.24.post1（#59794）；ROCm 内存 profiling 保留 config（#58014）；ROCM_ATTN sliding-window 边界（#59550）。详见 06。
- **模型**：GraniteMoeHybrid/FalconH1/Zamba2 spec decoding 下 Mamba page size AssertionError 修复（#59975）；非 gated MoE 加载 stacked expert 权重（#59031）；Qwen3ASR 声明 SupportsEagle3（#52824）；DeepSeek-V4 MegaMoE shared-expert finalize 独立于 linear post-load 顺序（#59927）。详见 03。
- **Qwen4Exp 系列**：QSA attention 尊重 `--kv-cache-dtype-skip-layers`（#60023）；HC up projection 留在 skinny GEMM 路径（#60027）；Quark checkpoint 下 PLE 表非量化加载（#59443）；PLE embedding 接受 INC（AutoRound）checkpoint（#59990）；QSA QKVG 与 indexer QK 投影合并（#59533）。详见 03/04。
- **Frontend**：model-not-found 404 列出已服务模型名（#59889）；流式错误在首 token 前返回（#40986）；scale-out token 流保留 abort finish_reason（#47933）；Responses API 复用流式 item id（#59859）。详见 07/08。
- **其他**：SM121 TP=2 skinny-GEMM plans（#59632）；int4_per_token_head 支持非 2 幂 head size（#56198）；DSv4.1 compressor ring 避开 null block（#58560）；XPU DeepSeek V4 FP8 sparse decode graph-capturable（#59159）；单声道音频归一化 1D（#56691）；FlashInfer all_reduce backend 选择修复（#56891）；batch-invariant mean 保留输出 dtype（#59106）；对齐 KV block size 对照全部 attention backend 校验（#58457）；Transformers backend 视频支持（#57441）；Transformers bump 5.18.0（#59621）；MoRIIO discovery heartbeat 在 worker 持 GIL 时保持运行（#59441）。
