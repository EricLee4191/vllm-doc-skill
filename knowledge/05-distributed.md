# 分布式并行（TP/PP/DP/EP）与 KV 传输

> 基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（V1 架构为当前引擎）。所有路径相对于仓库根 `/Users/baofeng/baofeng/github/vllm`。
> 核心目录：`vllm/distributed/`、`vllm/config/parallel.py`、`vllm/v1/executor/`、`vllm/v1/worker/`。

---

## 1. 总览
<!-- tags: distributed, overview, 分布式, 并行 -->

vLLM 的分布式层分四块：

- **`vllm/distributed/parallel_state.py`**：进程组（process group）的建立与 rank 划分，是全局通信的“总控”。它接管了 PyTorch 的分布式环境（`init_distributed_environment` + `initialize_model_parallel`）。
- **`vllm/distributed/device_communicators/`**：各后端的具体通信实现（NCCL / custom all-reduce / symm-mem / all2all / shm 等），由 `GroupCoordinator` 持有。
- **`vllm/v1/executor/`**：跨进程/跨节点拉起 worker 的编排层（`MultiprocExecutor` / `RayDistributedExecutor` / `UniProcExecutor`）。
- **`vllm/distributed/kv_transfer/`、`eplb/`、`elastic_ep/`、`ec_transfer/`、`weight_transfer/`**：KV 传输（PD 分离）、专家负载均衡、弹性 EP、encoder cache 传输、权重传输等专用子系统。

进程模型（V1）：
```
API Server (AsyncLLM)
   └─ EngineCoreProc（每个 DP rank 一个，跑 scheduler + 主循环）
        └─ Executor（Multiproc / Ray）
             └─ Worker × world_size（每个 worker 绑 1 个 GPU，跑 model runner）
```
`EngineCore` 与 `Worker` 之间通过 executor 的 `collective_rpc` / `execute_model` 通信；worker 之间通过 `parallel_state` 的进程组做集合通信。

---

## 2. 并行策略全景
<!-- tags: tp, pp, dp, ep, parallelism, 并行策略, tensor-parallel -->

`ParallelConfig`（`vllm/config/parallel.py`）定义的核心尺寸字段：

| 字段 | 含义 | 默认 |
|---|---|---|
| `tensor_parallel_size` (TP) | 张量并行组数 | 1 |
| `pipeline_parallel_size` (PP) | 流水线并行组数 | 1 |
| `prefill_context_parallel_size` (PCP) | prefill 序列切分 rank 数（不增加 KV 分片数） | 1 |
| `decode_context_parallel_size` (DCP) | decode KV cache 切分 rank 数（不扩展 world size） | 1 |
| `data_parallel_size` (DP) | 数据并行组数（MoE 层按 TP×PCP×DP 切分） | 1 |
| `enable_expert_parallel` | MoE 层用 EP 替代 TP | False |

**world_size 计算**（`ParallelConfig.__post_init__`）：
```python
self.world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size
# 跨 DP 的总进程数：
world_size_across_dp = world_size * data_parallel_size
```
注意：`world_size` 只含 TP×PP×PCP（决定单引擎内 worker 数），DP 是“多引擎”维度。

### 2.1 Tensor Parallelism (TP)
<!-- tags: tp, tensor-parallel, 张量并行, all-reduce, 单节点 -->
- 把单个 matmul 的权重矩阵按行/列切到多卡：`ColumnParallelLinear`（按输出维切，`vllm/model_executor/layers/linear.py:407`）、`RowParallelLinear`（按输入维切）。
- 通信模式：attention 的 QKV 投影用 column-parallel + all-reduce，o_proj 用 row-parallel；每层 attention 后有一次 all-reduce。
- 适用：单节点内（NVLink/PCIe P2P），是最常用的起步策略。`-tp 8` 单机 8 卡。
- TP 组内用 **custom all-reduce / NCCL symm-mem** 等低延迟后端（见 §4）。

### 2.2 Pipeline Parallelism (PP)
<!-- tags: pp, pipeline-parallel, 流水线并行, p2p, 跨节点 -->
- 把模型按层切成若干 stage，每个 stage 占一部分 GPU；stage 间用 P2P `send`/`recv` 传 hidden states。
- 适用：模型太大单机放不下，跨节点扩展。
- `MultiprocExecutor.supports_pp = True`；`RayDistributedExecutor` 也支持。PP 组内相邻 rank 走 `send_tensor_dict`/`recv_tensor_dict`（`parallel_state.py` 中 `GroupCoordinator`）。

### 2.3 Data Parallelism (DP)
<!-- tags: dp, data-parallel, 数据并行, lb, padding -->
- 每个 DP rank 是**一个完整的引擎副本**（独立 scheduler + 独立 KV cache），共享同一份权重。
- 关键约束（`parallel_state.py:1821` 注释）：同一 DP 组内所有 rank 的 `generate` 必须**同步调用**，否则会死锁（因为 MoE 层要做跨 DP 的 all2all / all-reduce）。
- 每步 forward 前，`vllm/v1/worker/dp_utils.py::coordinate_batch_across_dp` 用一次 all-reduce 同步各 DP rank 的 token 数、是否 ubatch、cudagraph mode，并做 **DP padding**（把各 rank pad 到相同 token 数以对齐集合通信）。
- DP 组是 **stateless** 的（`stateless_init_dp_group`，gloo 后端，`parallel.py:621`），因为 engine 进程可能没有 CUDA 设备。
- 适用：MoE 模型高吞吐（配合 EP，见 2.4）；dense 模型离线 DP 无意义（`parallel.py:900` 会报错）。
- DP 的三种 LB 模式（`parallel.py`）：
  - 默认（internal）：vLLM 内部在 DP rank 间负载均衡。
  - `data_parallel_hybrid_lb`：每节点一个 AsyncLLM + API server，vLLM 在本地 DP rank 间 LB，外部 LB 在节点/副本间 LB。
  - `data_parallel_external_lb`：K8s “one-pod-per-rank” 宽 EP 部署，仅 MoE。

### 2.4 Expert Parallelism (EP)
<!-- tags: ep, expert-parallel, 专家并行, all2all, moe -->
- 仅 MoE。把专家（expert）切到不同 rank，每个 rank 持有一组完整专家，token 通过 **all2all** 路由到持有目标专家的 rank。
- EP 组 = **DP × PCP × TP** 的笛卡尔积（`parallel_state.py:1928`）。即 EP 跨 DP、PCP、TP 三个维度。
- `enable_expert_parallel=True` 时，`FusedMoE` 层把 TP 维度“折叠”成 EP：`ep_size = tp_size`（`vllm/model_executor/layers/fused_moe/config.py:1240`），`tp_size` 置 1。
- 典型组合 **DP+EP**：TP=1, DP=N，则 EP=N，每个 DP rank 持 1/N 的专家，MoE 层做 all2all。这是 DeepSeek 类模型的推荐部署。
- **all2all 后端**（`all2all_backend`，`parallel.py:188`）：
  - `allgather_reducescatter`（默认，纯 NCCL 组合）
  - `deepep_high_throughput` / `deepep_low_latency` / `deepep_v2`（DeepEP kernel）
  - `mori_high_throughput` / `mori_low_latency`（MoRI，多节点）
  - `nixl_ep`（NIXL）
  - `flashinfer_nvlink_two_sided` / `flashinfer_nvlink_one_sided`（MNNVL）
  - 实现在 `vllm/distributed/device_communicators/all2all.py`（`AgRsAll2AllManager`、`DeepEPHTAll2AllManager`、`NixlEPAll2AllManager` 等）。
- `use_all2all`（`parallel.py:691`）：`data_parallel_size > 1` 或 `use_sequence_parallel_moe` 或 EP+PCP 时启用。

### 2.5 PCP / DCP（Context Parallel）
<!-- tags: pcp, dcp, context-parallel, 上下文并行, kv-split -->
- **PCP**（`prefill_context_parallel_size`）：切分 prefill 序列计算，扩展 world size 但不增加 KV 分片数。当前不支持与 DP 组合（`parallel.py:527`）。
- **DCP**（`decode_context_parallel_size`）：切分 decode KV cache，不扩展 world size；无 PCP 时复用 TP rank。`dcp_comm_backend` 可选 `ag_rs`（默认）或 `a2a`（MLA 模型把每层 3 次 NCCL 降到 2 次）。

### 2.6 组合方式小结
<!-- tags: 组合, tp-pp, dp-ep, sequence-parallel, elastic -->
- **TP+PP**：单机放不下 + 需要更多卡。
- **DP+EP**：MoE 高吞吐，跨节点。
- **TP+DP+EP**：`use_sequence_parallel_moe`（`parallel.py:673`）在 TP>1 且 DP>1 且 EP 时启用 sequence parallel，避免 all-reduce 后 token 在 TP rank 间重复计算。
- **Elastic EP**：EP 组可运行时扩缩容（见 §7）。

---

## 3. parallel_state：进程组建立与 rank 划分
<!-- tags: parallel-state, rank, process-group, 进程组 -->

### 3.1 初始化流程
<!-- tags: init, 初始化, process-group, world-group, rank -->
1. `init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend="nccl")`（`parallel_state.py:1586`）：
   - 若 `nnodes>1` 或 `data_parallel_size>1` 且非 external_launcher，会把 rank/world_size 扩展到跨 DP：`rank = data_parallel_rank * world_size + rank`，`world_size = world_size_across_dp`（`parallel_state.py:1618`）。
   - 调 `torch.distributed.init_process_group`（或 `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1` 时走 `split_group` 路径）。
   - 建立 `_WORLD` 组（`init_world_group`），并探测 `_NODE_COUNT`。
   - 若 `nnodes_within_dp>1` 且 DP>1，建 `_INNER_DP_WORLD` 组。
2. `initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, decode_context_model_parallel_size, backend)`（`parallel_state.py:1751`）：
   - 用 reshape 一次性切出所有维度的组。

### 3.2 rank 布局（核心）
<!-- tags: rank, 布局, reshape, 进程组, eplb-group -->
```python
# parallel_state.py:1826
all_ranks = torch.arange(world_size).reshape(
    -1,                                  # ExternalDP（verl 集成用）
    data_parallel_size,                   # DP
    pipeline_model_parallel_size,         # PP
    prefill_context_model_parallel_size,  # PCP
    tensor_model_parallel_size,           # TP
)
```
**布局顺序：ExternalDP × DP × PP × PCP × TP**（TP 是最内层/最快变化）。要取某维度的组，就把该维 transpose 到最后，reshape 成 2D 再 unbind：
- **TP 组**：`all_ranks.view(-1, tp_size).unbind(0)` → 相邻 tp_size 个 rank 一组。
- **PP 组**：`all_ranks.transpose(2,4).reshape(-1, pp_size).unbind(0)`。
- **PCP 组**：`all_ranks.transpose(3,4).reshape(-1, pcp_size).unbind(0)`。
- **DP 组**：`all_ranks.transpose(1,4).reshape(-1, dp_size).unbind(0)`。
- **EP 组**：`all_ranks.transpose(1,2).reshape(-1, dp_size*pcp_size*tp_size).unbind(0)`（跨 DP×PCP×TP）。
- **EPLB 组**：与 EP 组同 rank，但**独立进程组**，隔离 EPLB 通信与 MoE forward 的集合通信，避免死锁（`parallel_state.py:1957`）。
- **DCP 组**：无 PCP 时复用 TP rank；有 PCP 时跨 PCP 或整个 TP×PCP。

每个 rank 的组内身份：`rank_in_group`（组内 rank）、`local_rank`（本机设备号）、`rank`（全局）。

### 3.3 GroupCoordinator
<!-- tags: groupcoordinator, 进程组, 通信原语, all-reduce, broadcast -->
`GroupCoordinator`（`parallel_state.py:380`）是单个进程组的封装，持有：
- `cpu_group`（gloo 后端，CPU 协调）+ `device_group`（NCCL 等，设备通信）。
- `device_communicator`（`DeviceCommunicatorBase` 子类，见 §4）。
- `mq_broadcaster`（`MessageQueue`，共享内存广播，仅 TP 组启用，`parallel_state.py:519`）。

对外通信原语（`GroupCoordinator` 方法）：
- `all_reduce(input_)`、`all_gather(input_, dim)`、`reduce_scatter(input_, dim)`、`all_gatherv`、`reduce_scatterv`、`gather`、`broadcast`。
- `broadcast_object` / `broadcast_object_list`（CPU，走 gloo；TP 组可走 `mq_broadcaster` 共享内存）。
- `send_object`/`recv_object`（pickle 序列化，CPU）。
- `broadcast_tensor_dict` / `send_tensor_dict` / `recv_tensor_dict`：把 dict 拆成“元数据（CPU 广播）+ tensor（设备广播）”，PP 传 hidden states 用这个。`send_tensor_dict` 支持 `all_gather_group` 优化（TP 组内各 rank 发一片，接收端 all-gather 重组）。
- `graph_capture(...)`：CUDA graph 捕获上下文，捕获时接管 custom all-reduce / aiter 的 capture。

**自定义 op 注册**（`parallel_state.py:352`）：`all_reduce`/`reduce_scatter`/`all_gather` 注册为 `torch.ops.vllm.*`，带 `fake_impl`，供 Dynamo/torch.compile 追踪（`use_custom_op_call` 时走 `torch.ops.vllm.all_reduce(tensor, group_name=...)`）。

便捷函数（`vllm/distributed/communication_op.py`）：`tensor_model_parallel_all_reduce/all_gather/reduce_scatter/gather`、`broadcast_tensor_dict`，都是 `get_tp_group().xxx` 的薄封装。

获取组的 getter：`get_world_group()`、`get_tp_group()`、`get_pp_group()`、`get_dp_group()`、`get_ep_group()`、`get_eplb_group()`、`get_pcp_group()`、`get_dcp_group()`、`get_inner_dp_world_group()`。

### 3.4 Stateless 组（DP/EP，用于 elastic）
<!-- tags: stateless, 无状态, elastic, store, 动态组 -->
`StatelessGroupCoordinator`（`vllm/distributed/stateless_coordinator.py`）+ `StatelessProcessGroup`（`vllm/distributed/utils.py:199`）：不依赖 `torch.distributed` 全局状态，靠一个共享 `Store`（coord store）交换自选的组端口，rank 0 bind 3 个 socket 并把端口写入 store，其他 rank 读取后各自建 NCCL 通信子。用于 elastic EP 运行时动态建/拆 DP/EP 组。

---

## 4. 通信后端与 device_communicators
<!-- tags: nccl, allreduce, custom-allreduce, nvls, p2p, 通信 -->

`GroupCoordinator` 按平台实例化 `device_communicator`（`current_platform.get_device_communicator_cls()`）。CUDA 上是 `CudaCommunicator`（`vllm/distributed/device_communicators/cuda_communicator.py:29`）。

### 4.1 CudaCommunicator 持有的后端
<!-- tags: cuda-communicator, 后端, pynccl, custom-allreduce, symm-mem -->
- `pynccl_comm`（`PyNcclCommunicator`，`pynccl.py`）：直接调 NCCL C API 的通信子，是兜底/主力后端。
- `ca_comm`（`CustomAllreduce`，`custom_all_reduce.py`）：vLLM 自研低延迟 all-reduce（P2P 直写），**仅 TP 组**、**仅同节点**、world_size ∈ {2,4,6,8,16}。
- `qr_comm`（`QuickAllReduce`）：ROCm MI300 专用，补 custom allreduce。
- `fi_ar_comm`（`FlashInferAllReduce`）：FlashInfer all-reduce。
- `aiter_ar_comm`（`AiterCustomAllreduce`）：ROCm AITER。
- `symm_mem_comm`（`SymmMemCommunicator`）：torch symmetric memory。
- `all2all_manager`：EP 的 all2all（见 §2.4）。

### 4.2 all-reduce 分派顺序（`CudaCommunicator.all_reduce`，`cuda_communicator.py:278`）
<!-- tags: all-reduce, 分派, 顺序, 后端, dispatch -->
按输入 tensor 的 size/dtype 依次尝试，命中即返回：
1. **NCCL_SYMM_MEM**（`should_nccl_symm_mem_allreduce`）：`VLLM_USE_NCCL_SYMM_MEM=1` 时启用。
2. **QUICK_REDUCE**（ROCm）。
3. **FLASHINFER**。
4. **AITER_CUSTOM**（ROCm）。
5. **CUSTOM**（vLLM custom all-reduce）。
6. **SYMM_MEM**（torch symm mem）。
7. **PYNCCL**（兜底，再不行回退 `torch.distributed.all_reduce`）。

### 4.3 custom all-reduce 阈值
<!-- tags: custom-allreduce, 阈值, world-size, p2p, mnnvl -->
- `CustomAllreduce._SUPPORTED_WORLD_SIZES = [2,4,6,8,16]`；`max_size` 默认 8MB，但按 SM 架构 + world_size 查表 `CUSTOM_ALL_REDUCE_MAX_SIZES`（`all_reduce_utils.py:31`）：
  - SM 9.0（H100）：ws=2→64MB, ws=4→32MB, ws=6→512KB, ws=8→256KB。
  - SM 10.3/10.7（Blackwell/Rubin）：ws=2→4MB, ws=4→4MB, ws=6→8MB, ws=8→4MB。
- 超过阈值或 world_size 不支持 → 落回 PyNCCL。
- 需要 P2P 可访问（`gpu_p2p_access_check`，`VLLM_SKIP_P2P_CHECK` 可跳过检查）。
- 跨节点（非同节点）时走 **MNNVL**（multicast）路径，`mnnvl_only=True`。

### 4.4 NCCL symmetric memory（NVLS）
<!-- tags: symm-mem, nvls, nccl, 阈值, all-gather -->
- `VLLM_USE_NCCL_SYMM_MEM=1`（默认 0，`envs.py:1979`）启用；`pynccl_allocator.py::is_symmetric_memory_enabled()` 判断。
- all-reduce 阈值表 `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`（`all_reduce_utils.py:100`）：`min_world_size=4`，`custom_ar_preferred_ranges`（4 卡 16K–512K、8 卡 16K–128K 用 custom_AR），`always_use_above_world_size=8`（>8 卡全用 symm mem）。
- all-gather / reduce-scatter 用 `should_nccl_symm_mem_ag_rs()`，走 NVLS（`_all_gather_symm_mem`/`_reduce_scatter_symm_mem`），用持久预注册的 scratch buffer（`_get_symm_scratch`）避免每次注册的 ~0.5ms 开销。

### 4.5 其他 device_communicators
<!-- tags: device-communicators, shm, message-queue, cpu, xpu -->
- `pynccl.py` / `pynccl_wrapper.py`：NCCL C 封装（`PyNcclCommunicator`，支持 `group_start/group_end` 批量集合、`batch_isend_irecv`）。
- `shm_broadcast.py`：`MessageQueue`，基于共享内存 ring buffer 的广播（TP 组内广播 scheduler 元数据用，`VLLM_MQ_MAX_CHUNK_BYTES_MB` 控制分块，默认 16MB）。
- `shm_object_storage.py`：共享内存对象存储。
- `cpu_communicator.py` / `xpu_communicator.py`：CPU / Intel XPU 平台实现。
- `flashinfer_all_reduce.py`、`quick_all_reduce.py`、`aiter_custom_all_reduce.py`：各平台 all-reduce。
- `all_reduce_utils.py`：阈值表 + P2P 检查。

---

## 5. Executor 层
<!-- tags: executor, multiproc, ray, worker -->

`Executor`（`vllm/v1/executor/abstract.py:38`）是抽象基类，`get_class(vllm_config)` 按 `distributed_executor_backend` 选实现：
- `"ray"` → `RayDistributedExecutor`（`VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` 时 `RayExecutorV2`）
- `"mp"` → `MultiprocExecutor`
- `"uni"` → `UniProcExecutor`（world_size==1）
- `"external_launcher"` → `ExecutorWithExternalLauncher`（外部 torchrun 等拉起）
- 也可传自定义 `Executor` 子类或其 import 路径。

**后端自动选择**（`ParallelConfig.__post_init__`，`parallel.py:832`，选择逻辑在 ~912-956）：
- `world_size_across_dp>1` 时：TPU+SPMD→`uni`；CUDA 且 `nnodes>1`→`mp`；CUDA 且本机 GPU 数 < world_size→报错（提示用 ray 或设 nnodes）；`data_parallel_backend=="ray"`→`ray`；否则若 ray 已初始化且有 placement group→`ray`，默认 `mp`。
- `world_size==1`→`uni`。
- `nnodes>1` 只允许 `mp`/`uni`/`external_launcher`（`parallel.py:966-969`）。

### 5.1 MultiprocExecutor（`vllm/v1/executor/multiproc_executor.py:111`）
<!-- tags: multiproc-executor, worker, collective-rpc, 多进程, 多节点 -->
- `supports_pp = True`。
- 用 `mp` 起 `local_world_size` 个 `WorkerProc` 子进程（`WorkerProc.make_worker_process`，`multiproc_executor.py:705`），每个进程 `target=WorkerProc.worker_main`。
- **控制面**：leader 节点建 `rpc_broadcast_mq`（`MessageQueue`）广播 RPC；每个 worker 有 `worker_response_mq`。`collective_rpc`（`multiproc_executor.py:375`）把 `(method, args, kwargs, output_rank)` enqueue 到广播 MQ，再从各 `response_mqs` dequeue 结果。
- **多节点**：`node_rank_within_dp==0` 的节点是 leader，负责跨节点 response MQ（`peer_worker_response_mqs`）。
- 分布式初始化用 file store（`get_file_store_init_method`）或 aiter 需要时的 tcp store。
- worker 健康监控：`start_worker_monitor` 后台线程 + death pipe。
- 关系：`EngineCoreProc`（scheduler 进程）持有一个 `MultiprocExecutor`，通过它 `collective_rpc("execute_model"/"sample_tokens", ...)` 驱动所有 worker。

### 5.2 RayDistributedExecutor（`vllm/v1/executor/ray_executor.py:64`）
<!-- tags: ray-executor, ray, actor, placement-group, dag -->
- worker 是 **Ray actor**（`RayWorkerWrapper`，`vllm/v1/executor/ray_utils.py:56`，继承 `WorkerWrapperBase`）。
- `_init_workers_ray`（`ray_executor.py:156`）：按 placement group 的 GPU bundle 逐个 `ray.remote(...)(RayWorkerWrapper).remote(rpc_rank=rank)`；`VLLM_RAY_PER_WORKER_GPUS` 控制每 actor GPU 数。
- **rank 重排**：按“driver 节点优先 → 节点 worker 数少优先 → IP 小优先”排序（`sort_by_driver_then_worker_ip`），让同节点 worker rank 相邻，然后 `collective_rpc("adjust_rank", ...)` 调整 `rpc_rank`。
- 设备可见性不由 Ray 设置，worker 内部用 `local_rank` 索引（`_update_noset_device_env_vars`）。
- 支持 **Ray Compiled Graph (DAG)**：`_compiled_ray_dag` 把 PP 的 P2P 通信编译进 DAG（`pp_tp_workers` 按 PP rank 再按 TP rank 组织）。
- 环境变量传播：`vllm/ray/ray_env.py` 定义 `DEFAULT_ENV_VAR_PREFIXES`（`VLLM_`、`NCCL_`、`FLASH_ATTENTION_`、`LMCACHE_` 等）从 driver 拷贝到 ray worker，`VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` 可追加。
- `vllm/ray/lazy_utils.py`：`is_ray_initialized`/`is_in_ray_actor` 等惰性判断。

### 5.3 Worker 与并行
<!-- tags: worker, 并行, worker-base, init-device, numa -->
- `WorkerBase`（`vllm/v1/worker/worker_base.py:39`）：硬件无关接口（`init_device`/`load_model`/`execute_model`/`sample_tokens`/`determine_available_memory` 等）。
- `WorkerWrapperBase`（`worker_base.py:191`）：单进程包装，`init_worker` 时按 `worker_cls`（`parallel_config.worker_cls`，默认 `"auto"` 按平台解析）动态实例化真实 worker，并支持 `worker_extension_cls` 注入属性/方法（供 `collective_rpc` 调用）。
- GPU worker：`vllm/v1/worker/gpu_worker.py::Worker`（`init_device` 在 `:315`）。`init_device` 里做 DP local rank 到 `local_rank` 的映射（`local_rank += dp_local_rank * tp_pp_world_size`）、NUMA 绑定、物理 GPU id 映射等。
- worker 内调 `init_distributed_environment`（`gpu_worker.py:1423`）+ `ensure_model_parallel_initialized`（`gpu_worker.py:1432`）建立进程组。

---

## 6. KV Transfer / PD 分离（prefill-decode disaggregation）
<!-- tags: pd-disaggregation, kv-transfer, nixl, mooncake, lmcache, p-d分离 -->

### 6.1 抽象层（`vllm/distributed/kv_transfer/README.md`）
<!-- tags: kv-transfer, 抽象, pipe, lookup-buffer, connector -->
三层：
1. **KV pipe**：tensor 的 FIFO 管道（`send_tensor`/`recv_tensor`）。
2. **KV lookup buffer**：按 token 查 KV 的缓冲（`insert`/`drop_select`），解决 P/D 处理顺序不一致问题。
3. **KV connector**：把 pipe + buffer 接到 vLLM（`send_kv_caches_and_hidden_states`/`recv_...`）。
> pipe 层可绕过（若底层服务本身支持 KV 查找，如 Redis/RDMA DB）。

### 6.2 配置（`vllm/config/kv_transfer.py::KVTransferConfig`）
<!-- tags: kv-transfer-config, 配置, kv-role, kv-connector, 字段 -->
- `kv_connector`：连接器名（必填才启用）。
- `kv_role`：`kv_producer`（P）/ `kv_consumer`（D）/ `kv_both`。
- `engine_id`：默认 uuid4，TP/PP 组内从 rank0 广播同步（`kv_transfer_state.py::_sync_engine_id_across_tp`）。
- `kv_buffer_device`（cuda/cpu/xpu）、`kv_buffer_size`（默认 1e9 字节）。
- `kv_rank`/`kv_parallel_size`/`kv_ip`/`kv_port`（默认 14579）。
- `kv_connector_extra_config`：连接器自定义 JSON。
- `kv_connector_module_path`：外部连接器模块路径（V1）。
- `kv_load_failure_policy`：`recompute`（重算失败 block）/ `fail`（默认，直接失败）。
- CLI：`--kv-transfer-config '{"kv_connector": "...", "kv_role": "..."}'`（`arg_utils.py:1647`，支持 `80m` 这种人读数字）。

### 6.3 连接器工厂与注册（`kv_connector/factory.py`）
<!-- tags: connector, factory, 注册, nixl, mooncake -->
`KVConnectorFactory` 惰性注册/加载。已注册连接器（`factory.py:152` 起）：
- `NixlConnector`（= `NixlPullConnector`，pull/READ 模式）、`NixlPullConnector`、`NixlPushConnector`（push/WRITE 模式）—— 基于 NIXL（RDMA）。
- `LMCacheConnectorV1`、`LMCacheMPConnector`（LMCache 外部 KV 存储）。
- `MooncakeConnector`、`MooncakeStoreConnector`（Mooncake）。
- `FlexKVConnectorV1`、`HF3FSKVConnector`（HF3FS 文件系统）。
- `OffloadingConnector`、`SimpleCPUOffloadConnector`（CPU offload）。
- `MultiConnector`（组合多个子连接器）。
- `MoRIIOConnector`、`DecodeBenchConnector`、`ExampleConnector`、`ExampleHiddenStatesConnector`。

### 6.4 V1 连接器接口（`kv_connector/v1/base.py::KVConnectorBase_V1`）
<!-- tags: kv-connector, 接口, scheduler-role, worker-role, hma -->
连接器分两个 **role**（`KVConnectorRole`）：
- **SCHEDULER**（与 scheduler 同进程）：
  - `get_num_new_matched_tokens(request, num_computed_tokens)`：查远端 KV 能加载多少 token（可返回 None 表示稍后再查）。
  - `update_state_after_alloc(request, blocks, num_external_tokens)`：block 分配后更新状态。
  - `build_connector_meta(scheduler_output)`：构造本步发给 worker 的元数据。
  - `request_finished(request, block_ids)`：请求结束时决定是否异步保存（返回 True 则 block 延迟释放）。
  - `take_events()`：产出 KV 事件（`vllm/distributed/kv_events.py`，`BlockStored`/`BlockRemoved` 等，供外部 KV 索引）。
  - `on_new_request`、`update_connector_output`、`bind_gpu_block_pool`。
- **WORKER**（与 worker 同进程）：
  - `start_load_kv(forward_context)`：forward 前异步加载 KV。
  - `wait_for_layer_load(layer_name)`：attention 层内阻塞等待该层加载完（支持逐层流水）。
  - `save_kv_layer(layer_name, kv_layer, attn_metadata)`：forward 中异步保存某层 KV。
  - `wait_for_save()`：forward 结束前确保保存完成。
  - `get_finished(finished_req_ids)`：返回异步传输完成的请求 id。
  - `register_kv_caches` / `register_cross_layers_kv_cache`（NIXL 预注册）、`get_handshake_metadata`（P/D 带外握手）、`build_connector_worker_meta`。
- `SupportsHMA`（`base.py:85`）：支持 hybrid memory allocator 的连接器需实现 `request_finished_all_groups`；否则要 `--disable-hybrid-kv-cache-manager`。

### 6.5 集成点
<!-- tags: integration, 集成, model-runner-mixin, aggregator, 生命周期 -->
- `kv_transfer_state.py::ensure_kv_transfer_initialized`：worker 侧建 WORKER role 连接器（`_KV_CONNECTOR_AGENT` 全局单例）；`get_kv_transfer_group()`/`has_kv_transfer_group()` 访问。
- `vllm/v1/worker/kv_connector_model_runner_mixin.py::KVConnectorModelRunnerMixin`：在 `execute_model` 的 forward context 内封装连接器生命周期——`bind_connector_metadata` → `start_load_kv` → (yield 执行 forward) → `wait_for_save` + `get_finished` + 收集 stats/events/worker_meta → `clear_connector_metadata`。
- `Executor` 用 `KVOutputAggregator`（`kv_connector/utils.py`）把各 worker 的连接器输出聚合回 scheduler。
- 设计文档：`docs/design/nixl_kv_push_connector.md`（push 模式时序）、`docs/design/nixl_kv_cache_lease.md`（P 侧 KV block 租约 + D 侧心跳续租，避免 D 崩溃时 P 长期占用 block）。
- 示例：`examples/disaggregated/disaggregated_serving/`、`disaggregated_encoder/`、`lmcache/`、`mooncake_connector/`。

---

## 7. EPLB（Expert Parallel Load Balancing）
<!-- tags: eplb, moe, load-balancing, 专家负载均衡 -->

目录 `vllm/distributed/eplb/`。目的：MoE 专家负载不均时，动态把“热门”逻辑专家的**冗余副本**搬到不同 rank，均衡负载。

### 7.1 术语（`eplb/eplb_state.py` 头部注释）
<!-- tags: eplb, 术语, logical-expert, redundant, physical -->
- **Logical Expert**：模型逻辑结构里的专家（如 DeepSeek-R1 每层 256 个）。
- **Redundant Expert**：为热门逻辑专家额外复制的权重副本（`num_redundant_experts`）。
- **Physical Expert**：某设备上实例化的专家副本，可在设备间重排。
- 例：256 逻辑专家 + 32 冗余 = 288 物理专家；32 个 EP rank，每 GPU 持 288/32=9 个本地物理专家。

### 7.2 组件
<!-- tags: eplb, 组件, eplb-state, communicator, policy -->
- `EplbState`（`eplb_state.py:230`）：每模型一份（key=模型配置 hash）。维护滑动窗口负载统计（`expert_load_window_size`，默认 1000）、重排步计数（`expert_rearrangement_step`，`step_interval` 默认 3000）、`physical_to_logical_map`、异步 worker 线程。
- `EplbLayerState`（`eplb_state.py:1121`）：每 MoE 层一份，挂在 `FusedMoE` 层上（`fused_moe/layer.py:243`）。
- `EplbCommunicator`（`eplb_communicator.py`）：专家权重传输后端——`torch_nccl` / `torch_gloo`（CPU staging）/ `nixl`（RDMA 零拷贝）/ `pynccl`。自动选择（`parallel.py:974-989`）：优先 nixl，elastic EP 用 pynccl，静态 EP 用 torch_gloo。
- `async_worker.py::start_async_worker`：后台线程在独立 CUDA stream 上做权重传输（`use_async=True` 默认）。
- `policy/default.py::DefaultEplbPolicy`：`rebalance_experts`（入口）、`replicate_experts`（选哪些逻辑专家复制）、`balanced_packing`（把加权对象均衡打包到各 rank）、`rebalance_experts_hierarchical`（考虑节点内 NVLink 更快的分层放置）、`preserve_intragpu_slots`。算法改编自 DeepSeek EPLB。
- `rebalance_execute.py`：`rearrange_expert_weights_inplace`、`transfer_layer`、`move_from_buffer` 等实际搬权重。

### 7.3 集成
<!-- tags: eplb, 集成, 开关, eplb-config, 每步 -->
- 开关：`enable_eplb=True`（要求 `enable_expert_parallel=True` 且 TP×PCP×DP>1，`parallel.py:493`；仅 CUDA/ROCm）。
- 每步 `gpu_model_runner.py::eplb_step`（`:3502`）调 `eplb_state.step(...)` 更新统计、触发重排。
- EPLB 组（`get_eplb_group()`）与 EP 组同 rank 但独立，隔离通信。
- 配置 `EPLBConfig`（`parallel.py:59`）：`window_size`(1000)、`step_interval`(3000)、`num_redundant_experts`(0)、`use_async`(True)、`policy`("default")、`communicator`(None=自动)、`log_balancedness`。CLI：`--enable-eplb`、`--eplb-config`。

---

## 8. Elastic EP（弹性专家并行）
<!-- tags: elastic, ep, moe -->

目录 `vllm/distributed/elastic_ep/`。让 EP 组**运行时扩缩容**（scale up/down DP/EP），无需重启，配合 EPLB 在扩容时重排专家。

- 开关：`enable_elastic_ep=True`（要求 `enable_eplb=True`、`pipeline_parallel_size==1`、非 external/hybrid LB，`parallel.py:844`；async EPLB 需 NIXL）。
- 用 **stateless NCCL 组**（`StatelessGroupCoordinator`）管理 DP/EP，因为组会动态变化。`_init_elastic_ep_world`（`parallel_state.py:1551`）+ `_init_stateless_group`。
- `elastic_state.py::ElasticEPScalingState`：扩缩容状态机。
  - scale up：`ScaleUpNewEngineState`（新引擎：PRE_KV_INIT→PREPARE→COMPLETE）/ `ScaleUpExistingEngineState`（PREPARE→SYNC_KV_CACHE_MEMORY_SIZE→COMMIT_SCALE_UP→COMPLETE）。
  - scale down：`ScaleDownRemainingEngineState` / `ScaleDownRemovingEngineState`。
- `elastic_execute.py::ElasticEPScalingExecutor`：worker 侧执行 `prepare_reconfiguration`、`transfer_weights(old_dp_size, new_dp_size)`、`switch_and_prepare`/`switch_and_remove`（切换 active 组，`_replace_active_groups`）、`_perform_eplb_reshuffle`、`commit_scale_up/down`。
- `standby_state.py`：`create_standby_groups` 预建目标 dp_size 的 standby DP/EP/EPLB 组，扩容时快速切换。
- 入口：`DPEngineCoreProc`（`vllm/v1/engine/core.py:1986`）持 `eep_scaling_state`；`ReconfigureDistributedRequest`（`vllm/v1/engine/__init__.py`）承载扩缩容请求。
- env：`VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`、`VLLM_ELASTIC_EP_DRAIN_REQUESTS`。

---

## 9. 其他分布式子系统
<!-- tags: ec-transfer, weight-transfer, encoder-cache -->

### 9.1 EC Transfer（encoder cache 传输）
<!-- tags: ec-transfer, encoder-cache, 多模态, connector, 传输 -->
`vllm/distributed/ec_transfer/`。多模态场景下把 encoder（视觉等）输出 cache 在实例间传输，接口与 KV connector 平行：
- `ECConnectorBase`（`ec_connector/base.py:82`）+ `ECConnectorRole`（SCHEDULER/WORKER）。
- 方法：`register_caches`/`start_load_caches`/`save_caches`/`get_finished`/`has_cache_item`。
- 工厂 `ECConnectorFactory`（`ec_connector/factory.py`），配置 `ECTransferConfig`（`vllm/config/ec_transfer.py`：`ec_connector`/`ec_role`(ec_producer/ec_consumer/ec_both)/`ec_rank`/`ec_buffer_device`）。
- 集成：`ec_transfer_state.py::ensure_ec_transfer_initialized`、`vllm/v1/worker/ec_connector_model_runner_mixin.py`、`Executor` 的 `ECOutputAggregator`。

### 9.2 Weight Transfer（权重传输）
<!-- tags: weight-transfer, 权重传输, rlhf, nccl, ipc -->
`vllm/distributed/weight_transfer/`。训练→推理的权重同步（RLHF/在线学习场景）。
- `WeightTransferEngine` / `TrainerWeightTransferEngine`（`base.py`）；`WeightSource` 提供 `(name, tensor)` 流 + `metadata()`。
- 引擎（`factory.py:219` 注册，类在 `nccl_engine.py:105`）：`nccl`（`NCCLWeightTransferEngine`）、`ipc`（`IPCWeightTransferEngine`，同机共享内存）、`sparse_nccl`（`SparseNCCLWeightTransferEngine`，稀疏更新）。
- 配置 `WeightTransferConfig`（`vllm/config/weight_transfer.py`）。

### 9.3 KV Events
<!-- tags: kv-events, 事件, zmq, block-stored, 外部索引 -->
`vllm/distributed/kv_events.py`：`KVCacheEvent`（`BlockStored`/`BlockRemoved`）+ `EventBatch`，用 ZMQ 发布，供外部 KV 索引/缓存感知（如 LMCache、prefix cache 跨实例）。配置 `KVEventsConfig`（`vllm/config/kv_events.py`）。

---

## 10. 配置 / 调优旋钮
<!-- tags: tuning, knobs, flags, 分布式调优 -->

### 10.1 CLI / ParallelConfig 字段
<!-- tags: cli, parallel-config, 字段, flags, 旋钮 -->
| 旋钮 | 说明 |
|---|---|
| `--tensor-parallel-size` / `-tp` | TP 组数 |
| `--pipeline-parallel-size` / `-pp` | PP 组数 |
| `--data-parallel-size` / `-dp` | DP 组数（MoE） |
| `--data-parallel-rank` / `--data-parallel-start-rank` / `--data-parallel-size-local` | DP rank 定位（external/hybrid LB） |
| `--data-parallel-address` / `--data-parallel-rpc-port` | DP 集群头节点地址 / RPC 端口（默认 29550） |
| `--data-parallel-backend` | `mp` / `ray` |
| `--data-parallel-hybrid-lb` / `--data-parallel-external-lb` | DP 负载均衡模式 |
| `--enable-expert-parallel` | MoE 用 EP |
| `--all2all-backend` | EP all2all 后端（见 §2.4） |
| `--enable-eplb` / `--eplb-config` | EPLB 开关 / 参数 |
| `--enable-elastic-ep` | 弹性 EP |
| `--distributed-executor-backend` | `ray` / `mp` / `uni` / `external_launcher` / 自定义类 |
| `--nnodes` / `--node-rank` / `--master-addr` / `--master-port` | mp 多节点（master_port 默认 29501） |
| `--numa-bind` / `--numa-bind-nodes` / `--numa-bind-cpus` | GPU worker NUMA 绑定 |
| `--kv-transfer-config` | KV 传输（PD 分离）JSON |
| `--disable-custom-all-reduce` | 禁用 custom all-reduce，回退 NCCL |
| `--worker-cls` / `--worker-extension-cls` | 自定义 worker / 扩展类 |
| `--enable-dbo` / `--ubatch-size` | dual batch overlap / microbatch（DP 内） |

### 10.2 环境变量（`vllm/envs.py`）
<!-- tags: env-vars, 环境变量, dp, nccl, nixl -->
- `VLLM_DP_SIZE` / `VLLM_DP_RANK` / `VLLM_DP_RANK_LOCAL`：离线 SPMD 模式 DP。
- `VLLM_DP_MASTER_IP` / `VLLM_DP_MASTER_PORT`：DP master。
- `VLLM_USE_NCCL_SYMM_MEM`（默认 0）：NCCL symmetric memory all-reduce。
- `VLLM_ALLREDUCE_USE_SYMM_MEM`（默认 1）/ `VLLM_ALLREDUCE_USE_FLASHINFER`（默认 1）：all-reduce 后端开关。
- `VLLM_SKIP_P2P_CHECK`（默认 1）：跳过 P2P 检查。
- `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`（默认 0）：用 `torch.distributed.split_group` 建子组。
- `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`（默认 1）：Ray V2 executor。
- `VLLM_MQ_MAX_CHUNK_BYTES_MB`（默认 16）：共享内存 MQ 分块。
- `VLLM_RAY_PER_WORKER_GPUS` / `VLLM_RAY_BUNDLE_INDICES`：Ray worker GPU 数 / bundle 指定。
- `VLLM_NIXL_SIDE_CHANNEL_HOST`（localhost）/ `VLLM_NIXL_SIDE_CHANNEL_PORT`（5600）：NIXL 侧信道。
- `VLLM_NIXL_EP_MAX_NUM_RANKS`（32）：NIXL EP 最大 rank。
- `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` / `VLLM_ELASTIC_EP_DRAIN_REQUESTS`：弹性 EP。
- `VLLM_BATCH_INVARIANT`：批不变性（会禁用 custom all-reduce、symm mem）。
- `VLLM_DISABLE_ASYNC_SCHEDULING` 相关影响 `disable_nccl_for_dp_synchronization`（DP 同步用 Gloo 还是 NCCL）。
- NCCL 相关：`NCCL_*`（经 ray_env 传播到 worker）。

### 10.3 调优要点
<!-- tags: tuning, 调优, 单机, 跨节点, moe -->
- **单机**：优先 TP（`-tp 8`），custom all-reduce 自动启用（同节点、world_size∈{2,4,6,8,16}、size 低于阈值）。
- **跨节点 dense**：TP（节点内）+ PP（节点间）。
- **MoE 高吞吐**：DP+EP（`-dp N --enable-expert-parallel`），选合适 `--all2all-backend`（多节点常用 `deepep_low_latency` / `mori_*` / `nixl_ep`）。
- **MoE 负载不均**：开 `--enable-eplb` + `num_redundant_experts`。
- **需要弹性扩缩容**：`--enable-elastic-ep`（需 NIXL + EPLB）。
- **PD 分离**：P 实例 `kv_role=kv_producer`、D 实例 `kv_role=kv_consumer`，`kv_connector=NixlConnector`（RDMA）或 LMCache/Mooncake（外部存储）。
- **DP 死锁**：同 DP 组必须同步 `generate`；`coordinate_batch_across_dp` 的 all-reduce 是每步开销，`disable_nccl_for_dp_synchronization` 可在 async scheduling 下改走 CPU/Gloo 避免 GPU sync。

---

## 11. 关键文件
<!-- tags: files, 源码索引 -->

**进程组 / 通信**
- `vllm/distributed/parallel_state.py` — 进程组建立、rank 划分、`GroupCoordinator`、通信原语（核心，~2359 行）。
- `vllm/distributed/communication_op.py` — TP 通信便捷函数。
- `vllm/distributed/stateless_coordinator.py` — `StatelessGroupCoordinator`（elastic EP 用）。
- `vllm/distributed/utils.py` — `StatelessProcessGroup`、store 工具。
- `vllm/distributed/device_communicators/cuda_communicator.py` — CUDA 通信子 + all-reduce 分派。
- `vllm/distributed/device_communicators/custom_all_reduce.py` — 自研低延迟 all-reduce。
- `vllm/distributed/device_communicators/all_reduce_utils.py` — all-reduce 阈值表、P2P 检查。
- `vllm/distributed/device_communicators/pynccl.py` / `pynccl_wrapper.py` / `pynccl_allocator.py` — NCCL 封装 + symm-mem 分配器。
- `vllm/distributed/device_communicators/all2all.py` — EP all2all 各后端。
- `vllm/distributed/device_communicators/shm_broadcast.py` — 共享内存 `MessageQueue`。
- `vllm/distributed/device_communicators/{flashinfer_all_reduce,quick_all_reduce,aiter_custom_all_reduce,symm_mem,cpu_communicator,xpu_communicator,ray_communicator}.py`。

**配置**
- `vllm/config/parallel.py` — `ParallelConfig` / `EPLBConfig`（TP/PP/DP/EP/PCP/DCP、executor 后端、all2all、EPLB、elastic EP）。
- `vllm/config/kv_transfer.py` — `KVTransferConfig`。
- `vllm/config/ec_transfer.py` — `ECTransferConfig`。
- `vllm/config/weight_transfer.py` — `WeightTransferConfig`。
- `vllm/engine/arg_utils.py` — 上述配置对应的 CLI flag。

**Executor / Worker**
- `vllm/v1/executor/abstract.py` — `Executor` 基类 + 后端选择。
- `vllm/v1/executor/multiproc_executor.py` — `MultiprocExecutor` / `WorkerProc`。
- `vllm/v1/executor/ray_executor.py` / `ray_executor_v2.py` / `ray_utils.py` — Ray executor + `RayWorkerWrapper`。
- `vllm/v1/executor/uniproc_executor.py` — 单进程。
- `vllm/v1/worker/worker_base.py` — `WorkerBase` / `WorkerWrapperBase`。
- `vllm/v1/worker/gpu_worker.py` — GPU `Worker`（`init_device`/`load_model`/`execute_model`）。
- `vllm/v1/worker/dp_utils.py` — DP 步同步 / padding。
- `vllm/v1/engine/core.py` — `EngineCore` / `EngineCoreProc` / `DPEngineCoreProc`。
- `vllm/v1/engine/coordinator.py` — `DPCoordinator`（DP 多引擎协调）。
- `vllm/ray/ray_env.py` / `lazy_utils.py` — Ray 环境传播 / 惰性判断。

**KV / EC / EPLB / Elastic / Weight**
- `vllm/distributed/kv_transfer/kv_transfer_state.py` — worker 侧连接器单例。
- `vllm/distributed/kv_transfer/kv_connector/factory.py` — 连接器注册表。
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` — `KVConnectorBase_V1` 接口。
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` — NIXL pull/push 连接器。
- `vllm/distributed/kv_events.py` — KV 事件。
- `vllm/v1/worker/kv_connector_model_runner_mixin.py` — 连接器在 model runner 的集成。
- `vllm/distributed/eplb/{eplb_state,eplb_communicator,async_worker,rebalance_execute,policy/default}.py` — EPLB。
- `vllm/distributed/elastic_ep/{elastic_state,elastic_execute,standby_state}.py` — 弹性 EP。
- `vllm/distributed/ec_transfer/` — encoder cache 传输。
- `vllm/distributed/weight_transfer/` — 权重传输引擎。
- `vllm/distributed/nixl_utils.py` — NIXL 可用性/UCX 配置。

**模型层并行**
- `vllm/model_executor/layers/linear.py` — `ColumnParallelLinear` / `RowParallelLinear`（TP）。
- `vllm/model_executor/layers/fused_moe/{config,layer}.py` — `FusedMoEParallelConfig`（EP 尺寸计算）/ `FusedMoE`（EP/EPLB 集成）。
