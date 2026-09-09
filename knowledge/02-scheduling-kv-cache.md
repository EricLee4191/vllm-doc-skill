# 调度器与 KV Cache 管理

> 基于 vLLM main（`f32b17b6d6`，2026-08-21），最新 release tag **v0.28.0rc1**（v1 架构为默认且唯一的引擎）。所有路径相对于仓库根 `/Users/baofeng/baofeng/github/vllm`。

## 1. 总体架构
<!-- tags: scheduler, overview, 调度器 -->

v1 引擎中，调度器（`Scheduler`）与 KV cache 管理（`KVCacheManager`）都运行在 **EngineCore 进程**（CPU 侧），worker 进程只持有 KV cache 张量和 block table 的物理映射。调度器每步产出一个 `SchedulerOutput`（含 `num_scheduled_tokens`、`block_ids` 等），worker 据此执行 forward；`update_from_output()` 把采样结果写回请求状态。

核心对象关系：

```
Scheduler (vllm/v1/core/sched/scheduler.py)
 └── KVCacheManager (vllm/v1/core/kv_cache_manager.py)
      ├── KVCacheCoordinator (kv_cache_coordinator.py)
      │    ├── UnitaryKVCacheCoordinator        # 单 group（纯 full attention）
      │    ├── HybridKVCacheCoordinator         # 多 group（混合注意力 / mamba）
      │    └── KVCacheCoordinatorNoPrefixCache  # 关闭 prefix caching 时
      │    └── SingleTypeKVCacheManager × N     # 每个 kv_cache_group 一个
      └── BlockPool (block_pool.py)             # 物理块池 + 前缀缓存哈希表
```

- `KVCacheConfig`（`vllm/v1/kv_cache_interface.py:956`）：`num_blocks`（物理块数）、`kv_cache_tensors`（worker 如何初始化张量）、`kv_cache_groups`（`KVCacheGroupSpec` 列表，每组共享一张 block table）。
- 调度器通过 `SchedulerConfig.get_scheduler_cls()`（`vllm/config/scheduler.py:170`）选择 `Scheduler` 或 `AsyncScheduler`（`v1/core/sched/async_scheduler.py`）；`scheduler_cls` 字段可替换为自定义类。

## 2. Scheduler 调度算法
<!-- tags: scheduling, chunked-prefill, preemption, policy, 调度, 抢占 -->

### 2.1 统一 token 模型：没有 "prefill/decode 阶段"
<!-- tags: scheduler, token-model, continuous-batching, 统一模型, num-computed-tokens -->

`Scheduler.schedule()`（`scheduler.py:484`）的核心注释说明了 v1 的统一模型：

> 调度器里没有 "decoding phase / prefill phase"。每个请求只有 `num_computed_tokens` 与 `num_tokens_with_spec` 两个计数器，每步的目标是让 `num_computed_tokens` 追上 `num_tokens_with_spec`。这统一覆盖了 chunked prefill、prefix caching、speculative decoding 都只是这一模型的推论。

每步流程（`scheduler.py:484-1319`）：

1. **先调度 running 队列**：遍历 `self.running`，对每个请求计算 `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`，受 `token_budget`（= `max_num_scheduled_tokens`）与 `input_budget`（= `max_num_batched_tokens`）约束。
2. **再调度 waiting 队列**：在 token budget 有余量、且 `len(running) < max_num_seqs` 时，从 `waiting`/`skipped_waiting` 队列按策略取请求，做前缀缓存查找 + `allocate_slots()` 后加入 `running`。
3. 产出 `SchedulerOutput`（`v1/core/sched/output.py`）：`scheduled_new_reqs`（`NewRequestData`，含 `block_ids`）、`scheduled_cached_reqs`（`CachedRequestData`，增量更新）、`num_scheduled_tokens`、`num_common_prefix_blocks`（cascade attention 用）等。

**continuous batching** 即由此实现：每步（每次 forward）都重新组 batch，prefill 与 decode 请求**混合在同一 batch 中**（prefill 以 chunk 形式与 decode token 混排，即 prefill/decode 混合调度）。

### 2.2 队列与策略
<!-- tags: scheduler, queues, fcfs, priority, async-scheduler -->

- `self.waiting` / `self.running` / `self.skipped_waiting`（`scheduler.py:196-198`）：
  - `waiting`：新请求与抢占回来的请求（`prepend_request` 插到队首）。
  - `skipped_waiting`：本步因约束（LoRA 上限、encoder budget、KV connector 异步加载等）被跳过的请求，步末重新 prepend 回队首（`scheduler.py:1169`）。
  - 队列实现见 `v1/core/sched/request_queue.py`：`FCFSRequestQueue`（deque）与 `PriorityRequestQueue`（heapq，按 `(priority, arrival_time)` 排序，priority 数值小优先）。策略由 `SchedulerConfig.policy`（`"fcfs"` / `"priority"`）决定。
- `RequestStatus`（`v1/request.py:364`）：`WAITING` → `RUNNING` → `FINISHED_*`；另有 `PREEMPTED`、`WAITING_FOR_REMOTE_KVS`（KV connector 异步加载）、`WAITING_FOR_STREAMING_REQ`、`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`。
- **AsyncScheduler**（`async_scheduler.py`）：在 `_update_after_schedule` 中为 decode 请求预置 `num_output_placeholders`（1 个采样 token + spec tokens），使调度器可以在上一步 GPU 还在执行时就调度下一步，消除调度-执行重叠；`_update_request_with_output` 中提前 `cache_blocks` 新 token。由 `SchedulerConfig.async_scheduling` 控制（默认开启，`SchedulerConfig.get_scheduler_cls` 据此选类）。

### 2.3 Chunked prefill
<!-- tags: chunked-prefill, 分块, watermark, 准入, throttle -->

- `SchedulerConfig.enable_chunked_prefill` 默认 `True`（`vllm/config/scheduler.py:74`）。chunk 大小由剩余 `max_num_batched_tokens` 决定；`long_prefill_token_threshold`（默认 0=不限）可给单个请求的 prefill chunk 设上限（`scheduler.py:571`、`960`）。
- `scheduler_reserve_full_isl`（默认 `True`，`config/scheduler.py:130`）：准入时检查**整条序列**（而非首个 chunk）能否装下，防止 chunked prefill 过度准入导致 thrashing；对应 `allocate_slots(full_sequence_must_fit=...)`。
- `watermark`（默认 0.0，`config/scheduler.py:136`）：准入 waiting/preempted 请求时额外要求 `watermark * num_blocks` 的空闲块余量，避免频繁抢占。在 `KVCacheManager.allocate_slots` 中实现（`kv_cache_manager.py:466`，仅对 WAITING/PREEMPTED 且已有请求被调度时生效）。
- DP 部署的 prefill 节流：`prefill_schedule_interval` + `schedule(throttle_prefills=...)`，非对齐步只跑 decode、把 prefill chunk 推迟（`scheduler.py:528` 附近）。

### 2.4 抢占（preemption）：recompute，无 swap
<!-- tags: preemption, recompute, 抢占, 无swap, 释放 -->

v1 只有 **recompute**，没有 v0 的 swap（CPU 换出）。`_preempt_request()`（`scheduler.py:1347`）：

- 释放该请求全部 KV 块（`_free_request_blocks` → `kv_cache_manager.free`，带 hash 的块进入 LRU 可驱逐区，可被前缀缓存复用）；
- `request.num_computed_tokens = 0`，状态置 `PREEMPTED`，`num_preemptions += 1`，`waiting.prepend_request(request)` 放回队首；
- 异步调度下把在途输出标记为 stale（`num_stale_output_tokens`），恢复时丢弃或按序排空。

触发点：`allocate_slots()` 返回 `None`（块不够）时，FCFS 策略抢占 `running` 队尾（`running.pop()`），priority 策略抢占 `(priority, arrival_time)` 最大者（`scheduler.py:686` 附近）；被抢占者若是当前请求本身则停止。注意 v1 中**已调度请求不会被 swap 到 CPU**；需要"换出"语义时依赖 KV offload（见第 6 节）。

### 2.5 完成与释放
<!-- tags: update-from-output, 完成, 释放, spec-rollback, defer-free -->

- `update_from_output()`（`scheduler.py:1744`）：处理采样结果、spec token 拒绝回滚（`num_computed_tokens -= num_rejected`）、停止条件（`check_stop`）、`_free_request`（`scheduler.py:2422`）释放块。KV connector 场景下可延迟释放（`defer_block_free`，`deferred_frees` FIFO 按 step 序号 fence，`scheduler.py:342`）。

## 3. KV Cache 的 block 抽象
<!-- tags: kv-cache, block, blockpool, prefix-caching, apc, 前缀缓存 -->

### 3.1 物理块与 BlockPool
<!-- tags: blockpool, kv-cache-block, lru, 物理块, free-queue -->

`KVCacheBlock`（`v1/core/kv_cache_utils.py:160`）是纯元数据：`block_id`、`ref_cnt`、`_block_hash`（满块且被缓存时才有）、`_block_hash_num_tokens`（支持块内 partial 哈希）、`prev/next_free_block`（双向链表指针）、`is_null`。

`BlockPool`（`v1/core/block_pool.py:143`）：

- `blocks: list[KVCacheBlock]`：`num_gpu_blocks` 个物理块；`null_block`（block_id=0）是占位符，永不释放，用于 SWA/mamba 中"窗口外"的空槽位。
- `free_block_queue: FreeKVCacheBlockQueue`（`kv_cache_utils.py:226`）：手写双向链表（O(1) 中间删除），**按 LRU 顺序**组织空闲块——队首是最先被驱逐的。`free_blocks()`（`block_pool.py:719`）把无 hash 的块 prepend（LIFO 复用，GPU 局部性更好）、有 hash 的块 append（FIFO 复用，LRU 驱逐）。
- `cached_block_hash_to_block: BlockHashToBlockMap`（`block_pool.py:33`）：`{BlockHashWithGroupId: KVCacheBlock | dict[block_id, KVCacheBlock]}` 前缀缓存哈希表；同一 hash 可能对应多个物理块（不去重，保证 block table append-only）。
- 分配：`get_new_blocks()`（`block_pool.py:647`）从队首取块，`_maybe_evict_cached_block()` 顺带把被取走块的 hash 从缓存表移除（即 **LRU 驱逐**）。
- 命中复用：`touch()`（`block_pool.py:702`）ref_cnt+1 并把块从 free queue 中移除（ref_cnt=0 时）。
- `get_usage()`（`block_pool.py:808`）：`1 - free / (num_gpu_blocks - 1)`。

**block table**：worker 侧的 block table 就是每个请求在各 group 的 `block_id` 列表（`KVCacheBlocks.get_block_ids()`，`kv_cache_manager.py:77`），随 `SchedulerOutput` 下发；物理 KV 张量按 `block_id` 索引。

### 3.2 KVCacheManager 与 allocate_slots
<!-- tags: kvcachemanager, allocate-slots, 分配, single-type, coordinator -->

`KVCacheManager`（`kv_cache_manager.py:118`）是调度器与缓存系统的唯一接口，内部委托 `coordinator`（按 group 分发的 `SingleTypeKVCacheManager`）与 `block_pool`。

`allocate_slots()`（`kv_cache_manager.py:347`）是核心，布局注释（`kv_cache_manager.py:393-425`）：

```
| <comp> | <new_comp> | <ext_comp> | <new> | <lookahead> |
   已计算   本地前缀命中  外部(connector)  待计算   spec 预留
```

三阶段：
1. `coordinator.remove_skipped_blocks()`：释放注意力窗口外的块（SWA/mamba 等回收型 spec）；
2. `coordinator.get_num_blocks_to_allocate()`：计算需要的新块数（含对 `new_computed_blocks` 中"可驱逐候选块"的计数，`single_type_kv_cache_manager.py:217-222`），与 `free_blocks - reserved_blocks - watermark_blocks` 比较，不够则返回 `None`（触发抢占或拒绝准入）；
3. `allocate_new_computed_blocks()`（把命中的缓存块 touch 到请求）+ `allocate_new_blocks()`（从 pool 取新块），最后 `coordinator.cache_blocks()` 把新满块写入前缀缓存。

`SingleTypeKVCacheManager`（`v1/core/single_type_kv_cache_manager.py:36`）按注意力类型实现 `find_longest_cache_hit` / `get_num_blocks_to_allocate` / `remove_skipped_blocks` / `cache_blocks` 等：

| Manager | Spec | 特点 |
|---|---|---|
| `FullAttentionManager` | `FullAttentionSpec` | 块永不回收；命中查找从左到右线性扫描（downward-closing）；EAGLE 时丢弃最后一个匹配块（drop_eagle_block） |
| `SlidingWindowManager` | `SlidingWindowSpec` | 窗口外块被 `remove_skipped_blocks` 回收；命中需窗口内连续块连续 |
| `ChunkedLocalAttentionManager` | `ChunkedLocalAttentionSpec` | 只保留最近 `attention_chunk_size` 窗口 |
| `MambaManager` | `MambaSpec` | 状态块按 block 边界缓存（`mamba_cache_mode` all/align/none），窗口外置 null |
| `RSWAManager` | `RSWASpec` | Reference SWA：decode 时驱逐 prefill 尾部与当前窗口之间的 gap 块 |
| `CrossAttentionManager` | `CrossAttentionSpec` | encoder-decoder 交叉注意力 |
| `SinkFullAttentionManager` | `SinkFullAttentionSpec` | sink token 全注意力 |

`HybridKVCacheCoordinator.find_longest_cache_hit()`（`kv_cache_coordinator.py:560` 类、`757` 方法）用**迭代不动点算法**协调多 group 的命中长度：各 attention group 接受或缩短候选长度，任一缩短则重查所有 group，单调递减必收敛；full attention 组因 downward-closed 只需查一次。

### 3.3 自动前缀缓存（APC）
<!-- tags: apc, prefix-caching, 前缀缓存, block-hash, lru-eviction -->

- **块哈希**：`Request` 创建/追加 token 时增量计算 `block_hashes`（`v1/request.py:219,276`；`get_request_block_hasher`，`kv_cache_utils.py:712`）。哈希是**链式**的：`hash_block_tokens(parent_hash, tokens, extra_keys)`（`kv_cache_utils.py:618`），每个块哈希唯一标识"到该块结尾的整个前缀"。MM/LoRA/多模态特征通过 `extra_keys` 参与哈希。哈希算法由 `CacheConfig.prefix_caching_hash_algo`（`sha256`（默认）/ `sha256_cbor` / `xxhash` / `xxhash_cbor`）。
- **命中**：请求首次调度时 `KVCacheManager.get_computed_blocks()`（`kv_cache_manager.py:232`）→ `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`，`max_cache_hit_length = num_tokens - 1`（最后一个 token 必须重算以取 logits）。命中的块 `touch()` 后并入请求 block table，`num_computed_tokens` 直接跳到命中长度，跳过对应 prefill 计算。
- **写入**：`allocate_slots` 末尾与 `update_from_output`（异步调度路径）中 `cache_blocks()`，把新满块经 `BlockPool.cache_full_blocks()`（`block_pool.py:225`）插入哈希表。
- **驱逐**：LRU——空闲且有 hash 的块留在 free queue 尾部，分配新块时从队首驱逐（`_maybe_evict_cached_block`，`block_pool.py:679`）。
- **粒度**：`hash_block_size`（= `CacheConfig.prefix_match_unit`）可细于物理块（如 32 vs 1024），由 `resolve_kv_cache_block_sizes()`（`kv_cache_utils.py:648`）解析：单 group 时 = `block_size * dcp`；多 group 时 = `prefix_match_unit` 或各组 block size 的 GCD。细粒度 partial 命中（`enable_partial_hash_hits`）主要服务 Mamba "align" 模式。
- **重置**：`Scheduler.reset_prefix_cache()`（`scheduler.py:2545`）→ `BlockPool.reset_prefix_cache()`（`block_pool.py:764`），RLHF 权重更新后失效缓存用；`reset_running_requests=True` 时先抢占所有 running 请求。
- **统计**：`PrefixCacheStats`（`v1/metrics/stats.py`），`--log-stats` 时记录 query/hit token 数，暴露为 Prometheus 指标。

## 4. KV cache 容量计算（num_gpu_blocks）
<!-- tags: kv-cache, capacity, num-gpu-blocks, 显存, 容量 -->

启动时三步（`v1/worker/gpu_worker.py` + `v1/core/kv_cache_utils.py`）：

1. **可用显存**：`GPUWorker.determine_available_memory()`（`gpu_worker.py:475`）
   - 若设了 `kv_cache_memory_bytes`，直接用它（忽略 `gpu_memory_utilization`）；
   - 否则跑一次 `profile_run()`（dummy forward，按 `max_num_batched_tokens` 编译/捕获 CUDA graph），`available = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate`（`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`，v0.21+ 默认开）。
   - `requested_memory = total_gpu_memory * gpu_memory_utilization`（默认 **0.92**）。
2. **分组与块数**：`get_kv_cache_configs()`（`kv_cache_utils.py:2138`）
   - 合并所有 worker 的 per-layer spec（PP 各 stage 不同层名），`get_kv_cache_groups()`（`kv_cache_utils.py:1797`）按 spec 类型分组：全同 → 单 group；`UniformTypeKVCacheSpecs`（同类型不同 hidden size）→ 单 group 聚合；混合（full + SWA + mamba…）→ 多 group，物理页大小统一（`unify_kv_cache_spec_page_size`），`disable_hybrid_kv_cache_manager=True` 时强制全部按 full attention 分配。
   - `get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1377`）：一般情况 `num_blocks = available_memory // page_size // group_size`（`get_num_blocks`，`kv_cache_utils.py:1023`），`group_size = max(每组层数)`，每组各层共享同一 tensor 的不同部分；`num_gpu_blocks_override` 可覆盖（测试抢占用）。
   - 各 rank 取 `min(num_blocks)` 对齐（`kv_cache_utils.py:2278`）。
3. **容量报告**：`update_kv_cache_capacity()`（`kv_cache_utils.py:1917`）计算 `kv_cache_size_tokens = max_concurrency * max_model_len`（group-aware，`get_max_concurrency_for_kv_cache_config`，`kv_cache_utils.py:967`），启动日志打印 "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: X.XXx"。

**page_size**：`AttentionSpec.page_size_bytes = num_heads * storage_block_size * (head_size + head_size_v) * dtype_size`（`kv_cache_interface.py:255`），受 `cache_dtype`（fp8 等量化会缩小 page size，`KVQuantMode`）；`block_size` 默认 **16**（`CacheConfig.DEFAULT_BLOCK_SIZE`，`vllm/config/cache.py:59`）。

## 5. KVCacheSpec 体系与注册
<!-- tags: kv-cache, spec, mamba, sliding-window, hybrid -->

`KVCacheSpec`（`vllm/v1/kv_cache_interface.py:142`，frozen dataclass）描述**一层**的 KV 格式：`block_size`、`page_size_bytes`、`max_memory_usage_bytes(vllm_config)`（该层最坏占用）、`max_num_blocks_per_req`（worker block table 行宽）。`KVCacheSpecKind` 枚举：`full_attention` / `mla_attention` / `sliding_window` / `sliding_window_mla` / `mamba` / `chunked_local_attention` / `sink_full_attention` / `encoder_only_attention` / `cross_attention`。

主要 spec（`kv_cache_interface.py`）：

- `AttentionSpec`（:218）：`num_kv_heads`、`head_size`、`head_size_v`、`dtype`、`kv_quant_mode`、`page_size_padded`；
- `FullAttentionSpec`（:275）：+ `sliding_window`/`attention_chunk_size`（hybrid 关闭时 SWA 按 full 分配）、`non_causal`（Prefix LM，会禁用 chunked prefill/prefix caching）；
- `MLAAttentionSpec`（:382）：DeepSeek 系，单 latent 向量（`head_size_v=0`），`compress_ratio`、`storage_block_size = block_size // compress_ratio`；
- `SlidingWindowSpec`（:537）：`max_admission_blocks_per_request()`（:546）= `cdiv(sliding_window - 1 + extra_retained_tokens + max_in_flight_tokens, block_size) + 1`，是启动池容量与运行时准入的**单一事实来源**（防 #39734 死锁）；（注：`ChunkedLocalAttentionSpec` 也有同名方法 :500）
- `ChunkedLocalAttentionSpec`（:497）：`min(attention_chunk_size + max_in_flight_tokens, max_model_len)`；
- `MambaSpec`（:668）：`shapes/dtypes`（conv + ssm 状态），`max_memory_usage_bytes` 随 `mamba_cache_mode`：`all` → 全序列块数，`align` → `2 + num_speculative_blocks` 个状态块，`none` → 1 个；
- `UniformTypeKVCacheSpecs`（:797）：同类型多层聚合，`page_size_bytes` 为各层之和。

**注册机制**（`vllm/v1/kv_cache_spec_registry.py`）：`KVCacheSpecRegistry` 全局表 `{spec_cls: (manager_class, uniform_type_base_spec)}`，`@register_kv_cache_spec(manager_class=..., uniform_type_base_spec=...)` 装饰器支持 out-of-tree 自定义 spec；`get_manager_class` 沿 MRO 查找。内置注册在 `register_all_kvcache_specs()`（`single_type_kv_cache_manager.py:1897`）：`FullAttentionSpec→FullAttentionManager`、`SlidingWindowSpec→SlidingWindowManager`、`MambaSpec→MambaManager`、`ChunkedLocalAttentionSpec→ChunkedLocalAttentionManager`、`CrossAttentionSpec→CrossAttentionManager`、`MLAAttentionSpec/RSWASpec/HiddenStateCacheSpec/SinkFullAttentionSpec` 归入 full-attention 族；`current_platform.register_custom_kv_cache_specs()` 允许平台扩展。`uniform_type_base_spec` 相同的 spec 会被合并进同一个 kv cache group（如 MLA 与 full attention 可同组）。

## 6. KV Offload（换出到 CPU/磁盘/远端）
<!-- tags: kv-offload, cpu-offload, 换出, offloading -->

两条独立路径：

### 6.1 原生 offloading（`vllm/v1/kv_offload/`，默认 backend）
<!-- tags: kv-offload, native, offloading, tiering, cpu -->

- 配置：`CacheConfig.kv_offloading_size`（GiB，None=关闭）+ `kv_offloading_backend`（`"native"` 默认 / `"lmcache"`）。`VllmConfig._post_init_kv_transfer_config()`（`vllm/config/vllm.py:990`）把它翻译成 KV connector：native → `OffloadingConnector`（或 `VLLM_USE_SIMPLE_KV_OFFLOAD=1 时 → `SimpleCPUOffloadConnector`），`kv_role="kv_both"`，`cpu_bytes_to_use = size GiB`。
- 抽象（`kv_offload/base.py`）：`OffloadingManager`（:220）：`lookup/prepare_load/prepare_store/complete_store/complete_load/on_new_request/on_request_finished`；`OffloadKey = block_hash + group_idx`（:26）；`Medium`（CPU/STORAGE）、`Locality`（LOCAL/REMOTE）、`TierFilter` 支持分层；`OffloadPolicy`（BLOCK_LEVEL：只 offload 新算块 / 全量）。
- 后端注册（`kv_offload/factory.py`）：`CPUOffloadingSpec`（`kv_offload/cpu/`，含 `policies/lru.py`、`policies/arc.py` 驱逐策略、`swap_blocks_triton.py`）与 `TieringOffloadingSpec`（`kv_offload/tiering/`，多级：`fs/` 文件系统、`obj/` 对象存储、`p2p/` 实例间）。
- 语义：GPU 块被 LRU 驱逐前/后被异步写到 CPU（或更低层），后续请求的 prefix 查找可命中 offload 层并异步 load 回 GPU（请求进入 `WAITING_FOR_REMOTE_KVS`，`scheduler.py:1084-1114`）。这是 v1 中"swap"语义的正式实现。

### 6.2 simple_kv_offload（`vllm/v1/simple_kv_offload/`）
<!-- tags: simple-kv-offload, cpu, disk, cuda-stream, 换出 -->

更简单的 CPU（可选磁盘）offload connector：

- `SimpleCPUOffloadScheduler`（`manager.py:67`）：从 GPU `KVCacheConfig` 派生 CPU 侧配置（`_derive_cpu_config`），维护 CPU 块池；对 full-attention group 做前缀匹配，产出 `SimpleCPUOffloadMetadata`（load/store 的 `gpu_block_ids ↔ cpu_block_ids` 映射）。
- `SimpleCPUOffloadWorker`（`worker.py:26`）：独立 CUDA stream（`load_stream`/`store_stream`）异步 DMA（`copy_backend.py`），`cuda_mem_ops.py` 做 pinned memory 分配；`disk_backend.py` 支持落盘（`kv_offload_backend="disk"`、`disk_path`、`disk_buffer_slots`）。
- 与 6.1 的区别：实现简单、面向单机 CPU/磁盘两级，无 tiering/p2p。

## 7. 关键配置/调优旋钮
<!-- tags: tuning, knobs, flags -->

| 旋钮（CLI / 环境变量）

| 旋钮 | 位置 | 说明 |
|---|---|---|
| `max_num_seqs` | `SchedulerConfig:63` | 单步最大并发请求数；默认按硬件（`arg_utils.get_batch_defaults`：H100 类 1024，其他 256） |
| `max_num_batched_tokens` | `SchedulerConfig:49` | 单步最大 token 数（prefill chunk 上限）；默认 2048（测试值），实际按硬件：B200 16384 / H100 8192-16384 / 其他 2048-8192 |
| `max_num_scheduled_tokens` | `SchedulerConfig:56` | 调度器单步可发 token 上限，默认 = `max_num_batched_tokens`（spec decode 时可能更小） |
| `enable_chunked_prefill` | `SchedulerConfig:74` | 默认 True（encoder-decoder 强制 False） |
| `long_prefill_token_threshold` | `SchedulerConfig:70` | 单请求 prefill chunk 上限，0=不限 |
| `policy` | `SchedulerConfig:99` | `fcfs` / `priority` |
| `watermark` | `SchedulerConfig:136` | 准入保留空闲块比例，防频繁抢占 |
| `scheduler_reserve_full_isl` | `SchedulerConfig:130` | 整序列准入检查，默认 True |
| `async_scheduling` | `SchedulerConfig:148` | 调度-执行重叠 |
| `prefill_schedule_interval` | `SchedulerConfig:143` | DP 部署 prefill 节流周期 |
| `block_size` | `CacheConfig:61` | 物理块 token 数，默认 16；须与 attention backend 兼容 |
| `prefix_match_unit` | `CacheConfig:68` | 前缀哈希粒度（hash_block_size），可细于 block_size |
| `enable_prefix_caching` | `CacheConfig:107` | 默认 True |
| `prefix_caching_hash_algo` | `CacheConfig:109` | sha256 / xxhash（+`_cbor` 变体） |
| `gpu_memory_utilization` | `CacheConfig:80` | 默认 0.92；KV 显存 = 该比例 - 权重 - 激活 - cudagraph |
| `kv_cache_memory_bytes` | `CacheConfig:201` | 直接指定 KV 字节数，覆盖 gpu_memory_utilization |
| `cache_dtype`（CLI 名 `kv_cache_dtype`） | `CacheConfig:88` | auto/fp8/fp8_e5m2/int8_per_token_head/nvfp4/turboquant_* 等，量化省显存 |
| `kv_cache_dtype_skip_layers` | `CacheConfig:134` | 按层名/索引跳过 KV 量化 |
| `num_gpu_blocks_override` | `CacheConfig:101` | 覆盖 profiled 块数（测试抢占） |
| `mamba_block_size` / `mamba_cache_mode` / `mamba_cache_dtype` | `CacheConfig:145-157` | Mamba 状态块大小/缓存模式（all/align/none） |
| `kv_offloading_size` / `kv_offloading_backend` | `CacheConfig:210-219` | KV offload 容量(GiB) / native|lmcache |
| `disable_hybrid_kv_cache_manager` | `SchedulerConfig:122` | 混合模型按 full attention 统一分配 |
| `VLLM_USE_SIMPLE_KV_OFFLOAD` | `envs.py:2111` | native offload 走 simple connector |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | env | 是否把 cudagraph 显存计入 profile（默认开） |

**调优要点**：
- 吞吐：调大 `max_num_batched_tokens`（受显存/延迟权衡）；`max_num_seqs` 影响 batch 宽度。
- 延迟：小 `max_num_batched_tokens` 让 decode 更快混入；`watermark` 在显存紧张时减少抢占抖动。
- 长上下文/多轮对话：保持 `enable_prefix_caching=True` + 合理 `gpu_memory_utilization`（KV 容量 = 并发 × 序列长的乘积，见启动日志 "Maximum concurrency"）。
- 显存不足：`kv_cache_dtype=fp8`（约 2× KV 容量）、`kv_offloading_size` 把冷 KV 换出 CPU。
- 混合模型（SWA/mamba）：`disable_hybrid_kv_cache_manager` 影响分配方式；`prefix_match_unit` 影响前缀命中粒度。

## 8. 关键文件
<!-- tags: files -->

| 文件 | 内容 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | `Scheduler`：schedule()/update_from_output()、抢占、KV connector 集成（3007 行） |
| `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler`：调度-执行重叠 |
| `vllm/v1/core/sched/request_queue.py` | FCFS / Priority 请求队列 |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput` / `NewRequestData` / `CachedRequestData` |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`：allocate_slots/free/cache_blocks |
| `vllm/v1/core/kv_cache_coordinator.py` | `Unitary/Hybrid/NoPrefixCache` coordinator，多 group 命中协调 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 各注意力类型的 `SingleTypeKVCacheManager` 实现 + 内置注册 |
| `vllm/v1/core/block_pool.py` | `BlockPool`、`BlockHashToBlockMap`、LRU 驱逐、前缀缓存哈希表 |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`、`FreeKVCacheBlockQueue`、块哈希、`get_kv_cache_configs`、容量计算 |
| `vllm/v1/kv_cache_interface.py` | `KVCacheSpec` 及全部 spec 类、`KVCacheConfig`、`KVQuantMode` |
| `vllm/v1/kv_cache_spec_registry.py` | spec→manager 注册表与 `@register_kv_cache_spec` |
| `vllm/v1/kv_offload/` | 原生 offloading：base 抽象、cpu（LRU/ARC）、tiering（fs/obj/p2p） |
| `vllm/v1/simple_kv_offload/` | 简单 CPU/磁盘 offload connector（manager/worker/backends） |
| `vllm/config/scheduler.py` | `SchedulerConfig` 全部调度旋钮 |
| `vllm/config/cache.py` | `CacheConfig` 全部缓存旋钮 |
| `vllm/v1/worker/gpu_worker.py` | `determine_available_memory()`（:475）、`initialize_from_config()`（:665） |
| `vllm/v1/request.py` | `Request`、`RequestStatus`、`block_hashes` 增量计算 |
