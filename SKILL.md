---
name: vllm-skill
description: vLLM 架构知识库与部署优化助手。回答 vLLM 内部架构问题（引擎、调度、KV cache、attention、分布式、量化、投机解码），并给出部署与性能调优建议。当用户提到 vLLM、LLM 推理部署、推理服务调优、显存/OOM 排查、TP/PP/DP 并行选择时使用。
---

# vLLM 架构与部署优化

本 skill 提供两部分能力：

1. **架构信息**：vLLM（main / v0.30.x，v1 架构）的内部机制——引擎、调度、KV cache、模型执行、attention 后端、分布式并行、量化、投机解码、API 服务。
2. **部署优化**：给定模型规模、GPU 数量、目标（吞吐/延迟），给出具体的配置建议与调优排查路径。

## 知识库

结构化文档位于本 skill 的 `knowledge/` 目录：

| 文档 | 内容 |
|------|------|
| [00-overview.md](knowledge/00-overview.md) | 总览与架构地图（**先读这个**） |
| [01-architecture-core.md](knowledge/01-architecture-core.md) | 核心架构与请求生命周期 |
| [02-scheduling-kv-cache.md](knowledge/02-scheduling-kv-cache.md) | 调度器与 KV Cache 管理 |
| [03-model-execution.md](knowledge/03-model-execution.md) | 模型执行与编译优化 |
| [04-attention-kernels.md](knowledge/04-attention-kernels.md) | Attention 后端与底层算子 |
| [05-distributed.md](knowledge/05-distributed.md) | 分布式并行（TP/PP/DP/EP）与 KV 传输 |
| [06-quantization-hardware.md](knowledge/06-quantization-hardware.md) | 量化与多硬件平台 |
| [07-advanced-features.md](knowledge/07-advanced-features.md) | 投机解码与高级推理特性 |
| [08-deployment-optimization.md](knowledge/08-deployment-optimization.md) | 部署、API 服务与性能调优 |

## 使用方式

**检索约定**：每份文档的 `##` 级章节标题下都有一行 `<!-- tags: ... -->` 注释（英文关键词 + 中文别名）。定位章节时先 grep tags 再读对应区间，不要整篇读：

```bash
# 例：找"抢占"相关章节
grep -n "tags:.*preempt" knowledge/*.md
# 例：找 KV cache 显存容量
grep -n "tags:.*kv-cache.*capacity\|tags:.*显存" knowledge/*.md
# 命中后按行号 Read 对应区间
```

**回答架构问题**：
1. 先读 `knowledge/00-overview.md` 定位相关子系统。
2. 用 tags grep 定位到具体章节，Read 对应区间获取细节。
3. 若文档不够（如用户问某个具体类/函数/环境变量的最新行为），直接到源码仓库查证（默认路径 `/Users/baofeng/baofeng/github/vllm`，以用户实际 checkout 为准）。
4. 回答时引用具体文件路径与配置项名称，给出可验证的依据。

**部署/调优咨询**：
1. 先收集关键信息：模型（名称/参数量/是否 MoE/量化格式）、GPU（型号/数量/显存）、目标（吞吐优先 or 延迟优先）、当前配置（如有）、症状（OOM/慢/报错，如有）。
2. 读 `knowledge/08-deployment-optimization.md` 的调优指南与决策清单。
3. 涉及并行选择读 `05-distributed.md`，涉及显存读 `02-scheduling-kv-cache.md` 与 `06-quantization-hardware.md`，涉及 attention 后端/编译读 `04`/`03`。
4. 给出**具体可执行的配置**（CLI 参数或环境变量），并说明每个参数的作用与预期效果；有取舍时说明权衡。
5. 排查问题时给出分步诊断路径（先看什么日志/指标，再改什么参数）。

## 注意事项

- 知识库基于 vLLM main（`9f07d023d0`，2026-09-23）/ 最新 tag v0.30.1rc0（`153242a314`，2026-09-23，release candidate；上一正式 release 为 v0.30.0，`9ed533eb4a`，2026-09-20）源码整理。vLLM 迭代很快，回答"当前版本是否如此"类问题时以源码为准。
- 配置项名称以源码 `vllm/config/` 与 `vllm/envs.py` 为准，不要凭记忆猜测参数名。
- 给部署建议时，优先给保守可运行的基线配置，再给进阶优化项。

## 修复记录

- 2026-09-22：增量更新，基线 `8b98b7d0b4`（2026-09-21，v0.30.0rc2）→ `d90f0eade5`（2026-09-22，最新 tag **v0.30.0** 正式 release，`9ed533eb4a`，2026-09-20；main 领先该 release 分支 56+ commits），区间 56 commits。内容增量：Initialized engine snapshots（#51360，`vllm/snapshot/` 包 + `vllm snapshot create/restore` CLI，CRIU+CUDA checkpoint 捕获已初始化引擎，限 Linux x86-64/单 GPU/TP1，08 篇 §1.1 新增说明）、KV hints 请求信封（#53423，`vllm/v1/kv_hints/`，orchestrator 可编程 KV 管理提示贯通 InputProcessor→Request→KV offload tiering，02 篇 §3.3）、调度器 `long_prefill_token_threshold` 软化（#57951，batch 唯一请求时不截断 prefill chunk，02 篇 §2.3）+ `ParallelConfig.nnodes_within_dp` external LB 修复（#53743，05 篇 §2.3）、投机解码 DFlash async scheduling（#58065）+ draft 配置覆盖统一 `apply_draft_overrides` + draft 加载统一 `get_draft_load_config`（Fast Start 下 MTP draft 缓存到 daemon 独立 draft group，#57312，03 篇 §2.5/07 篇 §1.4）+ MTP draft KV group 位置标注通用化（#55390，02 篇 §5）、GLM5Next NoPE sparse-MLA head_size 512 接入 FA/FlashMLA（#55385，04 篇 §4.2）+ sparse MLA 准备开销削减（#57458）+ ROCm DSV4 自适应验证 flattened query lens（#52362）+ GDN stateless first-chunk 分类修复（#51565）、多模态 processor/receiver cache 重构为 `multimodal/cache/` 包 + `supports_multimodal_inputs` 移到 `ModelConfig` 缓存属性（#57913/#57967，07 篇 §5.1）、MXFP4 emulation 加载期反量化（#50814，`VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD`）、`VLLM_KIMI_K3_GEMM_RS` 更名 `VLLM_ENABLE_GEMM_RS` 并扩到 DSV4.1 `wo_b`（#57428，`kernels/linear/cute_dsl/gemm_rs_ar.py` +1177 行，03 篇 §3.2/06 篇）、`VLLM_PLE_CPU_OFFLOAD` 移除（#57937，07 篇 §7.2）、CPU `--device-memory-utilization` CLI 别名（#56547）、NIXL DCP 跨 MLA cache region pull 修复（#57389，05 篇 §6.3）、sleep mode level-2 保留冻结权重（#57891，`--sleep-preserve-parameter-names`，08 篇 §1.1）、安全校验三连（#57731 min_tokens≤填充后 max_tokens、#47450 拒绝空 structural_tag、#57006 prompt embeds is_token_ids 长度校验，01 篇 §6/07 篇 §2.2/§3.2）+ grammar poll 非阻塞（#55931，07 篇 §3.2）。锚点：本区间 diff 涉及 scheduler.py/kv_cache_utils.py/gpu_worker.py/worker_base.py/config/*.py 等被引用文件，行号锚点按符号 grep 抽查（scheduler.py 602-609/662-663/1102-1103、kv_cache_utils.py 2126/2145/2158、gpu_worker.py 245 起、config/scheduler.py 80、config/parallel.py 760、config/speculative.py 377/1787、model_loader/utils.py 46、sampling_params.py 1144、input_processor.py 319/397/540）全部命中；零 diff 文件锚点保持原值。INDEX.md 已重新生成。
- 2026-09-24：增量更新，基线 `d90f0eade5`（2026-09-22，v0.30.0）→ `9f07d023d0`（2026-09-23，最新 tag **v0.30.1rc0**，`153242a314`，2026-09-23 release candidate），区间 69 commits。内容增量：DSpark 支持 PP（#56956，`pp_utils.py` PPHandler.set_disabled + broadcast_drafts 折叠，05 篇 §2.2）、Kimi-K3 变长 decode（#52988，MLA/KDA metadata builder max_query_len，04 篇 §4.2）、异构 vocab spec decode 去 CPU-GPU 同步（#57396）、GLM MTP head 延迟加载（#55442，07 篇 §1.3）、MRV1+PP>1+async sched+structured output 禁止（#56250，08 篇 §4.6）、DiffusionGemma 结构化生成（#57250，DiffusionAsyncScheduler + validate_diffusion_sampling_params，01 篇 §6/07 篇 §3/08 篇 §4.6）、Granite 流式 tool-call 解析（#49648，`parser/granite.py`，07 篇 §1.6）、length finish_reason 流式 tool call 修复（#46303）、FIM completion 渲染（#44229，07 篇 §1.6）、MoE gate 统一 GateLinear（#58234，33 模型文件，03 篇 §3）、TritonExperts EP 丢弃远端 top-k slot（#58051，03 篇 §3）、Humming wNaM 非对称量化（#46528，zero_point，06 篇）、per-token NVFP4 MoE（#57176，06 篇）、FP8/MLA 权重变换重构（#57732，split_kv_b_proj 纯函数，06 篇）、DeepGEMM arch capability 优先（#58073，06 篇）、AllSpark INT8 W8A16 移除（#58001）、enforce_eager 禁用 JIT warmup（#58197/#55146，03 篇 §6.2/08 篇 §4.5）、set_torch_threads_for_runtime 移到 load_model 末尾（#55891，03 篇 §6.2）、Fast Start DP weight cache daemon（#57386，03 篇 §7）、with_hf_config 子模型视图跳过 __post_init__（#58212，01 篇 §6）、ModelState max_model_len 从 model config 读取（#58149）、AuxOutput KV connector 限制收窄为 5 个具名 PD connector（#58150，00/05/08 篇）、batch_invariant NCCL>=2.31 ring,tree;allreduce:tree（#58179，06 篇）、XPU batch-invariant（#55881，06 篇）、ROCm MRV2 sampler JIT warmup（#58092）、ROCm BF16 AsyncTP 融合（#58098）、ROCm AITER static FP8 attention output 融合（#58099）、RDNA3/4 narrow KV tile（#58225）、MiniMax MXFP8 zero blocks 修复（#58089）、DSV4/DSV4.1 inverse RoPE 融合（#57451/#57435）、GLM-5.2-MXFP4 ROCm（#51915）、GLM-5.3-Flash dense MLP sequence-parallel shard（#58061）、SM120 NoPE sparse MLA 修复（#55277，04 篇 §4.2）、encoder-only prefix caching 自动禁用（#58287，02 篇 §6）、Engram /dev/shm 回退（#57914，07 篇 §7.2）、Triton softcap NaN 修复（#56579）、EPD metadata-only audio（#57887）、dead code 清理（#58002，-585 行）。锚点重锚：13 处行号漂移修正（config/vllm.py 1213→1219/2237→2276、gpu_worker.py 532→566/778→807/1267→1323/1535→1564/1544→1573、mla_attention.py 1566→1569、sampling_params.py 1307→1319、input_processor.py 40→41、nvidia/model.py 388-395→413-420/559→586），零 diff 文件锚点保持原值。INDEX.md 已重新生成。
- 2026-09-13：全量锚点重锚（re-anchor），260 个锚点修正（256 行号修正 + 4 路径修正），92 个锚点保持准确，9 个锚点为补充新增；基于 vLLM main `2f59050eda`（v0.29.0）源码逐一定位，15 锚点随机抽检 100% 命中，INDEX.md 已重新生成。
- 2026-09-19：增量更新，基线 `2f59050eda`（2026-09-12，v0.29.0）→ `751f6807d9`（2026-09-19，最新 tag v0.30.0rc2），区间 419 commits。全量锚点重锚：360 个唯一锚点（365 处引用）中 101 个稳定、254 个漂移（245 自动验证 + 9 人工验证）、5 个歧义名人工消解，共应用 262 处替换，新基线下 365 处引用全部解析通过。内容增量：水印 dual_key_gumbel 支持投机解码、调度器 max_num_active_seqs RUNNING 准入上限、OnlineAcceptanceEstimator 自适应验证、KV offload back-pressure/KVCR/max_load_tokens、DBO FULL CUDA graph、MoonEP backend、PCP+DCP sparse-MLA、COMPOSITE attention backend、Quark W4A16 / CPU FP8 W8A8、`/release_kv_cache_memory` 端点、`--enable-scale-out` flag 取代环境变量、结构化输出 `_get_constraint_start`/`validate_tokens` 重构、Engram DP 分片 + 异步预取。INDEX.md 已重新生成。
- 2026-09-20：增量更新，基线 `751f6807d9`（2026-09-19，v0.30.0rc2）→ `4868312128`（2026-09-20，最新 tag 仍 v0.30.0rc2），区间 30 commits。锚点重锚：395 处引用 / 139 个被引用文件中，12 个文件在本区间有 diff，其中 13 处行号漂移已按 hunk 位移重锚（model_runner.py 1750→1756、185→186；input_processor.py 38→39；envs.py 1800→1801×3、2157→2158、1643→1644；_custom_ops.py 2821/2823/2876→2823/2836/2889；sampling_params.py 1305→1297、1199-1244→1191-1236），其余 127 个零 diff 文件锚点保持原值，新基线下全部解析通过。内容增量：Humming 特性整合（#56685，`utils/humming/` 包 + mxfp6 kernel + fallback 收紧 + Marlin/Humming 共享持久 workspace #57421）、Model Runner V2 支持自定义 logits processors（#56497）、`--enable-mamba-fine-grained-prefix-cache` 更名 `--enable-mamba-shared-prefix-checkpoint`（#57382）、generate API 暴露 per-request 投机解码指标（#43310）、EPD 动态注册（#54176）、DeepSeek-V4.1-flash encoder CUDA graph（#56625）、MiMo V2 bf16 MoE router + mxfp4 MoE（#57784）、GLM-5.3-Flash kpool/sparse-indexer 系列（#57546/#57534/#57477/#57701/#56810）、SM100 fp8_ds_mla cache scales 修复（#49435）、dead kernel code 清理（#57621）。INDEX.md 已重新生成。
- 2026-09-21：增量更新，基线 `4868312128`（2026-09-20，v0.30.0rc2）→ `86ce4d10e2`（2026-09-21，最新 tag 仍 v0.30.0rc2），区间 11 commits。内容增量：Profiler 统一为平台感知（#57460，torch profiling 从各 worker 收敛到 `vllm/profiler/wrapper.py` 工厂 `create_worker_profiler`，`ProfilerConfig` 新增 `torch_profiler_activities`，08 篇新增 §4.9 Profiling 小节）、sleep 时 KV connector cache reset 失败上抛（#54581，`core.py:874`）、MoRIIO KV connector READ 模式传输 hybrid mamba/KDA recurrent state（#51052）、spec decode dummy draft 步 stale block-table 修复（#56734）、DSV4.1 mHC 小 TP batch 系数 overlap（#57603）、ROCm Engram 表留 host 内存（#57491）、HY4 full CUDA graph indexer completion event（#57811）、XPU communicator world_size 修复（#57779）。锚点抽查（core.py:109/874、wrapper.py:60/675、config/profiler.py:38/55、gpu_worker.py:1281、arg_utils.py:1752）全部命中。INDEX.md 已重新生成。
- 2026-09-21（第二次）：增量更新，基线 `86ce4d10e2`（2026-09-21，v0.30.0rc2）→ `8b98b7d0b4`（2026-09-21，最新 tag 仍 v0.30.0rc2；main 已领先 v0.30.0 release 分支 310+ commits），区间 27 commits。内容增量：AuxOutput Connector block 键控 routed-expert 输出存储（#45635，`vllm/distributed/aux_output_connector/` + `config/aux_output.py`，mmap arena LRU + fail-closed，05 篇新增 §9.4）、DSV4.1 mHC 三连（#57643 TP all-reduce 与 mHC 输入准备融合 MNNVL Lamport kernel、#57874 overlap 收紧到 full CUDA graph、#57906 ROCm 禁用 SWA bounded replay）、ROCm/XPU 平台（#57526 Hy4 ROCm 路径 +718 行、#54535 MiniMax-M3 packed LBHNC AITER QK-norm 融合、#57277 XPU fused top-k/top-p sampler kernel `VLLM_XPU_USE_SAMPLER_KERNEL`、#57855 moe_align 7-arg 回退、#57554 DeepGEMM CUDA 12.9 构建修复）、Mamba/KDA prefill checkpoint 通用化（#57783）、多模态 receiver cache 安全语义（#57833，fresh payload 优先于 stale cache）+ Molmo2/Whisper/Mistral3/DiffusionGemma 预处理修复、xgrammar list-valued type 归一化（#48416）、Engram DP shared memory 默认开启（#57651）、derender 流式解析文档化（#57922）、sparse attention metadata 去冗余（#57885）。锚点重锚：12 个被引用文件在本区间有 diff，全部按符号 grep 验证重锚（scheduler.py 570→555/1922→1900/2526→2418/2446、gpu_model_runner.py 496→479/4197→4151/4578→4532 等、arg_utils.py 446→447/2055→2066/2717→2729 等、config/vllm.py 354→355/1162→1213 等），零 diff 文件锚点保持原值。INDEX.md 已重新生成（642 tags）。
