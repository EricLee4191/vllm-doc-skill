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

- 知识库基于 vLLM main（`751f6807d9`，2026-09-19）/ 最新 tag v0.30.0rc2（`fa6ff06066`，2026-09-18，release candidate）源码整理。vLLM 迭代很快，回答"当前版本是否如此"类问题时以源码为准。
- 配置项名称以源码 `vllm/config/` 与 `vllm/envs.py` 为准，不要凭记忆猜测参数名。
- 给部署建议时，优先给保守可运行的基线配置，再给进阶优化项。

## 修复记录

- 2026-09-13：全量锚点重锚（re-anchor），260 个锚点修正（256 行号修正 + 4 路径修正），92 个锚点保持准确，9 个锚点为补充新增；基于 vLLM main `2f59050eda`（v0.29.0）源码逐一定位，15 锚点随机抽检 100% 命中，INDEX.md 已重新生成。
- 2026-09-19：增量更新，基线 `2f59050eda`（2026-09-12，v0.29.0）→ `751f6807d9`（2026-09-19，最新 tag v0.30.0rc2），区间 419 commits。全量锚点重锚：360 个唯一锚点（365 处引用）中 101 个稳定、254 个漂移（245 自动验证 + 9 人工验证）、5 个歧义名人工消解，共应用 262 处替换，新基线下 365 处引用全部解析通过。内容增量：水印 dual_key_gumbel 支持投机解码、调度器 max_num_active_seqs RUNNING 准入上限、OnlineAcceptanceEstimator 自适应验证、KV offload back-pressure/KVCR/max_load_tokens、DBO FULL CUDA graph、MoonEP backend、PCP+DCP sparse-MLA、COMPOSITE attention backend、Quark W4A16 / CPU FP8 W8A8、`/release_kv_cache_memory` 端点、`--enable-scale-out` flag 取代环境变量、结构化输出 `_get_constraint_start`/`validate_tokens` 重构、Engram DP 分片 + 异步预取。INDEX.md 已重新生成。
