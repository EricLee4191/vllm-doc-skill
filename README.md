# vllm-skill

> vLLM 架构知识库 + 部署优化 Claude Skill。回答 vLLM 内部架构问题，并给出可落地的部署与性能调优建议。

本仓库是一个 [Claude Skill](https://docs.claude.com/en/docs/claude-code/skills)，把 **vLLM（v1 架构，main / v0.28.x）** 的内部机制整理成 9 份结构化中文文档，并配了一套基于 `tags` 倒排索引的检索约定，让 Claude 能**精准定位章节**回答架构问题、给出**具体可执行的部署配置**，而不是泛泛而谈。

---

## 它能做什么

1. **架构问答** — vLLM 的内部机制：引擎进程模型、调度器与 KV Cache、模型执行与编译、Attention 后端、分布式并行（TP/PP/DP/EP）、量化与多硬件、投机解码、API 服务。回答时引用具体文件路径与配置项，给出可验证的依据。
2. **部署调优** — 给定模型规模、GPU 数量、目标（吞吐 / 延迟），给出**具体的 CLI 参数 / 环境变量**配置，说明每个参数的作用与权衡，并提供分步排查路径（OOM、慢、报错）。

> 知识库基于 vLLM `main`（`f32b17b6d6`，2026-08-21）/ 最新 tag **v0.28.0rc1** 源码整理。vLLM 迭代很快，涉及"当前版本是否如此"的问题以源码为准。

## 目录结构

```
vllm-skill/
├── SKILL.md                     # Skill 入口：能力说明 + 使用流程 + 检索约定
├── README.md                    # 本文件
├── knowledge/                   # 9 份中文知识库文档（核心）
│   ├── 00-overview.md           #   总览与架构地图（入口，先读这个）
│   ├── 01-architecture-core.md  #   核心架构与请求生命周期
│   ├── 02-scheduling-kv-cache.md#   调度器与 KV Cache 管理
│   ├── 03-model-execution.md    #   模型执行与编译优化
│   ├── 04-attention-kernels.md  #   Attention 后端与底层算子
│   ├── 05-distributed.md        #   分布式并行（TP/PP/DP/EP）与 KV 传输
│   ├── 06-quantization-hardware.md#  量化与多硬件平台
│   ├── 07-advanced-features.md  #   投机解码与高级推理特性
│   ├── 08-deployment-optimization.md# 部署、API 服务与性能调优（最实用）
│   └── INDEX.md                 #   tags 倒排索引（自动生成，600 tag）
└── scripts/
    └── build_tag_index.py       # 重建 INDEX.md 的脚本
```

## 安装

本 skill 依赖 [Claude Code](https://claude.com/claude-code)。把仓库放到 skills 目录即可（软链或拷贝均可）：

```bash
# 1. 克隆
git clone https://github.com/EricLee4191/vllm-doc-skill.git

# 2. 软链到 Claude Code 的 skills 目录（推荐，更新方便）
ln -s "$(pwd)/vllm-doc-skill" ~/.claude/skills/vllm-skill

# 3. 验证：新开一个 Claude Code 会话，输入 / 应能看到 vllm-skill
```

> 若你的 Claude Code skills 目录不是 `~/.claude/skills`，按实际路径调整。

## 使用

**方式一：显式调用** — 在 Claude Code 会话里输入：

```
/vllm-skill  我有 1 张 A100 80G，要部署 Qwen2.5-7B 追求最大吞吐，怎么配？
```

**方式二：自动触发** — 当你的问题涉及 vLLM、LLM 推理部署、推理服务调优、显存 / OOM 排查、TP/PP/DP 并行选择时，skill 会自动被唤起，无需手动 `/`。

### 检索约定（skill 内部如何工作）

每份文档的 `##` 章节标题下都有一行 `<!-- tags: ... -->` 注释（英文关键词 + 中文别名）。Claude 定位章节时**先 grep tags、再按行号读对应区间**，不整篇读，既快又准：

```bash
# 例：找"抢占"相关章节
grep -n "tags:.*preempt" knowledge/*.md
# 例：找 KV cache 显存容量
grep -n "tags:.*kv-cache.*capacity\|tags:.*显存" knowledge/*.md
# 命中后按行号 Read 对应区间
```

`knowledge/INDEX.md` 是这些 tags 的倒排索引（600 个 tag → `文件:行号`），可直接 grep 定位。若文档改动后索引过期，重建：

```bash
python3 scripts/build_tag_index.py
```

## 具体示例

**场景**：单卡 A100 80G 部署 Qwen2.5-7B，目标是**最大吞吐**。

**提问**：

> /vllm-skill 我有 1 张 A100 80G，要部署 Qwen2.5-7B 追求最大吞吐，怎么配？

**skill 的处理流程**：

1. 读 `00-overview.md` 定位到"部署优化"子系统 → `08-deployment-optimization.md`。
2. 按决策树判断：7B bf16 权重约 14GB，单卡 80G 放得下 → **TP=1**，无需多卡切分。
3. 命中"目标=吞吐"分支，给出基线 + 进阶配置：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.93 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --performance-mode throughput
```

**skill 的回答要点**（节选）：

- `--gpu-memory-utilization 0.93`：把显存尽量留给 KV cache，吞吐场景可拉到 0.92~0.95。
- `--max-num-batched-tokens 16384`：小模型大卡可拉高，让每步 batch 更大。
- `--max-num-seqs 256`：拉高直到 KV cache 打满，以启动日志的 `Maximum concurrency` 为准。
- `--performance-mode throughput`：内部把相关默认值翻倍，偏吞吐。
- 上线前看启动日志确认 `GPU KV cache size` 与 `Maximum concurrency` 满足业务并发、无 preemption 警告；再用 `vllm bench serve` 压出 P99 TTFT/TPOT 基线。

> 完整决策树与上线检查清单见 `knowledge/08-deployment-optimization.md` 第 7 节。

## 注意事项

- 配置项名称以 vLLM 源码 `vllm/config/` 与 `vllm/envs.py` 为准，不要凭记忆猜测参数名。
- 给部署建议时优先给**保守可运行的基线配置**，再给进阶优化项。
- 文档不够时（如某个具体类 / 函数 / 环境变量的最新行为），直接到 vLLM 源码仓库查证。

## License

内部知识库，未开源授权。
