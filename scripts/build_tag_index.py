#!/usr/bin/env python3
"""扫描 knowledge/*.md 的 `<!-- tags: ... -->` 行，生成倒排索引 knowledge/INDEX.md。

用法: python3 scripts/build_tag_index.py
输出: knowledge/INDEX.md（tag → 文件:行号 列表，按 tag 字母序）
"""
import re
from pathlib import Path
from collections import defaultdict

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
TAG_RE = re.compile(r"^<!--\s*tags:\s*(.*?)\s*-->")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)")


def main():
    # tag -> [(file, line_no, heading)]
    index = defaultdict(list)
    # 每个文件的章节结构，用于校验 tags 覆盖: [file, line_no, heading, has_tags]
    coverage = []

    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        if md.name == "INDEX.md":
            continue
        lines = md.read_text(encoding="utf-8").splitlines()
        current_heading = None
        in_fence = False
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = HEADING_RE.match(line)
            if m:
                current_heading = m.group(2).strip()
                # 只把 ##/### 视为章节（H1 是文档标题，不需要 tags）
                if m.group(1) in ("##", "###"):
                    coverage.append([md.name, i, current_heading, False])
                continue
            t = TAG_RE.match(line)
            if t and current_heading:
                # 标记最近一个未标记的章节为"有 tags"
                for c in reversed(coverage):
                    if c[0] == md.name and c[1] <= i and not c[3]:
                        c[3] = True
                        break
                # 兼容两种写法：逗号分隔（vllm/sglang 风格，tag 可含空格）
                # 与空格分隔（pytorch-skill 风格）
                raw = t.group(1)
                if re.search(r"[,，、]", raw):
                    tags = [x.strip().lower() for x in re.split(r"[,，、]", raw)]
                else:
                    tags = [x.lower() for x in raw.split()]
                # 去重保序，避免同一行内重复词产生重复条目
                for tag in dict.fromkeys(x for x in tags if x):
                    index[tag].append((md.name, i, current_heading))

    out = []
    out.append("# Tags 倒排索引")
    out.append("")
    out.append("> 自动生成，勿手改。重建：`python3 scripts/build_tag_index.py`")
    out.append("> 检索约定：先在本文件 grep 关键词定位 `文件:行号`，再 Read 对应区间。")
    out.append("")
    n_files = len({f for v in index.values() for f, _, _ in v})
    out.append(f"共 {len(index)} 个 tag，覆盖 {n_files} 份文档。")
    out.append("")

    for tag in sorted(index):
        entries = index[tag]
        locs = ", ".join(f"{f}:{ln}" for f, ln, _ in entries)
        out.append(f"- **{tag}** → {locs}")

    (KNOWLEDGE_DIR / "INDEX.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    # 覆盖率报告到 stderr
    missing = [c for c in coverage if not c[3]]
    print(f"INDEX.md 已生成: {len(index)} tags, {n_files} 份文档")
    if missing:
        print(f"警告: {len(missing)} 个章节缺 tags:")
        for f, ln, h, _ in missing:
            print(f"  {f}:{ln} {h}")
    else:
        print("tags 覆盖率: 100%（所有章节均有 tags）")


if __name__ == "__main__":
    main()
