# -*- coding: utf-8 -*-
"""从 wiki 分条语料抽取分条首句 → samples.txt (每行一句)."""
import io, sys, re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
SRC = r"D:\llm-rerank\llm-rerank-research\article_corpus_shuffled.txt"
OUT = r"D:\llm-rerank\probe-scratch\ce_invest\samples.txt"
N = 800          # 样本句数
MIN_CHARS = 16   # 最短字符数(保证 8 token 前缀后有续 token)

seen = set()
out = []
with open(SRC, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if len(out) >= N:
            break
        if len(line) < MIN_CHARS + 8:
            continue
        # 分条首句 = 第一个句末标点之前的部分
        m = re.match(r"^(.+?)[。！？]", line)
        if not m:
            continue
        s = m.group(1).strip()
        # 过滤: 过短 / 含大量符号 / 重复
        if len(s) < MIN_CHARS or len(s) > 60:
            continue
        if sum(1 for ch in s if not ("\u4e00" <= ch <= "\u9fff")) > len(s) * 0.3:
            continue
        key = s[:20]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

with open(OUT, "w", encoding="utf-8") as f:
    for s in out:
        f.write(s + "\n")
print("样本数:", len(out))
print("示例:", out[:3])
