# -*- coding: utf-8 -*-
"""单音节窗：CE 融合序里前缀简拼字是否压过整读字（用 cohort 现成 CE）。"""
import json
from collections import Counter, defaultdict

EVAL = r"D:\llm-rerank\llm-rerank-research\eval"
READS = EVAL + r"\char_readings.tsv"

# 音节 → 整读字集合（覆盖范围 = char_readings 的 9457 行）
exact = defaultdict(set)
syll = set()
for line in open(READS, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        exact[p[1]].add(p[0])
        syll.add(p[1])

W = (1.0, 1.10, 0.60)


def ce_sum(ce3):
    s = 0.0
    for i, v in enumerate(ce3[:3]):
        if v is None or v < 0:
            break
        s += W[i] * v
    return s


rows = []
for line in open(EVAL + r"\luna_mixed\cohort.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["code"] not in syll:
        continue
    seen = {}
    for c in r["eng"]:
        text, s, e, typ, q, ce3, nt = c
        if len(text) != 1 or s != 0 or e != len(r["code"]):
            continue
        if ce3 and ce3[0] is not None and ce3[0] >= 0:
            seen[text] = ce_sum(ce3)
    if len(seen) < 2:
        continue
    ranked = sorted(seen.items(), key=lambda x: x[1])   # 成本升序 = 融合分降序
    isex = {t: (t in exact[r["code"]]) for t in seen}
    rows.append((r["code"], ranked, isex))

print("单音节窗（≥2 个满覆盖单字有 CE）:", len(rows))
hijack = [x for x in rows if not x[2][x[1][0][0]] and any(x[2][t] for t, _ in x[1])]
print("CE 序首位 = 前缀简拼字（窗内存在整读字）:", len(hijack),
      f"{100*len(hijack)/max(1,len(rows)):.1f}%")
n_exact_missing = sum(1 for x in rows if not any(x[2].values()))
print("窗内没有任何整读字（纯前缀窗，分层无作用）:", n_exact_missing)
c = Counter(x[0] for x in hijack)
print("被劫持的音节 top:", c.most_common(12))
for code, ranked, isex in hijack[:10]:
    top = ", ".join(f"{'[前]' if not isex[t] else ''}{t}({v:.1f})"
                    for t, v in ranked[:5])
    print(f"  {code}: {top}")
