# -*- coding: utf-8 -*-
"""单音节窗（cohort 现成 CE）——整读 vs 前缀简拼的分层诊断。
整读表取自**引擎词库** wanxiang.dict.yaml 的单字行（与引擎码空间一致）。"""
import json
from collections import Counter, defaultdict

EVAL = r"D:\llm-rerank\llm-rerank-research\eval"
DICT = r"C:\Users\Administrator\AppData\Roaming\Rime\wanxiang.dict.yaml"
exact = defaultdict(set)
syll = set()
with open(DICT, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or line[0] in "#-." or "\t" not in line:
            continue
        p = line.split("\t")
        if len(p) < 3 or len(p[0]) != 1:
            continue
        exact[p[1]].add(p[0])
        syll.add(p[1])
print("音节数:", len(syll), "单字-读音行:", sum(len(v) for v in exact.values()))

rows = []
for line in open(EVAL + r"\luna_mixed\cohort.jsonl", encoding="utf-8"):
    r = json.loads(line)
    code = r["code"]
    if code not in syll:
        continue
    seen = {}
    for c in r["eng"]:
        text, s, e, typ, q, ce3, nt = c
        if len(text) != 1 or s != 0 or e != len(code):
            continue
        if ce3 and ce3[0] is not None and ce3[0] >= 0:
            seen[text] = ce3[0]
    if len(seen) < 2:
        continue
    ranked = sorted(seen.items(), key=lambda x: x[1])
    isex = {t: (t in exact[code]) for t in seen}
    rows.append((code, ranked, isex))

print("单音节窗（≥2 个满覆盖单字有 CE）:", len(rows))
no_ex = [x for x in rows if not any(x[2].values())]
print("窗内无整读字:", len(no_ex))
hij = [x for x in rows if not x[2][x[1][0][0]] and any(x[2].values())]
print("CE 序首位 = 非整读字（窗内有整读字）:", len(hij),
      f"{100*len(hij)/max(1,len(rows)):.1f}%")


def contam(k):
    n = 0
    for code, ranked, isex in rows:
        top = [t for t, _ in ranked[:k]]
        if any(not isex[t] for t in top) and any(isex[t] for t in top):
            n += 1
    return n


for k in (1, 3, 5):
    c = contam(k)
    print(f"前 {k} 位混入非整读字: {c} ({100*c/max(1,len(rows)):.1f}%)")
print("被劫持音节:", Counter(x[0] for x in hij).most_common(10))
for code, ranked, isex in hij[:8]:
    print(" ", code, ", ".join(f"{'[前]' if not isex[t] else ''}{t}({v:.1f})"
                               for t, v in ranked[:5]))
