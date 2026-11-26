# -*- coding: utf-8 -*-
"""前缀简拼 vs 完整音节：按万象词库权重估算引擎序/频率序谁能排第一。
只读分析脚本（一次性），不改生产。"""
import math, sys
from collections import defaultdict

DICT = r"C:\Users\Administrator\AppData\Roaming\Rime\wanxiang.dict.yaml"

W = defaultdict(lambda: defaultdict(float))   # char -> toneless reading -> weight
syll = set()
n = 0
with open(DICT, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or line[0] in "#-." or "\t" not in line:
            continue
        p = line.split("\t")
        if len(p) < 3:
            continue
        t, code, w = p[0], p[1], p[2]
        try:
            w = float(w)
        except ValueError:
            continue
        if len(t) != 1:
            continue
        n += 1
        W[t][code] += w
        syll.add(code)
print("single-char entries:", n, " syllables:", len(syll), " chars:", len(W))

# ① 严格前缀关系
ext = {}
for S in syll:
    ex = [R for R in syll if R != S and R.startswith(S)]
    if ex:
        ext[S] = sorted(ex)
print("syllables that are a strict prefix of another:", len(ext))

# ② 每个音节：exact 最优 vs 前缀简拼最优（引擎序含 -3 惩罚 / 纯频率）
PEN = math.log(0.05)   # librime kCompletionPenalty
rows = []
for S in syll:
    exact = []
    comp = []
    for ch, rd in W.items():
        for R, w in rd.items():
            if R == S:
                exact.append((w, ch))
            elif R.startswith(S):
                comp.append((w, ch))
    if not exact or not comp:
        continue
    exact.sort(reverse=True)
    comp.sort(reverse=True)
    ew = math.log(exact[0][0]) if exact[0][0] > 0 else -99
    cw = math.log(comp[0][0]) if comp[0][0] > 0 else -99
    # 合并后位次（引擎口径：完成项 -3）
    merged = [(ew, 0, exact[0][1])]
    for w, ch in comp[:6]:
        merged.append((math.log(w) + PEN, 1, ch))
    for w, ch in exact[:6]:
        merged.append((math.log(w), 0, ch))
    merged.sort(key=lambda x: -x[0])
    # 去重保首现
    seen = set(); order = []
    for sc, isc, ch in merged:
        if ch in seen: continue
        seen.add(ch); order.append((isc, ch, sc))
    rows.append((S, order, ew, cw))

bad = [r for r in rows if r[1] and r[1][0][0] == 1]
print("syllables with both exact and prefix candidates:", len(rows))
print("engine order (with log0.05 penalty): prefix-only candidate is top-1:",
      len(bad), f"({len(bad)/max(1,len(rows))*100:.1f}%)")
# 纯频率（无惩罚）
pure = 0
for S, order, ew, cw in rows:
    if cw > ew:
        pure += 1
print("pure frequency (no penalty): prefix-only top-1:", pure,
      f"({pure/max(1,len(rows))*100:.1f}%)")
# 前缀项进前 3
in3 = 0
for S, order, ew, cw in rows:
    if any(isc == 1 for isc, ch, sc in order[:3]):
        in3 += 1
print("prefix-only candidate inside top-3:", in3,
      f"({in3/max(1,len(rows))*100:.1f}%)")

# ③ 例子
for S in ["shen", "shi", "zhu", "jing", "xin", "li", "ji", "yi", "yu", "wei", "zhi", "an"]:
    r = next((x for x in rows if x[0] == S), None)
    if not r:
        continue
    print(f"--- {S}: " + ", ".join(
        f"{'[前缀]' if isc else ''}{ch}({sc:.1f})" for isc, ch, sc in r[1][:5]))

# ④ 同一字两个读音互为前缀（前缀剥离的误命中类）
coll = [(ch, a, b) for ch, rd in W.items() for a in rd for b in rd
        if a != b and b.startswith(a)]
print("chars with readings where one is a strict prefix of the other:",
      len(coll), "pairs:", len(coll))
for x in coll[:12]:
    print("   ", x[0], x[1], "<", x[2])
