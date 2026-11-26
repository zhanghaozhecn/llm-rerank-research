# -*- coding: utf-8 -*-
"""池 64 复核：整读分层劫持率 + 前缀确认 gate 取法（结构口径）。

① 单音节窗：融合序首位 / 前 3 / 前 5 是否被「前缀简拼字」占据
   （整读表 = 引擎词库单字行，与生产 syll_chars.tsv 同源）。
② gate 的「同字数档首选」vs「同 span 首选」：现状能确认 / 改 span 才解锁 /
   两种都不是 lead。窗口按 CE 预算 12 截断（与生产同构）。
用法：python -I -X utf8 eval/calib/scratch/tier_gate_pool64.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL = Path(r"D:\llm-rerank\llm-rerank-research\eval")
DICT = Path(r"C:\Users\Administrator\AppData\Roaming\Rime\wanxiang.dict.yaml")
sys.stdout.reconfigure(encoding="utf-8")

# 整读表（引擎词库单字行）
exact = defaultdict(set)
with open(DICT, encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and len(p[0]) == 1:
            exact[p[1]].add(p[0])
syll = set(exact)
print(f"整读表：{len(syll)} 音节 / {sum(len(v) for v in exact.values())} 字-音节")

cands = json.load(open(EVAL / "luna_mixed" / "cands64.json", encoding="utf-8"))
cohort = [json.loads(l) for l in
          open(EVAL / "luna_mixed" / "cohort64.jsonl", encoding="utf-8")]
print(f"cohort64：{len(cohort)} 窗")


def budget_window(rows, budget=12):
    """引擎序（= rows 顺序）预算准入；返回进窗的 rows。"""
    rem = budget
    out = []
    for r in rows:
        nt = r[6] or 0
        cost = 0 if nt <= 1 else min(nt, 3) - 1
        if nt == 0:
            cost = 1
        if nt <= 1 or cost <= rem:
            out.append(r)
            if nt > 1:
                rem -= cost
    return out


# ── ① 整读分层劫持率 ────────────────────────────────────────────────
rows_all = []
for w in cohort:
    code = w["code"]
    if code not in syll:
        continue
    seen = {}
    for r in w["eng"]:
        t, s, e = r[0], r[1], r[2]
        if len(t) != 1 or s != 0 or e != len(code):
            continue
        if r[5] and r[5][0] is not None and r[5][0] >= 0:
            seen[t] = r[5][0]
    if len(seen) < 2:
        continue
    ranked = sorted(seen.items(), key=lambda x: x[1])
    rows_all.append((code, ranked))

print(f"\n① 单音节窗（≥2 个满覆盖单字有 CE）：{len(rows_all)}")
hij = [x for x in rows_all if x[1][0][0] not in exact[x[0]]]
print(f"  融合序首位 = 前缀简拼字：{len(hij)} "
      f"（{100*len(hij)/max(1,len(rows_all)):.1f}%）")
for k in (3, 5):
    n = 0
    for code, rk in rows_all:
        top = [t for t, _ in rk[:k]]
        if (any(t not in exact[code] for t in top)
                and any(t in exact[code] for t in top)):
            n += 1
    print(f"  前 {k} 位混入前缀简拼字：{n}（{100*n/max(1,len(rows_all)):.1f}%）")
print("  被劫持音节:", Counter(x[0] for x in hij).most_common(8))
for code, rk in hij[:6]:
    print("   ", code, ", ".join(f"{'[前]' if t not in exact[code] else ''}{t}"
                                 f"({v:.1f})" for t, v in rk[:5]))

# ── ② gate 取法 ────────────────────────────────────────────────────
tot = srok = cur_ok = unblock = neither = miss = 0
tail_rival = Counter()
ex = []
for w in cohort:
    if not w["gen"]:
        continue
    g = w["gen"][0]
    text, lead, le = g[0], g[2], g[3]
    if not lead or le is None:
        continue
    tot += 1
    if len(text) - len(lead) < 2:
        continue
    srok += 1
    win = budget_window(w["eng"][:64])
    lst = [(r[5][0], r[1], r[2], r[0]) for r in win
           if r[5] and r[5][0] is not None and r[5][0] >= 0]
    if not lst:
        continue
    lst.sort(key=lambda x: x[0])
    cur = next((x for x in lst if len(x[3]) == len(lead)), None)
    sp = next((x for x in lst if x[1] == 0 and x[2] == le), None)
    if cur is None or sp is None:
        miss += 1
        continue
    if cur[3] == lead:
        cur_ok += 1
    elif sp[3] == lead:
        unblock += 1
        tail_rival[cur[2] == len(w["code"])] += 1
        if len(ex) < 6:
            ex.append((w["code"], lead, le, [x[3] for x in lst[:5]], cur, sp))
    else:
        neither += 1

print(f"\n② gate（预算 12 @ 64 池）：有组句 {tot}，过音节守卫 {srok}")
print(f"  现状可确认（同字数档首选 == lead）: {cur_ok} "
      f"（{100*cur_ok/max(1,srok):.1f}%）")
print(f"  改同 span 才解锁: {unblock}（{100*unblock/max(1,srok):.1f}%）"
      f"  其中对手是尾部候选 {tail_rival[True]} / 头部·中部 {tail_rival[False]}")
print(f"  两种取法都不是 lead: {neither}  结构不成立: {miss}")
for code, lead, le, top, cur, sp in ex:
    print(f"   {code}: lead={lead}(0..{le}) 前5={top} "
          f"同字数档={cur[1:]} 同span={sp[1:]}")
