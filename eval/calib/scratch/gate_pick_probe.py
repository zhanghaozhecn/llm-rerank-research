# -*- coding: utf-8 -*-
"""③ 影响量化：gate 的「同字数档首选」vs「同 span 首选」——结构口径。
用 cohort 的 gen（lead / lead_end）+ 候选 span + CE，不跑门（P/τ/ctx 另算）。"""
import json
from collections import Counter

EVAL = r"D:\llm-rerank\llm-rerank-research\eval"
cands = json.load(open(EVAL + r"\luna_mixed\cands.json", encoding="utf-8"))

tot = sr_ok = 0
cur_ok = unblock = neither = lead_missing = 0
ex = []
rival_tail = Counter()
for line in open(EVAL + r"\luna_mixed\cohort.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if not r["gen"]:
        continue
    g = r["gen"][0]
    text, lead, le = g[0], g[2], g[3]
    if not lead or le is None:
        continue
    tot += 1
    if len(text) - len(lead) < 2:      # 音节守卫：剩余 ≥2 字
        continue
    sr_ok += 1
    ce = {c[0]: c[5][0] for c in r["eng"][:16]
          if c[5] and c[5][0] is not None and c[5][0] >= 0}
    lst = []
    for c in cands.get(r["code"], [])[:16]:
        t, s, e = c[0], c[1], c[2]
        if t in ce:
            lst.append((ce[t], s, e, t))
    if not lst:
        continue
    lst.sort(key=lambda x: x[0])       # CE 成本升序 = 融合序（冷账本）
    cur = next((x for x in lst if len(x[3]) == len(lead)), None)
    sp = next((x for x in lst if x[1] == 0 and x[2] == le), None)
    if not sp:
        lead_missing += 1
        continue
    if cur is None:
        lead_missing += 1
        continue
    if cur[3] == lead:
        cur_ok += 1
    elif sp[3] == lead:
        unblock += 1
        tail_rival = (cur[2] == len(r["code"]))
        rival_tail[tail_rival] += 1
        if len(ex) < 8:
            ex.append((r["code"], lead, le, [x[3] for x in lst[:5]],
                       (cur[1], cur[2], cur[3]), (sp[1], sp[2], sp[3])))
    else:
        neither += 1

print(f"有组句的窗 {tot}；过音节守卫 {sr_ok}；lead 在该 span 有候选且同字数档也有 {sr_ok-lead_missing}")
print(f"  现状已可确认（同字数档首选 == lead）: {cur_ok}")
print(f"  **改同 span 后解锁**（同字数档首选≠lead，但同 span 首选==lead）: {unblock}")
print(f"  两种取法都不是 lead（真冲突）: {neither}")
print(f"  结构不成立（lead 不在窗/无同字数档候选）: {lead_missing}")
print(f"  其中对手候选是**尾部候选**（span 到码尾 = 可能完成匹配）: "
      f"{rival_tail[True]}；是头部/中部候选（整读音节）: {rival_tail[False]}")
for code, lead, le, top, cur, sp in ex:
    print(f"  {code}: lead={lead}(span 0..{le}) 融合前5={top} "
          f"同字数档首选={cur} 同span首选={sp}")
