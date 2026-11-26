# -*- coding: utf-8 -*-
"""① CE 预算到底放进来多少行/多少 token；② 单字 lead 的 gate 是否被
「不完整音节字」占位（用户当面提的两件事）。"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EVAL = Path(r"D:\llm-rerank\llm-rerank-research\eval")
DICT = Path(r"C:\Users\Administrator\AppData\Roaming\Rime\wanxiang.dict.yaml")
sys.path.insert(0, str(EVAL / "calib"))
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_eval as E   # noqa: E402
import luna_mixed_lib as L    # noqa: E402

d = E.load(k_pool=64, k_probe=64, cohort=L.COHORT_EXT, cands_path=L.CANDS_EXT)
prod = E.prod_score(d)
ar = np.arange(d["n"])
full = (~d["pad"]) & (~d["gshow"]) & d["has_ce"]
ideal_full = np.where(full & d["good"], d["tlen"], -1).max(1)

print("=== ① 预算 12 的窗：行数 / token 数 ===")
m = L.budget_mask(d["nt"], d["gshow"], d["pad"], 12) & d["has_ce"]
rows = m.sum(1)
tok = np.where(m, d["nt"], 0).sum(1)
one = np.where(m & (d["nt"] == 1), 1, 0).sum(1)
multi = np.where(m & (d["nt"] >= 2), 1, 0).sum(1)
print(f"  行数   中位 {int(np.median(rows))} 均值 {rows.mean():.1f} max {rows.max()}")
print(f"  其中 1-token 行 中位 {int(np.median(one))} 均值 {one.mean():.1f}")
print(f"      多-token 行 中位 {int(np.median(multi))} 均值 {multi.mean():.1f}")
print(f"  CE token 总数（Σntok）中位 {int(np.median(tok))} 均值 {tok.mean():.1f} "
      f"max {tok.max()}")
print(f"  估算耗时 = 30ms + 8.5ms×Σntok：中位 "
      f"{(30 + 8.5*np.median(tok))/1000:.2f}s")

print("\n=== ①b 若预算按「候选行数」计（前 k 行 / 引擎序）===")
print(f"{'k 行':>5}{'Σntok中位':>10}{'wrong':>9}{'nonideal':>11}{'估算ms':>9}")
for k in (6, 8, 10, 12, 16, 20, 32, 39):
    mm = L.budget_mask(d["nt"], d["gshow"], d["pad"], 999)
    cnt = np.cumsum(mm, axis=1)
    mm = mm & (cnt <= k) & d["has_ce"]
    s = np.where(mm, prod, -1e18)
    top = s.argmax(1)
    g = d["good"][ar, top]
    tl = d["tlen"][ar, top]
    tokk = np.where(mm, d["nt"], 0).sum(1)
    print(f"{k:>5}{int(np.median(tokk)):>10}"
          f"{100*(~g).mean():>8.2f}%{100*((~g)|(tl<ideal_full)).mean():>10.2f}%"
          f"{30 + 8.5*np.median(tokk):>9.0f}")

print("\n=== ② 单字 lead：gate 的 1 字档首选是否可能是「不完整音节字」===")
exact = defaultdict(set)
with open(DICT, encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and len(p[0]) == 1:
            exact[p[1]].add(p[0])
cohort = [json.loads(l) for l in
          open(EVAL / "luna_mixed" / "cohort64.jsonl", encoding="utf-8")]
n_single_lead = 0
bad_pick = 0
unblock2 = 0
examples = []
for w in cohort:
    if not w["gen"]:
        continue
    g = w["gen"][0]
    text, lead, le = g[0], g[2], g[3]
    if not lead or le is None or len(lead) != 1:
        continue
    if len(text) - len(lead) < 2:
        continue
    n_single_lead += 1
    win = []
    rem = 12
    for r in w["eng"][:64]:
        t, s, e, typ, q, ce3, nt = r
        nt = nt or 0
        cost = 0 if nt <= 1 else min(nt, 3) - 1
        if nt == 0:
            cost = 1
        if nt <= 1 or cost <= rem:
            if ce3 and ce3[0] is not None and ce3[0] >= 0:
                win.append((ce3[0], s, e, t))
            if nt > 1:
                rem -= cost
    if not win:
        continue
    win.sort()
    singles = [x for x in win if len(x[3]) == 1]
    if not singles:
        continue
    pick = singles[0]
    code = w["code"]
    ok = code[:le] in exact and pick[3] in exact[code[:le]]
    if not ok:
        bad_pick += 1
        # 跳过不完整音节后，下一个 1 字档候选是不是 lead？
        rest = [x for x in singles[1:]
                if x[3] in exact.get(code[:le], set())]
        if rest and rest[0][3] == lead:
            unblock2 += 1
            if len(examples) < 5:
                examples.append((code, lead, [(x[3], x[0]) for x in singles[:5]]))
print(f"  单字 lead 且过音节守卫：{n_single_lead}")
print(f"  1 字档首选是**不完整音节字**：{bad_pick}")
print(f"  其中跳过它后紧接着就是 lead（会被新规则解锁）：{unblock2}")
for code, lead, top in examples:
    print(f"   {code}: lead={lead} 单字序 {top}")
