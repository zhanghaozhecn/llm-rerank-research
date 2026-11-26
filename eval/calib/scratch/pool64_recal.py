# -*- coding: utf-8 -*-
"""池 64 复核：CE 预算扫描 + 公式族（CE 权重 / 长度项 / β 之外的冷账本口径）。

沿用 luna_mixed_eval 的装载与评分，只换 cohort/cands 为 64 池产物。
用法：python -I -X utf8 eval/calib/scratch/pool64_recal.py
"""
import sys
from pathlib import Path

import numpy as np

EVAL = Path(r"D:\llm-rerank\llm-rerank-research\eval")
sys.path.insert(0, str(EVAL / "calib"))
sys.stdout.reconfigure(encoding="utf-8")

import luna_mixed_eval as E   # noqa: E402
import luna_mixed_lib as L    # noqa: E402
import luna_mixed_budget as B  # noqa: E402

POOL = 64
d = E.load(k_pool=POOL, k_probe=POOL, cohort=L.COHORT_EXT,
           cands_path=L.CANDS_EXT)
print(f"cohort64: {d['n']} 窗 · 池 {POOL}")

print("\n── ① CE 预算扫描（池 64，引擎序准入；生产点 w2=1.10 w3=0.60）")
print(f"{'预算':>5}{'窗宽中位':>8}{'wrong':>9}{'nonideal':>11}{'short':>9}"
      f"{'1tok理想':>10}")
prod = E.prod_score(d)
rows = []
for bud in (4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 20, 24, 32, 64):
    m = L.budget_mask(d["nt"], d["gshow"], d["pad"], budget=bud) & d["has_ce"]
    e = E.metrics(prod, m, d)
    has, tok, cov = E.classes(d, m)
    ideal = np.where(m & d["good"], d["tlen"], -1).max(1) >= 0
    t1 = has & (tok == 0)
    sizes = m.sum(1)
    rows.append((bud, int(np.median(sizes)), 100 * e["wrong"].mean(),
                 100 * e["nonideal"].mean(), 100 * e["short"].mean()))
    print(f"{bud:>5}{int(np.median(sizes)):>8}{100*e['wrong'].mean():>8.2f}%"
          f"{100*e['nonideal'].mean():>10.2f}%{100*e['short'].mean():>8.2f}%"
          f"{100*e['nonideal'][t1].mean():>9.2f}%(1tok n={int(t1.sum())})"
          f"  理想覆盖 {100*ideal.mean():.2f}%")
best = min(rows, key=lambda r: r[3])
print(f"  → 最优预算 {best[0]}（nonideal {best[3]:.2f}%，窗宽中位 {best[1]}）；"
      f"生产 12 = {next(r[3] for r in rows if r[0]==12):.2f}%")

print("\n── ①b 同上，但**理想候选固定为全 64 池内最长正确**（去掉「窗越小理想越短」"
      "的自指偏差；横比口径）")
full = (~d["pad"]) & (~d["gshow"]) & d["has_ce"]
ideal_full = np.where(full & d["good"], d["tlen"], -1).max(1)
ar = np.arange(d["n"])
print(f"{'预算':>5}{'窗宽中位':>8}{'wrong':>9}{'nonideal':>11}{'short':>9}")
rows2 = []
for bud in (4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 20, 24, 32, 64):
    m = L.budget_mask(d["nt"], d["gshow"], d["pad"], budget=bud) & d["has_ce"]
    s = np.where(m, prod, -1e18)
    top = s.argmax(1)
    g = d["good"][ar, top]
    tl = d["tlen"][ar, top]
    wrong = float((~g).mean())
    nonideal = float(((~g) | (tl < ideal_full)).mean())
    short = float((g & (tl < ideal_full)).mean())
    rows2.append((bud, wrong, nonideal))
    print(f"{bud:>5}{int(np.median(m.sum(1))):>8}{100*wrong:>8.2f}%"
          f"{100*nonideal:>10.2f}%{100*short:>8.2f}%")
b2 = min(rows2, key=lambda r: r[2])
print(f"  → 固定理想口径下的最优预算 {b2[0]}"
      f"（nonideal {100*b2[2]:.2f}%）；生产 12 = "
      f"{100*next(r[2] for r in rows2 if r[0]==12):.2f}%")

print("\n── ② 公式族复核（预算 12 @ 64 池；A 选点→B 评测，对称平均）")
B.formula_check(d, L.budget_mask(d["nt"], d["gshow"], d["pad"], 12) & d["has_ce"])
