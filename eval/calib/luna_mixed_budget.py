#!/usr/bin/env python3
"""luna 混合窗：**CE 预算准入**下的结构 / 生产点（2026-11-26）。

背景：2026-11-26 起生产窗口不再是「前 6 条」，而是 CE 预算 12 准入
（1-token 免费全收 + 多 token 花 min(ntok,3)−1）。旧研究
`luna_mixed_eval.py` 的 6/16 窗口径结论需要按预算窗重测；本脚本用同一
cohort（21,596 静态窗、同一 CE 缓存）给出：

  · 窗宽分布、窗内覆盖率（有正确候选/理想候选）；
  · 引擎序首选、融合分首选的 wrong / nonideal（生产点 w2=1.10, w3=0.60）；
  · 与 top6 / top16 并排，便于看「预算窗 ≈ 多少候选」以及旧结论的偏移。

输出：eval/luna_mixed_budget.txt
用法：python eval/calib/luna_mixed_budget.py
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_eval as E  # noqa: E402
import luna_mixed_lib as L  # noqa: E402


def count_mask(d, k):
    m = (~d["pad"]) & (~d["gshow"])
    return m & (np.cumsum(m, axis=1) <= k)


def report(name, d, mask):
    n = d["n"]
    ar = np.arange(n)
    has_win = mask.any(1)
    sizes = mask.sum(1)
    e_eng = E.metrics(np.zeros_like(d["c1"]), mask, d)
    e_prod = E.metrics(E.prod_score(d), mask, d)
    has, tok, cov = E.classes(d, mask)
    ideal_len = np.where(mask & d["good"], d["tlen"], -1).max(1)
    print(f"\n── {name}（窗宽 中位 {int(np.median(sizes))} / 均值 "
          f"{sizes.mean():.1f} / 空窗 {int((~has_win).sum())}）")
    print(f"  窗内有正确候选 {100*has.mean():.2f}%  "
          f"窗内有理想（最长正确）{100*(ideal_len >= 0).mean():.2f}%")
    print(f"  引擎序首选  : {E.rates(e_eng)}")
    print(f"  生产融合首选: {E.rates(e_prod)}")
    print(f"  理想类      : {E.by_class(e_prod, has, tok, cov)}")
    print(f"  按域（生产 nonideal）: wiki "
          f"{100*e_prod['nonideal'][d['dom'] == 'wiki'].mean():.2f}% / user "
          f"{100*e_prod['nonideal'][d['dom'] == 'user'].mean():.2f}%")
    cnt = Counter(sizes.tolist())
    print("  窗宽分布: " + " ".join(f"{k}:{cnt[k]}"
                                    for k in sorted(cnt) if k <= 17)
          + f"  >17:{sum(v for k, v in cnt.items() if k > 17)}")
    return e_prod


def formula_check(d, mask):
    """在预算窗上复核长度项/归一化（旧结论在 6 窗口径）。"""
    half = d["sent"] % 2 == 0
    rows = []
    for name, norm, al, lam in E.variants():
        tes, picks = [], []
        for tr, te in ((half, ~half), (~half, half)):
            best = None
            for w2 in E.W2S:
                for w3 in E.W3S:
                    s = E.score_of(d, w2, w3, norm, al, lam)
                    e = E.metrics(s, mask & tr[:, None], d)
                    v = float(e["nonideal"][tr].mean())
                    if best is None or v < best[0]:
                        best = (v, w2, w3)
            _, w2, w3 = best
            s = E.score_of(d, w2, w3, norm, al, lam)
            e = E.metrics(s, mask & te[:, None], d)
            tes.append((float(e["wrong"][te].mean()),
                        float(e["nonideal"][te].mean())))
            picks.append((w2, w3))
        rows.append((name, norm, al, lam, picks,
                     np.mean([x[0] for x in tes]),
                     np.mean([x[1] for x in tes])))
    rows.sort(key=lambda x: x[6])
    print("\n── 公式族复核（预算窗；A 选点→B 评测 + 对称平均）")
    print(f"{'变体':<6}{'norm':<6}{'α':>5}{'λ':>6}"
          f"{'选点(w2,w3)':>18}{'wrong':>8}{'nonideal':>10}")
    for name, norm, al, lam, picks, w, n in rows:
        pk = "/".join(f"{a:.2f},{b:.2f}" for a, b in picks)
        print(f"{name:<6}{norm:<6}{al:>5.2f}{lam:>+6.2f} {pk:>18}"
              f"{100*w:>7.2f}%{100*n:>9.2f}%")
    f0 = next(r for r in rows if r[0] == "F0")
    best = rows[0]
    print(f"  F0 现行: nonideal {100*f0[6]:.2f}%；最优 {best[0]} "
          f"({best[1]}, α={best[2]:.2f}, λ={best[3]:+.2f}) "
          f"{100*best[6]:.2f}%  Δ {100*(best[6]-f0[6]):+.2f}pp")


def main():
    d = E.load()
    print("=" * 100)
    print(f"luna 混合窗 · CE 预算准入 vs 旧 6/16 窗口径 · 静态窗 {d['n']} "
          f"（wiki {int((d['dom'] == 'wiki').sum())} / user "
          f"{int((d['dom'] == 'user').sum())}）· 冷账本")
    print("=" * 100)
    masks = (("top6（旧生产）", count_mask(d, 6)),
             ("top16（旧窗宽实验）", count_mask(d, 16)),
             ("CE 预算 12（收集 16 池）",
              L.budget_mask(d["nt"], d["gshow"], d["pad"]) & d["has_ce"]))
    for name, m in masks:
        report(name, d, m)
    formula_check(d, masks[-1][1])
    print("\n注：wrong = 首选不是正确前缀；nonideal = 首选不是窗内最长正确"
          "（正确但偏短也算，多付一次确认）。预算窗在本 cohort 的 **CE 收集池"
          "（16 条）内**模拟；线上收集 64，超出的候选未算 CE，因此本表是"
          "「收集池 16」口径（覆盖/窗宽会被低估），融合指标只在有 CE 的"
          "子集上算。预算 12 = 旧 6 候选的等价成本。")


if __name__ == "__main__":
    main()
