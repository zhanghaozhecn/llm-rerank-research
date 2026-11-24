#!/usr/bin/env python3
"""长度项实验的定点复核：把各公式族的最优点与生产点放在同一口径下比。

输入：eval/length_formula_sweep_<tag>.tsv（length_formula_sweep.py 产出）
      同一 cohort/模型 的 CE 缓存（复用 fusion_sweep_all 装载器）
输出：eval/lenformula_compare_<tag>.txt（整体/分半/按目标 ntok 分组）

用法:
  python eval/calib/lenformula_compare.py --chain big_user \
    --model 0.6B=ce_06bb_base_q4km_cuser_c --tsv eval/length_formula_sweep_user_06bb.tsv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import fusion_sweep_all as FS  # noqa: E402
from corpus_std import EVAL  # noqa: E402
from length_formula_sweep import long_score, ntok_class  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROD = dict(w2=1.10, w3=0.60, norm="none", alpha=0.0, lam=0.0,
            b_dec=14.0, b_cum=1.0)


def fmt_ntok(e, cls):
    parts = []
    for c, nm in ((0, "1tok"), (1, "2tok"), (2, "3+tok")):
        m = cls == c
        parts.append(f"{nm} {100*e[m].mean():.2f}%({int(m.sum())})"
                     if m.any() else f"{nm} n/a")
    return " ".join(parts)


def eval_point(pt, c1, c2, c3, nt, pad, du, lp1, widx, half, k):
    s1 = np.where(pad, -1e9, pt["b_dec"] * du + pt["b_cum"] * lp1)
    order = np.argsort(-s1, axis=1, kind="stable")[:, :k]
    s1k = np.take_along_axis(s1, order, axis=1)
    L = long_score(c1, c2, c3, nt, pad, pt["w2"], pt["w3"], pt["norm"],
                   pt["alpha"], pt["lam"])
    s2 = s1k + np.take_along_axis(L, order, axis=1)
    win = order[np.arange(len(order)), s2.argmax(1)]
    e = win != widx
    pres = order == widx[:, None]
    return dict(err=float(e.mean()), errA=float(e[half].mean()),
                errB=float(e[~half].mean()), e=e,
                miss=float(1.0 - pres.any(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tsv", default="", help="length_formula_sweep 的 TSV")
    ap.add_argument("--ctxs", default="20")
    ap.add_argument("--ks", default="5,6")
    ap.add_argument("--tau", type=float, default=400.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--isolate", action="store_true",
                    help="固定共识权重，隔离 α/λ 的影响（忽略 --tsv）")
    a = ap.parse_args()
    name, prefix = a.model.split("=", 1)
    ctxs = [int(x) for x in a.ctxs.split(",") if x.strip()]
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    rows = []
    if not a.isolate:
        rows = [r for r in csv.DictReader(open(a.tsv, encoding="utf-8"),
                                          delimiter="\t")
                if r["chain"] == a.chain and r["model"] == name
                and int(r["ctx"]) == ctxs[0] and int(r["k"]) in ks]
    else:
        # 隔离实验：固定共识权重 (1.15,0.60,β14,γ1)，只变 α/λ
        base = dict(chain=a.chain, model=name, ctx=str(ctxs[0]),
                    w2="1.15", w3="0.60", b_dec="14", b_cum="1.0")
        rows.append(dict(base, family="F0", norm="none", alpha="0.0",
                         lam="0.0"))
        for lm in (0.25, 0.5, 0.75, 1.0, 1.5):
            rows.append(dict(base, family="F3", norm="none", alpha="0.0",
                             lam=str(lm)))
        for al in (0.25, 0.5, 0.75, 1.0):
            rows.append(dict(base, family="F2", norm="pow", alpha=str(al),
                             lam="0.0"))
        rows.append(dict(base, family="F4", norm="pow", alpha="0.5",
                         lam="0.5"))

    data = FS.load_base(a.chain, 30000, f"{prefix}{ctxs[0]}")
    n, pad, widx, lp1 = data["n"], data["pad"], data["widx"], data["lp1"]
    du = FS.eff_by_tau(data, a.tau)
    ce = FS.load_ce(a.chain, f"{prefix}{ctxs[0]}", data)
    c1, c2, c3, nt = ce
    cls = ntok_class(c1, c2, c3, nt, widx, n)
    half = np.arange(n) % 2 == 0

    out = ["=" * 104,
           f"长度项定点复核 · cohort_{a.chain} · {name} · {n} 窗 · τ={a.tau:g}",
           "=" * 104]
    for k in ks:
        out.append(f"\n-- k={k} --")
        pe = eval_point(PROD, c1, c2, c3, nt, pad, du, lp1, widx, half, k)
        out.append(f"  生产点(1.10,0.60,β14,γ1) err={100*pe['err']:.2f}% "
                   f"A={100*pe['errA']:.2f} B={100*pe['errB']:.2f} "
                   f"miss={100*pe['miss']:.2f}%  {fmt_ntok(pe['e'], cls)}")
        for r in rows:
            if not a.isolate and int(r["k"]) != k:
                continue
            pt = dict(w2=float(r["w2"]), w3=float(r["w3"]), norm=r["norm"],
                      alpha=float(r["alpha"]), lam=float(r["lam"]),
                      b_dec=float(r["b_dec"]), b_cum=float(r["b_cum"]))
            e = eval_point(pt, c1, c2, c3, nt, pad, du, lp1, widx, half, k)
            out.append(
                f"  {r['family']:<4} err={100*e['err']:>6.2f}% "
                f"A={100*e['errA']:.2f} B={100*e['errB']:.2f} "
                f"(Δ{100*(e['err']-pe['err']):+.2f}pp)  "
                f"w2={pt['w2']:.2f} w3={pt['w3']:.2f} {r['norm']} "
                f"α={pt['alpha']:.2f} λ={pt['lam']:.1f} "
                f"β={pt['b_dec']:.0f} γ={pt['b_cum']:.1f}  "
                f"{fmt_ntok(e['e'], cls)}")
    txt = "\n".join(out)
    tag = a.tag or f"_{a.chain}_{name}"
    (EVAL / f"lenformula_compare{tag}.txt").write_text(txt + "\n",
                                                       encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
