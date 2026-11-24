#!/usr/bin/env python3
"""CE 位置权重 × 长度项 的联合标定（2026-11-24 统一窗口后重标）。

背景：统一准入后窗口里等长词与子词同池比较，`-(ce1 + w2·ce2 + w3·ce3)` 的
加权和随 token 数变负，单字/短子词系统性占优。本脚本在**现有 CE 缓存**
（原始 ce1..ce3 + 真实总 ntok）上网格搜索多种公式，不重跑模型。

分数结构（每候选；n = 真实总 token 数）:
    L = -(ce1 + w2·ce2 + w3·ce3) / f(n) + lam·n
    s2 = L + b_dec·log1p(eff_τ) + b_cum·log1p(uni)
公式族（--families）:
    F0  f=1                     lam=0   现行结构（重标 w2/w3/β/γ）
    F1  f=Σw（加权均值）         lam=0   per-token 平均
    F2  f=n^alpha               lam=0   幂归一（alpha>0）
    F3  f=1                     lam>0   线性长度项（正 λ 罚短）
    F4  f=n^alpha               lam>0   归一 + 线性

注：CE 整体尺度 a_ce 与 (β, γ) 存在退化（a 缩放等价于 β/γ 同比缩小），故
固定 a_ce=1，把 β/γ 网格拉宽覆盖等价区间。

指标：窗内 top-1 选重率（全窗 + 按目标 ntok 分组）、split-half 诚实误差
（A 调参→B 评测、B→A，取均值）、层1 覆盖缺口 miss。τ 固定 400（与生产一致）。

用法（每域一次；CE 目录前缀 + ctx 自动拼接）:
  python eval/calib/length_formula_sweep.py --chain big_user \
    --model 0.6B=ce_06bb_base_q4km_cuser_c --ctxs 20 --ks 5,6 --tag _user_06bb
  python eval/calib/length_formula_sweep.py --chain big_60k \
    --model 0.6B=ce_06bb_base_q4km_c60k_c --ctxs 20 --ks 5,6 --tag _wiki_06bb
输出: eval/length_formula_sweep_<tag>.txt / .tsv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import fusion_sweep_all as FS  # noqa: E402  复用 load_base / eff_by_tau / load_ce
from corpus_std import EVAL  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROD = dict(w2=1.10, w3=0.60, norm="none", alpha=0.0, lam=0.0,
            b_dec=14.0, b_cum=1.0)


def num(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def long_score(c1, c2, c3, nt, pad, w2, w3, norm, alpha, lam):
    """→ L [n,W]（负 CE 域，越大越好）；bad/pad 置 -1e9。"""
    c2v = np.where(c2 > 0, c2, 0.0)
    c3v = np.where(c3 > 0, c3, 0.0)
    base = c1 + np.where(nt >= 2, w2 * c2v, 0.0) + np.where(nt >= 3, w3 * c3v, 0.0)
    if norm == "none":
        den = np.ones_like(base)
    elif norm == "pow":
        den = np.power(np.maximum(nt, 1).astype(np.float64), alpha)
    else:  # wsum：按实际参与的权重和归一（加权均值）
        den = 1.0 + np.where(nt >= 2, w2, 0.0) + np.where(nt >= 3, w3, 0.0)
    L = -base / np.maximum(den, 1e-9) + lam * nt
    bad = (nt == 0) | ((nt >= 2) & (c2 < 0)) | ((nt >= 3) & (c3 < 0))
    return np.where(bad | pad, -1e9, L)


def family_of(norm, lam):
    if norm == "wsum":
        return "F1"
    if norm == "pow":
        return "F2" if lam == 0 else "F4"
    return "F0" if lam == 0 else "F3"


def ntok_class(c1, c2, c3, nt, widx, n):
    """目标候选的 token 类：0=1tok, 1=2tok, 2=3+tok。"""
    cls = np.zeros(n, dtype=int)
    idx = (np.arange(n), widx)
    cls = np.where(nt[idx] >= 3, 2, np.where(nt[idx] >= 2, 1, 0))
    return cls


def eval_L(L, s1, order, widx, k):
    o = order[:, :k]
    s1k = np.take_along_axis(s1, o, axis=1)
    s2 = s1k + np.take_along_axis(L, o, axis=1)
    win = o[np.arange(len(o)), s2.argmax(1)]
    return win != widx


def fmt_ntok(e, cls):
    parts = []
    for c, nm in ((0, "1tok"), (1, "2tok"), (2, "3+tok")):
        m = cls == c
        parts.append(f"{nm} {100*e[m].mean():.2f}%({int(m.sum())})"
                     if m.any() else f"{nm} n/a")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--arm", type=int, default=30000)
    ap.add_argument("--model", required=True, help="名称=CE 目录前缀")
    ap.add_argument("--ctxs", default="20")
    ap.add_argument("--ks", default="5,6")
    ap.add_argument("--tau", type=float, default=400.0)
    ap.add_argument("--w2", default="0.95,1.00,1.05,1.10,1.15,1.20")
    ap.add_argument("--w3", default="0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--w-dec", default="8,10,12,14,16,20")
    ap.add_argument("--w-cum", default="0.5,1.0,1.5")
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--lams", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--families", default="F0,F1,F2,F3,F4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fs = set(a.families.split(","))
    w2s, w3s = num(a.w2), num(a.w3)
    decs, cums = num(a.w_dec), num(a.w_cum)
    alphas, lams = num(a.alphas), num(a.lams)
    ctxs = [int(x) for x in a.ctxs.split(",") if x.strip()]
    ks = [int(x) for x in a.ks.split(",") if x.strip()]

    t0 = time.time()
    name, prefix = a.model.split("=", 1)
    # 用第一个存在且 ctx 匹配的 CE 目录装载 base（只借它算 uni/pad/widx）
    lp1_ce = f"{prefix}{ctxs[0]}"
    data = FS.load_base(a.chain, a.arm, lp1_ce)
    n = data["n"]
    pad, widx = data["pad"], data["widx"]
    lp1 = data["lp1"]
    du = FS.eff_by_tau(data, a.tau)
    print(f"base {n} 窗 · τ={a.tau:g} 重放完成 {time.time()-t0:.0f}s", flush=True)

    # 层1 序只与 (β, γ) 有关：预排 40 条序
    orders = []
    for bd in decs:
        for bc in cums:
            s1 = np.where(pad, -1e9, bd * du + bc * lp1)
            orders.append((bd, bc,
                           np.argsort(-s1, axis=1, kind="stable").astype(
                               np.int32), s1))
    print(f"层1 序预排 {len(orders)} 条", flush=True)

    half = np.arange(n) % 2 == 0
    out = ["=" * 112,
           f"CE×长度项联合标定 · cohort_{a.chain} · {name} ctx={ctxs} · "
           f"{n} 窗 · τ={a.tau:g} · a_ce≡1（与 β/γ 退化）",
           f"网格 w2{len(w2s)} × w3{len(w3s)} × β{len(decs)} × γ{len(cums)}"
           f" × α{len(alphas)} × λ{len(lams)}",
           "err = 全窗选重率；honest = split-half 留出（A→B, B→A 均值）",
           "=" * 112]
    tsv = ["chain\tmodel\tctx\tk\tfamily\terr\thonest\terrA\terrB\tmiss"
           "\tw2\tw3\tnorm\talpha\tlam\tb_dec\tb_cum\tntok_err"]
    all_rows = []

    for ctx in ctxs:
        sub = f"{prefix}{ctx}"
        ce = FS.load_ce(a.chain, sub, data)
        if ce is None:
            print(f"{name} ctx={ctx}: 缺 {sub}，跳过", flush=True)
            continue
        c1, c2, c3, nt = ce
        cls = ntok_class(c1, c2, c3, nt, widx, n)

        # 变体清单（family, norm, alpha, lam）
        variants = []
        if "F0" in fs:
            variants.append(("F0", "none", 0.0, 0.0))
        if "F1" in fs:
            variants.append(("F1", "wsum", 0.0, 0.0))
        if "F2" in fs:
            variants += [("F2", "pow", al, 0.0) for al in alphas]
        if "F3" in fs:
            variants += [("F3", "none", 0.0, lm) for lm in lams]
        if "F4" in fs:
            variants += [("F4", "pow", al, lm) for al in alphas for lm in lams]

        # 每 k/family 的三种最优（全量 / A 半 / B 半）
        best = {}
        prod_err = {}
        miss_of = {}
        nc = 0
        # 层1 序 + 窗内 s1 预取（每 (β,γ) 一次，取 max(k) 宽）
        kmax = max(ks)
        ords = []
        for bd, bc, order, s1 in orders:
            o = order[:, :kmax]
            ords.append((bd, bc, o, np.take_along_axis(s1, o, axis=1)))
        ar = np.arange(n)
        for w2 in w2s:
            for w3 in w3s:
                for fam, norm, al, lm in variants:
                    L = long_score(c1, c2, c3, nt, pad, w2, w3, norm, al, lm)
                    for bd, bc, o, s1k in ords:
                        s2 = s1k + np.take_along_axis(L, o, axis=1)
                        for k in ks:
                            win = o[ar, s2[:, :k].argmax(1)]
                            e = win != widx
                            ea = float(e[half].mean())
                            eb = float(e[~half].mean())
                            key = (k, fam)
                            rec = dict(err=float(e.mean()), errA=ea, errB=eb,
                                       w2=w2, w3=w3, norm=norm, alpha=al,
                                       lam=lm, b_dec=bd, b_cum=bc)
                            nc += 1
                            if key not in best or rec["err"] < best[key][0]["err"]:
                                best[key] = (rec, e)
                            ba = best.setdefault((k, fam, "A"), None)
                            if ba is None or ea < ba[0]["errA"]:
                                best[(k, fam, "A")] = (rec, e)
                            bb = best.setdefault((k, fam, "B"), None)
                            if bb is None or eb < bb[0]["errB"]:
                                best[(k, fam, "B")] = (rec, e)
                            if (w2 == PROD["w2"] and w3 == PROD["w3"]
                                    and norm == PROD["norm"] and al == 0.0
                                    and lm == 0.0 and bd == PROD["b_dec"]
                                    and bc == PROD["b_cum"]):
                                prod_err[k] = rec["err"]
                                pres = order == widx[:, None]
                                miss_of[k] = float(1.0 - pres[:, :k].any(1).mean())
        print(f"ctx={ctx}: {nc} 次评估 {time.time()-t0:.0f}s", flush=True)

        out.append(f"\n== cohort_{a.chain} · {name} ctx={ctx} · {n} 窗 ==")
        for k in ks:
            pe = prod_err.get(k)
            out.append(f"-- k={k}  现行点 err="
                       f"{(100*pe if pe is not None else float('nan')):.2f}%"
                       f"  miss={(100*miss_of.get(k, float('nan'))):.2f}%")
            out.append(f"  {'族':<4}{'最优err':>8}{'诚实err':>8}"
                       f"{'乐观差':>7}  {'w2':>5}{'w3':>5}{'α':>5}{'λ':>4}"
                       f"{'β':>4}{'γ':>4}  按目标 ntok 的选重率")
            for fam in ("F0", "F1", "F2", "F3", "F4"):
                key = (k, fam)
                if key not in best:
                    continue
                b, e = best[key]
                ha = best.get((k, fam, "A"))
                hb = best.get((k, fam, "B"))
                # A 半选参 → 用其在 B 的表现；这里近似取 A/B 两半选点的留出均值
                honest = 0.5 * (ha[0]["errB"] + hb[0]["errA"]) \
                    if (ha and hb) else float("nan")
                out.append(f"  {fam:<4}{100*b['err']:>7.2f}%{100*honest:>7.2f}%"
                           f"{100*(honest-b['err']):>6.2f}%  "
                           f"{b['w2']:>5.2f}{b['w3']:>5.2f}{b['alpha']:>5.2f}"
                           f"{b['lam']:>4.1f}{b['b_dec']:>4.0f}{b['b_cum']:>4.1f}"
                           f"  {fmt_ntok(e, cls)}")
                tsv.append(
                    f"{a.chain}\t{name}\t{ctx}\t{k}\t{fam}\t"
                    f"{100*b['err']:.3f}\t{100*honest:.3f}\t"
                    f"{100*b['errA']:.3f}\t{100*b['errB']:.3f}\t"
                    f"{100*miss_of.get(k, float('nan')):.3f}\t"
                    f"{b['w2']:.2f}\t{b['w3']:.2f}\t{b['norm']}\t"
                    f"{b['alpha']:.2f}\t{b['lam']:.1f}\t"
                    f"{b['b_dec']:.0f}\t{b['b_cum']:.2f}\t"
                    f"{fmt_ntok(e, cls)}")
                all_rows.append((b["err"], k, fam, b))
        # 跨族最优（供落地参考）
        for k in ks:
            pts = [(best[(k, f)][0]["err"], f, best[(k, f)][0])
                   for f in ("F0", "F1", "F2", "F3", "F4") if (k, f) in best]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            e, f, b = pts[0]
            out.append(f"  → k={k} 全族最优: {f} err={100*e:.2f}% "
                       f"(w2={b['w2']:.2f} w3={b['w3']:.2f} "
                       f"{b['norm']} α={b['alpha']:.2f} λ={b['lam']:.1f} "
                       f"β={b['b_dec']:.0f} γ={b['b_cum']:.1f})")

    txt = "\n".join(out)
    tag = a.tag or f"_{a.chain}_{name}"
    (EVAL / f"length_formula_sweep{tag}.txt").write_text(txt + "\n",
                                                         encoding="utf-8")
    (EVAL / f"length_formula_sweep{tag}.tsv").write_text(
        "\n".join(tsv) + "\n", encoding="utf-8")
    print(txt)
    print(f"\n报告: eval/length_formula_sweep{tag}.txt ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
