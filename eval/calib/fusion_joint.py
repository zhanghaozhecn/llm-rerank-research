#!/usr/bin/env python3
"""融合参数联合扫参（2026-11-18 用户定案：不做可分性验证，直接联合）。

分数结构（两层，k = 层1 窗口宽度；**与生产同一参数化**，三项各有一个系数）：
  层1:  s1 = b_dec·log1p(eff_τ) + b_cum·log1p(uni)
  层2:  s2 = a_ce·L(w2,w3) + b_dec·log1p(eff_τ) + b_cum·log1p(uni)
        L = −(ce1 + w2·ce2 + w3·ce3)（w1 固定 1 = 只定尺度）

生产系数 = a_ce=1, b_dec=14, b_cum=1（累计项系数正好是 1）。
扫参固定 a_ce=1（三者同乘不改变 argmax，故只需相对系数），
网格 = w2 × w3 × τ × b_dec × b_cum。τ 由 streams 重放复算（不依赖语料重跑）。
用法:
  python calib/fusion_joint.py --chain big --arm 30000 --ce ce_05bi --k 5
  python calib/fusion_joint.py --chain big_ctx10k --arm 30000 --ce ce_c10 --k 5
报告: eval/fusion_joint_<chain>_<ce>_k<k>.txt
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from corpus_std import FormulaReplay
from derive_pairfix_ratio import load_w2codes

sys.stdout.reconfigure(encoding="utf-8")
EVAL = corpus_std.EVAL
PROJ = corpus_std.PROJ


def load_matrices(chain, arm, ce_sub):
    """anchors + samples(eff_win) + CE 原始三分量 + streams 重放索引。"""
    d = PROJ / f"cohort_{chain}"
    anch = [json.loads(l) for l in open(d / "anchors.jsonl", encoding="utf-8")]
    smp = []
    sp = d / "samples.jsonl"
    if sp.exists():
        smp = [json.loads(l) for l in open(sp, encoding="utf-8")]
    keep = [i for i, a in enumerate(anch)
            if (a.get("dom") in (None, "big")) and a.get("arm") == arm]
    n = len(keep)
    codes = [(smp[i].get("code") if smp else None) or anch[i].get("code", "")
             for i in keep]
    W = 10
    wins = [anch[i]["win"] for i in keep]
    lens = np.array([len(w) for w in wins])
    widx = np.array([w.index(anch[i]["w"]) for w, i in zip(wins, keep)])
    pad = np.arange(W)[None, :] >= lens[:, None]
    c1 = np.full((n, W), -9.0); c2 = np.full((n, W), -9.0)
    c3 = np.full((n, W), -9.0); nt = np.zeros((n, W), dtype=int)
    ce_path = d / ce_sub / "results.jsonl"
    recs = {}
    for l in open(ce_path, encoding="utf-8"):
        try:
            r = json.loads(l); recs[r["sid"]] = (r.get("ce") or {},
                                                 r.get("ntok") or {})
        except Exception:
            pass
    for r, i in enumerate(keep):
        a = anch[i]; ce, ntok = recs.get(a["sid"], ({}, {}))
        for j, w in enumerate(a["win"]):
            if w in ce:
                v = ce[w]
                c1[r, j], c2[r, j], c3[r, j] = v[0], v[1], v[2]
                nt[r, j] = ntok.get(w, 0)
    by_ui = {}
    for r, i in enumerate(keep):
        by_ui.setdefault(anch[i]["ui"], {}).setdefault(anch[i]["ci"], []).append(r)
    return dict(anch=anch, keep=keep, n=n, wins=wins, widx=widx, pad=pad,
                codes=codes, c1=c1, c2=c2, c3=c3, nt=nt, by_ui=by_ui)


def eff_by_tau(data, tau):
    """按 τ 重放 streams → du[n,W] = log1p(eff_τ)（提交前取值）。"""
    _, first_code = load_w2codes()
    n, W = data["n"], 10
    du = np.zeros((n, W))
    need = set(data["by_ui"])
    with open(PROJ / ("cohort_" + data["chain"]) / "streams.jsonl",
              encoding="utf-8") as f:
        for ui, line in enumerate(f):
            wmap = data["by_ui"].get(ui)
            if not wmap:
                continue
            rp = FormulaReplay(tau)
            for ci, w in enumerate(json.loads(line)):
                for r in wmap.get(ci, ()):
                    a = data["anch"][data["keep"][r]]
                    code = data["codes"][r]
                    for j, x in enumerate(a["win"]):
                        e = rp.eff_of((x, code))
                        du[r, j] = math.log1p(e) if e else 0.0
                rp.bump((w, first_code.get(w, "")))
    return np.where(data["pad"], 0.0, du)


def ce_shape(data, w2, w3):
    c1, c2, c3, nt, pad = data["c1"], data["c2"], data["c3"], data["nt"], data["pad"]
    s = -c1
    s = np.where(nt >= 2, -(c1 + w2 * np.where(c2 > 0, c2, 0.0)), s)
    s = np.where(nt >= 3,
                 -(c1 + w2 * np.where(c2 > 0, c2, 0.0)
                   + w3 * np.where(c3 > 0, c3, 0.0)), s)
    bad = (nt == 0) | ((nt >= 2) & (c2 < 0)) | ((nt >= 3) & (c3 < 0))
    return np.where(bad | pad, -1e9, s)


def fused_err(data, L, du, lp1, a_ce, b_dec, b_cum, k):
    s1 = np.where(data["pad"], -1e9, b_dec * du + b_cum * lp1)
    order = np.argsort(-s1, axis=1, kind="stable")[:, :k]
    s1k = np.take_along_axis(s1, order, axis=1)
    s2 = s1k + a_ce * np.take_along_axis(L, order, axis=1)
    winner = order[np.arange(len(s2)), s2.argmax(1)]
    err = float((winner != data["widx"]).mean())
    miss = float((~np.any(order == data["widx"][:, None], axis=1)).mean())
    return err, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="big")
    ap.add_argument("--arm", type=int, default=30000)
    ap.add_argument("--ce", default="ce_05bi")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ce-coef", type=float, default=1.0,
                    help="CE 项系数 a_ce（尺度基准，默认 1 = 生产）")
    ap.add_argument("--w2", default="1.00,1.05,1.10,1.15,1.20,1.25")
    ap.add_argument("--w3", default="0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--tau", default="100,200,300,400,600")
    ap.add_argument("--w-dec", default="6,10,14,20,28",
                    help="衰减词频项系数 b_dec（生产 14）")
    ap.add_argument("--w-cum", default="0.5,0.75,1.0,1.25,1.5",
                    help="累计词频项系数 b_cum（生产 1）")
    a = ap.parse_args()
    num = lambda s: [float(x) for x in s.split(",") if x.strip()]

    t0 = time.time()
    data = load_matrices(a.chain, a.arm, a.ce)
    data["chain"] = a.chain
    # lp1：用 streams 的累计计数（与 cohort_matrices 同法；窗内常数不影响 argmax）
    m = None
    try:
        import summary_4x4 as S
        m = S.cohort_matrices(a.chain, {a.arm}, ce_dir=a.ce)[a.arm]
        lp1 = m["lp1"]
    except Exception:
        lp1 = None
    if lp1 is None:
        raise SystemExit("lp1 装载失败")
    del m
    print(f"loaded {data['n']} windows in {time.time()-t0:.1f}s", flush=True)

    taus = num(a.tau)
    dus = {t: eff_by_tau(data, t) for t in taus}
    print(f"eff replay done ({len(taus)} tau) {time.time()-t0:.1f}s", flush=True)

    a_ce = a.ce_coef
    best = (9.0, None)
    top = []
    for tau in taus:
        du = dus[tau]
        for w2 in num(a.w2):
            for w3 in num(a.w3):
                L = ce_shape(data, w2, w3)
                for b_dec in num(a.w_dec):
                    for b_cum in num(a.w_cum):
                        e, miss = fused_err(data, L, du, lp1, a_ce,
                                            b_dec, b_cum, a.k)
                        if e < best[0]:
                            best = (e, dict(tau=tau, w2=w2, w3=w3,
                                            b_dec=b_dec, b_cum=b_cum,
                                            miss=miss))
                        top.append((e, tau, w2, w3, b_dec, b_cum, miss))
    top.sort(key=lambda t: t[0])
    E = best[1]
    L = ["=" * 96,
         f"融合联合扫参 · cohort_{a.chain}@{a.arm} · CE={a.ce} · k={a.k}",
         f"参数化 = a_ce·L + b_dec·log1p(eff) + b_cum·log1p(uni)；"
         f"生产 = (1, 14, 1)；本次固定 a_ce={a_ce}",
         f"窗数 {data['n']} · 网格 τ{num(a.tau)} × w2{num(a.w2)} × w3{num(a.w3)} "
         f"× b_dec{num(a.w_dec)} × b_cum{num(a.w_cum)}",
         f"最优: τ={E['tau']}, w2={E['w2']}, w3={E['w3']}, "
         f"b_dec={E['b_dec']}, b_cum={E['b_cum']} → 选重率 {100*best[0]:.3f}% "
         f"(层1 miss {100*E['miss']:.3f}%)",
         "=" * 96, "top10:"]
    for e, tau, w2, w3, b_dec, b_cum, miss in top[:10]:
        L.append(f"  {100*e:7.3f}%  τ={tau:>4} w2={w2:.2f} w3={w3:.2f} "
                 f"b_dec={b_dec:>4} b_cum={b_cum:.2f} miss={100*miss:.2f}%")
    txt = "\n".join(L)
    out = EVAL / f"fusion_joint_{a.chain}_{a.ce}_k{a.k}.txt"
    out.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n报告: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
