#!/usr/bin/env python3
"""LLM × 词频 2×2 矩阵（2026-11-20 重测，0.6B base ctx20 生产口径）。

取代 README 旧 2×2（0.5B Instruct / ctx10 / k5 / 20,004 窗时代）。
四格（同窗同参，固定生产点 τ=400 / w2=1.10 / w3=0.60 / b_dec=14 / b_cum=1.0）：
  无LLM·无词频 = 引擎序（窗内第 0 位命中）
  无LLM·有词频 = 层1 词频 argmax（b_dec·log1p(eff)+b_cum·log1p(uni)，全宽 10）
  有LLM·无词频 = 纯 CE argmax（ce_shape 位置加权，全宽 10；判别力上限口径）
  有LLM·有词频 = 两层融合（层1 词频选前 k → 层2 CE+词频 argmax）
复用 fusion_sweep_all 的 load_base / eff_by_tau / load_ce（词码对累计口径）。
用法:
  python eval/calib/matrix_2x2.py --chain big_60k --arm 30000 \
    --ce ce_06bb_base_q4km_c60k_c20 [--ks 4,5,6] [--tag _06bb_20261120]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_sweep_all as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="big_60k")
    ap.add_argument("--arm", type=int, default=30000)
    ap.add_argument("--ce", required=True, help="CE 目录名（完整名，含 ctx 后缀）")
    ap.add_argument("--ks", default="4,5,6")
    ap.add_argument("--tau", type=float, default=400.0)
    ap.add_argument("--w2", type=float, default=1.10)
    ap.add_argument("--w3", type=float, default=0.60)
    ap.add_argument("--b-dec", type=float, default=14.0)
    ap.add_argument("--b-cum", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    data = F.load_base(a.chain, a.arm, a.ce)
    n, widx, pad, lp1 = data["n"], data["widx"], data["pad"], data["lp1"]
    du = F.eff_by_tau(data, a.tau)
    c1, c2, c3, nt = F.load_ce(a.chain, a.ce, data)
    L = F.ce_shape(c1, c2, c3, nt, pad, a.w2, a.w3)

    s1 = np.where(pad, -1e9, a.b_dec * du + a.b_cum * lp1)
    order = np.argsort(-s1, axis=1, kind="stable")
    have_ce = L > -1e8

    # 四格
    cell_engine = float((widx != 0).mean())
    cell_freq = float((order[:, 0] != widx).mean())
    any_ce = have_ce.any(1)
    cell_ce = float((L.argmax(1) != widx)[any_ce].mean())
    ce_cover = float(any_ce.mean())

    out = ["=" * 88,
           f"LLM × 词频 2×2 · cohort_{a.chain}@{a.arm} · {n} 窗 · 0.6B base ctx20 · "
           f"τ{a.tau:g}/w2 {a.w2}/w3 {a.w3}/b_dec {a.b_dec:g}/b_cum {a.b_cum:g}"
           + (f" · {a.tag}" if a.tag else ""),
           "=" * 88,
           f"无LLM·无词频（引擎序）          {cell_engine*100:.2f}%",
           f"无LLM·有词频（层1 argmax）      {cell_freq*100:.2f}%",
            f"有LLM·无词频（纯CE argmax，全宽）{cell_ce*100:.2f}%   "
            f"(CE 覆盖 {ce_cover*100:.2f}% 窗)",
            "-" * 88]
    for k in ks:
        o = order[:, :k]
        s1k = np.take_along_axis(s1, o, axis=1)
        s2 = s1k + np.take_along_axis(L, o, axis=1)
        win = o[np.arange(n), s2.argmax(1)]
        e = float((win != widx).mean())
        pres = (order[:, :k] == widx[:, None]).any(1)
        miss = float(1.0 - pres.mean())
        out.append(f"有LLM·有词频（两层 k={k}）       {e*100:.2f}%   (层1miss {miss*100:.2f}%)")
    report = "\n".join(out) + "\n"
    print(report, flush=True)
    fn = f"matrix_2x2_{a.chain}_{a.ce}{a.tag}.txt"
    (F.PROJ / "eval" / fn).write_text(report, encoding="utf-8")
    print(f"报告: eval/{fn}", flush=True)


if __name__ == "__main__":
    main()
