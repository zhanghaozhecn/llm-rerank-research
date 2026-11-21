#!/usr/bin/env python3
"""T_prefill(ctx) 线性拟合与 ctx* 选择（2026-11-18 用户定案口径）。

用户定案：t_prefill 与 ctx 近似线性 → 用线性函数拟合，取 **t_prefill<150ms 的最大 ctx**，
且 **ctx 上限 20 token**（再大没有必要）。

拟合用 OLS（p50 vs ctx）。ctx=1 是首窗/冷启动态（p50 明显低于线性段，p90 异常高），
默认排除在拟合之外，仅作敏感性对照（`--keep-ctx1`）。

用法:
  python tools/fit_prefill.py                # 读 eval/latency_synth_*_prep.tsv
  python tools/fit_prefill.py --keep-ctx1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[1]
FILES = {
    "0.5B-base": "latency_synth_05bb_q4km_prep.tsv",
    "0.6B-base": "latency_synth_06bb_q4km_prep.tsv",
    "0.8B-base": "latency_synth_08bb_q4km_prep.tsv",
    "2B": "latency_synth_2B_prep.tsv",
}
LIMIT_MS = 150.0
CTX_CAP = 20


def load(path):
    xs, ys = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            xs.append(float(p[1]))
            ys.append(float(p[4]))  # p50
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-ctx1", action="store_true", help="把 ctx=1 也纳入拟合")
    ap.add_argument("--extra", action="append", default=[],
                    help="额外拟合表：名称=latency_synth_*.tsv（可多次）")
    args = ap.parse_args()

    files = dict(FILES)
    for spec in args.extra:
        name, fn = spec.split("=", 1)
        files[name] = fn
    print(f"{'模型':<24}{'斜率':>8}{'截距':>8}{'R²':>7}"
          f"{'150ms 交点':>11}{'ctx*':>6}{'拟合@ctx*':>10}")
    for name, fn in files.items():
        path = EVAL / fn
        if not path.exists():
            print(f"{name:<24}  (缺 {fn})")
            continue
        x, y = load(path)
        m = np.ones_like(x, dtype=bool) if args.keep_ctx1 else (x > 1)
        k, b = np.polyfit(x[m], y[m], 1)
        pred = k * x + b
        ss_res = float(((y[m] - pred[m]) ** 2).sum())
        ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        cross = (LIMIT_MS - b) / k
        ctx_star = min(CTX_CAP, int(np.floor(cross)))
        fit_at = k * ctx_star + b
        print(f"{name:<14}{k:>8.2f}{b:>8.1f}{r2:>7.3f}"
              f"{cross:>11.1f}{ctx_star:>6d}{fit_at:>10.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
