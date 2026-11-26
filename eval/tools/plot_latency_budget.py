#!/usr/bin/env python3
"""③ 候选段延迟图（CE 预算口径）：x=ce_budget B（4–18），y=T_cand 最坏情况。

数据（2026-11-26：3 个 base Q4_K_M 部署档，ctx=20，step2+step3）：
  eval/latency_budget_05bb_q4km_budget.tsv
  eval/latency_budget_06bb_q4km_budget.tsv
  eval/latency_budget_08bb_q4km_budget.tsv
用法: python tools/plot_latency_budget.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

SERIES = [
    ("0.5B-base Q4_K_M", "latency_budget_05bb_q4km_budget.tsv", "C0"),
    ("0.6B-base Q4_K_M", "latency_budget_06bb_q4km_budget.tsv", "C2"),
    ("0.8B-base Q4_K_M", "latency_budget_08bb_q4km_budget.tsv", "C1"),
]


def read(name):
    p = corpus_std.EVAL / name
    d = {}
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        return d
    head = lines[0].split("\t")
    ib, iv = head.index("budget"), head.index("p50")
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) > max(ib, iv):
            d[int(f[ib])] = float(f[iv])
    return d


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (label, fname, color) in zip(axes, SERIES):
        d = read(fname)
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker="o", ms=3.5, color=color)
        ax.axhline(100, color="0.55", ls="--", lw=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("CE 预算 B（CE2+CE3 总 token 数）")
        ax.set_xlim(3, 19)
        ax.set_xticks([4, 6, 8, 10, 12, 14, 16, 18])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("T_cand 中位 (ms)")
    fig.suptitle("③ 候选段最坏情况延迟（预算全花在多 token 候选：3-token×2 / 2-token×1；"
                 "CPU 中位；虚线=100ms）")
    fig.tight_layout()
    out = corpus_std.EVAL / "fig_cand_budget.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("写出", out.name)


if __name__ == "__main__":
    main()
