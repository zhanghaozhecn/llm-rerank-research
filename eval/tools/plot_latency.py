#!/usr/bin/env python3
"""②③ 延迟曲线图：每模型一张子图，4 张拼一张（部署档 Q4_K_M）。

② eval/fig_prefill_ctx.png：x=上文 token 数（1–20），y=T_prefill（ms），虚线 150ms
③ eval/fig_cand_k.png      ：x=候选数 k（2–9），y=T_cand 最坏情况（ms），虚线 100ms

数据（2026-11-19 定稿：0.5B/0.6B/0.8B 全部 official base Q4_K_M 自建）：
  ② 0.5B `05bb_q4km_prep` / 0.6B `06bb_q4km_prep` / 0.8B `08bb_q4km_prep` / 2B `2B_prep`
  ③ 0.5B `05bb_q4km_s23` / 0.6B `06bb_q4km_s23` / 0.8B `08bb_q4km_s23`；
     2B `2B_step23`（取 ctx=20 行）
用法: python tools/plot_latency.py
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

MODELS = ["0.5Bi", "0.6B", "0.8B", "2B"]
LABEL = {"0.5Bi": "Qwen2.5-0.5B Base (Q4_K_M)",
         "0.6B": "Qwen3-0.6B Base (Q4_K_M)",
         "0.8B": "Qwen3.5-0.8B Base (Q4_K_M)",
         "2B": "Qwen3.5-2B (Q4_K_M)"}
PRE = {
    "0.5Bi": ["latency_synth_05bb_q4km_prep.tsv"],
    "0.6B": ["latency_synth_06bb_q4km_prep.tsv"],
    "0.8B": ["latency_synth_08bb_q4km_prep.tsv"],
    "2B": ["latency_synth_2B_prep.tsv"],
}
CAND = {
    "0.5Bi": ["latency_synth_05bb_q4km_s23.tsv"],
    "0.6B": ["latency_synth_06bb_q4km_s23.tsv"],
    "0.8B": ["latency_synth_08bb_q4km_s23.tsv"],
    "2B": [("latency_synth_2B_step23.tsv", 20)],
}


def read(files, xcol="ctx"):
    """files = [path] 或 [(path, ctx_filter)]；同一 x 后者覆盖前者。"""
    d = {}
    for item in files:
        path, ctx_filter = (item, None) if isinstance(item, str) else item
        p = corpus_std.EVAL / path
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        head = lines[0].split("\t")
        ix, iv = head.index(xcol), head.index("p50")
        ic = head.index("ctx") if "ctx" in head else None
        for line in lines[1:]:
            f = line.split("\t")
            if len(f) <= max(ix, iv):
                continue
            if ctx_filter is not None and ic is not None and int(f[ic]) != ctx_filter:
                continue
            d[int(f[ix])] = float(f[iv])
    return d


def main():
    pre = {m: read(PRE[m]) for m in MODELS}
    cand = {m: read(CAND[m], xcol="k") for m in MODELS}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m in zip(axes.flat, MODELS):
        d = pre[m]
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker="o", ms=3, color="C0")
        ax.axhline(150, color="0.55", ls="--", lw=1)
        ax.set_title(LABEL[m], fontsize=10)
        ax.set_xlabel("上文 token 数 (ctx)")
        ax.set_ylabel("T_prefill 中位 (ms)")
        ax.set_xlim(0.5, 20.5)
        ax.grid(alpha=0.25)
    fig.suptitle("② ctx 预填充延迟（Q4_K_M 部署档，CPU 中位；虚线=150ms）")
    fig.tight_layout()
    out1 = corpus_std.EVAL / "fig_prefill_ctx.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m in zip(axes.flat, MODELS):
        d = cand[m]
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker="o", ms=4, color="C1")
        ax.axhline(100, color="0.55", ls="--", lw=1)
        ax.set_title(LABEL[m], fontsize=10)
        ax.set_xlabel("候选数 k（进 CE 融合窗）")
        ax.set_ylabel("T_cand 中位 (ms)")
        ax.set_xlim(1.5, 9.5)
        ax.grid(alpha=0.25)
    fig.suptitle("③ 候选段最坏情况延迟（k 个候选×3 token、首 token 互不相同；"
                 "CPU 中位；虚线=100ms）")
    fig.tight_layout()
    out2 = corpus_std.EVAL / "fig_cand_k.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    print("写出", out1.name, out2.name)


if __name__ == "__main__":
    main()
