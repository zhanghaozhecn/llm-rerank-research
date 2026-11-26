#!/usr/bin/env python3
"""首选率网格（CE 预算口径）：上文 token 数 × 候选预算 4–18（2026-11-26）。

替代 `cpp/bench_sweep2`（tok 1–20 × cand 2–9）：2026-11-26 起生产成本单位是
**CE2+CE3 总预算 B**（1-token 免费、多 token 花 min(ntok,3)−1），候选数不再是
成本参数，旧网格失去实验意义。

口径（与 sweep2 严格同构，只把「取前 k 个」换成「预算准入」）：
  · 候选集合 = 引擎原窗（dict 序），正确词在其原有位置；
  · 分数  = 纯 CE 排序（CE = ce1 + w2·ce2 + w3·ce3，无词频项——与 sweep2 一致）；
  · 准入  = 层1 序（这里退化为 dict 序）扫描，1-token 免费，多 token 花
            min(ntok,3)−1，预算用尽即止（`luna_mixed_lib.budget_mask`，与
            llm_filter 同构）；
  · 首选率 = 准入集合内 CE 最高的候选 == 正确词。
计分窗 = `cohort_big_60k` 锚点的**真实候选窗**（dict 序）+ 锚点真实 CE 缓存
（0.6B/0.5B/0.8B/4B base Q4_K_M, ctx20，`--model 名称=CE目录`）。
**不含 tok 轴**：ctx 截断需逐 tok 重算 CE，生产固定 ctx=20，故只保留 ctx=20 一档；
旧 bench_sweep2 的 tok 行仅为历史（真实语料短文多，tok 对首选率的影响见 ② 的
cf 结论与 `eval_samples.tsv` 时期的记录）。

用法:
  python calib/sweep_budget.py --model 0.6B=ce_06bb_base_q4km_c60k_c20 \
      --budgets 4-18 --tag 06bb_base
输出: eval/sweep_budget_<tag>.tsv / .txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calib"))
import corpus_std
import fusion_sweep_all as FS
import luna_mixed_lib as L

sys.stdout.reconfigure(encoding="utf-8")
W2, W3 = 1.10, 0.60


def parse_range(s):
    out = []
    for p in s.split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="big_60k")
    ap.add_argument("--arm", type=int, default=30000)
    ap.add_argument("--model", required=True, help="名称=CE 目录名")
    ap.add_argument("--budgets", default="4-18")
    ap.add_argument("--w2", type=float, default=W2)
    ap.add_argument("--w3", type=float, default=W3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name, sub = a.model.split("=", 1)
    budgets = parse_range(a.budgets)
    data = FS.load_base(a.chain, a.arm, sub)
    n = data["n"]
    c1, c2, c3, nt = FS.load_ce(a.chain, sub, data)
    ce = FS.ce_shape(c1, c2, c3, nt, data["pad"], a.w2, a.w3)
    has = ce > -1e8
    # 空窗（无任何可评分候选）不参与
    ok = has.any(1)
    lab = data["widx"]
    rows = []
    for b in budgets:
        m = L.budget_mask(nt, np.zeros_like(data["pad"]), data["pad"],
                          budget=b) & has
        pick = np.where(m, ce, -1e9).argmax(1)
        hit = (pick == lab) & ok
        rows.append((b, ok.sum(), hit.sum() / max(ok.sum(), 1)))
    out = corpus_std.EVAL / f"sweep_budget_{a.tag or name}.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("model\tbudget\tn\thit_rate\n")
        for b, nn, r in rows:
            f.write(f"{name}\t{b}\t{nn}\t{100*r:.2f}\n")
    print(f"\n{name} · 纯 CE 首选率 × CE 预算（{a.chain}，n={int(ok.sum())}）")
    for b, nn, r in rows:
        print(f"  B={b:<3} {100*r:6.2f}%")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
