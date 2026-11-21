#!/usr/bin/env python3
"""上下文长度 × 候选数统一网格（2026-11-17 脚本合并）。

原 `ctx_eval_report.py`（ctx 10/17 × k 4/5）与 `sweep_maxcand / kdepth /
kmixed / 2layer / grid2layer / pack2x2` 的精度口径合并：同一份 cohort/user
矩阵上跑**生产两层语义** `layer_eval`（层1 = β·du + lp1 稳定降序前 k，
层2 = 前 k 的 S1 + CE 取 argmax），一次输出 err(k, ctx) 矩阵、相对生产点
(k=5, ctx=10) 的 Δ，以及可选的等延迟表。

不同语料/词典/CE 目录/参数全部走参数；CE 值只依赖 (ctx, 候选)，k 维是纯
numpy 回放，不重算推理。

用法:
  python grid_ctx_cand.py --ks 4,5 --ctx-ce 10:ce,17:ce_c17
  python grid_ctx_cand.py --ks 2,3,4,5,6,7,8,9,10 --domains user
  python grid_ctx_cand.py --domains cohort,user --beta 12 --row-ms 8.5 \
      --base-ms 27 --ms-per-token 0.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from cohort_layer1_feat import layer_eval
from summary_4x4 import cohort_matrices, user_matrices

sys.stdout.reconfigure(encoding="utf-8")
OUT = corpus_std.EVAL / "grid_ctx_cand_report.txt"


def parse_ks(text):
    ks = [int(x) for x in text.split(",") if x.strip()]
    if not ks:
        raise SystemExit("--ks 不能为空")
    return sorted(set(ks))


def parse_ctx_ce(text):
    out = []
    for item in (x.strip() for x in text.split(",") if x.strip()):
        if ":" not in item:
            raise SystemExit(f"--ctx-ce 需要 ctx:ce_dir 形式: {item!r}")
        ctx, ce = item.split(":", 1)
        out.append((int(ctx), ce))
    if not out:
        raise SystemExit("--ctx-ce 不能为空")
    return out


def load_matrices(domain, chain, arm, ce_dir):
    if domain == "cohort":
        return cohort_matrices(chain, {arm}, ce_dir=ce_dir)[arm]
    if domain == "user":
        return user_matrices(chain, ce_dir=ce_dir)
    raise SystemExit(f"未知 domain: {domain}")


def main():
    ap = argparse.ArgumentParser(description="ctx×候选数统一网格")
    ap.add_argument("--domains", default="cohort")
    ap.add_argument("--chain", default="big")
    ap.add_argument("--arm", type=int, default=corpus_std.DEFAULT_ARM,
                    help="cohort 年龄档（默认 3 万 = 用户默认年龄, 2026-11-18 定案）")
    ap.add_argument("--ks", default="4,5")
    ap.add_argument("--ctx-ce", default="10:ce,17:ce_c17",
                    help="ctx_tokens:CE 子目录（cohort 在链内, user 在项目根）")
    ap.add_argument("--beta", type=float, default=14.0)
    ap.add_argument("--row-ms", type=float, default=8.5,
                    help="S2 每行边际成本（仅等延迟表）")
    ap.add_argument("--base-ms", type=float, default=27.0,
                    help="S2 固定底座（仅等延迟表）")
    ap.add_argument("--ms-per-token", type=float, default=0.0,
                    help=">0 时输出等延迟表（prefill ms/token）")
    a = ap.parse_args()

    ks = parse_ks(a.ks)
    ctx_ce = parse_ctx_ce(a.ctx_ce)
    domains = [d.strip() for d in a.domains.split(",") if d.strip()]
    out = ["=" * 84,
           f"ctx×候选数网格 · 两层生产语义 · β={a.beta:g} · "
           f"k={ks} · ctx={[c for c, _ in ctx_ce]}",
           "=" * 84]

    for domain in domains:
        label = f"{domain}×{a.chain}"
        if domain == "cohort":
            label += f"@{a.arm}"
        out.append(f"\n== {label} ==")
        mats = {}
        for ctx, ce in ctx_ce:
            m = load_matrices(domain, a.chain, a.arm, ce)
            mats[ctx] = m
            print(f"loaded {domain} ctx={ctx} ({ce}): {m['n']} 窗", flush=True)
        base_m = mats[ctx_ce[0][0]]
        for ctx, m in mats.items():
            if not np.array_equal(m["widx"], base_m["widx"]):
                raise SystemExit(f"{domain} ctx={ctx} 窗序与首个 ctx 不一致")
        err = {}
        for ctx, m in mats.items():
            S1 = a.beta * m["du"] + m["lp1"]
            for k in ks:
                e, _miss, _brk = layer_eval(S1, m["s_ce"], m["widx"], k)
                err[(k, ctx)] = 100.0 * e.mean()

        hdr = "  {:<6}".format("k") + "".join(f"ctx{c:<6}" for c, _ in ctx_ce)
        out.append(hdr)
        for k in ks:
            out.append("  {:<6}".format(k) +
                       "".join(f"{err[(k, c)]:>9.2f}" for c, _ in ctx_ce))

        base_k = 5 if 5 in ks else ks[0]
        base_ctx = 10 if any(c == 10 for c, _ in ctx_ce) else ctx_ce[0][0]
        b = err[(base_k, base_ctx)]
        out.append(f"  基准点 k={base_k}, ctx={base_ctx}: {b:.2f}%")
        out.append("  Δ vs 基准（负=更好）:")
        for k in ks:
            out.append("  {:<6}".format(k) +
                       "".join(f"{err[(k, c)] - b:>+9.2f}" for c, _ in ctx_ce))

        if a.ms_per_token > 0:
            t0 = a.base_ms + a.row_ms * base_k
            out.append(f"\n  等延迟表（T = {a.base_ms:g} + {a.row_ms:g}·k + "
                       f"{a.ms_per_token:g}·(ctx-10) ms; 基准 T={t0:.1f}ms）")
            out.append("  {:<6}{:>9}{:>11}{:>10}".format(
                "ctx", "k_max", "T(ms)", "Δerr"))
            for ctx, _ in ctx_ce:
                kk = [k for k in ks
                      if a.base_ms + a.row_ms * k +
                      a.ms_per_token * max(0, ctx - 10) <= t0]
                if not kk:
                    continue
                k = max(kk)
                t = (a.base_ms + a.row_ms * k +
                     a.ms_per_token * max(0, ctx - 10))
                out.append(f"  {ctx:<6}{k:>9}{t:>11.1f}"
                           f"{err[(k, ctx)] - b:>+10.2f}")

    txt = "\n".join(out)
    OUT.write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    print(f"\n报告: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
