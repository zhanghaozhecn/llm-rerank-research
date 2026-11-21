#!/usr/bin/env python3
"""① 的延迟网格：ctx × 设备（CPU / GPU）每窗毫秒。

同一批窗（默认 `cohort_big_60k/anchors.jsonl` 前 N 个）按 `--ctx` 截断上文、
按 `--cands` 截断候选，逐组合实测 wall time（多轮取最小值消热漂移）。

用法:
  python bench/ctx_latency.py                    # ctx 1-20, k=5, CPU+GPU
  python bench/ctx_latency.py --ctxs 1,5,10,20 --cands 5 --n 100
  python bench/ctx_latency.py --sims cpu           # 只测 CPU（GPU 需单进程铁律）
报告: eval/ctx_latency_report.txt
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
SIM_CPU = PROJ / "cpp/build_sim/Release/sim_rerank.exe"
SIM_GPU = PROJ / "cpp/build_sim/Release/sim_rerank_gpu.exe"
TARGET = PROJ / "cohort_big_60k/anchors.jsonl"
OUT = corpus_std.EVAL / "ctx_latency_report.txt"
MODEL = Path(os.environ.get(
    "GGUF_MODEL", "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"))


def load_rows(n, cands):
    rows = []
    with open(TARGET, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["ctx"]) < 120:   # 60k 链含浅历史窗，上文不足无法截到 ctx20
                continue
            win = list(r["win"])[:cands] if cands else list(r["win"])
            if len(win) < 2:
                continue
            rows.append({"context": r["ctx"], "cands": win})
            if len(rows) >= n:
                break
    return rows


def run(sim, rows, ctx, ngl, rounds):
    inp = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    env = dict(os.environ)
    if ngl == 0:
        env["SIM_NO_CE_PIPE"] = "1"
    best, done = float("inf"), ""
    for _ in range(rounds):
        t0 = time.perf_counter()
        p = subprocess.run([str(sim), str(MODEL), "--ce", "--ngl", str(ngl),
                            "--ctx", str(ctx)],
                           input=inp.encode("utf-8"), capture_output=True,
                           env=env)
        dt = time.perf_counter() - t0
        lines = p.stdout.decode("utf-8", "replace").strip().split("\n")
        if len(lines) != len(rows):
            raise RuntimeError(f"{sim.name}: 输出 {len(lines)} != {len(rows)}\n"
                               f"{p.stderr.decode('utf-8','replace')[-400:]}")
        best = min(best, dt / len(rows) * 1000)
        d = [l for l in p.stderr.decode("utf-8", "replace").split("\n")
             if l.startswith("Done.")]
        if d:
            done = d[-1]
    return best, done


def main():
    ap = argparse.ArgumentParser(description="ctx 延迟网格")
    ap.add_argument("--ctxs", default=",".join(str(i) for i in range(1, 21)))
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--sims", default="cpu,gpu", help="cpu,gpu 子集")
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",") if x.strip()]
    rows = load_rows(a.n, a.cands)
    if not rows:
        raise SystemExit("锚点为空")
    devices = [d.strip() for d in a.sims.split(",") if d.strip()]

    out = ["=" * 78,
           f"① 延迟网格：{len(rows)} 窗 × ctx {ctxs[0]}-{ctxs[-1]} × k={a.cands}"
           f" × {a.rounds} 轮（min）",
           f"模型: {MODEL.name}；锚点: {TARGET.name}", "=" * 78,
           f"{'ctx':>4}" + "".join(f"{d:>12}" for d in devices) +
           f"{'  GPU/CPU':>10}" + "  分段（CPU）"]
    for ctx in ctxs:
        ms, seg = {}, {}
        for d in devices:
            sim = SIM_GPU if d == "gpu" else SIM_CPU
            ngl = 99 if d == "gpu" else 0
            ms[d], seg[d] = run(sim, rows, ctx, ngl, a.rounds)
        ratio = (ms["gpu"] / ms["cpu"]) if "gpu" in ms and "cpu" in ms else None
        line = f"{ctx:>4}" + "".join(f"{ms[d]:>11.1f}" for d in devices)
        line += f"{(f'{ratio:>9.2f}x' if ratio else ''):>10}"
        line += f"  {seg.get('cpu', '')[:90]}"
        out.append(line)
        print(line, flush=True)
    txt = "\n".join(out)
    OUT.write_text(txt + "\n", encoding="utf-8")
    print(f"\n报告: {OUT}")


if __name__ == "__main__":
    main()
