#!/usr/bin/env python3
"""延迟网格（中位/p90）：模型 × ctx × k × 设备，逐窗计时统计。

与 `ctx_latency.py` 的区别：那个按"总墙钟/窗数"给均值（适合单模型 ctx 曲线）；
本脚本要的是**中位延迟**（用户定案：最终目标是 (中位延迟, 正确率) 帕累托前沿，
阈值 50–100ms 可变），所以走 sim 的 `SIM_TIMES` 逐窗输出（2026-11-18 加），
并按窗取中位数/p90（多轮先在窗内取最小值消热漂移，再取分位数）。

口径：候选段（层2 评分窗）——`--ce` 模式、`--cands k` 截断候选；
CPU = ngl=0 + `SIM_NO_CE_PIPE=1`（生产单进程形态），GPU = ngl=99 单进程。
**测量必须在机器空闲时做**（GPU CE 批处理会抢核，CPU 数字会虚高）。

用法:
  python bench/latency_grid.py --model 0.5Bi=d:/gguf_models/...gguf \
      --ctxs 1,5,10,15,20 --cands 2,5,9 --n 60 --rounds 2 --sims cpu
输出（追加）: eval/latency_grid_<tag>.tsv
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
SIM = {"cpu": PROJ / "cpp/build_sim/Release/sim_rerank.exe",
       "gpu": PROJ / "cpp/build_sim/Release/sim_rerank_gpu.exe"}
TARGET = PROJ / "cohort_big_60k/anchors.jsonl"


def load_rows(n, cands):
    rows = []
    with open(TARGET, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["ctx"]) < 120:   # 60k 链含浅历史窗，上文不足无法截到 ctx20
                continue
            win = list(r["win"])[:cands]
            if len(win) < 2:
                continue
            rows.append({"context": r["ctx"], "cands": win})
            if len(rows) >= n:
                break
    return rows


def run_once(sim, model, rows, ctx, ngl):
    inp = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    env = dict(os.environ)
    if ngl == 0:
        env["SIM_NO_CE_PIPE"] = "1"
    fd, tp = tempfile.mkstemp(suffix=".times", prefix="simlat_")
    os.close(fd)
    env["SIM_TIMES"] = tp
    p = subprocess.run([str(sim), str(model), "--ce", "--ngl", str(ngl),
                        "--ctx", str(ctx)],
                       input=inp.encode("utf-8"), capture_output=True, env=env)
    try:
        vals = [float(x) for x in open(tp, encoding="utf-8").read().split()]
    except Exception:
        vals = []
    finally:
        try:
            os.unlink(tp)
        except OSError:
            pass
    if len(vals) != len(rows):
        raise RuntimeError(
            f"{sim.name}: 逐窗计时 {len(vals)} != {len(rows)}；"
            f"{p.stderr.decode('utf-8','replace')[-300:]}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="名称=gguf 路径")
    ap.add_argument("--ctxs", default="1,5,10,15,20")
    ap.add_argument("--cands", default="2,5,9")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--sims", default="cpu")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    name, path = a.model.split("=", 1)
    name = name.strip()
    ctxs = [int(x) for x in a.ctxs.split(",") if x.strip()]
    ks = [int(x) for x in a.cands.split(",") if x.strip()]
    devices = [d.strip() for d in a.sims.split(",") if d.strip()]

    tag = a.note or name
    out = corpus_std.EVAL / f"latency_grid_{tag}.tsv"
    hdr = ("model\tctx\tk\tdevice\tp50\tp90\tmean\tmin\tn\trounds\t"
           "sim_wall_ms_per_win\n")
    new = not out.exists()
    with open(out, "a", encoding="utf-8") as f:
        if new:
            f.write(hdr)
        print(f"{'ctx':>4}{'k':>4}" + "".join(f"{d:>10}" for d in devices)
              + "   (p50 ms)", flush=True)
        for k in ks:
            rows = load_rows(a.n, k)
            for ctx in ctxs:
                line = f"{ctx:>4}{k:>4}"
                for d in devices:
                    ngl = 99 if d == "gpu" else 0
                    per = None
                    t0 = time.perf_counter()
                    for _ in range(a.rounds):
                        vals = run_once(SIM[d], path, rows, ctx, ngl)
                        per = vals if per is None else [min(x, y) for x, y
                                                        in zip(per, vals)]
                    wall = (time.perf_counter() - t0) * 1000 / (
                        a.rounds * len(rows))
                    p50 = statistics.median(per)
                    s = sorted(per)
                    p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
                    mean = statistics.fmean(per)
                    line += f"{p50:>10.1f}"
                    f.write(f"{name}\t{ctx}\t{k}\t{d}\t{p50:.2f}\t{p90:.2f}\t"
                            f"{mean:.2f}\t{min(per):.2f}\t{len(per)}\t"
                            f"{a.rounds}\t{wall:.2f}\n")
                    f.flush()
                print(line, flush=True)
    print(f"\n追加写入: {out}")


if __name__ == "__main__":
    main()
