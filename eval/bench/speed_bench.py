#!/usr/bin/env python3
"""统一推理速度基准（2026-11-17 脚本合并）。

原 `latency_probe_model.py`（模型规模对比）与 `latency_probe_upgrade.py`
（llama.cpp 库版本对比）合并：同一份锚点窗、CPU 单进程
（SIM_NO_CE_PIPE=1, --ngl 0），按 `--models × --sims` 组合实测 wall time；
多组合按轮次交错、每组合取最小值消热漂移。

用法:
  python speed_bench.py --n 200
  python speed_bench.py --models "0.5B=<gguf>,0.8B=<gguf>" --rounds 3
  python speed_bench.py --sims "new=<sim>,base=<sim>" --models "0.5B=<gguf>"

默认模型 = `corpus_std.MODEL`（`GGUF_MODEL` 环境变量可覆盖）；默认 sim =
`cpp/build_sim/Release/sim_rerank.exe`（CPU）。报告写 `speed_bench_report.txt`。
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
SIM_CPU = PROJ / "cpp" / "build_sim" / "Release" / "sim_rerank.exe"
OUT = corpus_std.EVAL / "speed_bench_report.txt"


def parse_pairs(text, what):
    """'name=path,name2=path2' → [(name, path)]；允许 path 内含 '='。"""
    out = []
    for item in (x.strip() for x in text.split(",") if x.strip()):
        if "=" not in item:
            raise SystemExit(f"--{what} 需要 name=path 形式: {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), path.strip()))
    if not out:
        raise SystemExit(f"--{what} 不能为空")
    return out


def load_windows(path, n):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    if len(rows) < n:
        print(f"WARN: 锚点只有 {len(rows)} 行（请求 {n}）", flush=True)
    inp = "\n".join(json.dumps({"context": r["ctx"], "cands": list(r["win"])},
                               ensure_ascii=False) for r in rows) + "\n"
    return inp, len(rows)


def run_once(sim, model, inp, n, ngl):
    env = dict(os.environ)
    if ngl == 0:
        env["SIM_NO_CE_PIPE"] = "1"
    t0 = time.perf_counter()
    p = subprocess.run([str(sim), model, "--ce", "--ngl", str(ngl)],
                       input=inp.encode("utf-8"), capture_output=True, env=env)
    dt = time.perf_counter() - t0
    lines = p.stdout.decode("utf-8", "replace").strip().split("\n")
    if len(lines) != n:
        raise RuntimeError(
            f"{sim} / {model}: 输出行数 {len(lines)} != {n}\n"
            f"stderr: {p.stderr.decode('utf-8', 'replace')[-500:]}")
    done = [l for l in p.stderr.decode("utf-8", "replace").split("\n")
            if l.startswith("Done.")]
    return dt / n * 1000, (done[-1] if done else "")


def main():
    ap = argparse.ArgumentParser(description="统一推理速度基准")
    ap.add_argument("--anchors", default=str(PROJ / "cohort_big_user" /
                                             "anchors.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", default=f"default={corpus_std.MODEL}",
                    help="name=path,name=path；默认 corpus_std.MODEL"
                         "（GGUF_MODEL 环境变量可覆盖；生产 0.5B 请显式传）")
    ap.add_argument("--sims", default=f"new={SIM_CPU}", help="name=path,name=path")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ngl", type=int, default=0, help="0=CPU; >0=GPU 层数")
    a = ap.parse_args()

    models = parse_pairs(a.models, "models")
    sims = parse_pairs(a.sims, "sims")
    inp, n = load_windows(a.anchors, a.n)
    if n == 0:
        raise SystemExit("锚点为空")
    combos = [(mn, mp, sn, sp) for mn, mp in models for sn, sp in sims]

    out = ["=" * 100,
           f"速度基准：{n} 窗 × {len(models)} 模型 × {len(sims)} sim × "
           f"{a.rounds} 轮（ngl={a.ngl}）",
           f"锚点: {Path(a.anchors).name}", "=" * 100]
    print("\n".join(out), flush=True)
    best = {i: [] for i in range(len(combos))}
    done_line = {i: "" for i in range(len(combos))}
    for rnd in range(a.rounds):
        order = list(range(len(combos)))
        if rnd % 2:
            order.reverse()          # 交错顺序，抵消热漂移
        for i in order:
            mn, mp, sn, sp = combos[i]
            per, done = run_once(sp, mp, inp, n, a.ngl)
            best[i].append(per)
            done_line[i] = done
            msg = f"  r{rnd + 1} {mn} @ {sn}: {per:.1f} ms/窗"
            if done:
                msg += f" | {done}"
            print(msg, flush=True)

    base = min(best[0]) if best[0] else None
    out.append("")
    out.append(f"{'model':<16}{'sim':<12}{'min ms/窗':>11}{'median':>10}"
               f"{'vs 首组合':>11}")
    for i, (mn, _mp, sn, _sp) in enumerate(combos):
        ms = sorted(best[i])
        mn_ms, med = ms[0], ms[len(ms) // 2]
        ratio = (mn_ms / base) if base else float("nan")
        out.append(f"{mn:<16}{sn:<12}{mn_ms:>11.1f}{med:>10.1f}"
                   f"{ratio:>10.2f}x")
    out.append("")
    out.append("分段（sim stderr Done. 行）:")
    for i, (mn, _mp, sn, _sp) in enumerate(combos):
        out.append(f"  {mn} @ {sn}: {done_line[i] or '-'}")
    txt = "\n".join(out)
    OUT.write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    print(f"\n报告: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
