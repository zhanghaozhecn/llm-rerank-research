#!/usr/bin/env python3
"""合成最坏情况延迟网格：模型 × ctx × k，一次进程跑完全部结果。

用户定案（2026-11-18）：
  - 延迟 = **候选段两次 decode**（step2 首 token 后一次、step3 再一次）；
    **预解码（ctx prefill）不计**（生产里 commit 后异步完成，不在关键路径）。
  - 取最坏情况：**所有候选都按 3 token 算**（k 个候选 = k 条 3-token 行），
    且首 token 两两不同（前缀去重失效 → step2/step3 各 k 行，真最坏）。
  - 一次进程跑完该模型全部 (ctx, k)：模型只加载一次、机器状态一致；
    多轮取**逐位置最小值**消运行间噪声，再取该格中位数。

候选构造：从 anchors 的真实候选词里挑 ntok==3 且首 token 互不相同的词
（`eval/lib/gguf_bpe.py` 复刻的 llama.cpp 口径），并按 sim 的 `--ce` 输出
`ntok` 复核（不一致直接报错）。

用法:
  python bench/latency_synth.py --model 0.5Bi=path.gguf --ctxs 1-20 --ks 2-9 \
      --n 12 --rounds 3 --devs cpu
输出: eval/latency_synth_<tag>.tsv / .txt
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from gguf_bpe import BPE

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
SIM = {"cpu": PROJ / "cpp/build_sim/Release/sim_rerank.exe",
       "gpu": PROJ / "cpp/build_sim/Release/sim_rerank_gpu.exe"}
ANCH = PROJ / "cohort_big_60k/anchors.jsonl"


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


def pick_cands(bpe, need, anchors):
    """挑 need 个候选词：每个恰好 3 token、首 token 互不相同。"""
    out, heads, seen = [], set(), set()
    for a in anchors:
        for w in a["win"]:
            if w in seen or len(w) < 2:
                continue
            seen.add(w)
            ps = bpe.pieces(w)
            if len(ps) != 3 or ps[0] in heads:
                continue
            heads.add(ps[0])
            out.append(w)
            if len(out) >= need:
                return out
    raise SystemExit(f"只找到 {len(out)} 个 3-token 候选（需 {need}）")


def build_input(ctx_text, cands_by_k, ctxs, ks, n):
    """→ (lines, cells)；每个 cell = n 行同形窗口。"""
    lines, cells = [], []
    for ctx in ctxs:
        for k in ks:
            cells.append((ctx, k, len(lines), n))
            for _ in range(n):
                lines.append(json.dumps({"context": ctx_text, "ctx": ctx,
                                         "cands": cands_by_k[k]},
                                        ensure_ascii=False))
    return lines, cells


def run_round(sim, model, lines, ngl, metric="total", emit_ce=True):
    inp = "\n".join(lines) + "\n"
    env = dict(os.environ)
    if ngl == 0:
        env["SIM_NO_CE_PIPE"] = "1"
    fd, sp = tempfile.mkstemp(suffix=".lat", prefix="simlat_")
    os.close(fd)
    env["SIM_TIMES" if metric == "total" else
        ("SIM_PREP" if metric == "prep" else "SIM_STEP23")] = sp
    cmd = [str(sim), str(model)]
    if emit_ce:
        cmd.append("--ce")
    cmd += ["--ngl", str(ngl)]
    p = subprocess.run(cmd, input=inp.encode("utf-8"),
                       capture_output=True, env=env)
    try:
        vals = [float(x) for x in open(sp, encoding="utf-8").read().split()]
    finally:
        try:
            os.unlink(sp)
        except OSError:
            pass
    if len(vals) != len(lines):
        raise RuntimeError(f"{sim.name}: 计时 {len(vals)} != {len(lines)}；"
                           f"{p.stderr.decode('utf-8','replace')[-400:]}")
    return vals


def verify_ntok(sim, model, cands, ctx_text, ngl):
    """用 sim 的 --ce 输出复核：每个候选 ntok 必须都是 3。"""
    line = json.dumps({"context": ctx_text, "cands": cands},
                      ensure_ascii=False) + "\n"
    p = subprocess.run([str(sim), str(model), "--ce", "--ngl", str(ngl)],
                       input=line.encode("utf-8"), capture_output=True)
    out = json.loads(p.stdout.decode("utf-8", "replace").strip().split("\n")[0])
    nt = out.get("ntok") or {}
    bad = {w: nt.get(w) for w in cands if nt.get(w) != 3}
    return bad, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="名称=gguf 路径")
    ap.add_argument("--ctxs", default="1-20")
    ap.add_argument("--ks", default="2-9")
    ap.add_argument("--n", type=int, default=12, help="每格窗数")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--devs", default="cpu")
    ap.add_argument("--metric", default="total",
                    choices=["total", "step23", "prep"],
                    help="total = 整窗全程（预填充 + 两次 decode + CE 计算）；"
                         "step23 = 只算候选段两次 decode；prep = 只算 ctx 预填充")
    ap.add_argument("--no-ce", action="store_true",
                    help="不传 --ce（不计算 CE；仅排查用）")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name, path = a.model.split("=", 1)
    name = name.strip()
    ctxs = parse_range(a.ctxs)
    ks = parse_range(a.ks)
    devs = [d.strip() for d in a.devs.split(",") if d.strip()]

    anchors = [json.loads(l) for l in open(ANCH, encoding="utf-8")]
    # 固定上文取**足够长**的锚点（60k 链首窗可能只有 1 字，无法截到 ctx20）
    ctx_text = next(a["ctx"] for a in anchors if len(a["ctx"]) >= 120)
    bpe = BPE(path)
    cands = pick_cands(bpe, max(ks), anchors)
    ck = {k: cands[:k] for k in ks}
    _bad, nt = verify_ntok(SIM["cpu"], path, ck[max(ks)], ctx_text, 0)
    if _bad:
        raise SystemExit(f"候选 token 数不是 3：{_bad}")
    print(f"{name}: {len(cands)} 个 3-token 候选（首 token 互不相同），"
          f"ctx {len(ctxs)} 档 × k {len(ks)} 档 × {a.n} 窗 × {a.rounds} 轮",
          flush=True)

    lines, cells = build_input(ctx_text, ck, ctxs, ks, a.n)
    tag = a.tag or name
    out = corpus_std.EVAL / f"latency_synth_{tag}.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("model\tctx\tk\tdev\tp50\tp90\tmax\tmin\tn\trounds\n")
    rows = []
    for dev in devs:
        allv = []
        for r in range(a.rounds):
            vals = run_round(SIM[dev], path, lines, 99 if dev == "gpu" else 0,
                             a.metric, not a.no_ce)
            allv.append(vals)
            print(f"  {dev} round {r+1}/{a.rounds} 完成", flush=True)
            # 逐窗（=逐位置）中位数跨轮聚合：不用 min（min 会系统性偏低）
        for ctx, k, off, n in cells:
            seg = []
            for vals in allv:
                seg.extend(vals[off:off + n])
            s = sorted(seg)
            p50 = statistics.median(seg)
            p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
            rows.append((name, ctx, k, dev, p50, p90, max(seg), min(seg)))
        print(f"  {dev} 完成", flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for nm, ctx, k, dev, p50, p90, mx, mn in rows:
            f.write(f"{nm}\t{ctx}\t{k}\t{dev}\t{p50:.2f}\t{p90:.2f}\t"
                    f"{mx:.2f}\t{mn:.2f}\t{a.n}\t{a.rounds}\n")
    label = {"total": "整窗全程（预填充 + 两次 decode + CE 计算）",
             "prep": "ctx 预填充（step1）",
             "step23": "候选段两次 decode（预解码不计）"}[a.metric]
    L = [f"{name} · {label} · k 个 3-token 候选（首 token 互不相同）"]
    for dev in devs:
        L.append(f"\n[{dev}] 中位 ms（行=ctx, 列=k）")
        L.append("  ctx" + "".join(f"{k:>8}" for k in ks))
        for ctx in ctxs:
            row = "".join(f"{r[4]:>8.1f}" for r in rows
                          if r[1] == ctx and r[3] == dev)
            L.append(f"{ctx:>5}" + row)
    txt = "\n".join(L)
    (corpus_std.EVAL / f"latency_synth_{tag}.txt").write_text(txt + "\n",
                                                             encoding="utf-8")
    print(txt)
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
