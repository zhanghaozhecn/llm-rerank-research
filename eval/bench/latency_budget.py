#!/usr/bin/env python3
"""合成最坏情况延迟网格：模型 × CE 预算 B（4–18），一次进程跑完。

口径与 `latency_synth.py` 的 ③ 候选段一致（2026-11-18 定案），只是把成本
单位从「候选数 k」换成**生产口径 CE2+CE3 总预算 B**（2026-11-26）：

  - 1-token 候选免费（CE1 走 ctx logits 共享归一化），不进预算；
  - 3-token 候选花 2 预算（step2 一行 + step3 一行）；
  - 2-token 候选花 1 预算（只有 step2 一行）；
  - 只计候选段两次 decode（预解码不计）。

最坏情况构造（与旧 k 网格可对照）：
  - B 偶数：B/2 个 3-token 候选，首 token 两两不同、次 token 两两不同
    → step2/step3 各 B/2 行；等价旧 `k = B/2`；
  - B 奇数：再补 1 个 2-token 候选（首/次 token 都与 3-token 组不同）
    → step2 (B+1)/2 行、step3 (B−1)/2 行，总行数 = B。
  每个候选都按 sim `--ce` 的 `ntok` 复核，不符直接报错。

用法:
  python bench/latency_budget.py --model 0.6B=path.gguf --budgets 4-18 \
      --n 12 --rounds 3 --tag 06bb_q4km_budget
输出: eval/latency_budget_<tag>.tsv / .txt
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import corpus_std
from gguf_bpe import BPE
import latency_synth as LS

sys.stdout.reconfigure(encoding="utf-8")
ANCH = corpus_std.PROJ / "cohort_big_60k/anchors.jsonl"
CTX = 20


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


def pick_sets(bpe, anchors, max_b):
    """→ (three, two)：three 最多 ceil(max_b/2) 个 3-token（首/次 token 互不
    相同），two 一个 2-token（首/次 token 与 three 组都不撞）。"""
    n3 = (max_b + 1) // 2
    three, heads, seconds, seen = [], set(), set(), set()
    for a in anchors:
        for w in a["win"]:
            if len(three) >= n3:
                break
            if w in seen or len(w) < 2:
                continue
            seen.add(w)
            ps = bpe.pieces(w)
            if (len(ps) == 3 and ps[0] not in heads
                    and ps[1] not in seconds):
                heads.add(ps[0])
                seconds.add(ps[1])
                three.append(w)
        if len(three) >= n3:
            break
    if len(three) < n3:
        raise SystemExit(f"只找到 {len(three)} 个合格 3-token 候选（需 {n3}）")
    two = None
    for a in anchors:
        for w in a["win"]:
            if len(w) < 2:
                continue
            ps = bpe.pieces(w)
            if (len(ps) == 2 and ps[0] not in heads
                    and ps[1] not in seconds):
                two = w
                break
        if two:
            break
    if two is None:
        raise SystemExit("找不到首/次 token 都不撞的 2-token 候选")
    return three, two


def verify_ntok(sim, path, cands, ctx_text, want):
    line = json.dumps({"context": ctx_text, "cands": cands},
                      ensure_ascii=False) + "\n"
    p = subprocess.run([str(sim), str(path), "--ce", "--ngl", "0"],
                       input=line.encode("utf-8"), capture_output=True)
    out = json.loads(p.stdout.decode("utf-8", "replace").strip()
                     .split("\n")[0])
    nt = out.get("ntok") or {}
    bad = {w: nt.get(w) for w in cands if nt.get(w) != want[w]}
    if bad:
        raise SystemExit(f"候选 token 数不符：{bad}")
    return nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="名称=gguf 路径")
    ap.add_argument("--budgets", default="4-18")
    ap.add_argument("--n", type=int, default=12, help="每格窗数")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name, path = a.model.split("=", 1)
    name = name.strip()
    budgets = parse_range(a.budgets)

    anchors = [json.loads(l) for l in open(ANCH, encoding="utf-8")]
    ctx_text = next(a_["ctx"] for a_ in anchors if len(a_["ctx"]) >= 120)
    bpe = BPE(path)
    three, two = pick_sets(bpe, anchors, max(budgets))

    ck, want, rows23 = {}, {}, {}
    for b in budgets:
        k3 = b // 2
        cands = three[:k3] + ([two] if b % 2 else [])
        ck[b] = cands
        w = {x: 3 for x in three[:k3]}
        if b % 2:
            w[two] = 2
        want.update(w)
        rows23[b] = (len(cands), k3)
    verify_ntok(LS.SIM["cpu"], path,
                [w for w in three + [two] if w in want], ctx_text, want)
    print(f"{name}: 预算 {budgets[0]}–{budgets[-1]} × {a.n} 窗 × "
          f"{a.rounds} 轮（ctx {CTX}）", flush=True)

    lines, cells = LS.build_input(ctx_text, ck, [CTX], budgets, a.n)
    out = corpus_std.EVAL / f"latency_budget_{a.tag or name}.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("model\tctx\tbudget\tcands\ts3\tdev\tp50\tp90\tmax\tmin\t"
                "n\trounds\n")
    allv = []
    for r in range(a.rounds):
        vals = LS.run_round(LS.SIM["cpu"], path, lines, 0, "step23", True)
        allv.append(vals)
        print(f"  cpu round {r+1}/{a.rounds} 完成", flush=True)
    rows = []
    for ctx, b, off, n in cells:
        seg = []
        for vals in allv:
            seg.extend(vals[off:off + n])
        s = sorted(seg)
        p50 = statistics.median(seg)
        p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
        nc, s3 = rows23[b]
        rows.append((name, ctx, b, nc, s3, p50, p90, max(seg), min(seg)))
    with open(out, "a", encoding="utf-8") as f:
        for nm, ctx, b, nc, s3, p50, p90, mx, mn in rows:
            f.write(f"{nm}\t{ctx}\t{b}\t{nc}\t{s3}\tcpu\t{p50:.2f}\t"
                    f"{p90:.2f}\t{mx:.2f}\t{mn:.2f}\t{a.n}\t{a.rounds}\n")
    L = [f"{name} · 候选段两次 decode（预解码不计）· CE 预算 B（最坏情况）",
         "  B = CE2+CE3 总 token 预算；行数 = step2 + step3 = B", "",
         "  B   候选数(3t+2t)  step2  step3   p50 ms"]
    for nm, ctx, b, nc, s3, p50, p90, mx, mn in rows:
        L.append(f"{b:>3}   {nc:>2}（{s3}+{nc - s3}）      {nc:>2}     "
                 f"{s3:>2}   {p50:>7.1f}")
    txt = "\n".join(L)
    (corpus_std.EVAL / f"latency_budget_{a.tag or name}.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
