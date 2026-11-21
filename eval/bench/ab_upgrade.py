#!/usr/bin/env python3
"""llama.cpp 镜像升级 A/B 对拍统一入口（2026-11-18 子目录重组合并，2 脚本 → 1）。

吸收并取代（同一工作流的两半）:
  ab_llama_upgrade.py      → --phase run     新库 sim 跑 2000 窗锚点 → 与金标
                             cohort_big/ce（钉死旧库 86b94708 产出）逐候选对拍:
                             CE 分量 max|Δ|≤0.004 门禁 + 首选翻转（门禁 0）+ 吞吐
  rerate_impact_upgrade.py → --phase impact  两份 CE（86b 基线 vs 新库, 各自
                             --phase run --out 产出）全融合回放对比:
                             选重率/干预率/逐窗最终首选翻转（改对/改错/换错）

术语（2026-11-15 用户定案）: 选重率 = 错误率 = 1−正确率; 干预率 = 最终首选 ≠
引擎词典序首选。kCeW=(1,1.10,0.60)（0.5B 重标定）, β=14, k=5（生产）。
零改动 A/B: 同锚点同特征状态（samples.jsonl 真实 eff_win; uni 侧两边同为 0,
不影响新旧差值）; sim 跑批走 CPU 静态链逐窗可信路径。

用法:
  python bench/ab_upgrade.py                          # run + impact 全链
  python bench/ab_upgrade.py --phase run --sim <新库sim> --out ab_upgrade_new.jsonl
  python bench/ab_upgrade.py --phase impact           # 需先有 base/new 两份 jsonl
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
SIM = PROJ / "cpp" / "build_sim" / "Release" / "sim_rerank.exe"
ANCHORS = PROJ / "cohort_big" / "anchors.jsonl"
GOLD = PROJ / "cohort_big" / "ce" / "results.jsonl"
KW = (1.0, 1.10, 0.60)  # 0.5B 重标定 kCeW（翻转判据用, 与生产一致）
BETA = 14.0
K = 5  # max_candidates 生产值


def wsum(ce3):
    """kCeW 加权和（None/负值分量跳过）= 融合 CE 项; 协议 CE 分 = −wsum。"""
    s = 0.0
    for k, v in zip(KW, ce3):
        if v is not None and v >= 0:
            s += k * v
    return s


def load_jsonl(p, n=None):
    out = []
    with open(p, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if n is not None and i >= n:
                break
            if l.strip():
                out.append(json.loads(l))
    return out


# ─────────────────────────── run（原 ab_llama_upgrade）───────────────────────

def phase_run(a):
    ngl_cmd = ["--ngl", a.ngl] if a.ngl else []

    gold = {}
    with open(GOLD, encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
                gold[r["sid"]] = r
            except Exception:
                pass

    anchors = []
    with open(ANCHORS, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= a.n:
                break
            anchors.append(json.loads(l))
    inp = "\n".join(json.dumps({"context": x["ctx"], "cands": x["win"]},
                               ensure_ascii=False) for x in anchors) + "\n"

    sim = Path(a.sim)
    env = {**os.environ,
           "GGUF_MODEL": "d:/gguf_models/Qwen3.5-0.8B-Q4_K_M.gguf"}
    t0 = time.time()
    p = subprocess.run([str(sim), "--ce"] + ngl_cmd, input=inp.encode("utf-8"),
                       capture_output=True, env=env, cwd=str(sim.parent))
    dt = time.time() - t0
    err = p.stderr.decode("utf-8", "replace")
    (PROJ / a.out).write_bytes(p.stdout)
    (PROJ / (a.out + ".stderr.log")).write_text(err, encoding="utf-8")
    recs = [json.loads(l) for l in (PROJ / a.out)
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    # CPU 版 sim_rerank 输出无 sid 字段 → 按行序配对（gold 与 anchors 同序）
    n_cmp = n_diff = 0
    maxd = 0.0
    dsum = 0.0
    n_win_flip = 0
    n_win_cmp = 0
    flip_examples = []
    for i, r in enumerate(recs):
        g = gold.get(r.get("sid")) if r.get("sid") else (
            gold.get(anchors[i]["sid"]) if i < len(anchors) else None)
        if not g:
            continue
        n_win_cmp += 1
        cand_diff = False
        for w, v in r.get("ce", {}).items():
            gv = g.get("ce", {}).get(w)
            if gv is None:
                continue
            n_cmp += 1
            new3 = [x if x is not None and x >= 0 else None for x in v]
            old3 = [x if x is not None and x >= 0 else None for x in gv]
            for x, y in zip(new3, old3):
                if x is None or y is None:
                    continue
                d = abs(x - y)
                if d > maxd:
                    maxd = d
                dsum += d
                if x != y:
                    n_diff += 1
                    cand_diff = True
        # 首选翻转: 加权和 argmin（等价 argmax score）
        words = [w for w in r.get("ce", {}) if w in g.get("ce", {})]
        if len(words) >= 2:
            wn = min(words, key=lambda w: wsum(r["ce"][w]))
            wo = min(words, key=lambda w: wsum(g["ce"][w]))
            if wn != wo:
                n_win_flip += 1
                if len(flip_examples) < 8:
                    sid = r.get("sid") or (anchors[i]["sid"] if i < len(anchors) else "?")
                    flip_examples.append(
                        f"{sid[:8]} {wo}->{wn} "
                        f"new={wsum(r['ce'][wn]):.4f} old={wsum(g['ce'][wo]):.4f}")

    print(f"窗口: {len(recs)} 跑, 与金标可对照 {n_win_cmp}")
    print(f"CE 分量对: {n_cmp}, 值不同 {n_diff}, max|Δ| = {maxd:.4e} nats, "
          f"mean|Δ| = {dsum / max(n_cmp, 1):.4e}")
    print(f"首选翻转: {n_win_flip}/{n_win_cmp}")
    for s in flip_examples:
        print(f"  flip {s}")
    print(f"吞吐: {dt:.1f}s = {a.n / dt:.1f} 窗/s")
    gate = "PASS" if (maxd <= 0.004 and n_win_flip == 0) else "FAIL"
    print(f"门禁: max|Δ|≤0.004 且零翻转 → {gate}")


# ─────────────────────────── impact（原 rerate_impact_upgrade）───────────────

def replay(win, eff, ce_map):
    """→ 最终首选词。win = 词典序候选（含目标）; eff = word→eff; ce_map = word→ce3。"""
    s1 = {w: BETA * math.log(1.0 + eff.get(w, 0.0)) for w in win}
    # 层1 稳定降序选窗（同分保词典序）
    top = sorted(win, key=lambda w: -s1[w])[:K]
    fused = {}
    for w in top:
        c = ce_map.get(w)
        fused[w] = s1[w] - (wsum(c) if c is not None else 0.0)
    # 层2 融合 argmax（同分稳定 = 先到先得, 与层1 序一致）
    best, bv = None, None
    for w in top:
        if bv is None or fused[w] > bv:
            best, bv = w, fused[w]
    return best


def phase_impact(a):
    anchors = load_jsonl(ANCHORS, a.n)
    eff_by_sid = {r["sid"]: r.get("eff_win", {})
                  for r in load_jsonl(PROJ / "cohort_big" / "samples.jsonl")}
    ce_old = load_jsonl(PROJ / "ab_upgrade_basecpu.jsonl", a.n)
    ce_new = load_jsonl(PROJ / a.out, a.n)
    assert len(anchors) == len(ce_old) == len(ce_new), \
        f"{len(anchors)}/{len(ce_old)}/{len(ce_new)}"

    n = n_err_o = n_err_n = n_int_o = n_int_n = 0
    flip = fixed = broken = 0
    flip_examples = []
    for anc, co, cn in zip(anchors, ce_old, ce_new):
        win = anc["win"]
        tgt = anc["w"]
        eff = eff_by_sid.get(anc["sid"], {})
        fo = replay(win, eff, co.get("ce", {}))
        fn = replay(win, eff, cn.get("ce", {}))
        n += 1
        dict_first = win[0]
        if fo != dict_first:
            n_int_o += 1
        if fn != dict_first:
            n_int_n += 1
        eo, en = fo != tgt, fn != tgt
        n_err_o += eo
        n_err_n += en
        if fo != fn:
            flip += 1
            if eo and not en:
                fixed += 1
            elif en and not eo:
                broken += 1
            elif eo and en:
                pass  # 换错（两态都错, 换了错的错）
            if len(flip_examples) < 6:
                flip_examples.append(
                    f"{anc['sid'][:8]} 目标={tgt} 词典首={dict_first}: "
                    f"old→{fo} new→{fn}")

    print(f"窗口: {n}")
    print(f"{'口径':<16} {'选重率(=错误率)':>14} {'干预率':>10}")
    print(f"{'86b 基线':<16} {n_err_o / n * 100:>13.3f}% "
          f"{n_int_o / n * 100:>9.3f}%")
    print(f"{'9f26fe11c':<16} {n_err_n / n * 100:>13.3f}% "
          f"{n_int_n / n * 100:>9.3f}%")
    print(f"差值:        选重率 {(n_err_n - n_err_o) / n * 100:+.3f}pp  "
          f"干预率 {(n_int_n - n_int_o) / n * 100:+.3f}pp")
    print(f"\n最终首选翻转: {flip}/{n} ({flip / n * 100:.3f}%)  "
          f"改对 {fixed} 改错 {broken} 换错 {flip - fixed - broken}")
    for s in flip_examples:
        print(f"  {s}")


def main():
    ap = argparse.ArgumentParser(
        description="llama.cpp 升级 A/B（run=对拍门禁 / impact=选重率回放）")
    ap.add_argument("--phase", choices=["run", "impact", "both"],
                    default="both")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sim", default=str(SIM), help="sim_rerank exe 路径（run）")
    ap.add_argument("--out", default="ab_upgrade_new.jsonl",
                    help="新库 CE 输出（PROJ 相对; run 写 / impact 读）")
    ap.add_argument("--ngl", default="")
    a = ap.parse_args()
    if a.phase in ("run", "both"):
        phase_run(a)
    if a.phase in ("impact", "both"):
        phase_impact(a)


if __name__ == "__main__":
    sys.exit(main())
