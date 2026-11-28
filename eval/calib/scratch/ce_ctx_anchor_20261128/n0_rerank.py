#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真 n=0（文档起始）：「。」锚 CE 重排 vs 不用 CE（freq_only）——生产口径选重率。

用户 2026-11-28 追问：选重率怎么会 75%？——首版窗集 = 最难切片（句首+码≥6+
冷账本+真值允许补全词）。本版对齐生产口径：

  * 生产选重率定义：P(当前编码首选 ≠ 实际词)，实际词 = 用户提交的词——
    它必然在候选列表里（只能选显示过的）。模拟侧对应 = **条件在真值∈池**
    的窗上统计；真值不在池的窗生产语义下不存在，剔除并报告。
  * 真值 = 句子完整词前缀（segs 逐词累计音节字母 ≤ 码长，取池内最长的
    句子一致前缀）——码覆盖不到第一个完整词的格 = 生产不会在那时刻提交，
    自然剔除（no_commit）。
  * 码长切片 2-3 / 4-5 / 6-11 / 12+：短码切片 ≈ 日常词级打字，选重率应落
    在熟悉量级（词典序 13.65% / 现役整链 7.86% 的参照带）。
  * 两臂（冷账本，与全 cohort 口径一致）：
      freq_only（现行生产）无 CE → 全候选层1 恒 0 → 序 = 引擎原序，首选 = 池[0]；
      anchor（反事实）ctx="。" 裸「。」CE（= 守卫扩到 n≤1 后等效上下文），
      融合 = −CE 加权和（pool[:10]，≈ 预算 12 现实覆盖）；pool[10:] 无 CE →
      层1 + 窗内最低 CE 分（生产压制规则，llm_filter.cc 窗底定案）→ 沉底。
      **必须 SIM_CTX_ANCHOR=0 裸跑**（==1 锚会把「。」再垫成「。。」）且落
      临时目录（主缓存同 sid 行是「。。」口径，写回会顶掉正值）。

数据 = luna+big 全部 m=0 句首窗：L∈[2,5] 全收（首版被 ≥6 过滤排除的新窗）
+ ≥6 沿用首版同一 12,000 采样（scratch 缓存全复用）。硬子集三臂版（含 bare
臂与 n=1 参考臂）结果存档 n0_rerank_hard12k.txt。

跑法（研究库根目录）：
  python eval/calib/scratch/ce_ctx_anchor_20261128/n0_rerank.py            # 干跑
  ... --apply   # 补算缺口（单作业 workers=8）+ 切片评测
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # eval/calib
sys.path.insert(0, str(HERE.parents[2] / "lib"))  # eval/lib
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L  # noqa: E402

KCE = (1.0, 1.10, 0.60)
SCRATCH = HERE / "n0_bare_punct_ce"   # 裸「。」CE 临时目录（不入共享缓存）
SLICES = ((6, "2-3"), (11, "4-5"), (5, "6-11"))  # (上界次序标记) —— 见 slice_of
CAP_GE6 = 12000


def log(msg):
    print(f"[n0] {msg}", flush=True)


def slice_of(lc):
    return "2-3" if lc <= 3 else "4-5" if lc <= 5 else "6-11" if lc <= 11 else "12+"


def fused_of(c3):
    """生产 CE 加权和（state_of 同构）：缺 token 不计，哨兵 -1e8 跳过。→ −CE"""
    if not c3 or not c3[0]:
        return None
    s = 0.0
    for k in range(min(3, len(c3[0]))):
        x = c3[0][k]
        if x is not None and x > -1e8:
            s += KCE[k] * x
    return -s


def truth_of(s, poolset, lc):
    """句子完整词前缀：逐词累计音节字母 ≤ 码长，取池内最长句子一致前缀。"""
    segs, syls = s.get("segs"), s.get("syls")
    if not segs or not syls:
        return None
    ci = acc = 0
    txt = ""
    truth = None
    for word in segs:
        k = sum(len(x) for x in syls[ci:ci + len(word)])
        if acc + k > lc:
            break
        acc += k
        txt += word
        if txt in poolset:
            truth = txt
        ci += len(word)
    return truth


def mcnemar_p(b, c):
    """不一致对 b vs c 的显著性 p（双侧正态近似；精确二项 k>1000 溢出）。"""
    k = b + c
    if not k:
        return 1.0
    return math.erfc(abs(b - c) / math.sqrt(k) / math.sqrt(2))


def boot_ci(pa, pb, it=2000):
    """配对差值 (pb−pa) 的 bootstrap 95% CI（pp）。"""
    n = len(pa)
    if not n:
        return (0.0, 0.0)
    diffs = [y - x for x, y in zip(pa, pb)]
    rng = random.Random(20261128)
    ds = []
    for _ in range(it):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        ds.append(100.0 * s / n)
    ds.sort()
    return (ds[int(0.025 * it)], ds[int(0.975 * it)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    # ── 窗集：m=0 全体；≥6 部分复刻首版选句（同序同过滤取前 12000）──
    sents = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
        for x in L.jsonl_iter(d / "sentences.jsonl"):
            sents[x["sid"]] = x
    wins_short, wins_ge6 = [], []
    for d in (L.OUT, L.EVAL / "prefix_big"):
        for w in L.jsonl_iter(d / "windows.jsonl"):
            if w.get("dyn") or w["m"] != 0 or len(w["code"]) < 2:
                continue
            (wins_short if len(w["code"]) <= 5 else wins_ge6).append(w)
    ge6_pick = []
    for w in wins_ge6:
        s = sents.get(w["sid"])
        if s and len(s.get("before") or "") >= 20:
            ge6_pick.append(w)
    if len(ge6_pick) > CAP_GE6:
        ge6_pick = ge6_pick[::max(1, len(ge6_pick) // CAP_GE6)][:CAP_GE6]
    picked = wins_short + ge6_pick
    log(f"窗集 短码(2-5) {len(wins_short)} + ≥6 复刻首版 {len(ge6_pick)}"
        f" = {len(picked)}")

    merged = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
        merged.update(json.load(open(d / "cands.json", encoding="utf-8")))

    # ── 真值（池内完整词前缀）+ 缺口统计 ──
    pop = []          # (w, s, texts, truth)
    miss = defaultdict(int)
    sl_miss = defaultdict(int)
    for w in picked:
        s = sents.get(w["sid"])
        cands = merged.get(w["code"], [])[:L.K_COLLECT]
        if len(cands) < 2:
            miss["pool_lt2"] += 1
            continue
        texts = [c[0] for c in cands]
        truth = truth_of(s, set(texts), len(w["code"]))
        if truth is None:
            miss["no_commit"] += 1          # 码覆盖不到完整词 = 无提交时刻
            sl_miss[slice_of(len(w["code"]))] += 1
            continue
        pop.append((w, s, texts, truth))
    log(f"真值可提交窗 {len(pop)}；剔除 " +
        "；".join(f"{k}={v}" for k, v in sorted(miss.items())) +
        "（no_commit 分片 " +
        "；".join(f"{k}={v}" for k, v in sorted(sl_miss.items())) + "）")

    # ── 裸「。」缺口（bak = 锚前裸算正值；scratch = 本次补算）──
    bak = {**L.load_ce(L.CE_RESULTS_EXT.with_suffix(".jsonl.bak_preanchor")),
           **L.load_ce(L.CE_RESULTS.with_suffix(".jsonl.bak_preanchor"))}
    have = {**bak, **L.load_ce(SCRATCH / "results.jsonl")}
    todo, pool_miss = [], 0
    for w, s, texts, truth in pop:
        todo += [(ln, sid) for ln, sid in L.ce_lines("。", texts[:10])
                 if sid not in have]
    log(f"裸「。」需补算 {len(todo)} 行（去重 sid "
        f"{len({sid for _, sid in todo})}）")
    if a.apply and todo:
        import os
        os.environ["SIM_CTX_ANCHOR"] = "0"   # 「。」必须裸跑（==1 锚垫成「。。」）
        SCRATCH.mkdir(parents=True, exist_ok=True)
        L.run_ce(todo, workers=8, out_dir=SCRATCH)
        log(f"补算完成 → {SCRATCH.name}")
    punct = {**bak, **L.load_ce(SCRATCH / "results.jsonl")}
    if a.apply and todo:
        got = sum(1 for w, s, texts, truth in pop
                  for ln, sid in L.ce_lines("。", texts[:10])
                  if sid in punct)
        want = sum(len(L.ce_lines("。", texts[:10])) for w, s, texts, truth in pop)
        log(f"覆盖核验 {got}/{want}")
        if got != want:
            log("FATAL: 覆盖不全，断点续跑重试 --apply")
            return 1
    if not a.apply:
        log("dry-run（--apply 补算 + 评测）")
        return 0

    # ── 两臂评测（条件：真值∈池 已由 pop 保证；窗=池前 6；融合规则同生产）──
    stat = defaultdict(lambda: dict(n=0, okA=0, okC=0))
    rec = defaultdict(list)
    miss2 = defaultdict(int)
    for w, s, texts, truth in pop:
        sl = slice_of(len(w["code"]))
        lns = L.ce_lines("。", texts[:10])
        rows = {}
        for ln, sid in lns:
            rr = punct.get(sid)
            if rr:
                for t in ln["cands"]:
                    if t in rr and rr[t][0] is not None:
                        rows[t] = rr[t]
        f10 = {t: fused_of(rows.get(t)) for t in texts[:10]}
        if any(v is None for v in f10.values()):
            miss2[f"{sl}_ce_missing"] += 1
            continue
        # freq_only：层1 恒 0 → 引擎原序，首选 = 池[0]
        okA = texts[0] == truth
        # anchor：[:10] 按 −CE 加权和；[10:] = 层1 + 窗内最低 CE 分（沉底）；
        # 稳定序 = 并列保池序（生产 stable_sort 同构）
        min_valid = min(f10.values())
        def fused_full(i, t):
            if i < 10:
                return f10[t]
            return min_valid
        best_i, best_v = 0, fused_full(0, texts[0])
        for i in range(1, len(texts)):
            v = fused_full(i, texts[i])
            if v > best_v:
                best_i, best_v = i, v
        okC = texts[best_i] == truth
        e = stat[sl]
        e["n"] += 1
        e["okA"] += okA
        e["okC"] += okC
        rec[sl].append((okA, okC))

    # ── 汇总 ──
    out = []

    def emit(x=""):
        print(x, flush=True)
        out.append(x)

    emit("真 n=0 两臂选重率（生产口径：条件在真值∈池；冷账本；窗=池前6）")
    emit("评测中再剔除（CE 缺值）：" +
         ("；".join(f"{k}={v}" for k, v in sorted(miss2.items())) or "0"))
    emit("")
    emit(f"{'码长':<6}{'窗N':>7}{'freq_only选重率':>16}{'anchor选重率':>14}"
         f"{'Δ选重率(pp)':>14}  {'95%CI':>16}  翻转(对←错/错←对)  p")
    order = ["2-3", "4-5", "6-11", "12+"]
    alloka, allokc = [], []
    for sl in order:
        e = stat.get(sl)
        if not e or not e["n"]:
            emit(f"{sl:<6}    —（无有效窗）")
            continue
        n = e["n"]
        errA = 100.0 * (n - e["okA"]) / n
        errC = 100.0 * (n - e["okC"]) / n
        pa = [r[0] for r in rec[sl]]
        pb = [r[1] for r in rec[sl]]
        lo, hi = boot_ci(pa, pb)
        b = sum(1 for x, y in zip(pa, pb) if not x and y)
        c = sum(1 for x, y in zip(pa, pb) if x and not y)
        emit(f"{sl:<6}{n:>7}{errA:>15.2f}%{errC:>13.2f}%"
             f"{errC-errA:>+13.2f}  [{-(hi):+6.2f},{-(lo):+6.2f}]"
             f"  {b:>4}/{c:<4}  {mcnemar_p(b, c):.1e}")
        alloka += pa
        allokc += pb
    if alloka:
        n = len(alloka)
        b = sum(1 for x, y in zip(alloka, allokc) if not x and y)
        c = sum(1 for x, y in zip(alloka, allokc) if x and not y)
        lo, hi = boot_ci(alloka, allokc)
        errA = 100.0 * (n - sum(alloka)) / n
        errC = 100.0 * (n - sum(allokc)) / n
        emit(f"{'全体':<6}{n:>7}{errA:>15.2f}%{errC:>13.2f}%"
             f"{errC-errA:>+13.2f}  [{-(hi):+6.2f},{-(lo):+6.2f}]"
             f"  {b:>4}/{c:<4}  {mcnemar_p(b, c):.1e}")
    emit("")
    emit("Δ/CI 均为选重率变化（负 = 锚改善）。参照带（生产口径，正常上文窗）："
         "词典序 13.65% → 现役整链 7.86%（README 选重率节）——本模拟为冷账本 "
         "wiki 句首，绝对水平不可直接对齐，比 Δ。")
    emit("现网 nofreq 窗码长分布（luna+pdsp corpus，2,408 行）：2-3 = 0.1%，"
         "4-5 = 18.0%，6-11 = 32.0%，12+ = 49.9% → 按生产占比加权 Δ ≈ "
         "0.180×(+6.90) + 0.320×(−6.15) + 0.499×(−8.98) + 0.001×(+34.72)"
         " ≈ −5.2pp（该人群净改善）。")
    (HERE / "n0_rerank.txt").write_text("\n".join(out) + "\n",
                                        encoding="utf-8")
    log("报告 → n0_rerank.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
