#!/usr/bin/env python3
"""层1 选窗特征研究（2026-11-09; 纯 CPU——CE 全部预计算, 未用 GPU）。

未及桶 = 层1 窗丢失（正确词没进 LLM 前k, 层2 无从挽回）+ 窗内选错（CE/融合
仍选错）。big@10万 未及 3.26pp vs 改错 1.01pp = 剩余误差最大的一块。

阶段 A（分解与现状）:
  - k 扫描（k=3..10, 层1 S1=(12,1)）: err/窗丢失/改错 → 层1 过滤的总代价
    （k=10 = 不过滤全量进 LLM）;
  - 参考: S1≡0（词典序）k=5 应对齐 4×4 纯CE 行（7.28/5.41/3.31）;
  - 窗丢失的 widx 词典位次直方图（集中在 5-6=特征可救, 散到 8-9=没救）。

阶段 B（特征网格, k=5, 层2 不变, S1 = 12·du + 1·lp1 + w·feat）:
  B0 层1 权重单独重标定 (β1, δ1)（现行直接抄了融合权重, 未为选窗标定）
  B1 词典位次 j（生产对应 = 引擎序信任; 每候选, 非码级）
  B2 语料 unigram logp（cohort_unigram_<chain>.json; 生产可静态表/词典权重）
  B3 词长（字符数）
  B4 上下文回声（候选词出现在前文 = 刚写过会再写; 生产=子串查, 零成本）
  B5 短τ eff（τ2=50 爆发分量, log1p; 生产=第二个衰减计数器）
设置: cohort_big@DEFAULT_ARM / small6w@DEFAULT_ARM / user×big
（2026-11-18 定案默认龄 3 万次提交; 历史报告为 10 万龄产物, 不再重跑）。
γ=0/基准行必须复现 4.26/3.44/2.02。报告: cohort_layer1_feat_report.txt
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_std
from corpus_std import MAX_CAND, FormulaReplay, normalize_ctx
from summary_4x4 import PROJ, cohort_matrices, user_matrices

TAU2 = 50.0
K_GRID = range(3, 11)
B0_GRID = [(6, 1), (12, 0), (12, 0.5), (12, 1), (12, 2), (24, 1)]
W_POS = (-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2)
W_LOGLIKE = (-1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2)
W_LEN = (-0.5, -0.25, 0, 0.25, 0.5)
W_ECHO = (0, 0.5, 1, 2, 4, 8)
W_EFF2 = (0, 2, 4, 8, 12, 16)


def layer_eval(S1, s_ce, widx, k):
    """→ (err, miss, brk) 逐窗布尔。miss = widx 不在层1 top-k。"""
    n = len(widx)
    order = np.argsort(-S1, axis=1, kind="stable")[:, :k]
    miss = ~np.any(order == widx[:, None], axis=1)
    S1k = np.take_along_axis(S1, order, axis=1)
    sct = np.take_along_axis(s_ce, order, axis=1)
    S2k = S1k + np.where(sct > -1e8, sct, 0.0)
    winner = order[np.arange(n), S2k.argmax(1)]
    err = winner != widx
    return err, miss, (widx == 0) & err


def triple(err, miss, brk):
    return (f"{100 * err.mean():6.2f} {100 * miss.mean():6.2f} "
            f"{100 * brk.mean():6.2f}")


def zeros_like_m(m):
    return np.zeros((m["n"], MAX_CAND))


def feat_pos(m):
    F = zeros_like_m(m)
    for i, ws in enumerate(m["wins"]):
        for j in range(len(ws)):
            F[i, j] = j
    return F


def feat_logp(m, chain):
    payload = json.loads((corpus_std.EVAL /
                          f"cohort_unigram_{chain}.json").read_text(
                              encoding="utf-8"))
    logp, unseen = payload["logp"], float(payload["unseen_logp"])
    F = zeros_like_m(m)
    for i, ws in enumerate(m["wins"]):
        for j, w in enumerate(ws):
            F[i, j] = logp.get(w, unseen)
    return F


def feat_len(m):
    F = zeros_like_m(m)
    for i, ws in enumerate(m["wins"]):
        for j, w in enumerate(ws):
            F[i, j] = len(w)
    return F


def feat_echo(m, ctxs):
    F = zeros_like_m(m)
    for i, ws in enumerate(m["wins"]):
        cn = normalize_ctx(ctxs[i])
        for j, w in enumerate(ws):
            if w in cn:
                F[i, j] = 1.0
    return F


def feat_eff2_cohort(m, chain):
    """短τ eff: 逐用户重放 streams（同 lp1 口径）, 锚点取事前 log1p(eff2)。"""
    by = defaultdict(list)
    for r in range(m["n"]):
        by[(int(m["ui"][r]), int(m["ci"][r]))].append(r)
    F = zeros_like_m(m)
    with open(PROJ / f"cohort_{chain}" / "streams.jsonl",
              encoding="utf-8") as f:
        for ui, line in enumerate(f):
            rp = FormulaReplay(TAU2)
            for ci, w in enumerate(json.loads(line)):
                for r in by.get((ui, ci), ()):
                    for j, x in enumerate(m["wins"][r]):
                        F[r, j] = math.log1p(rp.eff_of(x))
                rp.bump(w)
    return F


def feat_eff2_user(m):
    """真实用户链短τ eff: 常开语料词流重放（与 user_matrices 同过滤口径）。"""
    words_all = set(corpus_std.load_pdsp_words())
    by = defaultdict(list)
    for r in range(m["n"]):
        by[int(m["ci"][r])].append(r)
    F = zeros_like_m(m)
    rp = FormulaReplay(TAU2)
    ci = 0
    for w in corpus_std.iter_word_stream():
        if w not in words_all:
            continue
        for r in by.get(ci, ()):
            for j, x in enumerate(m["wins"][r]):
                F[r, j] = math.log1p(rp.eff_of(x))
        rp.bump(w)
        ci += 1
    return F


def ctxs_for_arm(chain, arm):
    out = []
    with open(PROJ / f"cohort_{chain}" / "anchors.jsonl", encoding="utf-8") as f:
        for l in f:
            a = json.loads(l)
            if a["arm"] == arm:
                out.append(a["ctx"])
    return out


def ctxs_for_user():
    return [json.loads(l)["ctx"] for l in
            open(PROJ / "user_anchors_std.jsonl", encoding="utf-8")]


def grid_feat(L, label, S1base, pad, F, w_grid, s_ce, widx, base_err):
    L(f"  [{label}] w: err%/窗丢%/改错%")
    for w in w_grid:
        S1 = S1base + w * np.where(pad, 0.0, F)
        err, miss, brk = layer_eval(S1, s_ce, widx, 5)
        d = 100 * (err.mean() - base_err)
        mark = "  <--" if d < -0.05 else ""
        L(f"    w={w:>6g}: {triple(err, miss, brk)}  Δ{d:+.2f}pp{mark}")


def analyze(name, m, chain, ctxs, eff2, lines):
    L = lines.append
    du, lp1, s_ce, widx, pad = (m["du"], m["lp1"], m["s_ce"],
                                m["widx"], m["pad"])
    n = m["n"]
    S1base = 12.0 * du + 1.0 * lp1
    err0, miss0, brk0 = layer_eval(S1base, s_ce, widx, 5)
    L(f"== {name} (n={n}) ==")
    L(f"  基准 (12,1) k=5: err {100 * err0.mean():.2f}% "
      f"窗丢 {100 * miss0.mean():.2f}% 改错 {100 * brk0.mean():.2f}%")
    # A: k 扫描
    L("  [A|k扫描] k: err%/窗丢%/改错%")
    for k in K_GRID:
        err, miss, brk = layer_eval(S1base, s_ce, widx, k)
        L(f"    k={k:>2}: {triple(err, miss, brk)}")
    err, miss, brk = layer_eval(np.where(pad, -1e9, 0.0), s_ce, widx, 5)
    L(f"    [ref S1≡0 词典序 k=5: {triple(err, miss, brk)}"
      f"  <-- 应对齐 4×4 纯CE 行]")
    # A: 窗丢失位次直方图
    order = np.argsort(-S1base, axis=1, kind="stable")[:, :5]
    miss = ~np.any(order == widx[:, None], axis=1)
    hist = np.zeros(MAX_CAND, dtype=int)
    for r in np.where(miss)[0]:
        hist[widx[r]] += 1
    L(f"  [A|窗丢失 widx 词典位次] " + " ".join(
        f"j{j}:{c}" for j, c in enumerate(hist) if c) +
      f"   （总 {miss.sum()}）")
    # B0: 层1 权重重标
    L("  [B0|层1权重 (β1,δ1)] : err%/窗丢%/改错%")
    for b1, d1 in B0_GRID:
        S1 = b1 * du + d1 * lp1
        err, miss, brk = layer_eval(S1, s_ce, widx, 5)
        d = 100 * (err.mean() - err0.mean())
        L(f"    ({b1:g},{d1:g}): {triple(err, miss, brk)}  Δ{d:+.2f}pp")
    # B1..B5
    grid_feat(L, "B1 词典位次 j", S1base, pad, feat_pos(m), W_POS,
              s_ce, widx, err0.mean())
    grid_feat(L, "B2 语料unigram logp", S1base, pad, feat_logp(m, chain),
              W_LOGLIKE, s_ce, widx, err0.mean())
    grid_feat(L, "B3 词长(字符)", S1base, pad, feat_len(m), W_LEN,
              s_ce, widx, err0.mean())
    grid_feat(L, "B4 上下文回声", S1base, pad, feat_echo(m, ctxs), W_ECHO,
              s_ce, widx, err0.mean())
    grid_feat(L, f"B5 短τeff(τ={TAU2:g},log1p)", S1base, pad, eff2, W_EFF2,
              s_ce, widx, err0.mean())
    L("")


def main():
    L = ["=" * 74,
         "层1 选窗特征研究 · 层2 不变 (β,δ)=(12,1) · 纯 CPU",
         "未及分解 = 窗丢失(层1没选进) + 窗内选错; 特征网格 S1 = 12du+lp1+w·feat",
         "=" * 74]
    m = user_matrices("big")
    eff2u = feat_eff2_user(m)
    print("user feats done", flush=True)
    analyze("user×big（真实用户单流）", m, "big", ctxs_for_user(), eff2u, L)
    del m, eff2u
    arm = corpus_std.DEFAULT_ARM
    m = cohort_matrices("big", {arm})[arm]
    eff2b = feat_eff2_cohort(m, "big")
    print("big feats done", flush=True)
    analyze(f"cohort_big @{arm}", m, "big", ctxs_for_arm("big", arm),
            eff2b, L)
    del m, eff2b
    m = cohort_matrices("small6w", {arm})[arm]
    eff2s = feat_eff2_cohort(m, "small6w")
    print("small6w feats done", flush=True)
    analyze(f"cohort_small6w @{arm}", m, "small6w",
            ctxs_for_arm("small6w", arm), eff2s, L)
    txt = "\n".join(L)
    (corpus_std.EVAL / "cohort_layer1_feat_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
