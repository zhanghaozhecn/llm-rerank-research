#!/usr/bin/env python3
"""单字/词码对评测合并入口（2026-11-18 子目录重组，5 脚本 → 1）。

吸收并取代（数字口径与报告文件名均不变）:
  test_sc.py                 → test      用户域单字 3 码重排测试（236 窗, Wilson CI）
  analyze_sc_final.py        → final     定案口径逐例分析（235 干净窗四象限 + B/C 全例）
  resample_percode_clean.py  → clean     词码对记账定案 · 干净样本集重统计（两域）
  inspect_word_percode.py    → wordflip  多字词域 (词,码) 分键 vs 词合并 配对翻转
  inspect_codes_behavior.py  → behavior  编码行为验证（简码必打简码/多音规模/分层重算）

共享引擎（原 5 份重复实现收拢于此）:
  - 词典全码表解析 load_dict_full()（w2codes + code_seq, 原抄 3 份）
  - 双键重放 replay_scenes（单字域, 场景制 + ci 计数器）/ replay_flat
    （多字词域, 扁平词流含回撤, ci 直接对位）（原抄 4 份）
  - 两层 k=5 融合终选 tl_pick / score_rows（原抄 6 份变体）
  - 资格剔除集 excl_sets（一码字/二码窗首选字/一二码简词）（原抄 2 份）
定案口径三开关全保留: 分键记账 percode、干净集剔除、无 CE 填充
  fill_mode 'mince'（定案: 窗内已算 CE 最低分）/ 's1fallback'（旧生产对照）。

数据: user_sc_samples.jsonl + user_sc_ce/（单字域）;
      user_samples_std.jsonl + user_anchors_std.jsonl + user_ce_work/（多字词域）。
报告: eval/<原报告名>.txt（统一落 eval 根, 与历史产物同目录）。
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from corpus_std import FormulaReplay, MAX_CAND
from summary_4x4 import K, build_ce_matrix

PROJ = corpus_std.PROJ
EVAL = corpus_std.EVAL
TRAIN_LOG = corpus_std.CORPUS_PATH          # 统一语料 v2（env RIME_LLM_CORPUS 可换）
TAU = 400.0
BETA, DELTA, W1 = 12.0, 1.0, 1.0

sys.stdout.reconfigure(encoding="utf-8")


# ─────────────────────────── 共享引擎 ───────────────────────────

def wilson(wrong, n, z=1.96):
    """→ (err%下界, err%点值, err%上界) 95% Wilson 区间（wrong = 错误窗数）。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = 1 - wrong / n                              # 命中率
    e = wrong / n * 100
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ((1 - (ctr + hw)) * 100, e, (1 - (ctr - hw)) * 100)


def load_dict_full():
    """词典全码表（行格式 `词\\t码1\\t码2…`）:
    → w2codes（词 → 码文件序）, code_seq（码 → 词条按文件序去重）。"""
    w2codes = defaultdict(list)
    code_seq = defaultdict(list)
    with open(corpus_std.DICT_PATH, encoding="utf-8") as f:
        in_body = False
        for line in f:
            line = line.rstrip("\n\r")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line[0] == "#" or "\t" not in line:
                continue
            parts = line.split("\t")
            w = parts[0].strip()
            if not w:
                continue
            for c in parts[1:]:
                c = c.strip()
                if c and not c.startswith("\\"):
                    w2codes[w].append(c)
                    if w not in code_seq[c]:
                        code_seq[c].append(w)
    return w2codes, code_seq


def excl_sets(w2codes, code_seq):
    """资格剔除集: excl_single（一码字 / 2 码窗文件序首选字, 多音保守任一音）、
    excl_word（有一二码简码的多字词）——打法轮不到 CE, 目标窗剔出评测域。"""
    s1_words = {w for w, cs in w2codes.items() if any(len(c) == 1 for c in cs)}
    c2_first = {c: ws[0] for c, ws in code_seq.items() if len(c) == 2}
    excl_single = set(s1_words)
    for w, cs in w2codes.items():
        if len(w) == 1 and any(c2_first.get(c) == w for c in cs if len(c) == 2):
            excl_single.add(w)
    excl_word = {w for w, cs in w2codes.items()
                 if len(w) > 1 and any(len(c) <= 2 for c in cs)}
    return excl_single, excl_word


def load_ce(path):
    """CE 结果目录 results.jsonl → {sid: rec}（容错解析, 断行跳过）。"""
    recs = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
            except Exception:
                continue
            recs[r["sid"]] = r
    return recs


def sc_win(r):
    """样本行 → 词典序重建评分窗（目标词插回 pos）。"""
    return (r["cands"][:r["pos"]] + [r["word"]]
            + r["cands"][r["pos"]:])[:MAX_CAND]


def build_single(rows, keep_pred=None):
    """单字域窗数据 → (win_data, lens, pad, widx, n)。keep_pred(r) 为 False 剔除。"""
    win_data = []
    for r in rows:
        if keep_pred and not keep_pred(r):
            continue
        win_data.append((sc_win(r), r))
    n = len(win_data)
    lens = np.array([len(w) for w, _ in win_data])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    widx = np.array([win.index(r["word"]) for win, r in win_data])
    return win_data, lens, pad, widx, n


def build_word_keep():
    """多字词域 → (keep, lens, pad, widx, n)；keep 元组
    (ci, word, win, code, sid, ctx)。"""
    samples = [json.loads(l) for l in open(PROJ / "user_samples_std.jsonl",
                                           encoding="utf-8")]
    anchors = [json.loads(l) for l in open(PROJ / "user_anchors_std.jsonl",
                                           encoding="utf-8")]
    keep = []
    for s, an in zip(samples, anchors):
        keep.append((an["ci"], s["word"], sc_win(s), s.get("code", ""),
                     an["sid"], an["ctx"]))
    n = len(keep)
    lens = np.array([len(k[2]) for k in keep])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    widx = np.array([k[2].index(k[1]) for k in keep])
    return keep, lens, pad, widx, n


def ce_maps_single(win_data, ce_recs):
    """单字域: sid = window_key_ctx（词典序窗）; → CE 证据 map 列表。"""
    maps = []
    for win, r in win_data:
        rec = ce_recs.get(corpus_std.window_key_ctx(r["ctx"], sc_win(r)))
        maps.append({x: (rec["ce"][x], rec["ntok"][x]) for x in win
                     if rec and x in rec["ce"] and x in rec["ntok"]})
    return maps


def ce_maps_word(keep, ce_recs):
    """多字词域: sid 由 anchors 直给。"""
    maps = []
    for ci, w, win, code, sid, ctx in keep:
        rec = ce_recs.get(sid)
        maps.append({x: (rec["ce"][x], rec["ntok"][x]) for x in win
                     if rec and x in rec["ce"] and x in rec["ntok"]})
    return maps


def replay_scenes(win_data, pad, percode, words_all, want_mats=False):
    """单字域双键重放（场景制词流, ci 计数器对位, τ=400）。
    percode=False 键=词 / True 键=(词, 提交码)。want_mats 时附 eff/uni 矩阵。"""
    n = len(win_data)
    scenes = [[(c["word"], c["code"]) for c in scene]
              for scene in corpus_std.iter_scenes(TRAIN_LOG,
                                                  reset_mode="strict")]
    ci_to_idx = {r["ci"]: i for i, (win, r) in enumerate(win_data)}
    rp = FormulaReplay(TAU)
    uni = {}
    s1 = np.zeros((n, MAX_CAND))
    eff_mat = np.zeros((n, MAX_CAND))
    uni_mat = np.zeros((n, MAX_CAND))
    ci = 0
    for scene in scenes:
        for w, c in scene:
            ii = ci_to_idx.get(ci)
            if ii is not None:
                win = win_data[ii][0]
                for j, x in enumerate(win):
                    key = (x, c) if percode else x
                    eff_mat[ii, j] = rp.eff_of(key)
                    uni_mat[ii, j] = uni.get(key, 0)
                    s1[ii, j] = (BETA * math.log1p(eff_mat[ii, j])
                                 + DELTA * math.log(uni_mat[ii, j] + 1.0))
            if w in words_all:
                k = (w, c) if percode else w
                rp.bump(k)
                uni[k] = uni.get(k, 0) + 1
                ci += 1
    s1 = np.where(pad, -1e9, s1)
    if want_mats:
        return s1, eff_mat, uni_mat
    return s1


def replay_flat(keep, pad, percode, words_all):
    """多字词域双键重放（扁平词流含被回撤提交, 与 user_anchors_std 的 ci
    口径一致; percode 键 = (词, 窗码=样本自身 code)）。"""
    n = len(keep)
    by_ci = defaultdict(list)
    for i, entry in enumerate(keep):
        by_ci[entry[0]].append(i)
    seq = [(c["word"], c["code"])
           for c in corpus_std.iter_commits(TRAIN_LOG, include_retracted=True)
           if c["word"] in words_all]
    rp = FormulaReplay(TAU)
    uni = {}
    s1 = np.zeros((n, MAX_CAND))
    for ci, (w, c) in enumerate(seq):
        for i in by_ci.get(ci, ()):
            win, code = keep[i][2], keep[i][3]
            for j, x in enumerate(win):
                key = (x, code) if percode else x
                s1[i, j] = (BETA * math.log1p(rp.eff_of(key))
                            + DELTA * math.log(uni.get(key, 0) + 1.0))
        k = (w, c) if percode else w
        rp.bump(k)
        uni[k] = uni.get(k, 0) + 1
    return np.where(pad, -1e9, s1)


def tl_pick(S1, s_ce, w1=1.0):
    """批量两层 k=5 终选（向量化, 与 summary_4x4.tl_hit 同式）
    → (各窗终选列 index, 层1 order 矩阵)。"""
    n = S1.shape[0]
    order = np.argsort(-S1, axis=1, kind="stable")[:, :K]
    S1k = np.take_along_axis(S1, order, axis=1)
    sct = np.take_along_axis(s_ce, order, axis=1)
    S2k = S1k + np.where(sct > -1e8, sct * w1, 0.0)
    win = order[np.arange(n), S2k.argmax(1)]
    return win, order


def score_rows(s1, s_ce, widx, w1=1.0, fill_mode="s1fallback"):
    """逐窗两层 k=5 → (对错 0/1 列表, order 列表)。fill_mode:
    's1fallback' 无 CE 回退 0（旧生产）/ 'mince' 取窗内已算 CE 最低分（定案）。"""
    ce_ok = s_ce > -1e8
    n = s1.shape[0]
    oks, orders = [], []
    for i in range(n):
        order = np.argsort(-s1[i], kind="stable")[:K]
        if fill_mode == "mince":
            okr = ce_ok[i][order]
            min_ce = s_ce[i][order][okr].min() if okr.any() else 0.0
            s2 = s1[i][order] + np.where(okr, s_ce[i][order] * w1, min_ce)
        else:
            s2 = s1[i][order] + np.where(ce_ok[i][order],
                                         s_ce[i][order] * w1, 0.0)
        orders.append(order)
        oks.append(int(order[int(np.argmax(s2))] == widx[i]))
    return oks, orders


# ─────────────────────────── test ───────────────────────────

def cmd_test(a):
    rows = [json.loads(l) for l in open(PROJ / "user_sc_samples.jsonl",
                                        encoding="utf-8")]
    ce_recs = load_ce(PROJ / "user_sc_ce" / "results.jsonl")

    def build(use_real):
        keep = []
        for r in rows:
            if use_real:
                if not r.get("real_win"):
                    continue
                win = r["real_win"][:MAX_CAND]
                if r["word"] not in win:
                    continue
            else:
                win = sc_win(r)
            keep.append((r, win))
        n = len(keep)
        lens = np.array([len(k[1]) for k in keep])
        pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
        widx = np.array([k[1].index(k[0]["word"]) for k in keep])
        du = np.zeros((n, MAX_CAND))
        lp1 = np.zeros((n, MAX_CAND))
        ce_maps = []
        for k, (r, win) in enumerate(keep):
            for j, x in enumerate(win):
                du[k, j] = math.log1p(r["eff_win"].get(x, 0.0))
                u = r["uni_win"].get(x, 0)
                lp1[k, j] = math.log((u + 1) / (r["n_seen"] + 20000))
            rec = ce_recs.get(corpus_std.window_key_ctx(r["ctx"], sc_win(r)))
            ce_maps.append({w: (rec["ce"][w], rec["ntok"][w])
                            for w in win
                            if rec and w in rec["ce"] and w in rec["ntok"]})
        s_ce = build_ce_matrix(ce_maps, lens, pad)
        du = np.where(pad, -1e9, du)
        lp1 = np.where(pad, -1e9, lp1)
        return du, lp1, s_ce, pad, widx, n

    lines = ["=" * 84,
             "用户域单字 3 码重排测试（236 窗 = 真实打字流 CJK 3 码重码窗）",
             f"常数: 生产 (β={a.beta:g}, δ={a.delta:g}, w1={a.w1:g})",
             "=" * 84]
    for tag, use_real in (("词典序重建窗（与标定域同构）", False),
                          ("真实窗快照子集（引擎实况, 49/51 与词典序同序）", True)):
        du, lp1, s_ce, pad, widx, n = build(use_real)
        if n == 0:
            continue
        lines.append(f"\n== {tag} (n={n}) ==")
        wrong_fix = n - (widx == 0).sum()
        lo, e, hi = wilson(wrong_fix, n)
        lines.append(f"  基线（固定顺序）      err={e:7.4f}%  "
                     f"95%CI [{lo:.2f}, {hi:.2f}]  ({wrong_fix}/{n} 窗)")
        win_ce, _ = tl_pick(np.where(pad, -1e9, 0.0), s_ce)
        wrong_ce = int((win_ce != widx).sum())
        lo, e, hi = wilson(wrong_ce, n)
        lines.append(f"  纯CE                  err={e:7.4f}%  "
                     f"95%CI [{lo:.2f}, {hi:.2f}]  ({wrong_ce}/{n} 窗)")
        for name, (b, d, w1) in (
                ("生产点", (a.beta, a.delta, a.w1)),
                ("百科argmax", (a.wiki_beta if a.wiki_beta is not None
                                else a.beta, a.wiki_delta, a.wiki_w1))):
            S1 = b * du + d * lp1
            win5, _ = tl_pick(S1, s_ce, w1)
            w5 = int((win5 != widx).sum())
            lo, e, hi = wilson(w5, n)
            S10 = S1 + np.where(s_ce > -1e8, s_ce * w1, 0.0)
            w10 = int((S10.argmax(1) != widx).sum())
            lo10, e10, hi10 = wilson(w10, n)
            lines.append(f"  LLM+词频 {name:<8} err={e:7.4f}%  "
                         f"95%CI [{lo:.2f}, {hi:.2f}]  ({w5}/{n} 窗)"
                         f"  hit@10 err {e10:.2f}% ({w10})")
        # 用户域 argmax（过拟合上界, 仅参考）
        best = (1e9, None)
        for b in (0, 4, 8, 12, 16, 24):
            for d in (0.25, 0.5, 1.0, 2.0):
                if b == 0 and d == 0:
                    continue
                S1 = b * du + d * lp1
                for w1 in (0.0, 0.5, 1.0, 1.5):
                    win5, _ = tl_pick(S1, s_ce, w1)
                    w5 = int((win5 != widx).sum())
                    if w5 < best[0]:
                        best = (w5, (b, d, w1))
        lo, e, hi = wilson(best[0], n)
        lines.append(f"  用户argmax(过拟合界)  err={e:7.4f}%  "
                     f"95%CI [{lo:.2f}, {hi:.2f}]  ({best[0]}/{n} 窗)"
                     f"  (β={best[1][0]:g}, δ={best[1][1]:g}, "
                     f"w1={best[1][2]:g})")
    txt = "\n".join(lines)
    (EVAL / "test_sc_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    return 0


# ─────────────────────────── final ───────────────────────────

def cmd_final():
    w2codes, code_seq = load_dict_full()
    words_all = set(corpus_std.load_pdsp_words())
    rows = [json.loads(l) for l in open(PROJ / "user_sc_samples.jsonl",
                                        encoding="utf-8")]
    excl_single, _excl_word = excl_sets(w2codes, code_seq)
    # ② 干净集: 剔二码窗首选字目标（含一码字/多音保守）
    win_data, lens, pad, widx, n = build_single(
        rows, keep_pred=lambda r: r["word"] not in excl_single)
    dropped = [r for r in rows if r["word"] in excl_single]
    ce_recs = load_ce(PROJ / "user_sc_ce" / "results.jsonl")
    ce_maps = ce_maps_single(win_data, ce_recs)
    miss_by_cand = Counter()
    for (win, r), m in zip(win_data, ce_maps):
        for x in win:
            if x not in m:
                miss_by_cand[x] += 1
    s_ce = build_ce_matrix(ce_maps, lens, pad)
    ce_ok = s_ce > -1e8

    s1, eff_pc, uni_pc = replay_scenes(win_data, pad, True, words_all,
                                       want_mats=True)   # ① 分键定案口径

    out = ["=" * 96,
           "单字 3 码重排 · 定案口径逐例分析（分键记账 + 干净集 235 窗 + 无 CE = 最低分）",
           f"生产公式 β=12 δ=1 w1=1, 层1 k=5; 剔除一二码打法目标窗 "
           f"{len(dropped)}（{', '.join(f'ci={r[chr(99)+chr(105)]}{r[chr(99)+chr(111)+chr(100)+chr(101)]}「{r[chr(119)+chr(111)+chr(114)+chr(100)]}」' for r in dropped)}）",
           "=" * 96]

    def simulate(mode):
        """mode: 'mince' 无CE=窗内已算CE最低分（定案, 2026-11-12 用户定:
        −∞ 太夸张, 取评分窗口中算了 ce 的最低分）/ 's1fallback' 旧生产。"""
        cats = {}
        det = {}
        for i in range(n):
            order = np.argsort(-s1[i], kind="stable")[:K]
            if mode == "mince":
                if ce_ok[i][order].any():
                    min_ce = s_ce[i][order][ce_ok[i][order]].min()
                    fill = min_ce / W1          # 使融合恰 = S1+min_ce
                else:
                    fill = 0.0
                s2 = s1[i][order] + np.where(ce_ok[i][order],
                                             s_ce[i][order] * W1, min_ce)
            else:
                s2 = s1[i][order] + np.where(ce_ok[i][order],
                                             s_ce[i][order] * W1, 0.0)
            wj = int(order[int(np.argmax(s2))])
            wi = int(widx[i])
            cat = (("A" if wi == 0 else "C") if wj == wi
                   else ("B" if wi == 0 else "D"))
            cats[i] = cat
            det[i] = (wi, wj, order)
        return cats, det

    cats_f, det_f = simulate("mince")
    cats_o, det_o = simulate("s1fallback")
    cnt = Counter(cats_f.values())
    cnto = Counter(cats_o.values())
    out.append(f"\n== 四象限（定案口径: 无 CE = 窗内已算 CE 最低分） ==")
    out.append(f"  A 保持对 {cnt['A']}  B 改错 {cnt['B']}  C 改对 {cnt['C']}  "
               f"D 双错 {cnt['D']}")
    out.append(f"  基线 err {(cnt['C'] + cnt['D']) / n * 100:.4f}%"
               f"（{cnt['C'] + cnt['D']}/{n}）  重排 err "
               f"{(cnt['B'] + cnt['D']) / n * 100:.4f}%"
               f"（{cnt['B'] + cnt['D']}/{n}）  净改对 {cnt['C'] - cnt['B']:+d} 窗")
    out.append(f"\n== 对照旧口径（无 CE 回退 S1=0 参与终选） ==")
    out.append(f"  A {cnto['A']}  B {cnto['B']}  C {cnto['C']}  D {cnto['D']}  "
               f"重排 err {(cnto['B'] + cnto['D']) / n * 100:.4f}%")
    diff = [i for i in range(n) if (cats_f[i], det_f[i][1]) !=
            (cats_o[i], det_o[i][1])]
    out.append(f"  两口径终选差异 {len(diff)} 窗: "
               + ", ".join(f"ci={win_data[i][1]['ci']}{win_data[i][1]['code']}"
                           f"「{win_data[i][1]['word']}」{cats_o[i]}→{cats_f[i]}"
                           f" 旧#{det_o[i][1]}→新#{det_f[i][1]}" for i in diff)
               or "无")

    # 主导项分解
    dom = Counter()
    for i, (wi, wj, order) in det_f.items():
        if cats_f[i] in ("B", "C"):
            dom[(cats_f[i], "层1词频" if order[0] == wj else "层2CE")] += 1
    out.append(f"\n主导项: " + ", ".join(
        f"{c}类·{who}={v}" for (c, who), v in sorted(dom.items())))

    # 逐例打印 B/C 全部
    def dump(i, cat, wi, wj, order):
        win, r = win_data[i]
        out.append(f"\n[{cat}] ci={r['ci']} 码={r['code']} 目标=「{r['word']}」"
                   f" 基线#{wi} → 终选#{wj}")
        out.append(f"  ctx尾: …{r['ctx'][-24:]}")
        l1_rank = {int(x): k for k, x in enumerate(order)}
        ok_row = ce_ok[i]
        min_ce = s_ce[i][ok_row].min() if ok_row.any() else 0.0
        for j, w in enumerate(win):
            has = bool(ce_ok[i, j])
            ce_s = f"{s_ce[i, j]:7.2f}" if has else f" 取窗内最低{min_ce:6.2f}"
            f2 = (s1[i, j] + s_ce[i, j] * W1) if has else (s1[i, j] + min_ce)
            tag = ""
            if j == wi and j == wj:
                tag = " ←目标(保持)"
            elif j == wi:
                tag = " ←目标(被挤)"
            elif j == wj:
                tag = " ←重排选中"
            r1 = l1_rank.get(j)
            rs = f"层1#{r1}" if r1 is not None else "层1外"
            out.append(f"  #{j} 「{w}」 eff={eff_pc[i, j]:5.2f}"
                       f" uni={uni_pc[i, j]:.0f}"
                       f" S1={s1[i, j]:6.2f} CE={ce_s} 融合={f2:7.2f}"
                       f" [{rs}]{tag}")

    out.append(f"\n-- B 类全部（基线对、重排改错）--")
    for i, (wi, wj, order) in det_f.items():
        if cats_f[i] == "B":
            dump(i, "B", wi, wj, order)
    out.append(f"\n-- C 类全部（基线错、重排改对）--")
    for i, (wi, wj, order) in det_f.items():
        if cats_f[i] == "C":
            dump(i, "C", wi, wj, order)

    # CE 缺失候选清单（定案口径下取窗内最低 CE）
    out.append(f"\n== CE 缺失候选（定案口径: 取窗内已算 CE 的最低分） ==")
    for w, c in miss_by_cand.most_common():
        out.append(f"  「{w}」 {c} 窗缺失")

    # 平铺清单（用户逐条审阅用）
    out.append(f"\n{'=' * 96}")
    out.append("平铺清单: 改错（B 类 13 条, 按主导项分组）")
    out.append(f"{'=' * 96}")
    for who in ("层1词频", "层2CE"):
        out.append(f"\n[{who} 主导]")
        for i, (wi, wj, order) in det_f.items():
            if cats_f[i] == "B" and (("层1词频" if order[0] == wj
                                      else "层2CE") == who):
                win, r = win_data[i]
                sel = win[wj]
                out.append(f"  ci={r['ci']} {r['code']} 「{r['word']}」"
                           f"(基线#{wi}) → 选中「{sel}」#{wj}"
                           f" | 选中项 eff={eff_pc[i, wj]:.2f}"
                           f" uni={uni_pc[i, wj]:.0f}"
                           f" CE={s_ce[i, wj]:.2f} vs 目标 CE={s_ce[i, wi]:.2f}")
    out.append(f"\n{'=' * 96}")
    out.append("平铺清单: 改对（C 类 17 条, 按主导项分组）")
    out.append(f"{'=' * 96}")
    for who in ("层1词频", "层2CE"):
        out.append(f"\n[{who} 主导]")
        for i, (wi, wj, order) in det_f.items():
            if cats_f[i] == "C" and (("层1词频" if order[0] == wj
                                      else "层2CE") == who):
                win, r = win_data[i]
                out.append(f"  ci={r['ci']} {r['code']} 「{r['word']}」"
                           f"(基线#{wi}) 保持/升到 #{wj}"
                           f" | 目标 eff={eff_pc[i, wi]:.2f}"
                           f" uni={uni_pc[i, wi]:.0f}"
                           f" CE={s_ce[i, wi]:.2f} vs 原首选 CE={s_ce[i, 0]:.2f}")

    txt = "\n".join(out)
    (EVAL / "analyze_sc_final_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt)
    return 0


# ─────────────────────────── clean ───────────────────────────

def cmd_clean():
    w2codes, code_seq = load_dict_full()
    words_all = set(corpus_std.load_pdsp_words())
    excl_single, excl_word = excl_sets(w2codes, code_seq)
    out = ["=" * 96,
           "词码对记账定案 · 干净样本集重统计（分键口径为主, 生产公式 12/1/1.0 k=5）",
           "=" * 96]
    out.append(f"资格剔除集: 一二码打法目标剔除 单字 {len(excl_single)} / "
               f"一二码简词 {len(excl_word)}")

    # 单字域
    rows = [json.loads(l) for l in open(PROJ / "user_sc_samples.jsonl",
                                        encoding="utf-8")]
    win_data, lens, pad, widx, n = build_single(
        rows, keep_pred=lambda r: r["word"] not in excl_single)
    dropped = sum(1 for r in rows if r["word"] in excl_single)
    ce_recs = load_ce(PROJ / "user_sc_ce" / "results.jsonl")
    s_ce = build_ce_matrix(ce_maps_single(win_data, ce_recs), lens, pad)
    ok_p, _ = score_rows(replay_scenes(win_data, pad, True, words_all),
                         s_ce, widx)
    ok_m, _ = score_rows(replay_scenes(win_data, pad, False, words_all),
                         s_ce, widx)
    base_ok = int((widx == 0).sum())
    out.append(f"\n== 单字 3 码域（剔一二码打法目标后） ==")
    out.append(f"  剔除目标窗 {dropped}（原 236）→ n={n}")
    out.append(f"  基线（固定序）err {(n - base_ok) / n * 100:.4f}%（{n - base_ok} 错）")
    out.append(f"  词码分键（定案口径）err {(n - sum(ok_p)) / n * 100:.4f}%"
               f"（{n - sum(ok_p)} 错）")
    out.append(f"  词合并（仅对照）    err {(n - sum(ok_m)) / n * 100:.4f}%"
               f"（{n - sum(ok_m)} 错）")

    # 多字词域
    keep, _lens, _pad, _widx, n_all = build_word_keep()
    keep2 = [k for k in keep if k[1] not in excl_word]
    n2 = len(keep2)
    lens2 = np.array([len(k[2]) for k in keep2])
    pad2 = np.arange(MAX_CAND)[None, :] >= lens2[:, None]
    widx2 = np.array([k[2].index(k[1]) for k in keep2])
    dropped2 = n_all - n2
    ce_recs2 = load_ce(PROJ / "user_ce_work" / "results.jsonl")
    s_ce2 = build_ce_matrix(ce_maps_word(keep2, ce_recs2), lens2, pad2)
    ok_p2, _ = score_rows(replay_flat(keep2, pad2, True, words_all),
                          s_ce2, widx2)
    ok_m2, _ = score_rows(replay_flat(keep2, pad2, False, words_all),
                          s_ce2, widx2)
    base_ok2 = int((widx2 == 0).sum())
    out.append(f"\n== 多字词 4 码域（剔一二码简词目标后） ==")
    out.append(f"  剔除目标窗 {dropped2}（原 20,001）→ n={n2}")
    out.append(f"  基线（固定序）err {(n2 - base_ok2) / n2 * 100:.4f}%"
               f"（{n2 - base_ok2} 错）")
    out.append(f"  词码分键（定案口径）err {(n2 - sum(ok_p2)) / n2 * 100:.4f}%"
               f"（{n2 - sum(ok_p2)} 错）")
    out.append(f"  词合并（仅对照）    err {(n2 - sum(ok_m2)) / n2 * 100:.4f}%"
               f"（{n2 - sum(ok_m2)} 错）")
    flips_bp = sum(1 for a, b in zip(ok_m2, ok_p2) if a and not b)
    flips_pb = sum(1 for a, b in zip(ok_m2, ok_p2) if b and not a)
    out.append(f"  两口径翻转: 合并对→分键错 {flips_bp} / 分键对→合并错 {flips_pb}")

    txt = "\n".join(out)
    (EVAL / "resample_percode_clean_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt)
    return 0


# ─────────────────────────── wordflip ───────────────────────────

def cmd_wordflip():
    words_all = set(corpus_std.load_pdsp_words())
    keep, lens, pad, widx, n = build_word_keep()
    ce_recs = load_ce(PROJ / "user_ce_work" / "results.jsonl")
    s_ce = build_ce_matrix(ce_maps_word(keep, ce_recs), lens, pad)

    s1m = replay_flat(keep, pad, False, words_all)
    s1p = replay_flat(keep, pad, True, words_all)
    winm, _ = tl_pick(s1m, s_ce)
    winp, orderp = tl_pick(s1p, s_ce)
    wm, wp = winm != widx, winp != widx

    # 自检: merged 应复现 test_uniprior_user ρ=0 生产点
    lines = ["=" * 96,
             "多字词用户域 · (词,码) 分键 vs 词合并（20,001 窗, β=12 δ=1, k=5, τ=400）",
             "=" * 96,
             f"  merged 词合并:  err {wm.sum()}/{n} = {wm.mean() * 100:.4f}%"
             f"   （基准 405/20001 = 2.0249%）",
             f"  percode 分键:   err {wp.sum()}/{n} = {wp.mean() * 100:.4f}%",
             f"  配对翻转: 合并对→分键错 {int((~wm & wp).sum())} 窗 /"
             f" 合并错→分键对 {int((wm & ~wp).sum())} 窗"]

    flips = [i for i in range(n) if wm[i] != wp[i]]
    lines.append(f"\n== 翻转窗例子（前 12） ==")
    for i in flips[:12]:
        ci, ctx, w, win, code, sid = keep[i]
        better = "分键改对" if wm[i] and not wp[i] else "分键改错"
        jw = int(np.where(orderp[i] == widx[i])[0][0])
        lines.append(f"\n[{better}] ci={ci} 码={code} 目标=「{w}」"
                     f"（分键层1位次#{jw}, 终位见翻转方向） ctx尾:…{ctx[-24:]}")
        for j in [int(x) for x in orderp[i][:4] if x < len(win)]:
            x = win[j]
            lines.append(f"    #{j} 「{x}」 "
                         f"percode S1={s1p[i, j]:6.2f} CE={s_ce[i, j]:7.2f}")
    lines.append("")
    txt = "\n".join(lines)
    (EVAL / "inspect_word_percode_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt)
    return 0


# ─────────────────────────── behavior ───────────────────────────

def cmd_behavior():
    w2codes, _code_seq = load_dict_full()
    minlen = {w: min(len(c) for c in cs) for w, cs in w2codes.items()}
    words_all = set(corpus_std.load_pdsp_words())
    # ① 提交流（词典词, 退格已消解; iter_scenes 按 M.kind 处理回撤）
    seq = []
    for scene in corpus_std.iter_scenes(TRAIN_LOG, reset_mode="strict"):
        for c in scene:
            w, ccode = c["word"], c["code"]
            if w in words_all and ccode and not ccode.startswith("\\"):
                seq.append((w, ccode))
    out = ["=" * 96,
           "编码行为验证（简码必打简码 / 多音词 / 分键结论分层）",
           "=" * 96]

    sub = [(w, c) for w, c in seq if minlen.get(w, 9) <= 3]
    agree = sum(1 for w, c in sub if len(c) == minlen[w])
    cross = Counter((minlen[w], len(c)) for w, c in sub)
    out.append(f"\n① 简码行为（有更短码的词典词共 {len(sub)} 次提交）")
    out.append(f"  实际码长 == 最短码长: {agree}/{len(sub)} = "
               f"{agree / max(len(sub), 1) * 100:.2f}%")
    out.append(f"  交叉表 (最短码长, 实际码长) → 次数: " + ", ".join(
        f"({a},{b}):{v}" for (a, b), v in sorted(cross.items())))
    bad = Counter(w for w, c in sub if len(c) > minlen[w])
    out.append(f"  反例（有短码却打长码）词 top10: " + " | ".join(
        f"「{w}」min={minlen[w]}×{v}" for w, v in bad.most_common(10)))

    # ② 多音词（多个码的首码不同 = 第一字声母不同）
    multi_poly = {w: cs for w, cs in w2codes.items()
                  if len(cs) >= 2 and len({c[0] for c in cs}) >= 2
                  and len(w) > 1}
    multi_any = {w: cs for w, cs in w2codes.items() if len(cs) >= 2}
    poly1 = {w: cs for w, cs in w2codes.items()
             if len(cs) >= 2 and len({c[0] for c in cs}) >= 2 and len(w) == 1}
    out.append(f"\n② 多码词 {len(multi_any)} 个（词表全量）; 其中首码不同"
               f"（多音结构）: 多字词 {len(multi_poly)} 个 / 单字 {len(poly1)} 个")
    ex = list(multi_poly.items())[:10]
    out.append(f"  多字词多音例: " + " | ".join(
        f"「{w}」{','.join(cs[:4])}" for w, cs in ex))

    # ③ 分层重算双口径（两域）
    def quadrant_sc():
        """单字域 236 窗双口径重放 → (合并对列表, 分键对列表, 目标最短码长, n)。"""
        rows = [json.loads(l) for l in open(PROJ / "user_sc_samples.jsonl",
                                            encoding="utf-8")]
        ce_recs = load_ce(PROJ / "user_sc_ce" / "results.jsonl")
        win_data, lens, pad, widx, n = build_single(rows)
        s_ce = build_ce_matrix(ce_maps_single(win_data, ce_recs), lens, pad)
        ok_m, _ = score_rows(replay_scenes(win_data, pad, False, words_all),
                             s_ce, widx)
        ok_p, _ = score_rows(replay_scenes(win_data, pad, True, words_all),
                             s_ce, widx)
        short = [minlen.get(r["word"], 9) for _, r in win_data]
        return ok_m, ok_p, short, n

    def quadrant_word():
        """多字词域 20,001 窗双口径重放。"""
        keep, lens, pad, widx, n = build_word_keep()
        ce_recs = load_ce(PROJ / "user_ce_work" / "results.jsonl")
        s_ce = build_ce_matrix(ce_maps_word(keep, ce_recs), lens, pad)
        ok_m, _ = score_rows(replay_flat(keep, pad, False, words_all),
                             s_ce, widx)
        ok_p, _ = score_rows(replay_flat(keep, pad, True, words_all),
                             s_ce, widx)
        short = [minlen.get(k[1], 9) for k in keep]
        return ok_m, ok_p, short, n

    out.append(f"\n③ 分层重算（目标词按『有无更短码』分档）")
    for name, fn in (("单字 3 码 236 窗", quadrant_sc),
                     ("多字词 4 码 20,001 窗", quadrant_word)):
        cm, cp, ml_short, n = fn()
        # ml_short: 目标词词典最短码长（0=无记录）
        ml_short = np.array(ml_short)
        tiers = [("全体窗", np.ones(n, bool)),
                 ("剔一码目标（顶屏字/简词, 必走一码）", ml_short != 1),
                 ("剔一二码目标（声韵直达/简词）", ml_short > 2)]
        out.append(f"\n  == {name} ==")
        out.append(f"  目标词最短码长分布: " + ", ".join(
            f"{k}:{int(v)}" for k, v in
            sorted(Counter(ml_short.tolist()).items())))
        for tag, mask in tiers:
            nn = int(mask.sum())
            if nn == 0:
                continue
            m_ok = sum(c for c, s in zip(cm, mask) if s)
            p_ok = sum(c for c, s in zip(cp, mask) if s)
            out.append(f"  {tag} (n={nn}): 合并 err {(nn - m_ok) / nn * 100:.4f}%"
                       f"  分键 err {(nn - p_ok) / nn * 100:.4f}%"
                       f"  （对 {m_ok} vs {p_ok}）")
        flip_bad = sum(1 for m, p in zip(cm, cp) if m and not p)
        flip_good = sum(1 for m, p in zip(cm, cp) if p and not m)
        out.append(f"  合并对→分键错 {flip_bad} / 分键对→合并错 {flip_good}")

    txt = "\n".join(out)
    (EVAL / "inspect_codes_behavior_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt)
    return 0


# ─────────────────────────── 入口 ───────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="单字/词码对评测合并入口（test/final/clean/wordflip/behavior）")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("test", help="用户域单字 3 码重排测试（Wilson CI）")
    p.add_argument("--beta", type=float, default=12.0)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--wiki-beta", type=float, default=None,
                   help="百科域 argmax β（不传 = 只评生产点）")
    p.add_argument("--wiki-delta", type=float, default=1.0)
    p.add_argument("--wiki-w1", type=float, default=1.0)
    p.set_defaults(fn=lambda a: cmd_test(a))

    sub.add_parser("final", help="定案口径逐例分析（235 干净窗四象限）"
                   ).set_defaults(fn=lambda a: cmd_final())
    sub.add_parser("clean", help="词码对定案 · 干净样本集重统计（两域）"
                   ).set_defaults(fn=lambda a: cmd_clean())
    sub.add_parser("wordflip", help="多字词域分键 vs 合并配对翻转"
                   ).set_defaults(fn=lambda a: cmd_wordflip())
    sub.add_parser("behavior", help="编码行为验证 + 分层重算"
                   ).set_defaults(fn=lambda a: cmd_behavior())

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
