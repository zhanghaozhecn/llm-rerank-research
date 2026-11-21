#!/usr/bin/env python3
"""预置降分修正 · 样本拓展（池化单字+多字词双域）+ 比值门选法
（2026-11-13, 用户指令: 样例低频需拓展样本数; 综合改对/改错比值与
总数两项判断是否修正）。

数据: cohort_big_sc（单字 59,681 窗）+ cohort_big（多字词 120,018 窗）
池化 179,699 窗; 各域独立重放（sc 域 bump 用 canonical 3 码近似,
big 域 bump 用词典首码近似——streams 无码, 与既有链口径一致）。

选法家族（全在偶半推导 / 奇半评测, fade 门不变）:
- 比值门（用户提议）: 改对/改错 ≤ R 且 支持(改对+改错) ≥ M
  —— R=0 即纯负门, 是纯负门的推广; ratio 管"系统性", M 管"证据量"。
- net 门（对照）: 改错−改对 ≥ T 且 支持 ≥ M。
产出: 两门最优对比 / 选定配置分域分解 / 池化后支持度分布 / 最终表
TSV + 明细 / 用户域验证。

用法: python derive_pairfix_ratio.py
报告: eval/derive_pairfix_ratio_report.txt
表:   eval/pairfix_pool_ratio.tsv
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_std
from corpus_std import FormulaReplay, MAX_CAND
from summary_4x4 import K, build_ce_matrix

PROJ = corpus_std.PROJ
TRAIN_LOG = corpus_std.CORPUS_PATH          # 统一语料 v2（env RIME_LLM_CORPUS 可换）
FADE = 3
LAMBDAS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
RS = [0.0, 0.1, 0.2, 0.3, 0.5]           # 改对/改错 比
MS_R = [2, 3, 5, 8, 12]                  # 比值门支持数
TS = [1, 2, 3]                           # net 门净损阈值
MS_T = [1, 2, 3, 5]
BETA = 12.0

sys.stdout.reconfigure(encoding="utf-8")


def load_w2codes():
    w2codes = defaultdict(list)
    first_code = {}
    with open(corpus_std.DICT_PATH, encoding="utf-8") as f:
        in_body = False
        for line in f:
            line = line.rstrip("\n\r")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line[0] == "#":
                continue
            if "\t" not in line:
                continue
            parts = line.split("\t")
            w = parts[0].strip()
            if not w:
                continue
            for c in parts[1:]:
                c = c.strip()
                if c and not c.startswith("\\"):
                    if c not in w2codes[w]:
                        w2codes[w].append(c)
                    if w not in first_code:
                        first_code[w] = c
    return dict(w2codes), first_code


def load_domain(dirname, bump_mode, word_code_sc, first_code, ce_sub="ce"):
    """→ dict(win, code, widx, du, uni, s_ce, ce_ok, min_ce, ui, arm,
    ctx, isflip 所需元数据); bump_mode: 'sc' 单字 canonical / 'big'
    词典首码。ce_sub: CE 结果子目录（换模型评测 ce_2b 等, 词频重放
    与窗完全同源不受影响）。"""
    d = PROJ / dirname
    anchors_all = [json.loads(l) for l in open(d / "anchors.jsonl",
                                               encoding="utf-8")]
    samples_all = [json.loads(l) for l in open(d / "samples.jsonl",
                                               encoding="utf-8")]
    keep = list(range(len(anchors_all)))
    n = len(keep)
    codes = [samples_all[i]["code"] for i in keep]
    uis = [anchors_all[i]["ui"] for i in keep]
    wins = [anchors_all[i]["win"] for i in keep]
    ctxs = [anchors_all[i]["ctx"] for i in keep]
    widx = np.array([win.index(anchors_all[i]["w"]) for i, win in
                     zip(keep, wins)])
    lens = np.array([len(w) for w in wins])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    del samples_all

    ce_recs = {}
    with open(d / ce_sub / "results.jsonl", encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
                ce_recs[r["sid"]] = {"ce": r.get("ce") or {},
                                     "ntok": r.get("ntok") or {}}
            except Exception:
                pass
    ce_maps = []
    for i in keep:
        an = anchors_all[i]
        rec = ce_recs.get(an["sid"])
        ce_maps.append({x: (rec["ce"][x], rec["ntok"][x]) for x in an["win"]
                        if rec and x in rec["ce"] and x in rec["ntok"]})
    s_ce = build_ce_matrix(ce_maps, lens, pad)
    ce_ok = s_ce > -1e8
    min_ce = np.where(ce_ok, s_ce, np.inf).min(axis=1)
    min_ce = np.where(np.isinf(min_ce), 0.0, min_ce)
    print(f"{dirname}: windows {n}", flush=True)

    by_ac = defaultdict(list)
    for r, i in enumerate(keep):
        by_ac[(anchors_all[i]["ui"], anchors_all[i]["ci"])].append(r)

    du = np.zeros((n, MAX_CAND))
    uni_pc = np.zeros((n, MAX_CAND))
    with open(d / "streams.jsonl", encoding="utf-8") as f:
        for ui, line in enumerate(f):
            rp = FormulaReplay(400.0)
            uni_p = {}
            for ci, w in enumerate(json.loads(line)):
                for r in by_ac.get((ui, ci), ()):
                    c = codes[r]
                    for j, x in enumerate(wins[r]):
                        du[r, j] = math.log1p(rp.eff_of((x, c)))
                        uni_pc[r, j] = uni_p.get((x, c), 0)
                if bump_mode == "sc":
                    kc = (w, word_code_sc.get(w, ""))
                else:
                    kc = (w, first_code.get(w, ""))
                rp.bump(kc)
                uni_p[kc] = uni_p.get(kc, 0) + 1
    print(f"{dirname}: replay done", flush=True)
    return dict(wins=wins, codes=codes, ctxs=ctxs, widx=widx, pad=pad,
                ui=np.array(uis),
                arm=np.array([anchors_all[i]["arm"] for i in keep]),
                du=du, uni=uni_pc, s_ce=s_ce, ce_ok=ce_ok, min_ce=min_ce)


def main():
    out = ["=" * 96,
           "预置降分修正 · 样本拓展（池化单字+多字词 179,699 窗）+ 比值门",
           "比值门 = 改对/改错 ≤ R 且 支持 ≥ M（R=0 即纯负门; ratio 管"
           "系统性, M 管证据量）; net 门对照; fade 门不变",
           "=" * 96]

    w2codes, first_code = load_w2codes()

    def _has_variant(cs):
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if len(cs[i]) == len(cs[j]) and cs[i] != cs[j]:
                    return True
        return False

    struct = {w for w, cs in w2codes.items()
              if len(cs) >= 2 and _has_variant(cs)}

    _, word_code_sc = corpus_std.load_pdsp_dict_single3()
    dom_sc = load_domain("cohort_big_sc", "sc", word_code_sc, first_code)
    dom_big = load_domain("cohort_big", "big", word_code_sc, first_code)

    wins = dom_sc["wins"] + dom_big["wins"]
    codes = dom_sc["codes"] + dom_big["codes"]
    ctxs = dom_sc["ctxs"] + dom_big["ctxs"]
    dom_id = np.array([0] * len(dom_sc["wins"]) +
                      [1] * len(dom_big["wins"]))
    widx = np.concatenate([dom_sc["widx"], dom_big["widx"]])
    pad = np.vstack([dom_sc["pad"], dom_big["pad"]])
    ui = np.concatenate([dom_sc["ui"], dom_big["ui"]])
    arm = np.concatenate([dom_sc["arm"], dom_big["arm"]])
    n_all = len(widx)
    du = np.vstack([dom_sc["du"], dom_big["du"]])
    uni_pc = np.vstack([dom_sc["uni"], dom_big["uni"]])
    s_ce = np.vstack([dom_sc["s_ce"], dom_big["s_ce"]])
    ce_ok = np.vstack([dom_sc["ce_ok"], dom_big["ce_ok"]])
    min_ce = np.concatenate([dom_sc["min_ce"], dom_big["min_ce"]])
    del dom_sc, dom_big
    print(f"pooled {n_all} windows", flush=True)

    s1 = np.where(pad, -1e9, BETA * du + np.log(uni_pc + 1.0))
    ce_term = np.where(ce_ok, s_ce, min_ce[:, None])
    win_cols = np.argsort(-s1, axis=1, kind="stable")[:, :K]
    s2w = np.take_along_axis(s1, win_cols, 1) + \
        np.take_along_axis(ce_term, win_cols, 1)
    l1_win = win_cols[:, 0]
    f_win = win_cols[np.arange(n_all), np.argmax(s2w, axis=1)]
    uni_w = np.take_along_axis(uni_pc, win_cols, 1)
    flip_rows = np.where(f_win != l1_win)[0]
    out.append(f"\n池化 {n_all} 窗（单字 {int((dom_id == 0).sum())} / "
               f"多字词 {int((dom_id == 1).sum())}）, CE 翻案窗 "
               f"{len(flip_rows)}")

    led = defaultdict(lambda: [0, 0])
    examples = defaultdict(list)
    wants = defaultdict(Counter)      # pair → 被挤掉的目标字词统计
    dom_seen = defaultdict(set)       # pair → 证据域 {0单,1词}
    for r in flip_rows:
        fw = int(f_win[r])
        pair = (wins[r][fw], codes[r])
        if fw == int(widx[r]):
            led[pair][0] += 1
        else:
            led[pair][1] += 1
            wants[pair][wins[r][int(widx[r])]] += 1
            dom_seen[pair].add(int(dom_id[r]))
            if len(examples[pair]) < 2:
                examples[pair].append(
                    f"[{'单' if dom_id[r] == 0 else '词'}]"
                    f"…{ctxs[r][-14:]}｜want=「{wins[r][int(widx[r])]}」"
                    f"@{pair[1]} 顶上=「{pair[0]}」")
    tgt_all = Counter()
    code_traffic = Counter(codes[r] for r in range(n_all))
    for r in range(n_all):
        tgt_all[wins[r][int(widx[r])]] += 1

    is_even = ui % 2 == 0
    led_even = defaultdict(lambda: [0, 0])
    for r in flip_rows:
        if not is_even[r]:
            continue
        fw = int(f_win[r])
        pair = (wins[r][fw], codes[r])
        if fw == int(widx[r]):
            led_even[pair][0] += 1
        else:
            led_even[pair][1] += 1

    ev_rows = np.where(~is_even)[0]
    base_err = 1 - (f_win[ev_rows] == widx[ev_rows]).mean()

    def apply_table(table, lam):
        match = np.zeros((len(ev_rows), K), dtype=bool)
        for ki, r in enumerate(ev_rows):
            wcr = win_cols[r]
            for j in range(K):
                jn = int(wcr[j])
                if jn < len(wins[r]) and \
                        (wins[r][jn], codes[r]) in table:
                    match[ki, j] = True
        pmask = match & (uni_w[ev_rows] < FADE)
        s2 = s2w[ev_rows].copy()
        s2[pmask] -= lam
        fw2 = win_cols[ev_rows][np.arange(len(ev_rows)),
                                np.argmax(s2, axis=1)]
        return 1 - (fw2 == widx[ev_rows]).mean()

    def select_even(gate, a, b):
        sel = {}
        for p, (nw, nl) in led_even.items():
            sup = nw + nl
            if gate == "ratio":
                if nl > 0 and nw <= a * nl and sup >= b:
                    sel[p] = (nw, nl)
            else:                      # net
                if nw - nl <= -a and sup >= b:
                    sel[p] = (nw, nl)
        return sel

    best = {("net", None): (1e9, None)}
    for R0 in RS:
        best[("ratio", R0)] = (1e9, None)
    for R in RS:
        for M in MS_R:
            table = select_even("ratio", R, M)
            if not table:
                continue
            for lam in LAMBDAS:
                err = apply_table(table, lam)
                k = ("ratio", R)
                if err < best[k][0]:
                    best[k] = (err, (R, M, lam, len(table)))
        print(f"scan R={R} done", flush=True)
    for T in TS:
        for M in MS_T:
            table = select_even("net", T, M)
            if not table:
                continue
            for lam in LAMBDAS:
                err = apply_table(table, lam)
                if err < best[("net", None)][0]:
                    best[("net", None)] = (err, (T, M, lam, len(table)))
        print(f"scan T={T} done", flush=True)

    out.append(f"\n拆半扫描（池化, 奇评测 {len(ev_rows)} 窗）: "
               f"奇半基线 err {base_err * 100:.4f}%")
    best_ratio = min((v for k, v in best.items() if k[0] == "ratio"),
                     key=lambda t: t[0])
    err_n, cfg_n = best[("net", None)]
    err_r, cfg_r = best_ratio
    out.append(f"  最优[net 门]: T={cfg_n[0]} M={cfg_n[1]} λ={cfg_n[2]:g}"
               f"（{cfg_n[3]} 对）→ err {err_n * 100:.4f}%"
               f"（{(base_err - err_n) * 100:+.4f}pp）")
    out.append(f"  最优[比值门]: R={cfg_r[0]:g} M={cfg_r[1]} "
               f"λ={cfg_r[2]:g}（{cfg_r[3]} 对）→ err {err_r * 100:.4f}%"
               f"（{(base_err - err_r) * 100:+.4f}pp）")
    for R in RS:
        v = best[("ratio", R)]
        if v[1]:
            out.append(f"    R={R:g} 最优: M={v[1][1]} λ={v[1][2]:g}"
                       f"（{v[1][3]} 对）→ {v[0] * 100:.4f}%"
                       f"（{(base_err - v[0]) * 100:+.4f}pp）")

    # 选定比值门最终表 + 分域分解 + 支持度
    Rf, Mf, lf = cfg_r[0], cfg_r[1], cfg_r[2]
    final = {}
    for p, (nw, nl) in led.items():
        sup = nw + nl
        if nl > 0 and nw <= Rf * nl and sup >= Mf:
            final[p] = (nw, nl)
    f_s = {p: v for p, v in final.items() if p[0] in struct}
    dom_of_pair = {}
    sup_hist = Counter(v[0] + v[1] for v in final.values())
    tsv = corpus_std.EVAL / "pairfix_pool_ratio.tsv"
    tsv.write_text("".join(f"{p[0]}\t{p[1]}\t{lf:g}\n" for p in final),
                   encoding="utf-8")
    out.append(f"\n== 比值门最终表（全量账 R={Rf:g} M={Mf} λ={lf:g}）== "
               f"{len(final)} 对 → {tsv.name}")
    out.append(f"  结构集词 {len(f_s)} / 非结构 {len(final) - len(f_s)}; "
               f"支持度: 2 次 {sup_hist.get(2, 0)} / 3 次 {sup_hist.get(3, 0)}"
               f" / 4 次 {sup_hist.get(4, 0)} / ≥5 次 "
               f"{sum(v for k, v in sup_hist.items() if k >= 5)}")

    # 部署表（全量推导 193 对）奇半自检 + 分域分解
    match_f = np.zeros((len(ev_rows), K), dtype=bool)
    for ki, r in enumerate(ev_rows):
        wcr = win_cols[r]
        for j in range(K):
            jn = int(wcr[j])
            if jn < len(wins[r]) and (wins[r][jn], codes[r]) in final:
                match_f[ki, j] = True
    s2_f = s2w[ev_rows].copy()
    s2_f[match_f & (uni_w[ev_rows] < FADE)] -= lf
    fw_f = win_cols[ev_rows][np.arange(len(ev_rows)),
                             np.argmax(s2_f, axis=1)]
    ok0 = f_win[ev_rows] == widx[ev_rows]
    okf = fw_f == widx[ev_rows]
    out.append(f"  部署表奇半自检（全量推导, 过拟合参考）: "
               f"{base_err * 100:.4f}% → {(1 - okf.mean()) * 100:.4f}%"
               f"（{(okf.mean() - ok0.mean()) * 100:+.4f}pp）")
    for did, name in ((0, "单字域"), (1, "多字词域")):
        m = dom_id[ev_rows] == did
        if m.sum():
            out.append(f"    {name}（奇半 {int(m.sum())} 窗, 基线 "
                       f"{(1 - ok0[m].mean()) * 100:.4f}%）: "
                       f"{int((~ok0[m]).sum())} 错 → "
                       f"{int((~okf[m]).sum())} 错"
                       f"（{int((~ok0[m]).sum() - (~okf[m]).sum()):+d}）, "
                       f"{(okf[m].mean() - ok0[m].mean()) * 100:+.4f}pp")

    # 全量账本存档（免重放复用）+ 人工审核辅助表
    ledger_path = corpus_std.EVAL / "pairfix_ledger_pool.jsonl"
    with open(ledger_path, "w", encoding="utf-8") as f:
        for p, (nw, nl) in sorted(led.items(),
                                  key=lambda t: (t[1][0] - t[1][1],
                                                 -(t[1][0] + t[1][1]))):
            f.write(json.dumps({
                "word": p[0], "code": p[1], "win": nw, "loss": nl,
                "struct": p[0] in struct,
                "domains": sorted(dom_seen.get(p, [])),
                "heat": tgt_all.get(p[0], 0),
                "traffic": code_traffic.get(p[1], 0),
                "wants": dict(wants[p].most_common(5)),
                "examples": examples.get(p, []),
            }, ensure_ascii=False) + "\n")
    print(f"ledger {len(led)} pairs -> {ledger_path.name}", flush=True)

    review_path = corpus_std.EVAL / "pairfix_review.tsv"
    hdr = ["词", "码", "λ", "判定", "改对", "改错", "支持", "比值",
           "结构", "证据域", "词热度", "码位流量", "挤掉的目标",
           "例证1", "例证2", "备注"]
    rv = ["\t".join(hdr)]
    rows_r = sorted(final.items(),
                    key=lambda t: (0 if t[0][0] in struct else 1,
                                   -(t[1][0] + t[1][1]),
                                   t[1][0] - t[1][1]))
    for (w, c), (nw, nl) in rows_r:
        sup = nw + nl
        flags = []
        if sup == 1:
            flags.append("孤例")
        if nw > 0:
            flags.append("混合")
        want_str = " ".join(f"{k}×{v}"
                            for k, v in wants.get((w, c), {}).most_common(4))
        doms = "+".join("单" if x == 0 else "词"
                        for x in sorted(dom_seen.get((w, c), [])))
        exs = [re.sub(r"[\x00-\x1f\x7f]+", "⏎",
                      x.replace("\t", " "))
              for x in examples.get((w, c), [])]
        rv.append("\t".join([
            w, c, f"{lf:g}", "", str(nw), str(nl), str(sup),
            f"{(nw / nl if nl else 0):.2f}",
            "S" if w in struct else "N", doms,
            str(tgt_all.get(w, 0)), str(code_traffic.get(c, 0)),
            want_str, exs[0] if exs else "", exs[1] if len(exs) > 1 else "",
            "；".join(flags)]))
    review_path.write_text("\n".join(rv) + "\n", encoding="utf-8")
    print(f"review table {len(rows_r)} pairs -> {review_path.name}",
          flush=True)
    top = sorted(final.items(), key=lambda t: (t[1][0] - t[1][1],
                                               -t[1][1]))[:40]
    out.append("  top40（按净损; [S]/[N]=结构/非结构, [单]/[词]=证据域）:")
    for (w, c), (nw, nl) in top:
        kind = "S" if w in struct else "N"
        ex = examples.get((w, c), [])
        out.append(f"    [{kind}]「{w}」@{c}  改对{nw}/改错{nl} "
                   f"净{nw - nl:+d}  {('; '.join(ex)) if ex else ''}")

    # ── 用户域验证（选定比值门表, fade）──
    table_keys = set(final)
    if table_keys:
        words_all = set(corpus_std.load_pdsp_words())

        def parse_scenes():
            # v2 语料场景制（iter_scenes 已按 M.kind 处理回撤）
            return [[(c["word"], c["code"]) for c in scene]
                    for scene in corpus_std.iter_scenes(TRAIN_LOG,
                                                        reset_mode="strict")]

        def eval_domain(tag, win_data, sids, ce_recs_u, is_word):
            n = len(win_data)
            lens_u = np.array([len(w) for w, *_ in win_data])
            pad_u = np.arange(MAX_CAND)[None, :] >= lens_u[:, None]
            widx_u = np.array([t for _, _, t, _ in win_data])
            ce_maps = []
            for i, (win, _c, _t, _ci) in enumerate(win_data):
                rec = ce_recs_u.get(sids[i])
                ce_maps.append({x: (rec["ce"][x], rec["ntok"][x])
                                for x in win
                                if rec and x in rec["ce"] and x in rec["ntok"]})
            s_ce_u = build_ce_matrix(ce_maps, lens_u, pad_u)
            ce_ok_u = s_ce_u > -1e8
            min_ce_u = np.where(ce_ok_u, s_ce_u, np.inf).min(axis=1)
            min_ce_u = np.where(np.isinf(min_ce_u), 0.0, min_ce_u)
            ce_term_u = np.where(ce_ok_u, s_ce_u, min_ce_u[:, None])
            du_u = np.zeros((n, MAX_CAND))
            uni_mat = np.zeros((n, MAX_CAND))
            rp = FormulaReplay(400.0)
            uni_p = {}
            if is_word:
                by_ci = defaultdict(list)
                for i, (_, _, _, ci) in enumerate(win_data):
                    by_ci[ci].append(i)
                seq = [(c["word"], c["code"])
                       for c in corpus_std.iter_commits(
                           TRAIN_LOG, include_retracted=True)
                       if c["word"] in words_all]
                for ci, (w, c) in enumerate(seq):
                    for i in by_ci.get(ci, ()):
                        win = win_data[i][0]
                        for j, x in enumerate(win):
                            du_u[i, j] = math.log1p(rp.eff_of((x, c)))
                            uni_mat[i, j] = uni_p.get((x, c), 0)
                    rp.bump((w, c))
                    uni_p[(w, c)] = uni_p.get((w, c), 0) + 1
            else:
                scenes = parse_scenes()
                ci_to_idx = {ci: i
                             for i, (_, _, _, ci) in enumerate(win_data)}
                ci = 0
                for scene in scenes:
                    for w, c in scene:
                        ii = ci_to_idx.get(ci)
                        if ii is not None:
                            win = win_data[ii][0]
                            for j, x in enumerate(win):
                                du_u[ii, j] = math.log1p(rp.eff_of((x, c)))
                                uni_mat[ii, j] = uni_p.get((x, c), 0)
                        if w in words_all:
                            rp.bump((w, c))
                            uni_p[(w, c)] = uni_p.get((w, c), 0) + 1
                            ci += 1
            s1_u = np.where(pad_u, -1e9,
                            BETA * du_u + np.log(uni_mat + 1.0))
            cols_u = np.argsort(-s1_u, axis=1, kind="stable")[:, :K]
            s2w_u = np.take_along_axis(s1_u, cols_u, 1) + \
                np.take_along_axis(ce_term_u, cols_u, 1)
            f_u = cols_u[np.arange(n), np.argmax(s2w_u, axis=1)]
            base_wrong = int((f_u != widx_u).sum())
            col_of = {}
            for i in range(n):
                for kj, j in enumerate(cols_u[i].tolist()):
                    col_of[(i, j)] = kj
            hits = []
            max_uni = defaultdict(int)
            for i, (win, code, _t, _ci) in enumerate(win_data):
                for j, x in enumerate(win):
                    if (x, code) in table_keys:
                        kj = col_of.get((i, j))
                        u = int(uni_mat[i, j])
                        max_uni[(x, code)] = max(max_uni[(x, code)], u)
                        if kj is not None and u < FADE:
                            hits.append((i, kj, (x, code)))
            out.append(f"\n== 用户域 {tag}（比值门表 R={Rf:g} M={Mf}）: "
                       f"n={n}, 基线 err {base_wrong}/{n} ==")
            s2_l = s2w_u.copy()
            for i, kj, _p in hits:
                s2_l[i, kj] -= lf
            f2 = cols_u[np.arange(n), np.argmax(s2_l, axis=1)]
            wrong = int((f2 != widx_u).sum())
            out.append(f"  λ={lf:g}: err {wrong}/{n}"
                       f"（vs 基线 {base_wrong - wrong:+d}）")
            det = defaultdict(lambda: [0, 0, 0])
            for i, kj, p in hits:
                if int(f2[i]) == int(f_u[i]):
                    continue
                dd = det[p]
                dd[2] += 1
                if int(f2[i]) == int(widx_u[i]):
                    dd[0] += 1
                else:
                    dd[1] += 1
            for p, (nw, nl, nh) in sorted(
                    det.items(),
                    key=lambda t: (t[1][1] - t[1][0], -t[1][2]),
                    reverse=True):
                kind = "S" if p[0] in struct else "N"
                out.append(f"    [{kind}]「{p[0]}」@{p[1]}  窗{nh} "
                           f"改对{nw}/改错{nl}  uni峰值 "
                           f"{max_uni.get(p, 0)}")
            verdict = "无损失 ✓" if wrong <= base_wrong else "有损失 ✗"
            out.append(f"  判据: {verdict}（{base_wrong} → {wrong}）")
            print(f"user domain {tag} done", flush=True)

        s1_words = {w for w, cs in w2codes.items()
                    if any(len(c) == 1 for c in cs)}
        c2_first = {}
        code_seq = defaultdict(list)
        with open(corpus_std.DICT_PATH, encoding="utf-8") as f:
            in_body = False
            for line in f:
                line = line.rstrip("\n\r")
                if line == "...":
                    in_body = True
                    continue
                if not in_body or not line or line[0] == "#" or \
                        "\t" not in line:
                    continue
                parts = line.split("\t")
                w = parts[0].strip()
                for c in parts[1:]:
                    c = c.strip()
                    if c and not c.startswith("\\") and \
                            w not in code_seq[c]:
                        code_seq[c].append(w)
        c2_first = {c: ws[0] for c, ws in code_seq.items() if len(c) == 2}
        excl_single = set(s1_words)
        for w, cs in w2codes.items():
            if len(w) == 1 and any(c2_first.get(c) == w
                                   for c in cs if len(c) == 2):
                excl_single.add(w)
        excl_word = {w for w, cs in w2codes.items()
                     if len(w) > 1 and any(len(c) <= 2 for c in cs)}

        rows = [json.loads(l) for l in
                open(PROJ / "user_sc_samples.jsonl", encoding="utf-8")]
        sc_ce = {}
        with open(PROJ / "user_sc_ce" / "results.jsonl",
                  encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                sc_ce[r["sid"]] = r
        win_data, sids = [], []
        for r in rows:
            if r["word"] in excl_single:
                continue
            win = (r["cands"][:r["pos"]] + [r["word"]]
                   + r["cands"][r["pos"]:])[:MAX_CAND]
            win_data.append((win, r["code"], win.index(r["word"]),
                             r["ci"]))
            sids.append(corpus_std.window_key_ctx(
                r["ctx"], (r["cands"][:r["pos"]] + [r["word"]]
                           + r["cands"][r["pos"]:])[:MAX_CAND]))
        eval_domain("单字干净集", win_data, sids, sc_ce, False)

        samples = [json.loads(l) for l in
                   open(PROJ / "user_samples_std.jsonl", encoding="utf-8")]
        anchors_u = [json.loads(l) for l in
                     open(PROJ / "user_anchors_std.jsonl",
                          encoding="utf-8")]
        wd_ce = {}
        with open(PROJ / "user_ce_work" / "results.jsonl",
                  encoding="utf-8") as f:
            for l in f:
                try:
                    r = json.loads(l)
                    wd_ce[r["sid"]] = r
                except Exception:
                    pass
        win_data_w, sids_w = [], []
        for s, an in zip(samples, anchors_u):
            if s["word"] in excl_word:
                continue
            win = (s["cands"][:s["pos"]] + [s["word"]]
                   + s["cands"][s["pos"]:])[:MAX_CAND]
            win_data_w.append((win, s.get("code", ""),
                               win.index(s["word"]), an["ci"]))
            sids_w.append(an["sid"])
        eval_domain("多字词干净集", win_data_w, sids_w, wd_ce, True)

    txt = "\n".join(out)
    corpus_std.EVAL.joinpath("derive_pairfix_ratio_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
