#!/usr/bin/env python3
r"""pairfix v2 推导（扩样全枚举口径, 2026-11-16, 用户指令"增加样本量+扣分
集中高频"）。

数据:
  新链 cohort_big_full（anchors 含 dom/du/uni/code/win —— prep_cohort_full
  emit 产物, du/uni 内联免重放）+ CE 结果 cohort_big_full/<ce-sub>/
  results.jsonl（sid 键; 允许 [旧结果播种 + 新算] 混合）
  旧链 5 域（load_domain 全量装载 + streams 重放, CE 已有零成本）:
  cohort_big_sc/_x2/cohort_big/_x2/cohort_big_user —— 证据池化

反事实（与 derive_pairfix_cfrep 同口径, λ=5, FADE=3, K=5）:
  层1 s1 = β·du + ln(uni+1)（du = ln(1+eff) 事前值）稳定降序前 5;
  层2 s2 = s1 + 加权 CE（ntok 分档 (1, w2, w3), 缺 CE 用窗内最低有效 CE）
  → argmax 胜者; 胜者 uni<FADE 激活, 扣 λ 看新 argmax:
    cf_fix = 基线错 → want（真值, 需在前5）回首位; cf_brk = 基线对 → 被顶掉
  账本按 ui 奇偶切两个独立用户半（掩码守卫）。

门（全参数化, 用户定案"门控自然收敛"）:
  ① G3 双半复制: 两半各自 cf_fix≥1 且 cf_brk==0
  ② 样本门: 合计激活窗 sup ≥ --emin 且合计 cf_fix ≥ --mmin
  ③ 频次门: 罚词 DP 流暴露度 freq ≥ --fmin（count 阶段 _freq.json）
  ④ 总改错 == 0
  审核表按期望收益 = 合计 cf_fix × 码位流量 降序。

产出（--tag 05bx）:
  eval/pairfix_pool_cfrep_<tag>.tsv      词/码/λ
  eval/pairfix_cfrep_review_<tag>.tsv    24 列审核表（19 基础列 + 语料频次/
                                         修对率/读音还原/多音低读/读音倾向;
                                         判定列空, 备注旧判定）
  eval/pairfix_gate_ledger_<tag>.jsonl   全量账本（分半+uni/龄分层）
  eval/derive_pairfix_full_report_<tag>.txt

用法:
  python -X utf8 derive_pairfix_full.py --tag 05bx [--sweep] [--no-validate]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
import summary_4x4
import derive_pairfix_ratio
import pairfix_pron
from derive_pairfix_ratio import FADE, K, load_domain, load_w2codes
from summary_4x4 import build_ce_matrix
from corpus_std import MAX_CAND

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
EVAL = corpus_std.EVAL
LAM = 5.0
CHUNK = 300_000

CJK_RE = re.compile(r"[一-鿿]")


def ctx_ok(s):
    """上文有效性门（2026-11-16 用户审核反馈：标点/无 CJK 上文的窗过滤）。
    文章边界标点串上开的窗对生产场景无意义。"""
    return bool(s) and bool(CJK_RE.search(s))


OLD_DOMAINS = [
    # 2026-11-18 研究范围收窄：单字 3 码 / 用户域 / 小词库链已删除，
    # 旧域池化只剩 wiki 多字词大词库（全枚举链见 --chain cohort_big_full）
    ("cohort_big", "big", None, "多字词", "wiki"),
]


class Ledger:
    """逐对账本（跨域/跨块累加; halves: 0=偶半 1=奇半）。"""

    def __init__(self):
        self.d = defaultdict(lambda: dict(
            fix=[0, 0], brk=[0, 0], sup=[0, 0], nw=[0, 0], nl=[0, 0],
            wants=Counter(), doms=Counter(), arms=Counter(),
            uni_cnt=Counter(), ex=[]))

    def add(self, pair, half, fixed, broken, fw_ok, want_w, dom, arm, uni,
            depth, ctx):
        d = self.d[pair]
        h = 0 if half == 0 else 1
        d["fix"][h] += int(fixed)
        d["brk"][h] += int(broken)
        d["sup"][h] += 1
        d["nw"][h] += int(fw_ok)
        d["nl"][h] += int(not fw_ok)
        if not fw_ok and want_w:
            d["wants"][want_w] += 1
            d["doms"][dom] += 1
            if len(d["ex"]) < 2:
                d["ex"].append(f"[{dom}]…{ctx[-14:]}｜want=「{want_w}」"
                               f"@{p_code(pair)} 顶上=「{pair[0]}」")
        d["arms"][arm] += 1
        d["uni_cnt"][int(uni)] += 1
        _ = depth  # 保留接口; 分层看 uni/arms 已足够


def p_code(pair):
    return pair[1]


def layer12(du, uni, pad, s_ce, ce_ok, min_ce, beta, w2, w3):
    """→ s2w, win_cols, arg2, f_win, uni_w（与 cfrep 向量化逐位同口径）。"""
    n = len(du)
    s1 = np.where(pad, -1e9, beta * du + np.log(uni + 1.0))
    ce_term = np.where(ce_ok, s_ce, min_ce[:, None])
    win_cols = np.argsort(-s1, axis=1, kind="stable")[:, :K]
    s2w = np.take_along_axis(s1, win_cols, 1) + \
        np.take_along_axis(ce_term, win_cols, 1)
    arg2 = np.argmax(s2w, axis=1)
    f_win = win_cols[np.arange(n), arg2]
    uni_w = np.take_along_axis(uni, win_cols, 1)
    return s2w, win_cols, arg2, f_win, uni_w


def accumulate(led, wins, codes, dom_names, du, uni, pad, s_ce, ce_ok,
               min_ce, widx, ui, arm, depth, beta, w2, w3, ctxs,
               table=None, stat=None):
    """向量化层1/层2/反事实 → 逐行账本（table 非 None = 部署态验证模式:
    前5内所有命中表且 uni<FADE 的候选统一扣 λ, 返回 (ok0, ok1) 计数）。"""
    n = len(widx)
    s2w, win_cols, arg2, f_win, uni_w = layer12(du, uni, pad, s_ce, ce_ok,
                                                min_ce, beta, w2, w3)
    ok0 = ok1 = 0
    if table is not None:
        match = np.zeros((n, K), dtype=bool)
        for j in range(K):
            jn = win_cols[:, j]
            uu = uni_w[:, j]
            rowok = np.array([jn[r] < len(wins[r]) for r in range(n)])
            hit = np.array([wins[r][jn[r]] if rowok[r] else "" for r in range(n)])
            m = np.array([(hit[r], codes[r]) in table and uu[r] < FADE
                          and rowok[r] for r in range(n)])
            match[:, j] = m
        s2t = s2w.copy()
        s2t[match] -= LAM
        f2 = win_cols[np.arange(n), np.argmax(s2t, axis=1)]
        wv = np.array([wins[r][widx[r]] for r in range(n)])
        ok0 = int((f_win == widx).sum())
        ok1 = int((f2 == widx).sum())
        return ok0, ok1
    tj = win_cols == widx[:, None]
    want_pos = np.where(tj.any(axis=1), tj.argmax(axis=1), -1)
    act = uni_w[np.arange(n), arg2] < FADE
    s2mod = s2w.copy()
    aidx = np.where(act)[0]
    s2mod[aidx, arg2[aidx]] -= LAM
    new_arg = np.argmax(s2mod, axis=1)
    halves = (ui % 2 == 0).astype(np.int64)
    n_act = 0
    for r in np.where(act)[0]:
        row = wins[r]
        cr = codes[r]
        fw = int(f_win[r])
        fw_ok = fw == int(widx[r])
        fixed = (not fw_ok) and want_pos[r] >= 0 and new_arg[r] == want_pos[r]
        broken = fw_ok and new_arg[r] != arg2[r]
        led.add((row[fw], cr), int(halves[r]), fixed, broken, fw_ok,
                None if fw_ok else row[int(widx[r])],
                dom_names[r], int(arm[r]), float(uni_w[r, arg2[r]]),
                int(depth[r]), ctxs[r] if ctxs is not None else "")
        n_act += 1
    if stat is not None:
        stat["windows"] += n
        stat["act"] += n_act
    return ok0, ok1


def ce_store(path):
    """results.jsonl → sid → (词元组, ce[n×3] f32, ntok[n] i8) 紧凑索引。"""
    store = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ce = r.get("ce") or {}
            if not ce:
                continue
            ntok = r.get("ntok") or {}
            words = tuple(sys.intern(w) for w in ce.keys())
            arr = np.array([ce[w] for w in words], dtype=np.float32)
            nt = np.array([ntok.get(w, 0) for w in words], dtype=np.int8)
            store[sys.intern(r["sid"])] = (words, arr, nt)
    return store


def load_old_domains(beta, w2, w3, ce_sub, led, traffic, stat):
    _, wc_sc3 = corpus_std.load_pdsp_dict_single3()
    for dirname, bump_mode, map_kind, name, side in OLD_DOMAINS:
        wc = wc_sc3 if map_kind == "sc3" else None
        d = load_domain(dirname, bump_mode, wc or wc_sc3, first_code_g,
                        ce_sub=ce_sub)
        n = len(d["widx"])
        msk = np.array([ctx_ok(c) for c in d["ctxs"]])
        stat["ctx_filtered"] += int((~msk).sum())
        dom_names = [name] * n
        depth = np.full(n, -1, dtype=np.int64)
        traffic.update(np.array(d["codes"])[msk])
        accumulate(led, [w for w, m in zip(d["wins"], msk) if m],
                   np.array(d["codes"])[msk], [x for x, m in zip(dom_names, msk) if m],
                   d["du"][msk], d["uni"][msk], d["pad"][msk], d["s_ce"][msk],
                   d["ce_ok"][msk], d["min_ce"][msk], d["widx"][msk],
                   d["ui"][msk], d["arm"][msk], depth[msk], beta, w2, w3,
                   [c for c, m in zip(d["ctxs"], msk) if m], stat=stat)
        print(f"old {dirname}: {n} windows (ctx 过滤 {int((~msk).sum())})",
              flush=True)
        del d


def process_chunk(buf, store, beta, w2, w3, led, traffic, stat):
    n0 = len(buf)
    buf = [r for r in buf if ctx_ok(r["ctx"])]
    stat["ctx_filtered"] += n0 - len(buf)
    n = len(buf)
    lens = np.array([min(len(r["win"]), MAX_CAND) for r in buf])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    du = np.zeros((n, MAX_CAND))
    uni = np.zeros((n, MAX_CAND))
    widx = np.zeros(n, dtype=np.int64)
    ui = np.zeros(n, dtype=np.int64)
    arm = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    wins = []
    codes = []
    dom_names = []
    ctxs = []
    no_ce = 0
    for i, r in enumerate(buf):
        win = r["win"][:MAX_CAND]
        wins.append(win)
        codes.append(r["code"])
        dom_names.append(r["dom"])
        ctxs.append(r["ctx"])
        widx[i] = win.index(r["w"])
        ui[i] = r["ui"]
        arm[i] = r["arm"]
        depth[i] = r["depth"]
        d = r["du"][:MAX_CAND]
        u = r["uni"][:MAX_CAND]
        du[i, :len(d)] = d
        uni[i, :len(u)] = u
    ce_maps = []
    for i, r in enumerate(buf):
        rec = store.get(r["sid"])
        if rec is None:
            no_ce += 1
            ce_maps.append({x: None for x in wins[i]})
            continue
        words, arr, nt = rec
        idx = {w: j for j, w in enumerate(words)}
        ce_maps.append({x: ((arr[idx[x]].tolist(), int(nt[idx[x]]))
                            if x in idx else None) for x in wins[i]})
    s_ce = build_ce_matrix(ce_maps, lens, pad)
    ce_ok = s_ce > -1e8
    min_ce = np.where(ce_ok, s_ce, np.inf).min(axis=1)
    min_ce = np.where(np.isinf(min_ce), 0.0, min_ce)
    stat["no_ce"] += no_ce
    for dom in ("big", "sc"):
        sel = [i for i, x in enumerate(dom_names) if x == dom]
        if not sel:
            continue
        sel = np.array(sel)
        dn = [dom_names[i] for i in sel]
        accumulate(led, [wins[i] for i in sel], [codes[i] for i in sel], dn,
                   du[sel], uni[sel], pad[sel], s_ce[sel], ce_ok[sel],
                   min_ce[sel], widx[sel], ui[sel], arm[sel], depth[sel],
                   beta, w2, w3, [ctxs[i] for i in sel], stat=stat)
    traffic.update(codes)
    return n


def gates_pass(led, freq, emin, mmin, fmin):
    def strict(d):
        return d["fix"][0] >= 1 and d["brk"][0] == 0 and \
            d["fix"][1] >= 1 and d["brk"][1] == 0

    sel = []
    for p, d in led.d.items():
        sup = d["sup"][0] + d["sup"][1]
        fix = d["fix"][0] + d["fix"][1]
        brk = d["brk"][0] + d["brk"][1]
        if strict(d) and sup >= emin and fix >= mmin and brk == 0 and \
                freq.get(p[0], 0) >= fmin:
            sel.append(p)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="05bx")
    ap.add_argument("--ce-sub", default="ce_05b")
    ap.add_argument("--beta", type=float, default=14.0)
    ap.add_argument("--w2", type=float, default=1.10)
    ap.add_argument("--w3", type=float, default=0.60)
    ap.add_argument("--emin", type=int, default=10)
    ap.add_argument("--mmin", type=int, default=5)
    ap.add_argument("--fmin", type=int, default=200)
    ap.add_argument("--chain", default="cohort_big_full")
    ap.add_argument("--freq-file",
                    default="pairfix_full_pot_full_freq.json")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()

    global first_code_g
    w2codes, first_code_g = load_w2codes()
    summary_4x4.W2 = a.w2
    summary_4x4.W3 = a.w3
    derive_pairfix_ratio.BETA = a.beta

    led = Ledger()
    stat = Counter()
    traffic = Counter()

    print(f"== load old domains (ce={a.ce_sub}) ==", flush=True)
    load_old_domains(a.beta, a.w2, a.w3, a.ce_sub, led, traffic, stat)

    anchors = PROJ / a.chain / "anchors.jsonl"
    results = PROJ / a.chain / a.ce_sub / "results.jsonl"
    print(f"== ce store {results.name} ==", flush=True)
    store = ce_store(results)
    print(f"ce records {len(store)}", flush=True)
    print(f"== new chain {anchors.name} ==", flush=True)
    n_new = 0
    buf = []
    with open(anchors, encoding="utf-8") as f:
        for line in f:
            buf.append(json.loads(line))
            if len(buf) >= CHUNK:
                n_new += process_chunk(buf, store, a.beta, a.w2, a.w3, led,
                                       traffic, stat)
                buf = []
                print(f"  processed {n_new}", flush=True)
    if buf:
        n_new += process_chunk(buf, store, a.beta, a.w2, a.w3, led, traffic,
                               stat)
    print(f"new chain: {n_new} windows (缺CE窗 {stat['no_ce']}, ctx 过滤 "
          f"{stat['ctx_filtered']}), 激活行 {stat['act']}/{stat['windows']}",
          flush=True)

    freq = json.loads((EVAL / a.freq_file).read_text(encoding="utf-8"))

    out = ["=" * 96,
           f"pairfix v2 推导（扩样全枚举; β={a.beta:g} kCeW=1.0/{a.w2:g}/"
           f"{a.w3:g} λ={LAM:g} FADE={FADE}）",
           f"池化: 旧5域 + 新链 {n_new} 窗; 激活行 {stat['act']}",
           f"门: G3双半 + sup≥{a.emin} + fix≥{a.mmin} + brk==0 + "
           f"freq≥{a.fmin}",
           "=" * 96]

    sel = gates_pass(led, freq, a.emin, a.mmin, a.fmin)
    out.append(f"主门: {len(sel)} 对")
    if a.sweep:
        out.append("sweep（对数）:")
        for emin in (5, 10, 20, 40):
            row = []
            for mmin in (3, 5, 10):
                for fmin in (0, 200, 500):
                    n_s = len(gates_pass(led, freq, emin, mmin, fmin))
                    row.append(f"e{emin}/m{mmin}/f{fmin}={n_s}")
            out.append("  " + "  ".join(row))
        sel = gates_pass(led, freq, a.emin, a.mmin, a.fmin)

    if sel:
        sups = sorted(sum(led.d[p]["sup"]) for p in sel)
        fixs = sorted(sum(led.d[p]["fix"]) for p in sel)
        uni_all = Counter()
        for p in sel:
            uni_all.update(led.d[p]["uni_cnt"])
        out.append(f"入表对: sup 中位 {sups[len(sups) // 2]} / 最小 {sups[0]}"
                   f" / 最大 {sups[-1]}; fix 中位 {fixs[len(fixs) // 2]}")
        out.append(f"  事件 uni 分布 {dict(sorted(uni_all.items()))} "
                   f"（uni=0/1/2 = 部署 fade 内全域）")

    # 旧表对比
    old_fp = EVAL / "pairfix_pool_cfrep_05b.tsv"
    if old_fp.exists():
        old_pairs = set()
        for line in old_fp.read_text(encoding="utf-8").splitlines():
            c = line.rstrip("\r").split("\t")
            if len(c) >= 2 and c[0] and c[1]:
                old_pairs.add((c[0], c[1]))
        new_pairs = set(sel)
        out.append(f"与旧 G3 表（{len(old_pairs)} 对）: 留存 "
                   f"{len(new_pairs & old_pairs)}, 新增 "
                   f"{len(new_pairs - old_pairs)}, 退役 "
                   f"{len(old_pairs - new_pairs)}")

    # 部署态 in-sample 验证（第二遍流式; 表 = 本次产出）
    if not a.no_validate and sel:
        table = set(sel)
        print("== validate pass ==", flush=True)
        ok0 = ok1 = 0
        _, wc_sc3 = corpus_std.load_pdsp_dict_single3()
        for dirname, bump_mode, map_kind, name, side in OLD_DOMAINS:
            wc = wc_sc3 if map_kind == "sc3" else None
            d = load_domain(dirname, bump_mode, wc or wc_sc3, first_code_g,
                            ce_sub=a.ce_sub)
            msk = np.array([ctx_ok(c) for c in d["ctxs"]])
            o0, o1 = accumulate(led, [w for w, m in zip(d["wins"], msk) if m],
                                np.array(d["codes"])[msk],
                                [name] * int(msk.sum()),
                                d["du"][msk], d["uni"][msk], d["pad"][msk],
                                d["s_ce"][msk], d["ce_ok"][msk],
                                d["min_ce"][msk], d["widx"][msk],
                                d["ui"][msk], d["arm"][msk],
                                np.full(int(msk.sum()), -1, dtype=np.int64),
                                a.beta, a.w2, a.w3,
                                [c for c, m in zip(d["ctxs"], msk) if m],
                                table=table)
            ok0 += o0
            ok1 += o1
            del d
        buf = []
        with open(anchors, encoding="utf-8") as f:
            for line in f:
                buf.append(json.loads(line))
                if len(buf) >= CHUNK:
                    b0, b1 = _validate_chunk(buf, store, table, a)
                    ok0 += b0
                    ok1 += b1
                    buf = []
        if buf:
            b0, b1 = _validate_chunk(buf, store, table, a)
            ok0 += b0
            ok1 += b1
        tot = ok0 + (stat["windows"] - ok0)  # noqa
        n_all = stat["windows"]
        if n_all:
            out.append(f"\n部署态全量自测（in-sample）: err "
                       f"{(1 - ok0 / n_all) * 100:.4f}% → "
                       f"{(1 - ok1 / n_all) * 100:.4f}% "
                       f"（修对 +{ok1 - ok0} 窗）")

    # ── 表与审核表 ──
    sel_sorted = sorted(sel, key=lambda p: -sum(led.d[p]["fix"])
                        * traffic.get(p[1], 1))
    tbl_path = EVAL / f"pairfix_pool_cfrep_{a.tag}.tsv"
    tbl_path.write_text("".join(f"{p[0]}\t{p[1]}\t{LAM:g}\n"
                                for p in sel_sorted), encoding="utf-8")
    out.append(f"\n表 → {tbl_path.name}: {len(sel_sorted)} 对, λ={LAM:g}")

    struct = {w for w, cs in w2codes.items()
              if len(cs) >= 2 and any(
                  len(cs[i]) == len(cs[j]) and cs[i] != cs[j]
                  for i in range(len(cs)) for j in range(i + 1, len(cs)))}
    old_verdict = {}
    audited = Path(r"D:/OneDrive/typing/拼读双拼/"
                   "pairfix_review_union_05b_audited.tsv")
    if audited.exists():
        rows = audited.read_text(encoding="utf-8-sig").split("\n")
        hdr = rows[0].rstrip("\r").split("\t")
        iw, ic, iv, ir = (hdr.index("词"), hdr.index("码"),
                          hdr.index("判定"), hdr.index("备注"))
        for line in rows[1:]:
            c = line.rstrip("\r").split("\t")
            if len(c) > ir:
                old_verdict[(c[iw], c[ic])] = (c[iv].strip(), c[ir])

    hdr19 = ["词", "码", "λ", "判定", "cf修对偶", "cf改错偶", "cf修对奇",
             "cf改错奇", "合计修对", "v2旧态", "被挤目标", "证据域", "结构",
             "词热度", "码位流量", "例证1", "例证2", "来源", "备注",
             # 2026-11-16 审核准则追加（用户指令: 例证外补词频先验 + 读音证据,
             # 系统性偏高才修正）——追加在尾部, 前 19 列号稳定兼容旧链
             "语料频次", "修对率", "读音还原", "多音低读", "读音倾向"]
    DOM_DISP = {"big": "多字词", "sc": "单字3码"}  # 新链原始域名 → 展示名
    cc, _cr = pairfix_pron.load_pron()
    cfreq = pairfix_pron.corpus_freq([p[0] for p in sel_sorted])
    rv = ["\t".join(hdr19)]
    for p in sel_sorted:
        d = led.d[p]
        wants = "、".join(f"{w}×{n}" for w, n in d["wants"].most_common(3))
        ov = old_verdict.get((p[0], p[1]))
        note = f"[旧判定:{ov[0] or '空'}]" if ov else ""
        if ov and ov[1]:
            note += f"[旧备注]{ov[1]}"
        side = "user" if "用户域" in d["doms"] and len(d["doms"]) == 1 \
            else ("wiki+user" if "用户域" in d["doms"] else "wiki")
        ev = pairfix_pron.pair_evidence(p[0], p[1], cc, _cr)
        n_fix = d["fix"][0] + d["fix"][1]
        n_sup = d["sup"][0] + d["sup"][1]
        rv.append("\t".join([
            p[0], p[1], f"{LAM:g}", "",
            str(d["fix"][0]), str(d["brk"][0]),
            str(d["fix"][1]), str(d["brk"][1]),
            str(d["fix"][0] + d["fix"][1]),
            "留存" if (p[0], p[1]) in old_verdict else "新增",
            wants, "、".join(sorted(DOM_DISP.get(x, x) for x in d["doms"])),
            "S" if p[0] in struct else "",
            str(freq.get(p[0], 0)), str(traffic.get(p[1], 0)),
            d["ex"][0] if d["ex"] else "",
            d["ex"][1] if len(d["ex"]) > 1 else "",
            side, note,
            str(cfreq.get(p[0], 0)),
            f"{n_fix / n_sup * 100:.0f}%({n_fix}/{n_sup})" if n_sup else "",
            pairfix_pron.fmt_recover(ev), pairfix_pron.fmt_flagged(ev),
            ev["tendency"]]))
    rv_path = EVAL / f"pairfix_cfrep_review_{a.tag}.tsv"
    rv_path.write_text("\n".join(rv) + "\n", encoding="utf-8")
    out.append(f"审核表 → {rv_path.name}: {len(rv) - 1} 行")

    led_path = EVAL / f"pairfix_gate_ledger_{a.tag}.jsonl"
    with open(led_path, "w", encoding="utf-8") as f:
        for p in sorted(led.d):
            d = led.d[p]
            f.write(json.dumps(
                {"word": p[0], "code": p[1],
                 "even": {"fix": d["fix"][0], "brk": d["brk"][0],
                          "sup": d["sup"][0]},
                 "odd": {"fix": d["fix"][1], "brk": d["brk"][1],
                         "sup": d["sup"][1]},
                 "uni": dict(d["uni_cnt"]),
                 "doms": dict(d["doms"]), "ex": d["ex"]},
                ensure_ascii=False) + "\n")
    out.append(f"账本 → {led_path.name}: {len(led.d)} 对")

    rpt = EVAL / f"derive_pairfix_full_report_{a.tag}.txt"
    rpt.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def _validate_chunk(buf, store, table, a):
    """部署态验证: 新链分块 apply 表。"""
    buf = [r for r in buf if ctx_ok(r["ctx"])]
    n = len(buf)
    lens = np.array([min(len(r["win"]), MAX_CAND) for r in buf])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    du = np.zeros((n, MAX_CAND))
    uni = np.zeros((n, MAX_CAND))
    widx = np.zeros(n, dtype=np.int64)
    ui = np.zeros(n, dtype=np.int64)
    wins = []
    codes = []
    for i, r in enumerate(buf):
        win = r["win"][:MAX_CAND]
        wins.append(win)
        codes.append(r["code"])
        widx[i] = win.index(r["w"])
        ui[i] = r["ui"]
        d = r["du"][:MAX_CAND]
        u = r["uni"][:MAX_CAND]
        du[i, :len(d)] = d
        uni[i, :len(u)] = u
    ce_maps = []
    for i, r in enumerate(buf):
        rec = store.get(r["sid"])
        if rec is None:
            ce_maps.append({x: None for x in wins[i]})
            continue
        words, arr, nt = rec
        idx = {w: j for j, w in enumerate(words)}
        ce_maps.append({x: ((arr[idx[x]].tolist(), int(nt[idx[x]]))
                            if x in idx else None) for x in wins[i]})
    s_ce = build_ce_matrix(ce_maps, lens, pad)
    ce_ok = s_ce > -1e8
    min_ce = np.where(ce_ok, s_ce, np.inf).min(axis=1)
    min_ce = np.where(np.isinf(min_ce), 0.0, min_ce)
    arm = np.zeros(n, dtype=np.int64)
    depth = np.full(n, -1, dtype=np.int64)
    led_dummy = Ledger()
    o0, o1 = accumulate(led_dummy, wins, codes, [""] * n, du, uni, pad,
                        s_ce, ce_ok, min_ce, widx, ui, arm, depth,
                        a.beta, a.w2, a.w3, None, table=table)
    return o0, o1


if __name__ == "__main__":
    main()
