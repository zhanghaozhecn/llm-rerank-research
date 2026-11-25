#!/usr/bin/env python3
"""前缀确认：同档比较（class-first）大样本评估 + 档位 CE 差特征（2026-11-26）。

背景：2026-11-26 结构证据从「窗口融合序第 1」改为「lead 的同字数档首选」
（1 字档 / 多字档）；小样本（luna 混合窗 62 个可判窗）显示新增 15 次触发、
结构精度 86.67%、原有 31 次不丢。用户要求：① 看**经过树模型之后**新增触发
的精度/召回；② 树模型里体现「档位不同（一字/多字）」与「跨档 CE 差」，
用大样本（召回、精度）决定是否保留分档比较。

做法：
  * 重建全部 class-first 行（**不筛融合序第 1**）——特征按**生产口径以 lead
    为锚**（ce1/ce2/ce3/ce_w/r2/ss 簇都是 lead 的），与 llm_filter.cc 的
    i1 = lead 一致；旧脚本 `prefix_confirm_model.py` 只收 rank0 行，新增触发
    根本不在缓存里。
  * 模型臂（诚实分半 OOF，句子奇偶 A/B，5 折作稳健性复核）：
      V1 旧口径模型：53 特征、只在**旧规则行**上训练 → 给新规则全集打分
         （≈ 现行已发布模型遇到新触发的行为）
      V2 同档规则 + 53 特征
      V3 同档规则 + 53 + 档位/跨档 CE 差特征
  * 等量口径：现行生产结构触发量 N0（旧规则 ∧ P≥0.95 ∧ ctx≥4），各臂按
    OOF 分取 90% / 95% / 100% N0 条，比精度、K=5 净收益、新增占比；
    再给「同精度最大触发量」（召回上限）与句子 cluster bootstrap CI。
  * 另给**现行导出模型**（rime-llm-rerank/user/prefix_gate_model.lua，
    训练时只见过旧规则行）直接作用于新增行的结果（row-level OOS，句子
    可能共享，仅作参考）。

跑法（研究库根目录）：
  python eval/calib/prefix_class_gate.py build [--window top6|budget]
  python eval/calib/prefix_class_gate.py eval --folds 5 --window budget
  python eval/calib/prefix_class_gate.py cross   # 跨口径 90/95/100% 量表
  python eval/calib/prefix_class_gate.py current # 固定发布 τ：整体/分档召回精度
  python eval/calib/prefix_class_gate.py ablate  # 53 特征分组消融
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.path.insert(0, "eval/lib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L            # noqa: E402
import prefix_confirm_model as M      # noqa: E402

KCE = (1.0, 1.10, 0.60)
K_WINDOW = L.K_WINDOW
DATA = [("luna", L.OUT, "gen20.tsv"),
        ("big", L.EVAL / "prefix_big", "gen.tsv")]
MODE = "top6"
CACHE = L.EVAL / "prefix_class_rows.jsonl"
LUA_MODEL = (L.PROJ.parent / "rime-llm-rerank" / "user" /
             "prefix_gate_model.lua")

BASE = [c for c in M.FEATS
        if c not in {"m", "n_syl_typed", "partial_syl", "n_syl_total",
                     "is_user"}] + M.DIST_SS

# 档位 / 跨档 CE 差特征（全部生产可算：窗口 + CE + lead 文本）
CLASS_FEATS = [
    "lead_is_global_first", "lead_rank_global",
    "n_multi", "multi_ratio",
    "cf_other_ce", "cf_other_ce1", "cf_other_chars",
    "same2_ce_gap_w", "same2_ce_gap_1", "same2_chars",
    "single_first_ce", "multi_first_ce",
    "single_multi_gap_w", "single_multi_gap_1",
    "lead_class_n", "lead_class_mass_share",
]


# ── 一窗 → 生产口径（lead 锚定）特征 ─────────────────────────────────
def ce_parts(rec):
    """缓存记录 (components, ntok) → (加权 CE, ce1, 有效 token 数)。

    **必须按 ntok 截断**：缓存里未用的 token 槽是 `-1.0` 占位哨兵（不是
    -1e8），直接当数值会把 1-token 词的 CE 和算成负的（生产只加真有该
    token 的分量）。"""
    comps = (rec[0] if rec else None) or []
    nt = min(int(rec[1] or 0), 3) if rec else 0
    s, first = 0.0, None
    for k in range(min(nt, len(comps))):
        v = comps[k]
        if v is None or v <= -1e8:
            continue
        if k == 0:
            first = v
        s += KCE[k] * v
    return (s, first, nt) if nt else None


def window_ced(ctx, pool, ce):
    """全局 CE 缓存（sid → {text: (分量, ntok)}）→ 本窗 text → (分量, ntok)。"""
    d = {}
    for ln, sid in L.ce_lines(ctx, [c[0] for c in pool]):
        rr = ce.get(sid)
        if not rr:
            continue
        for t in ln["cands"]:
            if t in rr and rr[t][0] is not None:
                d[t] = rr[t]
    return d


def build_window(w, pool, gen, ced, syls, cfreq, tag, mode="top6"):
    n = len(w["code"])
    texts = [c[0] for c in pool]
    starts = [c[1] for c in pool]
    ends = [c[2] for c in pool]
    gshow = gen[0]["text"] if gen else None
    usable = [i for i, t in enumerate(texts) if t != gshow]
    if mode == "top6":
        win = usable[:K_WINDOW]
    elif mode == "full":
        win = usable
    else:
        # 现行生产准入（2026-11-26）：层1 序扫描（冷账本 = 引擎序），
        # 1-token 免费、多 token 花 min(ntok,3)-1，预算 12。
        win, budget = [], 12
        for i in usable:
            r = ce_parts(ced.get(texts[i]))
            if r is None:
                continue
            nt = r[2]
            cost = 0 if nt <= 1 else min(nt, 3) - 1
            if cost <= budget:
                win.append(i)
                budget -= cost
    if len(win) < 2:
        return None, "win_lt2"
    cew, ce1v, ntok = {}, {}, {}
    for i in win:
        r = ce_parts(ced.get(texts[i]))
        if r is None:
            return None, "ce"
        cew[i], ce1v[i], ntok[i] = r
    order = sorted(win, key=lambda i: (cew[i], i))
    lead, p, le, sent, d = M.lead_stats(gen)
    lead_i = None
    for i in win:
        if texts[i] == lead and ends[i] == le and starts[i] == 0:
            lead_i = i
            break
    if lead_i is None:
        return None, "lead_not_in_window"
    lead_single = len(lead) == 1
    cf = None
    for i in order:
        if (len(texts[i]) == 1) == lead_single:
            cf = i
            break
    if cf != lead_i:
        return None, "not_class_first"
    r2i = next((i for i in order if i != lead_i), None)

    i1 = lead_i
    lead_ce = cew[i1]
    lead_nt = ntok[i1]
    lead_len = len(lead)
    r2_ce = cew[r2i] if r2i is not None else 0.0
    r2_ce1 = ce1v.get(r2i, -1.0) if r2i is not None else -1.0
    r2_chars = len(texts[r2i]) if r2i is not None else 0
    wm = (r2_ce - lead_ce) if r2i is not None else 0.0
    # n_single / n_same_span 沿用旧脚本口径（收集池，不是窗）——与已发布
    # 模型的训练分布一致；生产实现扫 candidates_（窗），是既有小偏差。
    n_single = sum(1 for c in pool if len(c[0]) == 1)
    n_same_span = sum(1 for c in pool if c[2] == le)
    syl = len(sent)
    cum, n_syl_typed = 0, 0
    for sy in syls[w["m"]:]:
        cum += len(sy)
        if cum <= n:
            n_syl_typed += 1
        else:
            break
    partial = 1.0 if cum > n else 0.0
    cf_min = min((cfreq.get(ch, -13.0) for ch in lead), default=-13.0)
    cf_mean = sum(cfreq.get(ch, -13.0) for ch in lead) / max(lead_len, 1)
    cov = sum(len(sy) for sy in syls[w["m"]:w["m"] + lead_len])
    same_py = (le == cov) and (w["text"][:lead_len] != lead)
    rec = ced[texts[i1]]
    comps = (rec[0] if rec else None) or []

    def comp(k):
        return (comps[k] if len(comps) > k and comps[k] is not None
                and comps[k] > -1e8 else -1.0)

    f = dict(
        p=p, logit_p=math.log(max(p, 1e-6) / max(1 - p, 1e-6)),
        d=min(d, 40.0), top_score=min(-gen[0]["score"], 60.0),
        n_leads=len({x["lead"] for x in gen}),
        cnt_lead=sum(1 for x in gen if x["lead"] == lead),
        n_gen=len(gen), sent_len=syl, lead_len=lead_len,
        srem=syl - lead_len, lead_end=le,
        lead_chars_cov=lead_len / max(le, 1),
        wm=wm, n_window=len(order), n_pool=len(pool), code_len=n,
        n_syl_typed=n_syl_typed, partial_syl=partial,
        n_syl_total=len(syls[w["m"]:]),
        ctx_len=len(w["ctx"]), m=w["m"],
        is_user=1.0 if w["domain"] == "user" else 0.0,
        ce1=comp(0), ce2=comp(1), ce3=comp(2),
        ce_w=lead_ce, ntok=lead_nt,
        chars_per_tok=lead_len / max(lead_nt, 1),
        ce_per_tok=lead_ce / max(lead_nt, 1),
        ce_per_char=lead_ce / max(lead_len, 1),
        multi_tok=1.0 if lead_nt > 1 else 0.0,
        ce1_r2=r2_ce1, ce_w_r2=r2_ce,
        ntok_r2=ntok.get(r2i, 0) if r2i is not None else 0,
        ce_per_tok_r2=(r2_ce / max(ntok.get(r2i, 1), 1)
                       if r2i is not None else 0.0),
        ntok_gap=(ntok.get(r2i, 0) - lead_nt) if r2i is not None else 0,
        ce1_per_char=(ce1v[i1] / max(lead_len, 1)
                      if ce1v.get(i1) is not None else -1.0),
        r2_chars=r2_chars, r2_single=1.0 if r2_chars == 1 else 0.0,
        ce1_gap=(r2_ce1 - ce1v[i1]) if (r2i is not None
                                     and ce1v.get(i1) is not None
                                     and r2_ce1 is not None
                                     and r2_ce1 >= 0.0) else 0.0,
        n_single=n_single, n_same_span=n_same_span,
        lead_single=1.0 if lead_single else 0.0,
        single_ratio=n_single / max(len(pool), 1),
        char_freq=cf_min, char_freq_mean=cf_mean,
    )
    g = ss_feats(order, texts, ends, cew, ce1v, lead_i, lead_ce, le)
    c = class_feats(order, texts, cew, ce1v, lead_i, lead_single, win)
    ok = w["text"].startswith(lead)
    return dict(
        wid=w["wid"], sid=w["sid"], tag=tag, domain=w["domain"],
        ok=ok, fire=p >= M.PGATE, old_rule=(order[0] == lead_i),
        lead=lead, lead_syl=w["code"][:le] if le > 0 else "",
        text=w["text"], m=w["m"], same_py=same_py,
        truth_prefix=w["text"][:lead_len],
        f=f, g=g, c=c), None


def ss_feats(order, texts, ends, cew, ce1v, lead_i, lead_ce, le):
    """同 span CE 分布（lead 锚定，复刻 llm_filter.cc 的 ss12）。"""
    win = list(order)
    S = []
    for i in order:
        if ends[i] != le:
            continue
        S.append(dict(ce=cew[i],
                      ce1=ce1v[i] if ce1v.get(i) is not None else 0.0,
                      is_lead=(i == lead_i), has1=ce1v.get(i) is not None,
                      txt=texts[i]))
    if not any(x["is_lead"] for x in S):
        S.append(dict(ce=lead_ce,
                      ce1=ce1v[lead_i] if ce1v.get(lead_i) is not None else 0.0,
                      is_lead=True, has1=ce1v.get(lead_i) is not None,
                      txt=texts[lead_i]))
    S.sort(key=lambda x: (x["ce"], x["txt"]))
    if len(win) < 2:
        return dict(ss_n=float(len(win)), ss_lead_rank_w=0.0,
                    ss_lead_rank_1=0.0, ss_wgap=0.0, ss_1gap=0.0,
                    ss_ent=0.0, ss_pmass=1.0, ss_range=0.0, ss_top2gap=0.0,
                    ss_best_is_lead=1.0, ss_near=0.0, ss_gap_ratio=0.0,
                    win_ent=0.0, win_pmass=1.0, win_range=0.0,
                    win_n=float(len(win)))
    pos = next(k for k, x in enumerate(S) if x["is_lead"])
    mn, mx = S[0]["ce"], S[-1]["ce"]
    near = sum(1 for x in S if x["ce"] <= lead_ce + 1.0)
    arr = np.array([x["ce"] for x in S])
    ws = np.exp(-(arr - mn))
    pmass = ws[pos] / ws.sum()
    ent = float(-(ws / ws.sum() * np.log(ws / ws.sum() + 1e-12)).sum())
    wgap = (S[1]["ce"] - lead_ce) if len(S) > 1 else 0.0
    cs = [k for k, x in enumerate(S) if x["has1"]]
    cs.sort(key=lambda k: (S[k]["ce1"], S[k]["txt"]))
    lead_c = next((t for t, k in enumerate(cs) if S[k]["is_lead"]), None)
    rank1 = float(lead_c) if lead_c is not None else -1.0
    gap1 = (S[cs[1]]["ce1"] - S[cs[lead_c]]["ce1"]
            if lead_c is not None and len(cs) > 1 else 0.0)
    allce = np.array([cew[i] for i in win])
    pwin = np.exp(-(allce - allce.min()))
    pw = pwin / pwin.sum()
    lead_pos = win.index(lead_i)
    return dict(ss_n=float(len(S)), ss_lead_rank_w=float(pos),
                ss_lead_rank_1=rank1, ss_wgap=float(wgap),
                ss_1gap=float(gap1), ss_ent=ent, ss_pmass=float(pmass),
                ss_range=float(mx - mn),
                ss_top2gap=float(S[1]["ce"] - S[0]["ce"] if len(S) > 1 else 0.0),
                ss_best_is_lead=1.0 if pos == 0 else 0.0,
                ss_near=float(near),
                ss_gap_ratio=float(wgap / (mx - mn + 1e-6)),
                win_ent=float(-(pw * np.log(pw + 1e-12)).sum()),
                win_pmass=float(pw[lead_pos]),
                win_range=float(allce.max() - allce.min()),
                win_n=float(len(win)))


def class_feats(order, texts, cew, ce1v, lead_i, lead_single, win):
    """档位 / 跨档 CE 差（全部只用窗口内已评分候选）。"""
    singles = [i for i in order if len(texts[i]) == 1]
    multis = [i for i in order if len(texts[i]) > 1]
    lead_cls = singles if lead_single else multis
    other_cls = multis if lead_single else singles
    same2 = next((i for i in lead_cls if i != lead_i), None)
    other1 = other_cls[0] if other_cls else None
    s1 = singles[0] if singles else None
    m1 = multis[0] if multis else None
    ce_all = np.array([cew[i] for i in win])
    q = np.exp(-(ce_all - ce_all.min()))
    q = q / q.sum()
    pos = {i: k for k, i in enumerate(win)}
    lead_ce = cew[lead_i]
    same2_gap_w = (cew[same2] - lead_ce) if same2 is not None else 0.0
    same2_gap_1 = ((ce1v[same2] - ce1v[lead_i])
                   if same2 is not None and ce1v.get(same2) is not None
                   and ce1v.get(lead_i) is not None else 0.0)
    return dict(
        lead_is_global_first=1.0 if order[0] == lead_i else 0.0,
        lead_rank_global=float(next(k for k, i in enumerate(order)
                                    if i == lead_i)),
        n_multi=float(len(multis)),
        multi_ratio=len(multis) / max(len(win), 1),
        cf_other_ce=float(cew[other1]) if other1 is not None else -1.0,
        cf_other_ce1=(float(ce1v[other1]) if (other1 is not None
                                             and ce1v.get(other1) is not None)
                      else -1.0),
        cf_other_chars=float(len(texts[other1])) if other1 is not None else 0.0,
        same2_ce_gap_w=float(same2_gap_w),
        same2_ce_gap_1=float(same2_gap_1),
        same2_chars=float(len(texts[same2])) if same2 is not None else 0.0,
        single_first_ce=float(cew[s1]) if s1 is not None else -1.0,
        multi_first_ce=float(cew[m1]) if m1 is not None else -1.0,
        single_multi_gap_w=(float(cew[m1] - cew[s1])
                            if (s1 is not None and m1 is not None) else 0.0),
        single_multi_gap_1=(float(ce1v[m1] - ce1v[s1])
                            if (s1 is not None and m1 is not None
                                and ce1v.get(s1) is not None
                                and ce1v.get(m1) is not None) else 0.0),
        lead_class_n=float(len(lead_cls)),
        lead_class_mass_share=float(sum(q[pos[i]] for i in lead_cls)),
    )


# ── 构建 ─────────────────────────────────────────────────────────────
def build(limit=None, mode=None):
    global CACHE
    mode = mode or MODE
    CACHE = (L.EVAL / ("prefix_class_rows.jsonl" if mode == "top6"
                       else "prefix_class_rows_budget.jsonl"))
    cfreq = M.char_freq()
    rows, miss = [], defaultdict(int)
    for tag, out, gf in DATA:
        cands = json.load(open(out / "cands.json", encoding="utf-8"))
        gen = M.load_gen_map(out / gf)
        ce = L.load_ce()
        syls = {s["sid"]: s["syls"]
                for s in L.jsonl_iter(out / "sentences.jsonl")}
        nw = 0
        for w in L.jsonl_iter(out / "windows.jsonl"):
            if w.get("dyn") or len(w["code"]) < 4:
                continue
            gs = gen.get(w["wid"])
            if not gs:
                miss["gen"] += 1
                continue
            lead, p, le, sent, d = M.lead_stats(gs)
            if not (le < len(w["code"]) and len(sent) - len(lead) >= 2):
                miss["guard"] += 1
                continue
            pool = cands.get(w["code"], [])[:L.K_COLLECT]
            if len(pool) < 2:
                miss["pool"] += 1
                continue
            if not any(c[0] == lead and c[2] == le and c[1] == 0
                       for c in pool):
                miss["lead_absent"] += 1
                continue
            row, why = build_window(w, pool, gs,
                                    window_ced(w["ctx"], pool, ce),
                                    syls.get(w["sid"], []), cfreq, tag,
                                    mode)
            if row is None:
                miss[why] += 1
                continue
            rows.append(row)
            nw += 1
            if limit and len(rows) >= limit:
                break
        print(f"[load] {tag}: rows={nw} miss={dict(miss)}", flush=True)
        if limit and len(rows) >= limit:
            break
    with open(CACHE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_old = sum(1 for r in rows if r["old_rule"])
    print(f"[save] {CACHE}：new-rule 行 {len(rows)}（旧规则 {n_old}，"
          f"新增 {len(rows)-n_old}）")
    return rows


def load_rows(mode=None):
    global CACHE
    if mode:
        CACHE = (L.EVAL / ("prefix_class_rows.jsonl" if mode == "top6"
                           else "prefix_class_rows_budget.jsonl"))
    if not CACHE.exists():
        return build()
    return [json.loads(l) for l in open(CACHE, encoding="utf-8")]


# ── 模型 ─────────────────────────────────────────────────────────────
def load_lua_model(path):
    text = open(path, encoding="utf-8").read()
    feats = re.search(r"feats=\{([^}]*)\}", text).group(1)
    feats = [x.strip().strip('"') for x in feats.split(",") if x.strip()]
    tau = float(re.search(r"tau=([0-9.eE+-]+)", text).group(1))
    trees = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{f={"):
            continue
        d = {}
        for key in "ftlrv":
            m = re.search(r"%s=\{([^}]*)\}" % key, line)
            vals = [x for x in m.group(1).split(",") if x]
            d[key] = ([int(x) for x in vals] if key in "flr"
                      else [float(x) for x in vals])
        trees.append(d)
    return feats, tau, trees


def lua_predict(feats, trees, row):
    x = np.array([float(row["f"].get(c, row["g"].get(c, row["c"].get(c, 0.0)))
                  or 0.0) for c in feats])
    s = 0.0
    for t in trees:
        node = 0
        f, th, lc, rc, val = t["f"], t["t"], t["l"], t["r"], t["v"]
        while f[node] != 0:
            node = (lc[node] if x[f[node] - 1] <= th[node] else rc[node]) - 1
        s += val[node]
    return 1.0 / (1.0 + math.exp(-s))


def feat_matrix(rows, cols):
    return np.array([[r["f"].get(c, r["g"].get(c, r["c"].get(c, 0.0)))
                      for c in cols] for r in rows], float)


def oof_scores(rows, cols, folds=2):
    """句子奇偶分半（folds=2，旧协议）或 5 折（按句子序号分组）。"""
    X = feat_matrix(rows, cols)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    if folds == 2:
        grp = np.array([M.half_of(r) for r in rows])
    else:
        uniq = sorted({r["sid"] for r in rows})
        gid = {k: kk % folds for kk, k in enumerate(uniq)}
        grp = np.array([gid[r["sid"]] for r in rows])
    sc = np.zeros(len(rows))
    for g in range(folds):
        ite = grp == g
        itr = ~ite
        clf = M.fit_lgb(X[itr], y[itr])
        sc[ite] = clf.predict_proba(X[ite])[:, 1]
    return sc, y


def tab(rows, name):
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    bad = n - ok
    net = {k: ok - k * bad for k in (2, 5, 12)}
    print(f"{name:<40}{n:>7}{100*ok/max(n,1):>8.2f}%{bad:>6}"
          f"{net[2]:>8}{net[5]:>8}{net[12]:>8}")


def head():
    print(f"{'策略':<40}{'触发':>7}{'精度':>9}{'错':>6}"
          f"{'K2':>8}{'K5':>8}{'K12':>8}")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def pick(sc, idx, n):
    return sorted(idx, key=lambda i: -sc[i])[:n]


def cluster_diff_ci(a, b, B=600, seed=7):
    """两个选择集的精度差（a−b，句子 cluster bootstrap）。"""
    by = defaultdict(list)
    for i in set(a) | set(b):
        by[sid_of[i]].append(i)
    keys = list(by)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        ch = rng.choice(len(keys), len(keys), replace=True)
        keep = set()
        for k in ch:
            keep.update(by[keys[k]])
        ha = [i for i in a if i in keep]
        hb = [i for i in b if i in keep]
        if ha and hb:
            pa = sum(1 for i in ha if sel_row_ok[i]) / len(ha)
            pb = sum(1 for i in hb if sel_row_ok[i]) / len(hb)
            vals.append(pa - pb)
    if not vals:
        return 0.0, 0.0
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def prec(sel):
    return sum(1 for i in sel if sel_row_ok[i]) / max(len(sel), 1)


sel_row_ok = []


def cluster_ci(sel_sets, B=600, seed=5):
    """句子 cluster bootstrap：对若干选择集分别给精度 95% CI（同一重采样）。"""
    by = defaultdict(list)
    allsel = set()
    for s in sel_sets:
        allsel.update(s)
    for i in allsel:
        by[sid_of[i]].append(i)
    keys = list(by)
    rng = np.random.RandomState(seed)
    out = [[] for _ in sel_sets]
    for _ in range(B):
        ch = rng.choice(len(keys), len(keys), replace=True)
        keep = set()
        for k in ch:
            keep.update(by[keys[k]])
        for a, s in enumerate(sel_sets):
            hit = [i for i in s if i in keep]
            if hit:
                out[a].append(sum(1 for i in hit if sel_row_ok[i]) / len(hit))
    return [(float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for v in out]


sid_of = []


# ── 评估 ─────────────────────────────────────────────────────────────
def evaluate(folds=2, mode=None):
    global sel_row_ok, sid_of
    rows = load_rows(mode)
    sel_row_ok = [r["ok"] for r in rows]
    sid_of = [r["sid"] for r in rows]
    n_new = len(rows)
    n_old_r = sum(1 for r in rows if r["old_rule"])
    print("=" * 108)
    print(f"同档比较大样本 · 新规则行 {n_new}（旧规则 {n_old_r}，"
          f"新增 {n_new-n_old_r}）")
    wtag = "引擎序前 6" if mode in (None, "top6") else "CE 预算 12 准入"
    print(f"口径：冷账本；窗口 = {wtag}（排除组句显示第 1 条）；0.6B base；"
          "P≥0.5 预筛；剩余≥2 音节；ctx≥4")
    comp = defaultdict(int)
    for r in rows:
        comp[(r["tag"], r["domain"])] += 1
    print("数据构成：" + " ".join(f"{k[0]}/{k[1]}={v}"
                                 for k, v in sorted(comp.items())))
    print("=" * 108)

    pop = [r for r in rows if r["f"]["ctx_len"] >= 4
           and r["f"]["p"] >= 0.5]
    old = [r for r in pop if r["old_rule"]]
    add = [r for r in pop if not r["old_rule"]]
    print(f"\n① 结构触发域（P≥0.5 ∧ ctx≥4）：{len(pop)} 窗"
          f"（旧规则 {len(old)} / 新增 {len(add)}）")
    head()
    tab(old, "旧规则（lead=融合序第 1）")
    tab(add, "新增（lead=同档首选但非第 1）")
    tab(pop, "合计")
    for nm, s in (
            ("新增·lead 单字", [r for r in add if r["f"]["lead_single"] > 0]),
            ("新增·lead 多字", [r for r in add if r["f"]["lead_single"] == 0]),
            ("新增·wiki", [r for r in add if r["domain"] == "wiki"]),
            ("新增·user", [r for r in add if r["domain"] == "user"]),
            ("新增·P≥0.95", [r for r in add if r["fire"]]),
            ("新增·0.5≤P<0.95", [r for r in add if not r["fire"]]),
            ("新增·单字 P≥0.95", [r for r in add
                                if r["f"]["lead_single"] > 0 and r["fire"]]),
            ("新增·多字 P≥0.95", [r for r in add
                                if r["f"]["lead_single"] == 0 and r["fire"]]),
            ("旧规则·P≥0.95", [r for r in old if r["fire"]])):
        tab(s, nm)
    if add:
        b = sum(1 for r in add if not r["ok"])
        lo, hi = wilson(b, len(add))
        print(f"  新增结构精度 {100*(1-b/len(add)):.2f}%"
              f"（Wilson 95% CI 精度下界 {100*(1-hi):.1f}%）")

    n0 = sum(1 for r in old if r["fire"])
    print(f"\n现行生产结构触发量 N0 = {n0}（旧规则 ∧ P≥0.95 ∧ ctx≥4）")

    # ② 现行导出模型直接作用于新增行（row-level OOS 参考）
    if LUA_MODEL.exists():
        feats, tau, trees = load_lua_model(LUA_MODEL)
        sc = {r["wid"]: lua_predict(feats, trees, r) for r in rows}
        print(f"\n② 现行导出模型（53 特征，τ={tau:.6f}；训练只见过旧规则行）")
        head()
        for nm, s in (
                ("旧规则·过门", [r for r in old if sc[r["wid"]] >= tau]),
                ("新增·过门", [r for r in add if sc[r["wid"]] >= tau]),
                ("合计·过门", [r for r in pop if sc[r["wid"]] >= tau])):
            tab(s, nm)
        for v, nm in ((1.0, "新增·单字"), (0.0, "新增·多字")):
            tab([r for r in add if r["f"]["lead_single"] == v
                 and sc[r["wid"]] >= tau], nm + "·过门")
        a = np.array([sc[r["wid"]] for r in add])
        if len(a):
            print(f"  新增行模型分：中位 {np.median(a):.4f}，"
                  f"≥τ 比例 {100*(a >= tau).mean():.1f}%")

    # ③ 各臂 OOF（folds=2 旧协议；folds=5 稳健性）
    print(f"\n③ 等量对比（OOF folds={folds}；目标量 = k × N0 = {n0}）")
    idx_old = [i for i, r in enumerate(rows) if r["old_rule"]
               and r["f"]["ctx_len"] >= 4 and r["f"]["p"] >= 0.5]
    idx_new = [i for i, r in enumerate(rows) if r["f"]["ctx_len"] >= 4
               and r["f"]["p"] >= 0.5]
    idx_all = list(range(len(rows)))
    sc0 = np.full(len(rows), np.nan)
    sc2 = np.full(len(rows), np.nan)
    sc3 = np.full(len(rows), np.nan)
    sc0[idx_old] = oof_scores([rows[i] for i in idx_old], BASE, folds=folds)[0]
    sc2[idx_new] = oof_scores([rows[i] for i in idx_new], BASE, folds=folds)[0]
    sc3[idx_new] = oof_scores([rows[i] for i in idx_new],
                              BASE + CLASS_FEATS, folds=folds)[0]
    # V1：旧规则行训练 → 新集（含新增行）折外打分
    sc1 = np.full(len(rows), np.nan)
    if folds == 2:
        splits = [((0, 1)), ((1, 0))]
        grp = np.array([M.half_of(r) for r in rows])
    else:
        uniq = sorted({r["sid"] for r in rows})
        gid = {k: kk % folds for kk, k in enumerate(uniq)}
        grp = np.array([gid[r["sid"]] for r in rows])
        splits = [(g, g) for g in range(folds)]
    for A, B in splits:
        itr = [i for i in idx_old if grp[i] == A]
        ite = [i for i in idx_all if grp[i] == B]
        clf = M.fit_lgb(feat_matrix([rows[i] for i in itr], BASE),
                        np.array([1 if rows[i]["ok"] else 0 for i in itr]))
        sc1[ite] = clf.predict_proba(feat_matrix([rows[i] for i in ite],
                                                 BASE))[:, 1]

    n90 = int(round(0.9 * n0))
    s0 = pick(sc0, idx_old, n90)
    base_prec = prec(s0)
    print(f"\n── V0 旧规则+53（只旧规则行；当前生产等价）")
    head()
    for k in (0.90, 0.95, 1.00):
        tab([rows[i] for i in pick(sc0, idx_old, int(round(k * n0)))],
            f"k={k:.2f}×N0")
    print(f"  基线精度（k=0.90）= {100*base_prec:.2f}%")

    arms = [("V0 旧规则+53", idx_old, sc0),
            ("V1 旧规则训练模型→新集", idx_new, sc1),
            ("V2 同档+53 特征", idx_new, sc2),
            ("V3 同档+53+档位特征", idx_new, sc3)]
    for label, idx, sc in arms[1:]:
        print(f"\n── {label}（可触发域 {len(idx)} 窗）")
        head()
        for k in (0.90, 0.95, 1.00):
            n = int(round(k * n0))
            s = pick(sc, idx, n)
            addn = [i for i in s if not rows[i]["old_rule"]]
            tab([rows[i] for i in s],
                f"k={k:.2f}×N0（新增 {len(addn)} / "
                f"{100*len(addn)/max(len(s),1):.0f}%）")
            if addn:
                print(f"      其中新增行 {len(addn)} 条，精度 "
                      f"{100*prec(addn):.2f}%（错 "
                      f"{sum(1 for i in addn if not rows[i]['ok'])}）")
        from sklearn.metrics import roc_auc_score
        ai = [i for i in idx if not rows[i]["old_rule"]]
        ya = [1 if rows[i]["ok"] else 0 for i in ai]
        if len(set(ya)) > 1:
            print(f"  新增行 AUC = {roc_auc_score(ya, sc[ai]):.4f}"
                  f"（n={len(ai)}）")

    # ④ 同精度（≥基线）最大触发量 / 召回
    print("\n④ 同精度（≥V0 基线 %.2f%%）最大触发量 / 召回（相对 N0）"
          % (100 * base_prec))
    head()
    for label, idx, sc in arms:
        o = sorted(idx, key=lambda i: -sc[i])
        okc, best = 0, 0
        for j, i in enumerate(o, 1):
            okc += 1 if rows[i]["ok"] else 0
            if okc / j >= base_prec:
                best = j
        s = o[:best]
        addn = [i for i in s if not rows[i]["old_rule"]]
        tab([rows[i] for i in s],
            f"{label}（新增 {len(addn)}；{100*best/max(n0,1):.0f}% N0）")
    print("  注：同精度量按点估计逐条下扫；跨臂比较看触发器数/精度/K5。")

    # ⑤ 新增子集的句子 cluster bootstrap（V1 在 90% 量点）
    sets = []
    for label, idx, sc in arms[1:]:
        sets.append(pick(sc, idx, n90))
    ci = cluster_ci(sets)
    print("\n⑤ 90% 量点各臂精度 95% CI（句子 cluster bootstrap）")
    for (label, _, sc), sel, (lo, hi) in zip(arms[1:], sets, ci):
        addn = [i for i in sel if not rows[i]["old_rule"]]
        print(f"  {label:<26} 全体 {100*prec(sel):.2f}% "
              f"[{100*lo:.2f},{100*hi:.2f}]  新增 "
              f"{len(addn)} 条 {100*prec(addn):.2f}%" if addn else
              f"  {label:<26} 全体 {100*prec(sel):.2f}% "
              f"[{100*lo:.2f},{100*hi:.2f}]")
    s2 = sets[1]
    a2 = [i for i in s2 if not rows[i]["old_rule"]]
    o2 = [i for i in s2 if rows[i]["old_rule"]]
    if a2 and o2:
        d = 100 * (prec(a2) - prec(o2))
        lo, hi = cluster_diff_ci(a2, o2)
        print(f"  V2 新增−旧规则 精度差 {d:+.2f}pp "
              f"[{100*lo:+.2f},{100*hi:+.2f}]（新增 {len(a2)} / 旧 {len(o2)}）")
    a3 = [i for i in sets[2] if not rows[i]["old_rule"]]
    if a3 and a2:
        d = 100 * (prec(a3) - prec(a2))
        lo, hi = cluster_diff_ci(a3, a2)
        print(f"  V3−V2（新增子集）精度差 {d:+.2f}pp "
              f"[{100*lo:+.2f},{100*hi:+.2f}]（{len(a3)} / {len(a2)}）")


def _load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def cross_scores(folds=5):
    """top6 旧规则行 → 预算窗（含新增行）的折外分数。
    ≈现行源码版部署：模型在旧 6 窗分布上学，线上吃预算准入后的更大窗口。"""
    top = _load_rows(L.EVAL / "prefix_class_rows.jsonl")
    bud = _load_rows(L.EVAL / "prefix_class_rows_budget.jsonl")
    tr_all = [r for r in top if r["old_rule"] and r["f"]["ctx_len"] >= 4]
    uniq = sorted({r["sid"] for r in bud} | {r["sid"] for r in tr_all})
    gid = {k: kk % folds for kk, k in enumerate(uniq)}
    te_idx = [i for i, r in enumerate(bud) if r["f"]["ctx_len"] >= 4
              and r["f"]["p"] >= 0.5]
    sc = np.full(len(bud), np.nan)
    Xte_all = feat_matrix(bud, BASE)
    for g in range(folds):
        itr = [r for r in tr_all if gid[r["sid"]] != g]
        clf = M.fit_lgb(feat_matrix(itr, BASE),
                        np.array([1 if r["ok"] else 0 for r in itr]))
        ite = [i for i in te_idx if gid[bud[i]["sid"]] == g]
        sc[ite] = clf.predict_proba(Xte_all[ite])[:, 1]
    return bud, sc, te_idx, len(tr_all)


def cross_eval(folds=5):
    """跨口径等量表（top6 旧规则训练 → 预算窗，90/95/100% N0）。"""
    global sel_row_ok, sid_of
    bud, sc, te_idx, ntr = cross_scores(folds)
    sel_row_ok = [r["ok"] for r in bud]
    sid_of = [r["sid"] for r in bud]
    n0 = sum(1 for i in te_idx if bud[i]["old_rule"] and bud[i]["fire"])
    print(f"\n跨口径：top6 旧规则训练（{ntr} 行）→ 预算窗测试"
          f"（{len(te_idx)} 行，N0={n0}）")
    head()
    for k in (0.90, 0.95, 1.00):
        s = pick(sc, te_idx, int(round(k * n0)))
        addn = [i for i in s if not bud[i]["old_rule"]]
        tab([bud[i] for i in s],
            f"k={k:.2f}×N0（新增 {len(addn)} / "
            f"{100*len(addn)/max(len(s),1):.0f}%）")
        if addn:
            print(f"      其中新增行 {len(addn)} 条，精度 "
                  f"{100*prec(addn):.2f}%")
    from sklearn.metrics import roc_auc_score
    ai = [i for i in te_idx if not bud[i]["old_rule"]]
    ya = [1 if bud[i]["ok"] else 0 for i in ai]
    print(f"  新增行 AUC = {roc_auc_score(ya, sc[ai]):.4f}（n={len(ai)}）")


def llm_single(r):
    """LLM（窗口融合序第 1）候选是否单字。"""
    ls = r["f"]["lead_single"] > 0
    gf = r["c"]["lead_is_global_first"] > 0
    return ls if gf else (not ls)


def current_report(folds=5):
    """现行门的整体/分档召回与精度（固定发布 τ；诚实 OOF 跨口径）。"""
    global sel_row_ok, sid_of
    bud, sc, te_idx, ntr = cross_scores(folds)
    sel_row_ok = [r["ok"] for r in bud]
    sid_of = [r["sid"] for r in bud]
    _, tau, _ = load_lua_model(LUA_MODEL)
    # 生产模型门只在 p≥0.5 才判（训练域下界）——从判定区分母里剔除 p<0.5
    te_idx = [i for i in te_idx if bud[i]["f"]["p"] >= 0.5]
    N = len(te_idx)
    sel = [i for i in te_idx if sc[i] >= tau]
    ok = sum(1 for i in sel if bud[i]["ok"])
    print(f"\n现行门（τ={tau:.6f}；跨口径 OOF：top6 旧规则 {ntr} 行 → "
          f"预算窗 {N} 行）")
    print(f"整体：可判 {N}，确认 {len(sel)}，精度 {100*ok/max(len(sel),1):.2f}%，"
          f"召回 {100*len(sel)/max(N,1):.2f}%，K5={ok-5*(len(sel)-ok)}")
    lo, hi = wilson(len(sel) - ok, len(sel))
    print(f"  精度 Wilson 95% CI [{100*(1-hi):.2f},{100*(1-lo):.2f}]")
    for nm, cond in (
            ("旧规则（lead=融合序第 1）", lambda r: r["old_rule"]),
            ("新增（lead=同档首选但非第 1）", lambda r: not r["old_rule"])):
        sub = [i for i in te_idx if cond(bud[i])]
        cf = [i for i in sub if sc[i] >= tau]
        okk = sum(1 for i in cf if bud[i]["ok"])
        print(f"  {nm:<30}可判 {len(sub):>6}  确认 {len(cf):>6}  "
              f"精度 {100*okk/max(len(cf),1):6.2f}%  召回 "
              f"{100*len(cf)/max(len(sub),1):5.2f}%")
    print("\n四种首词组合（ngram lead 字数 × LLM 融合序第 1 字数）")
    print(f"{'组合':<22}{'可判':>7}{'确认':>7}{'精度':>9}"
          f"{'召回':>8}{'占确认':>8}{'旧/新增':>10}")
    for ls, lnm in ((True, "1 字"), (False, "多字")):
        for lls, llnm in ((True, "1 字"), (False, "多字")):
            sub = [i for i in te_idx
                   if (bud[i]["f"]["lead_single"] > 0) == ls
                   and llm_single(bud[i]) == lls]
            cf = [i for i in sub if sc[i] >= tau]
            okk = sum(1 for i in cf if bud[i]["ok"])
            oldn = sum(1 for i in cf if bud[i]["old_rule"])
            print(f"ngram {lnm} / LLM {llnm:<6}{len(sub):>7}{len(cf):>7}"
                  f"{100*okk/max(len(cf),1):>8.2f}%"
                  f"{100*len(cf)/max(len(sub),1):>7.2f}%"
                  f"{100*len(cf)/max(len(sel),1):>7.1f}%"
                  f"{oldn:>6}/{len(cf)-oldn:<5}")
            if cf and okk < len(cf):
                lo, hi = wilson(len(cf) - okk, len(cf))
                print(f"    精度 CI [{100*(1-hi):.2f},{100*(1-lo):.2f}]，"
                      f"错 {len(cf)-okk}")
    return bud, sc, te_idx, tau


# ── 53 特征分组消融（2026-11-26 用户提问）──────────────────────────
GROUPS = {
    "ngram/lead 先验": ["p", "logit_p", "d", "top_score", "n_leads",
                     "cnt_lead", "n_gen", "sent_len"],
    "lead 形状": ["lead_len", "srem", "lead_end", "lead_chars_cov"],
    "窗口/margin": ["wm", "n_window", "n_pool", "code_len", "ctx_len"],
    "lead CE 结构": ["ce1", "ce2", "ce3", "ce_w", "ntok", "chars_per_tok",
                   "ce_per_tok", "ce_per_char", "multi_tok", "ce1_per_char"],
    "r2 竞争": ["ce1_r2", "ce_w_r2", "ntok_r2", "ce_per_tok_r2", "ntok_gap",
              "r2_chars", "r2_single", "ce1_gap"],
    "歧义计数": ["n_single", "n_same_span", "lead_single", "single_ratio"],
    "字频": ["char_freq", "char_freq_mean"],
    "ss12 CE 分布": list(M.DIST_SS),
}


def ablate(folds=5):
    """53 特征按语义组 leave-one-group-out（5 折 OOF，90% N0 等量）。"""
    global sel_row_ok
    rows = [r for r in _load_rows(L.EVAL / "prefix_class_rows_budget.jsonl")
            if r["f"]["ctx_len"] >= 4 and r["f"]["p"] >= 0.5]
    sel_row_ok = [r["ok"] for r in rows]
    idx = list(range(len(rows)))
    n0 = sum(1 for r in rows if r["old_rule"] and r["fire"])
    k = int(round(0.9 * n0))
    from sklearn.metrics import roc_auc_score
    y = np.array([1 if r["ok"] else 0 for r in rows])

    def run(cols):
        sc, _ = oof_scores(rows, cols, folds)
        return prec(pick(sc, idx, k)), float(roc_auc_score(y, sc))

    print(f"\n53 特征消融（budget 窗，{len(rows)} 行，5 折 OOF，"
          f"90% N0={k}）")
    bp, ba = run(BASE)
    print(f"{'变体':<24}{'精度':>9}{'Δpp':>8}{'AUC':>8}{'ΔAUC':>9}{'特征数':>7}")
    print(f"{'全 53 特征':<24}{100*bp:>8.2f}%{'—':>8}{ba:>8.4f}{'—':>9}"
          f"{len(BASE):>7}")
    for nm, cols in GROUPS.items():
        sub = [c for c in BASE if c not in cols]
        p, a = run(sub)
        print(f"{'−' + nm:<24}{100*p:>8.2f}%{100*(p-bp):>+8.2f}"
              f"{a:>8.4f}{a-ba:>+9.4f}{len(sub):>7}")
    for nm, cols in (("最小 4 特征 p/d/ce1/wm", ["p", "d", "ce1", "wm"]),
                     ("仅 ss12 + p/d/ce1", ["p", "d", "ce1"] + list(M.DIST_SS))):
        p, a = run(cols)
        print(f"{nm:<24}{100*p:>8.2f}%{100*(p-bp):>+8.2f}"
              f"{a:>8.4f}{a-ba:>+9.4f}{len(cols):>7}")
    print("  注：Δ = 去掉该组后 − 全特征；负值 = 该组有贡献。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="eval",
                    choices=["build", "eval", "all", "cross", "current",
                             "ablate"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--window", default="top6", choices=["top6", "budget"])
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        build(a.limit or None, a.window)
    if a.cmd in ("eval", "all"):
        evaluate(a.folds, a.window)
    if a.cmd == "cross":
        cross_eval(a.folds)
    if a.cmd == "current":
        current_report(a.folds)
    if a.cmd == "ablate":
        ablate(a.folds)
