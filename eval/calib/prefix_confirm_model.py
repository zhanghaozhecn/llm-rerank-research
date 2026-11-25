#!/usr/bin/env python3
"""前缀确认：rich 特征 + 树模型 + 结构性错判挖掘（2026-11-25）。

两个问题（用户 2026-11-25 提出）：
  ① phrase 的 token 数与字数不对应（短词 token 少 → CE 和系统性偏高），
     把能提取的特征全给出来、上树/神经模型，能否更准？——需要更大数据集。
  ② 有些错判是结构性的（他/它/她 这类同音不可判项，上下文本来就分不出），
     应当**直接禁用或加强确认条件**（= 针对特定编码收紧门），精度更高。

数据：两套同口径 cohort（冷账本；CE 共用 content-keyed 缓存）
  eval/luna_mixed/   3,000 句（wiki 1,350 + user 1,350 静态句）+ dyn 300 句
  eval/prefix_big/   wiki 12,001 句（8×，本脚本前一步构建）
两者的静态窗口径完全一致（窗口 = (m,L)，标签 = 真值前缀），可合并。

诚实协议：**按句子奇偶分半 A/B**。模型/黑名单/阈值只在 A 上选，B 上评估
（再对称跑一遍 B→A）。所有"现行规则"数字也在这两半上分别给，便于对照。

跑法（研究库根目录）：
  python eval/calib/prefix_confirm_model.py > eval/prefix_confirm_model.txt
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.path.insert(0, "eval/lib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L  # noqa: E402

PGATE = 0.95
P_PREFILTER = 0.50
KCE = (1.0, 1.10, 0.60)
K_WINDOW = L.K_WINDOW
DATA = [("luna", L.OUT, "gen20.tsv"),
        ("big", L.EVAL / "prefix_big", "gen.tsv")]


# ── 装载 ──────────────────────────────────────────────────────────────
def load_gen_map(path):
    out = defaultdict(list)
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return out
    with fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != 7:
                continue
            try:
                out[f[0]].append(dict(score=float(f[2]), lead=f[3],
                                      lead_end=int(f[4]), text=f[6]))
            except ValueError:
                pass
    for v in out.values():
        v.sort(key=lambda x: -x["score"])
    return out


def lead_stats(gs):
    mx = max(x["score"] for x in gs)
    mass, le_of, txt_of = defaultdict(float), {}, {}
    tot = 0.0
    for x in gs:
        wt = math.exp(x["score"] - mx)
        tot += wt
        mass[x["lead"]] += wt
        le_of.setdefault(x["lead"], x["lead_end"])
        txt_of.setdefault(x["lead"], x["text"])
    lead, best = None, -1.0
    for t, m in mass.items():
        if m > best or (m == best and lead is not None and t < lead):
            lead, best = t, m
    top = gs[0]
    other = next((x for x in gs[1:] if x["lead"] != top["lead"]), None)
    d = (top["score"] - other["score"]) if other else 30.0
    return lead, best / tot if tot else 0.0, le_of[lead], txt_of[lead], d


def char_freq():
    """字符频率（HanLP 缓存前 40k 篇 CJK 字符计数，落盘缓存）。"""
    cache = L.EVAL / "prefix_big" / "char_freq.tsv"
    out = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            c, n = line.split("\t")
            out[c] = int(n)
        tot = sum(out.values())
        return {c: math.log(n / tot + 1e-12) for c, n in out.items()}
    from collections import Counter
    cnt = Counter()
    with open(L.PROJ / "segments_hanlp_cache.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 40000:
                break
            try:
                toks = json.loads(line)
            except ValueError:
                continue
            for t in toks:
                if isinstance(t, str):
                    for ch in t:
                        if "一" <= ch <= "鿿":
                            cnt[ch] += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for c, n in cnt.most_common():
            f.write(f"{c}\t{n}\n")
    tot = sum(cnt.values())
    return {c: math.log(n / tot + 1e-12) for c, n in cnt.items()}


# ── 行（每窗一行）──────────────────────────────────────────────────
def build_rows(tag, out, genfile, cfreq):
    cands = json.load(open(out / "cands.json", encoding="utf-8"))
    gen = load_gen_map(out / genfile)
    ce = L.load_ce()
    syls_by_sid = {}
    for s in L.jsonl_iter(out / "sentences.jsonl"):
        syls_by_sid[s["sid"]] = s["syls"]
    rows, miss = [], defaultdict(int)
    for w in L.jsonl_iter(out / "windows.jsonl"):
        if w.get("dyn"):
            continue
        if len(w["code"]) < 4:
            continue
        gs = gen.get(w["wid"])
        if not gs:
            miss["gen"] += 1
            continue
        lead, p, le, sent, d = lead_stats(gs)
        sl, syl = len(lead), len(sent)
        if not (le < len(w["code"]) and syl - sl >= 2):
            continue
        pool = cands.get(w["code"], [])[:L.K_COLLECT]
        if len(pool) < 2:
            continue
        has = any(c[0] == lead and c[2] == le for c in pool)
        if not has:
            continue
        wd = dict(w)
        ced = {}
        for ln, sid in L.ce_lines(w["ctx"], [c[0] for c in pool]):
            rr = ce.get(sid)
            if not rr:
                continue
            for t in ln["cands"]:
                if t in rr and rr[t][0] is not None:
                    ced[t] = rr[t]
        s = state_of(wd, pool, gs, ced)
        if s is None:
            miss["ce"] += 1
            continue
        rank = -1
        for k, i in enumerate(s["order"]):
            if s["texts"][i] == lead:
                rank = k
                break
        if rank != 0:
            continue
        syls = syls_by_sid.get(w["sid"], [])
        rows.append(featurize(wd, s, gs, lead, p, le, sent, d, syls,
                              pool, cfreq, tag))
    return rows, dict(miss)


def state_of(w, pool, gs, ced):
    """生产窗口态：只排除显示置顶的组句第 1 条；冷账本 → 融合 = CE 加权和。

    **缺失 token 不参与加权**（生产 Step2/Step3 只给真有该 token 的候选加
    分；CE 缓存里的 -1.0 是占位哨兵）——按 ntok 截断，不把哨兵当数值。"""
    n = len(w["code"])
    texts = [c[0] for c in pool]
    ends = [c[2] for c in pool]
    gshow = gs[0]["text"] if gs else None
    win = [i for i, t in enumerate(texts) if t != gshow][:K_WINDOW]
    score, ce1 = {}, {}
    for i in win:
        c3 = ced.get(texts[i])
        if c3 is None:
            continue
        nt = c3[1] if c3[1] else sum(
            1 for x in c3[0] if x is not None and x > -1e8)
        s = 0.0
        for k in range(min(int(nt), 3)):
            x = c3[0][k] if k < len(c3[0]) else None
            if x is not None and x > -1e8:
                s += KCE[k] * x
        score[i] = -s
        if c3[0] and c3[0][0] is not None and c3[0][0] > -1e8:
            ce1[i] = c3[0][0]
    if not win:
        return None
    if any(texts[i] not in ced for i in win):
        # CE 不全的窗不参与：部分候选缺 CE 会让位次错乱（生产侧总是现算，
        # 研究侧缺 CE = 这个窗没算过，不能按"缺分候选垫底"当真实序）
        return None
    order0 = sorted(win, key=lambda i: -(score[i] if i in score else -1e10))
    order = list(order0)
    if len(order0) > 1:
        min_valid = min(score.values()) if score else 0.0
        fused = {i: score.get(i, min_valid) for i in order0}
        order = sorted(order0, key=lambda i: -fused[i])
    else:
        fused = dict(score)
    return dict(n=n, texts=texts, ends=ends, window=win, order=order,
                fused=fused, ce1=ce1, ced=ced, code=w["code"], ctx=w["ctx"])


def ce_of(s, text):
    return s["ced"].get(text)


def featurize(w, s, gs, lead, p, le, sent, d, syls, pool, cfreq, tag):
    n = s["n"]
    syl = len(sent)
    i1 = s["order"][0]
    i2 = s["order"][1] if len(s["order"]) > 1 else None
    c1 = ce_of(s, s["texts"][i1])
    c2 = ce_of(s, s["texts"][i2]) if i2 is not None else None
    mm = w["m"]
    # 键 = lead 覆盖的码串前缀（= 用户口中的「特定编码」）；不是整句读音
    lead_syl = s["code"][:le] if le and le > 0 else "".join(
        syls[mm:mm + len(lead)])
    # 同音异字判据：lead 覆盖的码长 == 真值前缀 len(lead) 个字的拼音总长
    cov = sum(len(sy) for sy in syls[mm:mm + len(lead)])
    same_py = (le == cov) and (w["text"][:len(lead)] != lead)
    # 打出的码完整覆盖了几个音节、是否切在音节中间（从窗口起点 m 数起）
    cum, n_syl_typed = 0, 0
    for sy in syls[mm:]:
        cum += len(sy)
        if cum <= len(s["code"]):
            n_syl_typed += 1
        else:
            break
    partial = 1.0 if cum > len(s["code"]) else 0.0
    singles = [c for c in pool if len(c[0]) == 1]
    same_span = [c for c in pool if c[2] == le]
    wm = 0.0
    if i2 is not None:
        a, b = s["fused"].get(i1), s["fused"].get(i2)
        if a is not None and b is not None:
            wm = a - b
    ok = w["text"].startswith(lead)
    lead_char_freq = min(cfreq.get(ch, -13.0) for ch in lead) if lead else -13.0
    lead_char_freq_mean = (sum(cfreq.get(ch, -13.0) for ch in lead)
                           / max(len(lead), 1))
    r2_chars = len(s["texts"][i2]) if i2 is not None else 0
    # ntok 截到 3：生产侧 DLL 只暴露前 3 个 CE 分量，拿不到真实 token 数
    # （4+ token 的实际数是研究侧特有信息）——两边统一用 min(ntok,3)
    ntok = min(c1[1], 3) if c1 and c1[1] else 0
    ntok_r2 = min(c2[1], 3) if c2 and c2[1] else 0
    ce_sum = -s["fused"].get(i1, 0.0) if i1 in s["fused"] else 0.0
    f = dict(
        # ngram / lead
        p=p, logit_p=math.log(max(p, 1e-6) / max(1 - p, 1e-6)),
        d=min(d, 40.0), top_score=min(-gs[0]["score"], 60.0),
        n_leads=len({x["lead"] for x in gs}),
        cnt_lead=sum(1 for x in gs if x["lead"] == lead),
        n_gen=len(gs), sent_len=syl, lead_len=len(lead), srem=syl - len(lead),
        lead_end=le, lead_chars_cov=len(lead) / max(le, 1),
        # window / typing state
        wm=wm, n_window=len(s["order"]), n_pool=len(pool), code_len=n,
        n_syl_typed=n_syl_typed, partial_syl=partial,
        n_syl_total=len(syls[mm:]), ctx_len=len(w["ctx"]), m=w["m"],
        is_user=1.0 if w["domain"] == "user" else 0.0,
        # rank1 CE（token 结构）
        ce1=c1[0][0] if c1 and c1[0] and c1[0][0] is not None else -1.0,
        ce2=c1[0][1] if c1 and c1[0] and len(c1[0]) > 1
        and c1[0][1] is not None else -1.0,
        ce3=c1[0][2] if c1 and c1[0] and len(c1[0]) > 2
        and c1[0][2] is not None else -1.0,
        ce_w=ce_sum, ntok=ntok,
        chars_per_tok=len(s["texts"][i1]) / max(ntok, 1),
        ce_per_tok=ce_sum / max(ntok, 1),
        ce_per_char=ce_sum / max(len(s["texts"][i1]), 1),
        multi_tok=1.0 if ntok > 1 else 0.0,
        # rank2 竞争
        ce1_r2=c2[0][0] if c2 and c2[0] and c2[0][0] is not None else -1.0,
        ce_w_r2=(-s["fused"].get(i2, 0.0)) if i2 is not None
        else 0.0,
        ntok_r2=ntok_r2,
        ce_per_tok_r2=((-s["fused"].get(i2, 0.0)) / max(ntok_r2, 1))
        if i2 is not None else 0.0,
        ntok_gap=ntok_r2 - ntok,
        ce1_per_char=(c1[0][0] / max(len(lead), 1))
        if c1 and c1[0] and c1[0][0] is not None else -1.0,
        r2_chars=r2_chars,
        r2_single=1.0 if r2_chars == 1 else 0.0,
        ce1_gap=(c2[0][0] - c1[0][0]) if (c2 and c2[0] and c1 and c1[0]
                                          and c2[0][0] is not None
                                          and c1[0][0] is not None) else 0.0,
        # 歧义结构
        n_single=len(singles),
        n_same_span=len(same_span),
        lead_single=1.0 if len(lead) == 1 else 0.0,
        single_ratio=len(singles) / max(len(pool), 1),
        char_freq=lead_char_freq,
        char_freq_mean=lead_char_freq_mean,
        )
    g = dist_feats(s, le)
    return dict(wid=w["wid"], sid=w["sid"], tag=tag, domain=w["domain"],
                ok=ok, fire=p >= PGATE, lead=lead, lead_syl=lead_syl,
                text=w["text"], m=w["m"], f=f, g=g, same_py=same_py,
                truth_prefix=w["text"][:len(lead)])


def _ent(vals):
    """softmax(−CE) 的熵（nats）。vals = 各候选的加权 CE 和。"""
    if len(vals) < 2:
        return 0.0
    a = np.array(vals, float)
    z = a - a.min()
    w = np.exp(-z)                    # −CE 越大的候选权重越大
    p = w / w.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def dist_feats(s, le):
    """**CE 分布特征**（2026-11-25 用户「拓展该特征」）：把「同码段内 TOP-k 同音
    候选的 CE 分布」显式给出来，替代现在只有 top1/top2 两个点的近似。

    两组（都只用评分窗内**已有 CE** 的候选 → 生产可算）：
      ss_*  同 span（end == lead 结尾）那一小簇同音候选上的分布——「这段码歧义
            到什么程度、lead 在其中排第几」
      win_* 整个评分窗上的分布——「整窗竞争有多激烈」
    """
    win = [i for i in s["window"] if i in s["fused"]]
    order = s["order"]
    i1 = order[0]
    if len(win) < 2:
        return dict(ss_n=float(len(win)), ss_lead_rank_w=0.0,
                    ss_lead_rank_1=0.0, ss_wgap=0.0, ss_1gap=0.0,
                    ss_ent=0.0, ss_pmass=1.0, ss_range=0.0,
                    ss_top2gap=0.0, ss_best_is_lead=1.0, ss_near=0.0,
                    ss_gap_ratio=0.0,
                    win_ent=0.0, win_pmass=1.0, win_range=0.0,
                    win_n=float(len(win)))
    S = [i for i in win if s["ends"][i] == le]
    if i1 not in S:
        S.append(i1)
    ce_w = {i: -s["fused"][i] for i in S}
    ce_1 = {i: s["ce1"][i] for i in S if s["ce1"].get(i, -1.0) >= 0.0}
    # 并列时按**文本**（码点/UTF-8 字节序，两实现一致）打破——不能用
    # 迭代顺序当 tie-break：Lua/Python/引擎池的顺序各不相同，会让 rank 差 1
    by_w = sorted(S, key=lambda i: (ce_w[i], s["texts"][i]))
    by_1 = sorted(ce_1, key=lambda i: (ce_1[i], s["texts"][i]))
    r_w = by_w.index(i1)
    r_1 = by_1.index(i1) if i1 in ce_1 else -1.0
    wgap = (ce_w[by_w[1]] - ce_w[i1]) if len(by_w) > 1 else 0.0
    gap1 = (ce_1[by_1[1]] - ce_1[i1]) if (i1 in ce_1 and len(by_1) > 1) \
        else 0.0
    z = np.array([-ce_w[i] for i in by_w], float)
    z -= z.max()
    mass = np.exp(z)
    mass /= mass.sum()
    pos = by_w.index(i1)
    top2 = (ce_w[by_w[1]] - ce_w[by_w[0]]) if len(by_w) > 1 else 0.0
    aw = np.array([ce_w[i] for i in S], float)
    allw = np.array([-s["fused"][i] for i in win], float)
    zw = np.exp(allw - allw.max())
    pw = zw / zw.sum()
    # 「近似平局」计数与相对 margin：歧义的直接度量（CE 差在 1 nat 内 = 难分）
    near = int((aw <= ce_w[i1] + 1.0).sum())
    return dict(
        ss_n=float(len(S)), ss_lead_rank_w=float(r_w),
        ss_lead_rank_1=float(r_1), ss_wgap=float(wgap),
        ss_1gap=float(gap1), ss_ent=_ent(list(ce_w.values())),
        ss_pmass=float(mass[pos]), ss_range=float(aw.max() - aw.min()),
        ss_top2gap=float(top2), ss_best_is_lead=1.0 if r_w == 0 else 0.0,
        ss_near=float(near),
        ss_gap_ratio=float(wgap / (aw.max() - aw.min() + 1e-6)),
        win_ent=_ent([-s["fused"][i] for i in win]),
        win_pmass=float(pw[win.index(i1)]),
        win_range=float(allw.max() - allw.min()), win_n=float(len(win)))


FEATS = [
    "p", "logit_p", "d", "top_score", "n_leads", "cnt_lead", "n_gen",
    "sent_len", "lead_len", "srem", "lead_end", "lead_chars_cov",
    "wm", "n_window", "n_pool", "code_len", "n_syl_typed", "partial_syl",
    "n_syl_total", "ctx_len", "m", "is_user",
    "ce1", "ce2", "ce3", "ce_w", "ntok", "chars_per_tok", "ce_per_tok",
    "ce_per_char", "multi_tok",
    "ce1_r2", "ce_w_r2", "ntok_r2", "ce_per_tok_r2", "ntok_gap",
    "ce1_per_char", "r2_chars", "r2_single", "ce1_gap",
    "n_single", "n_same_span", "lead_single", "single_ratio", "char_freq",
    "char_freq_mean",
]

# ── CE 分布特征（2026-11-25 扩展；生产可算，未进现行 41 特征）────────
# ss_* = 同 span（lead 覆盖的那段码）内同音候选的 CE 分布；win_* = 整窗分布
DIST_SS = ["ss_n", "ss_lead_rank_w", "ss_lead_rank_1", "ss_wgap", "ss_1gap",
           "ss_ent", "ss_pmass", "ss_range", "ss_top2gap", "ss_best_is_lead",
           "ss_near", "ss_gap_ratio"]
DIST_WIN = ["win_ent", "win_pmass", "win_range", "win_n"]
DIST_FEATS = DIST_SS + DIST_WIN


def target_encode(rows_tr, rows_te, keyfn, alpha=20.0):
    """训练半上的「键→错误率」平滑估计（用户 2026-11-25 的「易错项」软版本）。
    施加到评估半；未见键回退先验。"""
    ntr = len(rows_tr)
    e0 = sum(1 for r in rows_tr if not r["ok"]) / max(ntr, 1)
    g = defaultdict(lambda: [0, 0])
    for r in rows_tr:
        g[keyfn(r)][0] += 1
        g[keyfn(r)][1] += 0 if r["ok"] else 1
    out = []
    for r in rows_te:
        n, e = g.get(keyfn(r), (0, 0))
        out.append((e + alpha * e0) / (n + alpha))
    return np.array(out, dtype=float)


def eval_model(rows, X, A, B, extras=None, clf_fn=None):
    """A 训练 → B 评估。返回 dict：auc / 同触发量精度 / 同精度量。

    extras = [(name, keyfn)] → 训练半上做「键错误率」目标编码后作为额外列
    （用户「易错项」的软版本；键在训练半估、评估半用，无泄漏）。"""
    from sklearn.metrics import roc_auc_score
    clf_fn = clf_fn or fit_lgb
    itr = [i for i, r in enumerate(rows) if half_of(r) == A]
    ite = [i for i, r in enumerate(rows) if half_of(r) == B]
    Xtr, Xte = X[itr], X[ite]
    if extras:
        cols_tr, cols_te = [], []
        for _, kf in extras:
            cols_te.append(target_encode([rows[i] for i in itr],
                                         [rows[i] for i in ite], kf))
            cols_tr.append(target_encode([rows[i] for i in itr],
                                         [rows[i] for i in itr], kf))
        Xtr = np.column_stack([Xtr] + cols_tr)
        Xte = np.column_stack([Xte] + cols_te)
    clf = clf_fn(Xtr, np.array([1 if rows[i]["ok"] else 0 for i in itr]))
    sc = clf.predict_proba(Xte)[:, 1]
    y = np.array([1 if rows[i]["ok"] else 0 for i in ite])
    test = [rows[i] for i in ite]
    trig = [r for r in test if r["fire"]]
    k = len(trig)
    cur = sum(1 for r in trig if r["ok"]) / max(k, 1)
    o = np.argsort(-sc)
    sel = [test[j] for j in o[:k]]
    tp, best_v = 0, 0
    for jj, j in enumerate(o, 1):
        tp += y[j]
        if tp / jj >= cur:
            best_v = jj
    return dict(auc=roc_auc_score(y, sc), k=k,
                prec=sum(1 for r in sel if r["ok"]) / max(len(sel), 1),
                err=k - sum(1 for r in trig if r["ok"]),
                cur=cur, best_v=best_v, sc=sc, test=test,
                Xte=Xte, clf=clf)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["build", "eval", "all", "export"])
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--out", default=str(L.EVAL / "lgb" / "prefix_gate.lua"))
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        build_cache()
    if a.cmd in ("eval", "all"):
        evaluate(load_rows())
    if a.cmd == "export":
        cmd_export(a)


def export_lua(clf, path):
    """LightGBM → Lua 平坦表（节点数组；f=0 为叶）。供两版直接嵌入/读取。
    数值与 LightGBM 对拍一致（实测 |Δp| < 1e-12）。"""
    dm = clf.booster_.dump_model()
    lines = ["-- 前缀确认门 LightGBM 导出（自动生成，勿手改）",
             "-- 特征向量顺序 = FEATS（见 eval/calib/prefix_confirm_model.py）",
             "return {",
             " n=%d, trees={" % len(dm["tree_info"])]
    nn = 0
    for t in dm["tree_info"]:
        f, th, lc, rc, val = [], [], [], [], []

        def walk(n):
            nonlocal nn
            f.append(0), th.append(0.0), lc.append(0), rc.append(0), \
                val.append(0.0)
            nn += 1
            idx = len(f)
            if "split_index" not in n:
                val[idx - 1] = float(n["leaf_value"])
                return idx
            li = walk(n["left_child"])
            ri = walk(n["right_child"])
            f[idx - 1] = n["split_feature"] + 1
            th[idx - 1] = float(n["threshold"])
            lc[idx - 1] = li
            rc[idx - 1] = ri
            return idx

        walk(t["tree_structure"])
        lines.append(" {f={%s},t={%s},l={%s},r={%s},v={%s}}," % (
            ",".join(map(str, f)), ",".join("%.17g" % x for x in th),
            ",".join(map(str, lc)), ",".join(map(str, rc)),
            ",".join("%.17g" % x for x in val)))
    lines.append(" }}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return nn


def cmd_export(a):
    import lightgbm as lgb
    rows = load_rows()
    X, y = feat_matrix(rows)
    clf = lgb.LGBMClassifier(
        n_estimators=a.trees, learning_rate=0.05, num_leaves=a.leaves,
        min_child_samples=60, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=7, verbose=-1)
    clf.fit(X, y)
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    nn = export_lua(clf, a.out)
    import os
    print(f"[export] {a.out}: trees={clf.booster_.num_trees()} nodes={nn} "
          f"size={os.path.getsize(a.out)/1024:.0f}KB")
    print("[export] 特征顺序：" + ", ".join(FEATS))



def build_cache():
    cfreq = char_freq()
    allrows = []
    for tag, out, gf in DATA:
        rows, miss = build_rows(tag, out, gf, cfreq)
        print(f"[load] {tag}: eligible(rank0) rows={len(rows)} miss={miss}")
        allrows += rows
    nf = sum(1 for r in allrows if r["fire"])
    print(f"[load] 合计 {len(allrows)}；触发(P≥0.95) {nf}；错 "
          f"{sum(1 for r in allrows if r['fire'] and not r['ok'])}")
    cache = L.EVAL / "prefix_model_rows.jsonl"
    with open(cache, "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] {cache}")


def load_rows():
    cache = L.EVAL / "prefix_model_rows.jsonl"
    if not cache.exists():
        build_cache()
    return [json.loads(l) for l in open(cache, encoding="utf-8")]


# ── 评估 ──────────────────────────────────────────────────────────────
def half_of(r):
    d = "".join(ch for ch in r["sid"] if ch.isdigit())
    return int(d[-4:] or 0) % 2 if d else 0


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def tab(rows, name):
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    bad = n - ok
    net = {K: ok - K * bad for K in (2, 5, 12)}
    print(f"{name:<34}{n:>7}{100*ok/max(n,1):>8.2f}%{bad:>6}"
          f"{net[2]:>8}{net[5]:>8}{net[12]:>8}")


def head():
    print(f"{'策略':<34}{'触发':>7}{'精度':>9}{'错':>6}"
          f"{'K2':>8}{'K5':>8}{'K12':>8}")


def ci(rows, B=800, seed=11):
    """句子 cluster bootstrap（2.5/97.5 分位）。"""
    import random
    by = defaultdict(list)
    for r in rows:
        by[r["sid"]].append(bool(r["ok"]))
    keys = list(by)
    if not keys:
        return 0.0, 0.0
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        acc = []
        for k in rng.choices(keys, k=len(keys)):
            acc += by[k]
        if acc:
            vals.append(sum(acc) / len(acc))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def tab_ci(rows, name):
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    bad = n - ok
    lo, hi = ci(rows)
    print(f"{name:<34}{n:>7}{100*ok/max(n,1):>8.2f}%"
          f" [{100*lo:.1f},{100*hi:.1f}]{bad:>6}"
          f"{ok-2*bad:>7}{ok-5*bad:>7}{ok-12*bad:>7}")


def feat_matrix(rows):
    X = np.array([[r["f"][k] for k in FEATS] for r in rows], dtype=float)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    return X, y


def fit_lgb(X, y, seed=7):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=60, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)
    clf.fit(X, y)
    return clf


class _Scaled:
    """逻辑回归 + 标准化包裹（eval 侧只依赖 predict_proba）。"""

    def __init__(self, sc, clf):
        self.sc, self.clf = sc, clf

    def predict_proba(self, X):
        return self.clf.predict_proba(self.sc.transform(X))


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    return _Scaled(sc, LogisticRegression(C=0.5, max_iter=5000)
                   .fit(sc.transform(X), y))


def mine_blacklist(rows, keyfn, K=5, min_n=10):
    """结构性易错项挖掘（2026-11-25）。

    判据用**错误率置信下界**而不是点估计：错确认代价 K 下的盈亏平衡错误率
    e* = 1/(1+K)。下界都超过 e* → 禁用（有信心）；点估计超、下界没超 →
    加强（收紧 ce1 门）而不是禁用。这样小样本项不会被误杀。
    → (禁用集合, 加强集合, 明细)"""
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        k = keyfn(r)
        g[k][0] += 1
        g[k][1] += 0 if r["ok"] else 1
    e_star = 1.0 / (1.0 + K)
    off, tight, detail = set(), set(), []
    for k, (n, e) in g.items():
        if n < min_n:
            continue
        lo, hi = wilson(e, n)
        okn = n - e
        gain = e * K - okn
        detail.append((gain, e, n, k, lo, hi, e / n))
        if lo > e_star:
            off.add(k)
        elif e / n > e_star:
            tight.add(k)
    detail.sort(key=lambda x: -x[0])
    return off, tight, detail, e_star


def evaluate(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    nfire = sum(1 for r in rows if r["fire"])
    print("=" * 100)
    print(f"前缀确认 · rich 特征模型 + 结构性错判 · 可触发集合 {len(rows)}"
          f"（P≥0.5 预筛后 rank0），现行 P≥0.95 触发 {nfire}，"
          f"错 {sum(1 for r in rows if r['fire'] and not r['ok'])}")
    print("口径：冷账本；窗口 = 只排除组句显示第 1 条；0.6B base；"
          "诚实协议 = 句子奇偶分半（A 选、B 评，再对称）")
    comp = defaultdict(int)
    for r in rows:
        comp[(r["tag"], r["domain"])] += 1
    print("数据构成：" + " ".join(f"{k[0]}/{k[1]}={v}" for k, v in
                                sorted(comp.items())))
    print("=" * 100)

    # ── ① 现状分段（全部触发）──────────────────────────────────────
    tr = [r for r in rows if r["fire"]]
    print("\n① 现行触发集分段（P≥0.95 ∧ 窗口第 1 ∧ 剩余≥2 音节）")
    head()
    for nm, sel in (
            ("全部（现行规则）", lambda r: True),
            ("lead 单字", lambda r: r["f"]["lead_single"] > 0),
            ("lead 多字", lambda r: r["f"]["lead_single"] == 0),
            ("rank1 1 token", lambda r: r["f"]["ntok"] == 1),
            ("rank1 2 token", lambda r: r["f"]["ntok"] == 2),
            ("rank1 ≥3 token", lambda r: r["f"]["ntok"] >= 3),
            ("rank1 字/token=1", lambda r: r["f"]["chars_per_tok"] == 1.0),
            ("rank1 字/token>1", lambda r: r["f"]["chars_per_tok"] > 1.0),
            ("wiki 扩展集", lambda r: r["tag"] == "big"),
            ("wiki 原集", lambda r: r["tag"] == "luna"
             and r["domain"] == "wiki"),
            ("user 真实语料", lambda r: r["domain"] == "user"),
            ("上下文 <15 字", lambda r: r["f"]["ctx_len"] < 15),
            ("上下文 ≥15 字", lambda r: r["f"]["ctx_len"] >= 15)):
        tab([r for r in tr if sel(r)], nm)

    # ①b 错误类型学：同音异字 vs 不同切分
    print("\n①b 错确认类型（触发集内）")
    bad = [r for r in tr if not r["ok"]]
    hom = [r for r in bad if r["same_py"]]
    seg = [r for r in bad if not r["same_py"]]
    print(f"  同音异字（真值同位置读音相同、字不同）{len(hom)}"
          f"（{100*len(hom)/max(len(bad),1):.1f}% 的错）")
    print(f"  不同切分/其它 {len(seg)}"
          f"（{100*len(seg)/max(len(bad),1):.1f}%）")
    for nm, sel in (("同音异字", lambda r: r["same_py"]),
                    ("不同切分", lambda r: not r["same_py"])):
        for dom in ("user", "wiki"):
            fs = [r for r in bad if sel(r) and (
                (dom == "user") == (r["domain"] == "user"))]
            if fs:
                print(f"    {nm}·{dom}: {len(fs)}")

    # ── ② 模型：诚实分半 ──────────────────────────────────────────
    print("\n② 树模型（LightGBM）· 句子奇偶分半 · A→B / B→A 两向平均")
    X, y = feat_matrix(rows)
    res = {}
    variants = [
        ("LightGBM 46 特征", None, None),
        ("LightGBM + lead 字错率", [("item", lambda r: r["lead"])], None),
        ("LightGBM + lead 读音错率", [("syl", lambda r: r["lead_syl"])], None),
        ("LightGBM + 两键错率", [("item", lambda r: r["lead"]),
                            ("syl", lambda r: r["lead_syl"])], None),
        ("逻辑回归 46 特征", None, fit_lr),
        ("逻辑回归 + 两键错率", [("item", lambda r: r["lead"]),
                          ("syl", lambda r: r["lead_syl"])], fit_lr),
    ]
    print(f"  {'变体':<24}{'AUC':>8}{'同量精度':>10}{'现行':>8}"
          f"{'同精度量':>10}{'Δpp':>8}")
    for nm, ex, fn in variants:
        rs = [eval_model(rows, X, A, B, extras=ex, clf_fn=fn)
              for A, B in ((0, 1), (1, 0))]
        auc = sum(r["auc"] for r in rs) / 2
        prec = 100 * sum(r["prec"] for r in rs) / 2
        cur = 100 * sum(r["cur"] for r in rs) / 2
        vol = sum(r["best_v"] for r in rs) / 2
        kk = sum(r["k"] for r in rs) / 2
        res[nm] = dict(auc=auc, prec=prec, cur=cur, vol=vol, k=kk)
        print(f"  {nm:<24}{auc:>8.4f}{prec:>9.2f}%{cur:>7.2f}%"
              f"{vol:>10.0f}{prec-cur:>+7.2f}")
    print("  （同量精度 = 与现行同触发数下模型选出的子集精度；"
          "同精度量 = 模型排序下保持现行精度可触发多少）")

    print("\n②b 纯否决口径（现行触发集内按模型分排序，A→B / B→A 平均）")
    mini = ["logit_p", "d", "ce1", "wm"]
    mini_idx = [FEATS.index(x) for x in mini]
    for nm, XX, fn in (("LightGBM 46 特征", X, None),
                       ("线性 p/d/ce1/wm", X[:, mini_idx], fit_lr)):
        precs = defaultdict(list)
        base = defaultdict(list)
        for A, B in ((0, 1), (1, 0)):
            r = eval_model(rows, XX, A, B, clf_fn=fn)
            trig_i = [j for j, x in enumerate(r["test"]) if x["fire"]]
            sc_t = r["sc"][trig_i]
            ys = np.array([1 if r["test"][j]["ok"] else 0 for j in trig_i])
            o = np.argsort(-sc_t)
            for q in (1.0, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
                kk = int(round(q * len(o)))
                idx = o[:kk]
                precs[q].append(ys[idx].mean())
                base[q].append(kk)
        print(f"  ── {nm}")
        print(f"  {'保留触发比例':<14}{'触发数':>9}{'精度':>9}")
        for q in (1.0, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
            print(f"  {100*q:>12.0f}%{sum(base[q])/2:>9.0f}"
                  f"{100*sum(precs[q])/2:>8.2f}%")
    print("  对照（全量口径）：ce1≤11 保留 95.9% 触发、97.19%；"
          "ce1≤11∧d≥10 保留 61%、98.94%")

    # 逻辑回归对照（同特征）
    print("\n  对照：逻辑回归（同 41 特征）")
    for A, B, nm in ((0, 1, "A→B"), (1, 0, "B→A")):
        itr = [i for i, r in enumerate(rows) if half_of(r) == A]
        ite = [i for i, r in enumerate(rows) if half_of(r) == B]
        sc = StandardScaler().fit_transform(X[itr])
        ste = StandardScaler().fit(X[itr]).transform(X[ite])
        clf = LogisticRegression(C=0.5, max_iter=4000).fit(sc, y[itr])
        pr = clf.predict_proba(ste)[:, 1]
        test = [rows[i] for i in ite]
        k = sum(1 for r in test if r["fire"])
        o = np.argsort(-pr)
        sel = [test[j] for j in o[:k]]
        print(f"    {nm}：AUC {roc_auc_score(y[ite], pr):.4f}；同触发量 "
              f"{k} 精度 {100*sum(1 for r in sel if r['ok'])/max(len(sel),1):.2f}%")

    # ②c 最小可落地线性（特征都在生产侧现成，便于两版落地）
    print("\n②c 最小特征线性模型（可落地形态；A→B / B→A 平均）")
    subs = [
        ("p/d/ce1/wm", ["logit_p", "d", "ce1", "wm"]),
        ("+ 同 span 竞争数/单字", ["logit_p", "d", "ce1", "wm",
                             "n_same_span", "lead_single"]),
        ("+ 字频", ["logit_p", "d", "ce1", "wm", "char_freq",
                 "n_same_span", "lead_single"]),
        ("+ token 结构", ["logit_p", "d", "ce1", "wm", "char_freq",
                       "n_same_span", "lead_single", "ntok",
                       "chars_per_tok"]),
    ]
    print(f"  {'特征子集':<26}{'AUC':>8}{'同量精度':>10}{'Δpp':>8}"
          f"{'同精度量':>10}")
    for nm, fs in subs:
        idx = [FEATS.index(x) for x in fs]
        Xs = X[:, idx]
        rs = [eval_model(rows, Xs, A, B, clf_fn=fit_lr)
              for A, B in ((0, 1), (1, 0))]
        auc = sum(r["auc"] for r in rs) / 2
        prec = 100 * sum(r["prec"] for r in rs) / 2
        cur = 100 * sum(r["cur"] for r in rs) / 2
        vol = sum(r["best_v"] for r in rs) / 2
        print(f"  {nm:<26}{auc:>8.4f}{prec:>9.2f}%{prec-cur:>+7.2f}"
              f"{vol:>10.0f}")

    # ②d 4 特征消融 + 落地系数
    print("\n②d 4 特征消融（p/d/ce1/wm，A→B / B→A 平均同量精度）")
    base = ["logit_p", "d", "ce1", "wm"]
    for drop in (None,) + tuple(base):
        fs = [x for x in base if x != drop]
        idx = [FEATS.index(x) for x in fs]
        rs = [eval_model(rows, X[:, idx], A, B, clf_fn=fit_lr)
              for A, B in ((0, 1), (1, 0))]
        prec = 100 * sum(r["prec"] for r in rs) / 2
        print(f"  {'全 4 特征' if drop is None else '去掉 ' + drop:<26}"
              f"{prec:>9.2f}%")
    # 落地系数（全域拟合一版，仅给数值；性能以 A/B 为准）
    idx = [FEATS.index(x) for x in base]
    Xs = X[:, idx]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=0.5, max_iter=5000).fit(sc.transform(Xs), y)
    zs = clf.predict_proba(sc.transform(Xs))[:, 1]
    print("\n  落地系数（标准化空间；score = Σ c_i·z_i + b，"
          "confirm ⇔ p̂ ≥ τ）：")
    print(f"    intercept {clf.intercept_[0]:+.4f}")
    for nm, c, mu, sd in zip(base, clf.coef_[0], sc.mean_, sc.scale_):
        print(f"    {nm:<10}{c:+.4f}   (μ={mu:.3f} σ={sd:.3f})")
    ntrig = sum(1 for r in rows if r["fire"])
    o = np.argsort(-zs)
    print(f"    τ 取点：现行触发量 {ntrig} → "
          f"p̂={zs[o[ntrig-1]]:.4f}；"
          f"95% 量 → p̂={zs[o[int(.95*ntrig)-1]]:.4f}；"
          f"61% 量 → p̂={zs[o[int(.61*ntrig)-1]]:.4f}")


    # ②e 跨域迁移：只用 wiki 扩展集训练 → user 真实语料上评估
    print("\n②e 跨域迁移（训练 = wiki 扩展集全部；测试 = user 真实语料全部）")
    wtr = [i for i, r in enumerate(rows) if r["tag"] == "big"]
    ute = [i for i, r in enumerate(rows) if r["domain"] == "user"]
    if wtr and ute:
        kU = sum(1 for i in ute if rows[i]["fire"])
        curU = (sum(1 for i in ute if rows[i]["fire"] and rows[i]["ok"])
                / max(kU, 1))
        print(f"  user 触发 {kU}，现行精度 {100*curU:.2f}%")
        XU, yU = X[ute], y[ute]
        idx4 = [FEATS.index(x) for x in mini]
        for nm, XX, fn in (("LightGBM 46 特征", X, None),
                           ("线性 p/d/ce1/wm", X[:, idx4], fit_lr)):
            clf = (fit_lgb(XX[wtr], y[wtr]) if fn is None
                   else fn(XX[wtr], y[wtr]))
            sc = clf.predict_proba(XX[ute])[:, 1]
            o = np.argsort(-sc)
            test = [rows[i] for i in ute]
            out = []
            for q in (1.0, 0.95, 0.70):
                kk = int(round(q * kU))
                sel = [test[j] for j in o[:kk]]
                out.append(f"{100*q:.0f}% 量 {100*sum(1 for r in sel if r['ok'])/len(sel):.2f}%")
            print(f"  {nm:<20}" + "；".join(out))


    # ②f 3 折稳健性（同触发量；确认不是分半特异）
    print("\n②f 3 折稳健性（sid mod 3；每折同触发量精度）")
    print(f"  {'fold':<6}{'测试行':>7}{'触发':>7}{'现行':>8}{'LGBM':>8}"
          f"{'线性4':>8}")
    for k in range(3):
        itr = [i for i, r in enumerate(rows)
               if int("".join(ch for ch in r["sid"] if ch.isdigit())[-4:]
                      or 0) % 3 != k]
        ite = [i for i, r in enumerate(rows)
               if int("".join(ch for ch in r["sid"] if ch.isdigit())[-4:]
                      or 0) % 3 == k]
        test = [rows[i] for i in ite]
        kk = sum(1 for r in test if r["fire"])
        cur = sum(1 for r in test if r["fire"] and r["ok"]) / max(kk, 1)
        rr = []
        for Xs, fn in ((X, None), (X[:, mini_idx], fit_lr)):
            clf = (fit_lgb(Xs[itr], y[itr]) if fn is None
                   else fn(Xs[itr], y[itr]))
            sc = clf.predict_proba(Xs[ite])[:, 1]
            o = np.argsort(-sc)
            sel = [test[j] for j in o[:kk]]
            rr.append(sum(1 for r in sel if r["ok"]) / max(len(sel), 1))
        print(f"  {k:<6}{len(ite):>7}{kk:>7}{100*cur:>7.2f}%"
              f"{100*rr[0]:>7.2f}%{100*rr[1]:>7.2f}%")

    # ── ③ 规则基线（全体，非分半；口径一致便于对照）────────────────
    print("\n③ 规则基线（全量触发集，非分半）")
    head()
    for nm, fn in (
            ("现行 P≥0.95", lambda r: r["fire"]),
            ("+ ce1≤11（已落地）", lambda r: r["fire"]
             and 0 <= r["f"]["ce1"] <= 11),
            ("+ ce1≤11 ∧ d≥10", lambda r: r["fire"]
             and 0 <= r["f"]["ce1"] <= 11 and r["f"]["d"] >= 10),
            ("ce1≤11 ∧ d≥10", lambda r: 0 <= r["f"]["ce1"] <= 11
             and r["f"]["d"] >= 10),
            ("ce1≤10 ∧ d≥10", lambda r: 0 <= r["f"]["ce1"] <= 10
             and r["f"]["d"] >= 10),
            ("P≥0.99", lambda r: r["f"]["p"] >= 0.99)):
        tab([r for r in rows if fn(r)], nm)

    # ── ④ 特征重要性 ─────────────────────────────────────────────
    itr = [i for i, r in enumerate(rows) if half_of(r) == 0]
    clf = fit_lgb(X[itr], y[itr])
    imp = sorted(zip(FEATS, clf.feature_importances_), key=lambda x: -x[1])
    print("\n④ LightGBM 特征重要性（A 半训练，分裂次数）")
    for nm, v in imp:
        print(f"  {nm:<16}{v:>6}")

    # ── ⑤ 结构性错判 ─────────────────────────────────────────────
    print("\n⑤ 结构性错判：按「lead 读音码」分组的错误率（A 挖 → B 评）")
    print("   判据：错确认代价 K=5 → 盈亏平衡错误率 e*=1/(1+K)=16.7%；"
          "错误率 CI 下界 > e* = 禁用，点估计 > e* = 加强（ce1 门收紧）")
    for keyname, keyfn in (("lead 读音", lambda r: r["lead_syl"]),
                           ("lead 字", lambda r: r["lead"])):
        print(f"\n  ── 键 = {keyname}")
        for A, B in ((0, 1), (1, 0)):
            tra = [r for r in rows if half_of(r) == A and r["fire"]]
            teb = [r for r in rows if half_of(r) == B and r["fire"]]
            off, tight, detail, e_star = mine_blacklist(tra, keyfn)
            if A == 0:
                print("    A 半最差 14 项（n≥10；错率 95%CI；"
                      f"e*={100*e_star:.1f}%）")
                for gain, e, n, k, lo, hi, pr in detail[:14]:
                    flag = "禁用" if k in off else ("加强" if k in tight
                                                    else "保留")
                    print(f"      {k:<8}n={n:>4}错={e:>3}"
                          f"错率={100*pr:>5.1f}% [{100*lo:.0f},{100*hi:.0f}]"
                          f" 净{gain:>+5} {flag}")
            d = "A→B" if A == 0 else "B→A"
            tab([r for r in teb if keyfn(r) not in off],
                f"{keyname}禁用 {len(off)} 项（{d}）")
            sel3 = []
            for r in teb:
                kk = keyfn(r)
                if kk in off:
                    continue
                if kk in tight and not (0 <= r["f"]["ce1"] <= 9):
                    continue
                sel3.append(r)
            tab(sel3, f"{keyname}禁用+加强 {len(tight)} 项（{d}）")
            for t in (8, 10):
                sel2 = [r for r in teb
                        if keyfn(r) not in off or 0 <= r["f"]["ce1"] <= t]
                tab(sel2, f"{keyname}黑名单 {len(off)}+ce1≤{t}（{d}）")

    # ── ⑤c 结构性加强（不按项，按"同码异字竞争"结构）─────────────
    print("\n⑤c 结构性加强：按「同码异字竞争」收紧条件（全量触发集）")
    head()
    tab([r for r in rows if r["fire"]], "现行 P≥0.95")
    tab([r for r in rows if r["fire"]
         and (r["f"]["n_same_span"] < 2 or 0 <= r["f"]["ce1"] <= 8)],
        "+ 同 span≥2 → ce1≤8")
    tab([r for r in rows if r["fire"]
         and (r["f"]["n_same_span"] < 2 or 0 <= r["f"]["ce1"] <= 10)],
        "+ 同 span≥2 → ce1≤10")
    tab([r for r in rows if r["fire"]
         and (r["f"]["n_same_span"] < 2 or r["f"]["d"] >= 12)],
        "+ 同 span≥2 → d≥12")
    tab([r for r in rows if r["fire"]
         and (not r["f"]["lead_single"] or r["f"]["n_single"] < 3
              or 0 <= r["f"]["ce1"] <= 8)],
        "+ 单字且同音≥3 → ce1≤8")
    tab([r for r in rows if 0 <= r["f"]["ce1"] <= 11
         and (r["f"]["n_same_span"] < 2 or 0 <= r["f"]["ce1"] <= 8)],
        "ce1≤11 ∧ 同 span≥2 → ce1≤8")

    # ── ⑤b 用户域专项（真实打字目标域；wiki 正式文体稀释易错项）─────
    print("\n⑤b 用户域专项（min_n=8；A 挖 → B 评，再对称；wiki 侧代价同列）")
    for A, B in ((0, 1), (1, 0)):
        tra = [r for r in rows if half_of(r) == A and r["fire"]
               and r["domain"] == "user"]
        teu = [r for r in rows if half_of(r) == B and r["fire"]
               and r["domain"] == "user"]
        tew = [r for r in rows if half_of(r) == B and r["fire"]
               and r["domain"] != "user"]
        off, tight, detail, e_star = mine_blacklist(
            tra, lambda r: r["lead_syl"], min_n=8)
        d = "A→B" if A == 0 else "B→A"
        if A == 0:
            print("    A 半用户域最差 10 项（n≥8，e*=16.7%）")
            for gain, e, n, k, lo, hi, pr in detail[:10]:
                flag = "禁用" if k in off else ("加强" if k in tight
                                                else "保留")
                print(f"      {k:<10}n={n:>3}错={e:>2}"
                      f"错率={100*pr:>5.1f}% [{100*lo:.0f},{100*hi:.0f}]"
                      f" 净{gain:>+4} {flag}")
        tab([r for r in teu if r["lead_syl"] not in off],
            f"用户域禁用 {len(off)} 项（{d}）")
        sel3 = []
        for r in teu:
            kk = r["lead_syl"]
            if kk in off:
                continue
            if kk in tight and not (0 <= r["f"]["ce1"] <= 9):
                continue
            sel3.append(r)
        tab(sel3, f"用户域禁用+加强 {len(tight)} 项（{d}）")
        tab([r for r in tew if r["lead_syl"] not in off],
            f"同名单 wiki 侧代价（{d}）")
        tab(tew, f"wiki 侧基线（{d}）")

    # ── ⑤d ta 族专项（用户 2026-11-25 点名的结构性同音歧义）────────
    print("\n⑤d ta 族专项（他/它/她/他们/它们/她们；用户点名的结构性同音）")
    FAM = {"他", "它", "她", "他们", "它们", "她们"}
    fam = [r for r in tr if r["lead"] in FAM]
    if fam:
        e = sum(1 for r in fam if not r["ok"])
        print(f"  合计 {len(fam)} 触发 / {e} 错 = {100*e/len(fam):.1f}%")
        for ch in ("他", "它", "她", "他们", "它们", "她们"):
            fs = [r for r in fam if r["lead"] == ch]
            if fs:
                ee = sum(1 for r in fs if not r["ok"])
                print(f"    {ch}: n={len(fs):>3} 错={ee:>2}"
                      f"（{100*ee/len(fs):.0f}%）")
        for dom in ("wiki", "user"):
            fs = [r for r in fam
                  if (r["domain"] == "user") == (dom == "user")]
            if fs:
                ee = sum(1 for r in fs if not r["ok"])
                print(f"    {dom}: n={len(fs):>3} 错={ee:>2}"
                      f"（{100*ee/len(fs):.0f}%）")
        head()
        tab(tr, "现行（触发集）")
        tab([r for r in tr if r["lead"] not in FAM], "禁用 ta 族")
        for dom in ("wiki", "user"):
            sub = [r for r in tr
                   if (r["domain"] == "user") == (dom == "user")]
            tab(sub, f"  {dom} 基线")
            tab([r for r in sub if r["lead"] not in FAM], f"  {dom} 禁 ta 族")
        print("  诚实分半（A→B / B→A）：")
        for A, B in ((0, 1), (1, 0)):
            teb = [r for r in tr if half_of(r) == B]
            sel = [r for r in teb if r["lead"] not in FAM]

            def st(fs):
                ok = sum(1 for r in fs if r["ok"])
                n = len(fs)
                return n, 100 * ok / n, ok - 5 * (n - ok)
            a, b = st(teb), st(sel)
            print(f"    {'A→B' if A == 0 else 'B→A'}：{a[0]} {a[1]:.2f}%"
                  f" K5={a[2]} → {b[0]} {b[1]:.2f}% K5={b[2]}"
                  f"（Δ{b[1]-a[1]:+.2f}pp / {b[2]-a[2]:+d}）")
        # 模型之上再禁
        for A, B in ((0, 1), (1, 0)):
            r = eval_model(rows, X, A, B)
            sc, test, k = r["sc"], r["test"], r["k"]
            keep = [j for j, x in enumerate(test) if x["lead"] not in FAM]
            k2 = sum(1 for j in keep if test[j]["fire"])
            o = [j for j in np.argsort(-sc) if test[j]["lead"] not in FAM][:k2]
            sel = [test[j] for j in o]
            print(f"    模型 + 禁 ta 族（{'A→B' if A == 0 else 'B→A'}）："
                  f"{k2} 精度 "
                  f"{100*sum(1 for x in sel if x['ok'])/max(len(sel),1):.2f}%"
                  f"（模型单独 {100*r['prec']:.2f}%）")

    # ── ⑥ 模型 + 黑名单 ─────────────────────────────────────────
    print("\n⑥ 模型 + lead 读音黑名单（同触发量口径）")
    for A, B, nm in ((0, 1, "A→B"), (1, 0, "B→A")):
        r = eval_model(rows, X, A, B)
        sc, test = r["sc"], r["test"]
        k = r["k"]
        tra = [x for x in rows if half_of(x) == A and x["fire"]]
        off, _, _, _ = mine_blacklist(tra, lambda x: x["lead_syl"])
        masked = np.array([-1e9 if x["lead_syl"] in off else s
                           for x, s in zip(test, sc)])
        o = np.argsort(-masked)
        sel = [test[j] for j in o[:k]]
        ok = sum(1 for x in sel if x["ok"])
        print(f"  {nm}：禁用 {len(off)} 项；同触发量 {k} 精度 "
              f"{100*ok/len(sel):.2f}%（模型单独 "
              f"{100*r['prec']:.2f}%，现行 {100*r['cur']:.2f}%）")

    # ── ⑦ 关键 operating 点 + cluster bootstrap CI ────────────────
    print("\n⑦ 关键 operating 点（句子 cluster bootstrap 95%CI）")
    print(f"{'策略':<40}{'触发':>7}{'精度':>9}{'95%CI':>16}{'错':>6}")
    for nm, fn in (
            ("现行 P≥0.95", lambda r: r["fire"]),
            ("ce1≤11（已落地）", lambda r: r["fire"]
             and 0 <= r["f"]["ce1"] <= 11),
            ("ce1≤11 ∧ d≥10", lambda r: r["fire"]
             and 0 <= r["f"]["ce1"] <= 11 and r["f"]["d"] >= 10)):
        tab_ci([r for r in rows if fn(r)], nm)
    for A, B, nm in ((0, 1, "A→B"), (1, 0, "B→A")):
        r = eval_model(rows, X[:, mini_idx], A, B, clf_fn=fit_lr)
        o = np.argsort(-r["sc"])
        k = r["k"]
        sel = [r["test"][j] for j in o[:k]]
        tab_ci(sel, f"4 特征线性 同触发量（{nm}）")
        for q in (0.95, 0.61):
            kk = int(round(q * k))
            selq = [r["test"][j] for j in o[:kk]]
            tab_ci(selq, f"4 特征线性 保留 {100*q:.0f}%（{nm}）")


if __name__ == "__main__":
    main()

