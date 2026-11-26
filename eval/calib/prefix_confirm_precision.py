"""前缀确认精度的上下文/边距分析（2026-11-25）。

问题：现行规则（P(lead)≥0.95 ∧ 窗口融合序第 1 ∧ 同文本同 span ∧ 剩余≥2
音节）的精度还能否改善；上文长/短/没有、一句多次确认各是什么表现。

本脚本覆盖静态窗与上下文变体：
  ① 静态窗（真实上文 = 句前文 + 已确认前缀）：上下文分档、特征 AUC、
     规则对照（ngram 句分差 d / 窗口融合 margin wm）、句级 cluster bootstrap；
  ② 空上文（新文档句首，m=0，ctx 置空）：无 CE → 窗口序 = 引擎序；
  ③ 仅前缀（新文档写到第 m 字，ctx = 真值[:m]）：CE/组句按该上下文补算。
一句多次确认的链式仿真见 prefix_confirm_chain.py。

数据（本机、不入库，均由 luna 混合窗 cohort 派生）：
  eval/luna_mixed/gen20.tsv             静态窗 top-20 组句（真实上文）
  eval/luna_mixed/gen20_emptyctx.tsv    静态 m=0 窗 top-20 组句（ctx 为空）
  eval/luna_mixed/prefix_variants.jsonl 仅前缀变体窗（3,764）
  eval/luna_mixed/gen20_prefixctx.tsv   变体窗 top-20 组句（ctx=真值[:m]）
  eval/luna_mixed/ce/results.jsonl      窗口 3-token CE 缓存（含变体）

跑法（研究库根目录）：
  python eval/calib/prefix_confirm_precision.py > eval/prefix_confirm_precision.txt
"""
import math
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L
import prefix_confirm_2x2 as P
import prefix_confirm_chain as C
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PGATE = 0.95
OUT = L.OUT


def load_gen(path):
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


def static_rows(st, g20):
    sents = {s["sid"]: s for s in L.jsonl_iter(L.SENTENCES)}
    miss = {"ce": [], "gen": [], "full": []}
    rows = []
    for w in L.jsonl_iter(L.WINDOWS):
        if w["dyn"]:
            continue
        gs = g20.get(w["wid"])
        if not gs or len(w["code"]) < 4:
            continue
        s = st.state(w["wid"], miss)
        if s is None:
            continue
        n = len(w["code"])
        lead, p, le, sent, d = C.lead_stats(gs)
        sl, syl = len(lead), len(sent)
        rank = -1
        for k, i in enumerate(s["order"]):
            if s["texts"][i] == lead:
                rank = k
                break
        pool_has = any(t == lead and c[2] == le
                       for t, c in zip(s["texts"], st.cands[w["code"]][:16]))
        # 生产可触发集合 = 位次/span/守卫成立（P 阈值由规则决定）
        if not (rank == 0 and pool_has and le < n and syl - sl >= 2):
            continue
        wm = 0.0
        if len(s["order"]) >= 2:
            a = s["fused"].get(s["order"][0])
            b = s["fused"].get(s["order"][1])
            if a is not None and b is not None:
                wm = a - b
        ce1 = s.get("ce1", {}).get(s["order"][0]) if s["order"] else None
        rows.append(dict(
            ce1=ce1,
            wid=w["wid"], sid=w["sid"], domain=w["domain"], m=w["m"],
            L=w["L"], n=n, ctx_len=len(w["ctx"]), lead=lead, truth=w["text"],
            ok=bool(w["text"].startswith(lead)), p=p, d=d, wm=wm,
            fire=p >= PGATE, syl=syl, sl=sl, srem=syl - sl,
            nlead=len({x["lead"] for x in gs}),
            cnt_lead=sum(1 for x in gs if x["lead"] == lead),
            topscore=max(min(-gs[0]["score"], 60.0), 0.0)))
    return rows


def boot(rows, name, B=2000):
    by = defaultdict(list)
    for r in rows:
        by[r["sid"]].append(r["ok"])
    keys = list(by)
    if not keys:
        return
    rng = random.Random(7)
    vals = []
    for _ in range(B):
        acc = []
        for k in rng.choices(keys, k=len(keys)):
            acc += by[k]
        if acc:
            vals.append(sum(acc) / len(acc))
    vals.sort()
    ok = sum(1 for r in rows if r["ok"])
    print(f"  {name:<24}{len(rows):>6}{100*ok/len(rows):>8.2f}%"
          f"  95%CI [{100*vals[int(.025*len(vals))]:.1f},"
          f"{100*vals[int(.975*len(vals))]:.1f}]")


def rule_table(rows, name, wm_ok=True, ce1_ok=True):
    base = len(rows)
    print(f"\n{name}：触发 {base}；精度 "
          f"{100*sum(1 for r in rows if r['ok'])/max(base,1):.2f}%")
    print(f"{'规则':<26}{'触发':>6}{'占现行':>7}{'精度':>8}{'错':>4}"
          f"{'K=2':>7}{'K=5':>7}{'K=12':>7}")

    def rep(nm, sel):
        fs = [r for r in rows if sel(r)]
        ok = sum(1 for r in fs if r["ok"])
        bad = len(fs) - ok
        print(f"{nm:<26}{len(fs):>6}{100*len(fs)/max(base,1):>6.1f}%"
              f"{100*ok/max(len(fs),1):>7.2f}%{bad:>4}"
              f"{ok-2*bad:>7}{ok-5*bad:>7}{ok-12*bad:>7}")
    rep("现行 P≥0.95", lambda r: True)
    for d in (5, 8, 10, 12, 15):
        rep(f"+ d≥{d}", lambda r, d=d: r["d"] >= d)
    if wm_ok:
        for w in (1, 2, 3, 5):
            rep(f"+ wm≥{w}", lambda r, w=w: r["wm"] >= w)
        rep("+ wm≥2 ∨ d≥10", lambda r: r["wm"] >= 2 or r["d"] >= 10)
        rep("+ wm≥3 ∨ d≥10", lambda r: r["wm"] >= 3 or r["d"] >= 10)
        rep("+ wm≥2 ∧ d≥10", lambda r: r["wm"] >= 2 and r["d"] >= 10)
    rep("+ m≤2 → d≥10", lambda r: r["m"] > 2 or r["d"] >= 10)
    rep("+ ctx≥2 ∧ (wm≥2∨d≥10)",
        lambda r: r["ctx_len"] >= 2 and (r["wm"] >= 2 or r["d"] >= 10))
    if ce1_ok:
        for t in (9, 10, 11, 12, 13):
            rep(f"+ ce1≤{t}", lambda r, t=t:
                r.get("ce1") is not None and r["ce1"] <= t)
        rep("+ ce1≤11 ∧ (wm≥2∨d≥10)",
            lambda r: r.get("ce1") is not None and r["ce1"] <= 11
            and (r["wm"] >= 2 or r["d"] >= 10))
        rep("+ ce1≤11 ∧ d≥10",
            lambda r: r.get("ce1") is not None and r["ce1"] <= 11
            and r["d"] >= 10)


def main():
    st = C.Store2()
    g20 = load_gen(OUT / "gen20.tsv")
    eligible = static_rows(st, g20)
    rows = [r for r in eligible if r["fire"]]
    print("=" * 92)
    print(f"前缀确认精度 · 静态窗 · 可触发集合 {len(eligible)}，"
          f"现行规则触发 {len(rows)}（正确 "
          f"{sum(1 for r in rows if r['ok'])}）")
    print("口径：P(lead)≥0.95 ∧ 窗口融合序第 1 ∧ 同文本同 span ∧ 剩余≥2 音节；"
          "冷账本；0.6B base")
    print("=" * 92)
    print("\n① 上下文分档（现行触发内）")
    for nm, sel in (("真实上文 ≥15 字", lambda r: r["ctx_len"] >= 15),
                    ("真实上文 <15 字", lambda r: r["ctx_len"] < 15),
                    ("句首 m=0", lambda r: r["m"] == 0),
                    ("句内 m>0", lambda r: r["m"] > 0),
                    ("m≤2", lambda r: r["m"] <= 2),
                    ("m>2", lambda r: r["m"] > 2),
                    ("wiki", lambda r: r["domain"] == "wiki"),
                    ("user", lambda r: r["domain"] == "user")):
        boot([r for r in rows if sel(r)], nm)

    print("\n② 特征单变量 AUC（触发内，越大越准）")
    feats = [("p(lead)", lambda r: r["p"]),
             ("ngram 句分差 d", lambda r: min(r["d"], 30.0)),
             ("窗口融合 margin wm", lambda r: max(min(r["wm"], 30.0), 0.0)),
             ("上文长度", lambda r: r["ctx_len"]),
             ("m", lambda r: r["m"]),
             ("域=user", lambda r: 1.0 if r["domain"] == "user" else 0.0),
             ("总音节", lambda r: r["syl"]),
             ("lead 音节", lambda r: r["sl"]),
             ("剩余音节", lambda r: r["srem"]),
             ("码长 L", lambda r: r["L"]),
             ("top-20 lead 种数", lambda r: r["nlead"]),
             ("lead 命中句数", lambda r: r["cnt_lead"]),
             ("组句 top 分", lambda r: r["topscore"]),
             ("首 token CE(反向)", lambda r: 99.0 if r.get("ce1") is None
              else r["ce1"])]
    ys = np.array([1 if r["ok"] else 0 for r in eligible])
    X = np.array([[fn(r) for _, fn in feats] for r in eligible], dtype=float)
    for (nm, _), x in zip(feats, X.T):
        v = roc_auc_score(ys, x)
        if nm.startswith("首 token"):
            v = 1.0 - v          # CE 越低越准 → 翻成正向 AUC
        print(f"  {nm:<18}{v:.3f}")

    print("\n③ 规则对照（K = 错确认代价，键）")
    rule_table(rows, "静态窗")

    print("\n④ 逻辑回归（14 特征，句子分组 5 折 OOF）")
    Xs = StandardScaler().fit_transform(X)
    ids = np.array([r["sid"] for r in eligible])
    oof = np.zeros(len(eligible))
    coefs = []
    for tr, te in GroupKFold(n_splits=5).split(Xs, ys, ids):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[tr], ys[tr])
        oof[te] = clf.predict_proba(Xs[te])[:, 1]
        coefs.append(clf.coef_[0])
    print(f"  AUC {roc_auc_score(ys, oof):.4f}；标准化系数：")
    for (nm, _), c in sorted(zip(feats, np.mean(coefs, 0)),
                             key=lambda x: -abs(x[1])):
        print(f"    {nm:<18}{c:+.3f}")
    k = len(rows)
    cur_prec = sum(1 for r in rows if r["ok"]) / max(len(rows), 1)
    idx = np.argsort(-oof)[:k]
    print(f"  同触发数 {k}：精度 {100*ys[idx].mean():.2f}%"
          f"（现行 {100*cur_prec:.2f}%）")
    order = np.argsort(-oof)
    tp = 0
    best = None
    for j, i in enumerate(order):
        tp += ys[i]
        if tp / (j + 1) >= cur_prec:
            best = (j + 1, tp / (j + 1))
    if best:
        print(f"  同精度下可触发 {best[0]}（现行 {k}，"
              f"{100*(best[0]/k-1):+.1f}%），实际精度 {100*best[1]:.2f}%")
    print("  PR（OOF，按触发量）：")
    for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        kk = int(q * len(eligible))
        idx = np.argsort(-oof)[:kk]
        print(f"    触发 {kk:>5}（{100*q:>3.0f}% 集合）："
              f"精度 {100*ys[idx].mean():>6.2f}%")

    # ── 空上文 ─────────────────────────────────────────────────────
    g20e = load_gen(OUT / "gen20_emptyctx.tsv")
    if g20e:
        print("\n⑤ 空上文（新文档句首，ctx 置空；无 CE → 窗口序 = 引擎序）")
        st2 = P.Store()
        er = []
        for w in L.jsonl_iter(L.WINDOWS):
            if w["dyn"] or w["m"] != 0 or len(w["code"]) < 4:
                continue
            gs = g20e.get(w["wid"], [])
            pool = st2.cands.get(w["code"], [])[:L.K_COLLECT]
            if not pool or not gs:
                continue
            order = [i for i, c in enumerate(pool)
                     if c[0] != gs[0]["text"]][:L.K_WINDOW]
            lead, p, le, sent, d = C.lead_stats(gs)
            sl, syl = len(lead), len(sent)
            rank = -1
            for k2, i in enumerate(order):
                if pool[i][0] == lead:
                    rank = k2
                    break
            ph = any(c[0] == lead and c[2] == le for c in pool)
            if not (p >= PGATE and rank == 0 and ph and le < len(w["code"])
                    and syl - sl >= 2):
                continue
            er.append(dict(sid=w["sid"], ok=w["text"].startswith(lead),
                           d=d, wm=0.0, sl=sl, m=0, ctx_len=0, L=w["L"],
                           domain=w["domain"]))
        rule_table(er, "空上文", wm_ok=False, ce1_ok=False)
        for nm, sel in (("wiki", lambda r: r["domain"] == "wiki"),
                        ("user", lambda r: r["domain"] == "user")):
            boot([r for r in er if sel(r)], nm)

    # ── 仅前缀 ─────────────────────────────────────────────────────
    vs = list(L.jsonl_iter(OUT / "prefix_variants.jsonl"))
    g20p = load_gen(OUT / "gen20_prefixctx.tsv")
    if vs and g20p:
        print("\n⑥ 仅前缀（新文档写到第 m 字，ctx = 真值[:m]；CE 按该上下文）")
        ce = L.load_ce_multi()
        pr = []
        for v in vs:
            pool = [c[0] for c in st.cands.get(v["code"], [])[:L.K_COLLECT]]
            if len(pool) < 2:
                continue
            tset = set(pool)
            ced = {}
            for ln, sid in L.ce_lines(v["ctx"], pool):
                rr = ce.get(sid)
                if not rr:
                    continue
                for t in ln["cands"]:
                    if t in tset and t in rr and rr[t][0] is not None:
                        ced[t] = rr[t]
            gs = g20p.get(v["wid"], [])
            if not gs:
                continue
            w = dict(code=v["code"], text=v["text"], m=v["m"])
            s = C.Store2.build(w, st.cands[v["code"]][:L.K_COLLECT], gs,
                               ced, {})
            lead, p, le, sent, d = C.lead_stats(gs)
            sl, syl = len(lead), len(sent)
            rank = -1
            for k2, i in enumerate(s["order"]):
                if s["texts"][i] == lead:
                    rank = k2
                    break
            ph = any(t == lead and e == le
                     for t, e in zip(s["texts"], s["ends"]))
            if not (p >= PGATE and rank == 0 and ph and le < len(v["code"])
                    and syl - sl >= 2):
                continue
            wm = 0.0
            if len(s["order"]) >= 2:
                a = s["fused"].get(s["order"][0])
                b = s["fused"].get(s["order"][1])
                if a is not None and b is not None:
                    wm = a - b
            ce1 = s.get("ce1", {}).get(s["order"][0]) if s["order"] else None
            pr.append(dict(sid=v["sid"], ok=v["text"].startswith(lead),
                           d=d, wm=wm, m=v["m"], ctx_len=v["m"],
                           L=len(v["code"]), ce1=ce1))
        rule_table(pr, "仅前缀")
        print("  按前缀长度：")
        for lo, hi, nm in ((1, 1, "m=1"), (2, 2, "m=2"),
                           (3, 4, "m=3-4"), (5, 99, "m≥5")):
            fs = [r for r in pr if lo <= r["m"] <= hi]
            if fs:
                ok = sum(1 for r in fs if r["ok"])
                fd = [r for r in fs if r["d"] >= 10]
                okd = sum(1 for r in fd if r["ok"])
                fc = [r for r in fs if r.get("ce1") is not None
                      and r["ce1"] <= 11]
                okc = sum(1 for r in fc if r["ok"])
                print(f"    {nm:<6} 现行 {len(fs):>4} 精度 {100*ok/len(fs):>6.2f}%"
                      f"；d≥10 {len(fd):>4} 精度 {100*okd/max(len(fd),1):>6.2f}%"
                      f"；ce1≤11 {len(fc):>4} 精度 {100*okc/max(len(fc),1):>6.2f}%")


if __name__ == "__main__":
    main()
