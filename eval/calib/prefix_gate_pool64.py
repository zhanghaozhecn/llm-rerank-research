#!/usr/bin/env python3
"""前缀确认门 53 特征 + τ 按 64 池窗重训（2026-11-26）。

现役模型（τ=0.977630192）重训后的两个口径变化：
  · 收集池 64 → CE 预算 12 准入窗更深（中位窗 39 行 vs 16 池 ~17），
    n_window / wm / r2 / ss_* / n_single / n_pool 等特征分布整体位移；
  · 结构触发域 = 「同 span [0,le) 首选 ∧ 单字 lead 跳过音节不完整字」
    （2026-11-26 定案③ + tier skip），取代训练时的「同字数档首选」。

行来源 = cohort64（luna 混合语料 21,596 窗，eng 内嵌 64 池 CE + ngram gen）。
诚实协议 = 句子奇偶分半（A 选 B 评，再对称），对比三条臂：
  A 新配方：64 池 span 域训练行，53 特征折外打分（本脚本）
  B 旧配方：16 池 class 域行（prefix_class_rows_budget.jsonl）折外重训
    → 同一批 64 池测试行打分（≈ 现役模型遇到新口径的诚实版本）
  C 现役部署模型直接打分（训练句与测试句重叠，仅作 in-sample 参考）

跑法（研究库根目录）：
  python eval/calib/prefix_gate_pool64.py build   # 重建 span 域训练行
  python eval/calib/prefix_gate_pool64.py train   # OOF + 对比 + 导出
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "eval/calib")
sys.path.insert(0, "eval/lib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L            # noqa: E402
import prefix_confirm_model as M      # noqa: E402
import prefix_class_gate as G         # noqa: E402
import export_prefix_gate as E        # noqa: E402

ROWS64 = L.EVAL / "prefix_class_rows_budget64.jsonl"
OLD_ROWS = L.EVAL / "prefix_class_rows_budget.jsonl"
SENT = L.OUT / "sentences.jsonl"
BIG = L.EVAL / "prefix_big"
SOURCES = [(L.COHORT_EXT, SENT, "luna"),
           (BIG / "cohort64.jsonl", BIG / "sentences.jsonl", "big")]
OUT_LUA = L.EVAL / "lgb" / "prefix_gate_model_pool64.lua"
OUT_H = L.EVAL / "lgb" / "prefix_gate_model_pool64.h"


def load_syll():
    """整读表 syll_chars.tsv → {码串: 字集合}（tier 判定 = 字 ∈ 表[码]）。"""
    p = os.environ.get("RIME_LLM_SYLL_CHARS")
    if not p:
        p = os.path.join(os.environ.get("APPDATA", ""), "Rime", "llm_rerank",
                         "luna_pinyin", "syll_chars.tsv")
    tbl = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            code, chars = line.rstrip("\n").split("\t", 1)
            tbl[code] = set(chars)
    return tbl


# ── build ─────────────────────────────────────────────────────────────
def build_source(path, syls, cfreq, syll, tag):
    rows, miss = [], Counter()
    nw = nwin = 0
    for w in L.jsonl_iter(path):
        nwin += 1
        if w.get("dyn") or len(w["code"]) < 4:
            continue
        if not w.get("gen"):
            miss["gen"] += 1
            continue
        gs = [dict(text=g[0], score=g[1], lead=g[2], lead_end=g[3])
              for g in w["gen"] if g[2] and g[3] is not None]
        if not gs:
            miss["gen_empty"] += 1
            continue
        gs.sort(key=lambda x: -x["score"])
        lead, p, le, sent, d = M.lead_stats(gs)
        # 与生产门同构：le < 码长（严格前缀）∧ 剩余 ≥2（研究口径 = ngram 句
        # 字数差，音节数 ≈ 字数）
        if not (le < len(w["code"]) and len(sent) - len(lead) >= 2):
            miss["guard"] += 1
            continue
        eng = [e for e in w["eng"] if e[5] is not None]
        pool = [(e[0], e[1], e[2]) for e in eng]
        if len(pool) < 2:
            miss["pool"] += 1
            continue
        if not any(c[0] == lead and c[1] == 0 and c[2] == le for c in pool):
            miss["lead_absent"] += 1
            continue
        # ngram lead 是完整音节词：单字 lead 必须整读（异常窗剔除）
        if len(lead) == 1 and lead not in syll.get(w["code"][:le], set()):
            miss["lead_tier1"] += 1
            continue
        ced = {e[0]: (e[5], e[6]) for e in eng}
        row, why = G.build_window(w, pool, gs, ced, syls.get(w["sid"], []),
                                  cfreq, tag, "budget", rule="span",
                                  syll=syll)
        if row is None:
            miss[why] += 1
            continue
        rows.append(row)
        nw += 1
    print(f"[load] {tag}: 窗 {nwin} → 行 {nw}；miss={dict(miss)}", flush=True)
    return rows, miss


def build():
    cfreq = M.char_freq()
    syll = load_syll()
    rows = []
    for path, sent_path, tag in SOURCES:
        if not path.exists():
            print(f"[skip] {tag}: {path} 不存在（先跑回填 + assemble）",
                  flush=True)
            continue
        syls = {s["sid"]: s["syls"] for s in L.jsonl_iter(sent_path)}
        r, _ = build_source(path, syls, cfreq, syll, tag)
        rows += r
    with open(ROWS64, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    okc = sum(1 for r in rows if r["ok"])
    fire = sum(1 for r in rows if r["fire"])
    add = sum(1 for r in rows if not r["cls_first"])
    comp = Counter((r["tag"], r["domain"]) for r in rows)
    print(f"[save] {ROWS64}")
    print(f"[stat] 合计 span 域行 {len(rows)} {dict(comp)}（结构精度 "
          f"{100*okc/max(len(rows),1):.2f}%；P≥0.95 触发 {fire}；"
          f"span 域新增（非同字数档首选）{add}）")
    nw_arr = np.array([r["f"]["n_window"] for r in rows])
    print(f"[stat] 预算窗宽：中位 {np.median(nw_arr):.0f} / "
          f"p25 {np.percentile(nw_arr, 25):.0f} / p75 "
          f"{np.percentile(nw_arr, 75):.0f}")
    return rows


def load64():
    rows = [json.loads(l) for l in open(ROWS64, encoding="utf-8")]
    rows = [r for r in rows if r["f"]["ctx_len"] >= 4
            and r["f"]["p"] >= 0.5]
    return rows


# ── train + 对比 ──────────────────────────────────────────────────────
def tau_at(oof, fire, vol=0.90):
    n = int(round(vol * int(fire.sum())))
    order = np.argsort(-oof)
    return float(oof[order[n - 1]]), n


def head():
    print(f"  {'臂':<30}{'确认':>7}{'精度':>9}{'错':>6}{'K5':>8}")


def line(name, sel, ys):
    n = len(sel)
    ok = int(ys[sel].sum())
    print(f"  {name:<30}{n:>7}{100*ok/max(n,1):>8.2f}%{n-ok:>6}"
          f"{ok-5*(n-ok):>8}")
    return ok, n - ok


def _fit_score(Xtr, tr_rows, tr_grp, g, Xte, te_idx, self_idx=None):
    """在训练折（grp != g）上拟合旧配方 → 给测试下标打分（可带回本域自分）。"""
    itr = np.where(tr_grp != g)[0]
    clf = M.fit_lgb(
        Xtr[itr], np.array([1 if tr_rows[i]["ok"] else 0 for i in itr]))
    out = clf.predict_proba(Xte[te_idx])[:, 1]
    slf = (clf.predict_proba(Xtr[self_idx])[:, 1]
           if self_idx is not None and len(self_idx) else None)
    return out, slf


def train(folds=2, export=True):
    from sklearn.metrics import roc_auc_score
    rows = load64()
    old = [r for r in G._load_rows(OLD_ROWS)
           if r["f"]["ctx_len"] >= 4 and r["f"]["p"] >= 0.5]
    old_luna = [r for r in old if r["tag"] == "luna"]
    print("=" * 104)
    print(f"64 池 span 域重训 · 行 {len(rows)}（64 池 span 域，ctx≥4 ∧ "
          f"p≥0.5）· 对照旧配方行 {len(old)}（16 池 class 域）")
    comp = Counter((r["tag"], r["domain"]) for r in rows)
    print("数据构成：" + " ".join(f"{k[0]}/{k[1]}={v}"
                                 for k, v in sorted(comp.items())))
    print("口径：冷账本；CE 预算 12 准入窗（64 池）；0.6B base；"
          f"诚实协议 = 句子奇偶 {folds} 折")
    print("=" * 104)

    y = np.array([1 if r["ok"] else 0 for r in rows])
    X = G.feat_matrix(rows, E.DEP)
    grp = np.array([M.half_of(r) for r in rows])
    grp_o = np.array([M.half_of(r) for r in old])
    grp_ol = np.array([M.half_of(r) for r in old_luna])
    Xo_all = G.feat_matrix(old, E.DEP)
    Xol_all = G.feat_matrix(old_luna, E.DEP)

    # A：新配方 OOF；B：旧配方（全量）折外训练；B2：旧配方只 luna
    # （与 A 同量级数据 → 分离「口径对齐」与「数据量」两个因素）
    sc_new = np.zeros(len(rows))
    sc_old = np.zeros(len(rows))
    sc_old_l = np.zeros(len(rows))
    sc_old_self = np.zeros(len(old))
    sc_ol_self = np.zeros(len(old_luna))
    for g in range(folds):
        ite = grp == g
        itr = ~ite
        sc_new[ite] = M.fit_lgb(X[itr], y[itr]).predict_proba(X[ite])[:, 1]
        tro = np.where(grp_o != g)[0]
        clf_o = M.fit_lgb(
            Xo_all[tro], np.array([1 if old[i]["ok"] else 0 for i in tro]))
        te = np.where(ite)[0]
        sc_old[te] = clf_o.predict_proba(X[te])[:, 1]
        sc_old_l[te], slf = _fit_score(Xol_all, old_luna, grp_ol, g, X,
                                       te, self_idx=np.where(grp_ol == g)[0])
        sc_ol_self[np.where(grp_ol == g)[0]] = slf
        teo = np.where(grp_o == g)[0]
        sc_old_self[teo] = clf_o.predict_proba(Xo_all[teo])[:, 1]

    fire = np.array([r["fire"] for r in rows])
    fire_o = np.array([r["fire"] for r in old])
    fire_ol = np.array([r["fire"] for r in old_luna])
    tau_new, n_new = tau_at(sc_new, fire)
    tau_old, n_old = tau_at(sc_old_self, fire_o)
    tau_ol, n_ol = tau_at(sc_ol_self, fire_ol)
    print(f"\nτ 推导（OOF 90% × P≥0.95 结构触发量）：")
    print(f"  A 新配方（64 池 span，{len(rows)} 行）：触发 {int(fire.sum())}"
          f" → 目标 {n_new}，τ_new = {tau_new:.9f}")
    print(f"  B 旧配方（16 池 class 全量 {len(old)} 行）：触发 "
          f"{int(fire_o.sum())} → 目标 {n_old}，τ_old = {tau_old:.9f}"
          f"（现役 {G.load_lua_model(G.LUA_MODEL)[1]:.9f}）")
    print(f"  B2 旧配方只 luna（{len(old_luna)} 行，与 A 同量级数据）："
          f"触发 {int(fire_ol.sum())} → 目标 {n_ol}，τ_b2 = {tau_ol:.9f}")

    print(f"\n操作点对比（同一批 64 池测试行 {len(rows)}）：")
    head()
    sel_n = np.where(sc_new >= tau_new)[0]
    sel_o = np.where(sc_old >= tau_old)[0]
    sel_ol = np.where(sc_old_l >= tau_ol)[0]
    a = line("A 新配方 64池span @ τ_new", sel_n, y)
    b = line("B 旧配方全量 @ τ_old", sel_o, y)
    b2 = line("B2 旧配方只luna @ τ_b2", sel_ol, y)
    sel_n2 = np.array(sorted(np.argsort(-sc_new)[:len(sel_o)]))
    sel_o2 = np.array(sorted(np.argsort(-sc_old)[:len(sel_n)]))
    line("A @ B 的确认量（等量）", sel_n2, y)
    line("B @ A 的确认量（等量）", sel_o2, y)
    print(f"  AUC：新 {roc_auc_score(y, sc_new):.4f} vs 旧全量 "
          f"{roc_auc_score(y, sc_old):.4f} vs 旧只luna "
          f"{roc_auc_score(y, sc_old_l):.4f}")

    # 分段（新配方 @ τ_new）
    print(f"\n新配方 @ τ_new 分段：")
    for nm, cond in (
            ("lead 单字", lambda r: r["f"]["lead_single"] > 0),
            ("lead 多字", lambda r: r["f"]["lead_single"] == 0),
            ("domain wiki", lambda r: r["domain"] == "wiki"),
            ("domain user", lambda r: r["domain"] == "user"),
            ("span 域新增（非同档首选）", lambda r: not r["cls_first"]),
            ("同档首选（旧域内）", lambda r: r["cls_first"]),
            ("P≥0.95（结构触发）", lambda r: r["fire"])):
        idx = np.array([i for i, r in enumerate(rows) if cond(r)])
        sel = idx[sc_new[idx] >= tau_new]
        if len(idx):
            line(nm, sel, y)
    add_mask = np.array([not r["cls_first"] for r in rows])
    n_add = int(add_mask.sum())
    n_add_fire = int((sc_new[add_mask] >= tau_new).sum())
    print(f"  （span 域新增行共 {n_add}，其中过门 {n_add_fire}）")

    # C：现役部署模型直接打分（in-sample 参考）
    feats, tau_dep, trees = G.load_lua_model(G.LUA_MODEL)
    sc_dep = np.array([G.lua_predict(feats, trees, r) for r in rows])
    sel_d = np.where(sc_dep >= tau_dep)[0]
    print(f"\nC 现役部署模型 @ τ={tau_dep:.9f}（in-sample 参考）：")
    line("C 确认", sel_d, y)
    both = int(((sc_dep >= tau_dep) & (sc_new >= tau_new)).sum())
    print(f"  与 A 重叠 {both}；A 独有 "
          f"{len(sel_n) - both}；C 独有 {len(sel_d) - both}")

    if not export:
        return
    # 导出：全量拟合 + τ（与对比臂同协议，应与 tau_new 一致）
    clf, tau = E.train(rows, folds=folds)
    if abs(tau - tau_new) > 1e-9:
        print(f"[warn] τ 不一致：train()={tau:.9f} vs 对比臂={tau_new:.9f}")
    trees = E.dump_trees(clf)
    cfreq = M.char_freq()
    E.write_lua(OUT_LUA, trees, tau, cfreq)
    E.write_h(OUT_H, trees, tau, cfreq)
    print(f"[export] {OUT_LUA} {os.path.getsize(OUT_LUA)/1024:.0f} KB")
    print(f"[export] {OUT_H} {os.path.getsize(OUT_H)/1024:.0f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train",
                    choices=["build", "train", "all"])
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        build()
    if a.cmd in ("train", "all"):
        if not ROWS64.exists():
            build()
        train(a.folds, export=not a.no_export)
