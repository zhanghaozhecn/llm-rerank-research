# -*- coding: utf-8 -*-
"""「同字数档首选」vs「同 span 首选」——带**现役 53 特征模型门**的离线评估。

对每个有组句、过音节守卫的窗：
  · 结构层：old pick（同字数档首个）与 new pick（span == [0, le) 首个）是否
    等于 lead；
  · 置信层：p（gen softmax）+ p̂（现役树，τ=0.9776）+ ctx≥4 —— 与 gate 同门。
输出四格（both / new only / old only / neither）× 触发率 × 确认正确率。
"""
import json
import sys
from collections import Counter

EVAL = r"D:\llm-rerank\llm-rerank-research\eval"
sys.path.insert(0, EVAL + r"\calib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L          # noqa: E402
import prefix_class_gate as G       # noqa: E402
import prefix_confirm_model as M    # noqa: E402
import export_prefix_gate as E      # noqa: E402

cands = json.load(open(L.CANDS_EXT, encoding="utf-8"))
ce = L.load_ce_multi()
cfreq = M.char_freq()
feats, tau, trees = G.load_lua_model(G.LUA_MODEL)
print(f"model: {len(trees)} 树 · τ={tau}")

cohort = [json.loads(l) for l in open(L.COHORT_EXT, encoding="utf-8")]
print(f"cohort64: {len(cohort)} 窗")


def budget_win(pool, ced, budget=12):
    """生产准入（引擎序 = pool 顺序；1-token 免费）。"""
    out, rem = [], budget
    for c in pool:
        rec = ced.get(c[0])
        if not rec or rec[0] is None:
            continue
        nt = rec[1] or 0
        cost = 0 if nt <= 1 else min(nt, 3) - 1
        if cost <= rem:
            out.append(c)
            rem -= cost
    return out


stat = Counter()
fire = Counter()
ok = Counter()
examples = {}
for w in cohort:
    if not w["gen"]:
        continue
    gs = [dict(text=g[0], score=g[1], lead=g[2], lead_end=g[3])
          for g in w["gen"] if g[2] and g[3] is not None]
    if not gs:
        continue
    lead, le = gs[0]["lead"], gs[0]["lead_end"]
    truth = w["text"]
    if len(truth) - len(lead) < 2:          # 音节守卫
        continue
    pool = cands.get(w["code"], [])[:64]
    ced = {}
    for ln, sid in L.ce_lines(w["ctx"], [c[0] for c in pool]):
        rr = ce.get(sid) or {}
        for t in ln["cands"]:
            if t in rr and rr[t][0] is not None:
                ced[t] = rr[t]
    win = budget_win(pool, ced)
    if len(win) < 2:
        continue
    # 融合序（冷账本）：单 token 用 ce1；多 token 用加权和（与生产同权重）
    def score(c):
        r = ced[c[0]]
        s = 0.0
        for k, wt in enumerate((1.0, 1.10, 0.60)):
            if k < len(r[0]) and r[0][k] is not None:
                s += wt * r[0][k]
        return s
    order = sorted(win, key=lambda c: (score(c), c[0]))
    old = next((c for c in order if len(c[0]) == len(lead)), None)
    new = next((c for c in order if c[1] == 0 and c[2] == le), None)
    a = old is not None and old[0] == lead
    b = new is not None and new[0] == lead
    cls = ("both" if a and b else "new_only" if b else
           "old_only" if a else "neither")
    stat[cls] += 1
    # 置信层（与 gate 同门）
    if cls in ("both", "new_only", "old_only"):
        row, why = G.build_window(w, pool, gs, ced, [], cfreq, "luna",
                                  "budget")
        if row is None:
            stat[cls + "_nofeat"] += 1
            continue
        p_hat = G.lua_predict(feats, trees, row)
        p = row["f"].get("p", 0.0)
        fires = p >= 0.5 and p_hat >= tau and len(w["ctx"]) >= 4
        fire[cls] += int(fires)
        good = truth.startswith(lead)
        ok[cls] += int(fires and good)
        if cls == "old_only" and len(examples.get("old_only", [])) < 6:
            examples.setdefault("old_only", []).append(
                (w["code"], lead, le, round(p_hat, 4), good,
                 [(c[0], c[1], c[2]) for c in order[:5]]))
        if cls == "new_only" and fires and len(examples.get("fire", [])) < 8:
            examples.setdefault("fire", []).append(
                (w["code"], lead, round(p_hat, 4), good,
                 [c[0] for c in order[:5]]))

print("\n结构分类（old = 同字数档首个，new = 同 span 首个）:")
for k in ("both", "new_only", "old_only", "neither"):
    if stat[k]:
        print(f"  {k:<9} {stat[k]:>6}")
print("\n过置信层（p≥0.5 ∧ p̂≥τ ∧ ctx≥4）:")
for k in ("both", "new_only", "old_only"):
    if stat[k]:
        print(f"  {k:<9} 触发 {fire[k]:>5} / {stat[k]:>5}"
              f"（{100*fire[k]/stat[k]:.1f}%）· 正确 {ok[k]:>5}"
              f"（触发内 {100*ok[k]/max(1,fire[k]):.2f}%）")
print("\nnew_only 中过门的例子（code, lead, p̂, 正确, 融合前5）:")
for e in examples.get("fire", []):
    print("  ", e)
print("")
print("old_only 例子（old 过、new 不过：同 span 上有更强的同音候选）:")
for e in examples.get("old_only", []):
    print("  ", e)
