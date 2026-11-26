#!/usr/bin/env python3
"""现行 53 特征门：**按句内第 x 个词**的确认率与正确率（2026-11-25）。

用户问法：「最新模型在 wiki 语料上测试得到的第 x 个词确认率和正确率」。
数据 = 研究 cohort（`eval/prefix_model_rows_v2.jsonl`，wiki 扩样 12,694 窗 +
luna 3,052 窗）——每行是一个**判定窗**（P≥0.5 预筛、lead 在引擎池同 span、
剩余 ≥2 音节、且窗口融合序第 1 就是该 lead）。

口径：
  * **x（第几个词）** = lead 在真值句里的词序号 = 1 + 真值句 `[:m]` 的词数
    （`m` = 窗口在句中的起始字偏移；词典 = `pdsp_dict.yaml` 贪心最长匹配）。
  * **确认** = 现行门的完整条件：OOF 分数 ≥ τ（0.973286）∧ p ≥ 0.5 ∧
    lead 不在 ta 族黑名单；τ/分数用**句子奇偶分半 OOF**（不 in-sample，
    否则高估）。
  * **确认率(x)** = 第 x 个词位上被确认的窗 / 该位置上的判定窗数
    （条件于"这里存在一个可判定窗"，不是每个键位都算机会）。
  * **正确率(x)** = 确认里 lead 真是真值前缀的比例。

跑法（研究库根目录）：
  python eval/calib/confirm_by_position.py [--domain wiki|user|all]
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402
import ss_ce_dist as S  # noqa: E402

TA = {"他", "它", "她", "他们", "它们", "她们"}
TAU = 0.973286164
P_FLOOR = 0.5
UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
DEP = [c for c in M.FEATS if c not in UNAVAIL] + M.DIST_SS


def load_sents():
    """sid → (text, segs 的累积长度列表)。segs = HanLP 分词（sentences.jsonl）。"""
    out = {}
    for d in (M.L.OUT, M.L.EVAL / "prefix_big"):
        try:
            it = M.L.jsonl_iter(d / "sentences.jsonl")
        except FileNotFoundError:
            continue
        for x in it:
            ends, cum = [], 0
            for w in x["segs"]:
                cum += len(w)
                ends.append(cum)
            out[x["sid"]] = (x["text"], ends)
    return out


def pos_of(sid, m, sents):
    """第 x 个词 = 1 + 在偏移 m 之前**完整结束**的词数。"""
    t = sents.get(sid)
    if not t:
        return None
    ends = t[1]
    return 1 + sum(1 for e in ends if e <= m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="wiki",
                    choices=["wiki", "user", "all"])
    ap.add_argument("--cohort", default="v2", choices=["v2", "class"],
                    help="class = 2026-11-26 预算窗新规则 cohort + 新树")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    if a.cohort == "class":
        import prefix_class_gate as G
        rows = [r for r in G._load_rows(
            G.L.EVAL / "prefix_class_rows_budget.jsonl")
            if r["f"]["ctx_len"] >= 4 and r["f"]["p"] >= 0.5]
        if a.domain != "all":
            rows = [r for r in rows if r["domain"] == a.domain]
        sents = load_sents()
        y = np.array([1 if r["ok"] else 0 for r in rows])
        oof = G.oof_scores(rows, G.BASE, folds=a.folds)[0]
        TAU = G.load_lua_model(G.LUA_MODEL)[1]
    else:
        rows = S.load_v2()
        if a.domain != "all":
            rows = [r for r in rows if r["domain"] == a.domain]
        sents = load_sents()
        X = S.mat(rows, DEP)
        y = np.array([1 if r["ok"] else 0 for r in rows])
        import lightgbm as lgb
        P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                 min_child_samples=60, subsample=0.9, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
                 verbose=-1)
        half = np.array([M.half_of(r) for r in rows])
        oof = np.zeros(len(rows))
        for A, B in ((0, 1), (1, 0)):
            itr, ite = half == A, half == B
            oof[ite] = lgb.LGBMClassifier(**P).fit(
                X[itr], y[itr]).predict_proba(X[ite])[:, 1]
    pos = np.array([pos_of(r["sid"], r["m"], sents) for r in rows])
    fire = np.array([(s >= TAU) and (r["f"]["p"] >= P_FLOOR)
                     and (r["lead"] not in TA)
                     for s, r in zip(oof, rows)])
    print(f"[data] {a.domain}: 判定窗 {len(rows)}；真机触发(旧门) "
          f"{sum(r['fire'] for r in rows)}；现行门确认 {int(fire.sum())} "
          f"（{100*fire.mean():.1f}%）")
    if fire.sum():
        print(f"[总] 确认正确率 {100*y[fire].mean():.2f}%"
              f"（错 {int((y[fire]==0).sum())}）")
    print("\n第x个词  判定窗   确认   确认率   正确率   错")
    tot = defaultdict(lambda: [0, 0, 0])
    for p, f, o in zip(pos, fire, y):
        tot[int(p)][0] += 1
        if f:
            tot[int(p)][1] += 1
            tot[int(p)][2] += 1 if o else 0
    for k in sorted(tot):
        n, c, ok = tot[k]
        if n < 20 and k > 8:
            continue
        acc = f"{100*ok/c:>6.2f}%" if c else "     —"
        print(f"   {k:<6}{n:>7}{c:>7}   {100*c/n:>5.1f}%   {acc}"
              f"   {c-ok if c else 0:>2}")
    # 句位口径：每个 (sid, 第x词) 只算一次；取该位上**最早**触发的窗
    # （生产里确认后组合拆段，后面同位的更大窗不会再有）
    slot = {}
    for r, p_, f_, o_ in zip(rows, pos, fire, y):
        if p_ is None:
            continue
        k = (r["sid"], int(p_))
        cur = slot.get(k)
        if cur is None:
            slot[k] = dict(n=1, fire=bool(f_), ok=bool(o_) if f_ else False,
                           le=r["f"]["lead_end"])
        else:
            cur["n"] += 1
            if f_ and (not cur["fire"] or r["f"]["lead_end"] < cur["le"]):
                cur["fire"], cur["ok"] = True, bool(o_)
                cur["le"] = r["f"]["lead_end"]
    print()
    print("[句位口径] 每个（句, 第x词）算一次，取该位最早触发的窗")
    print("第x个词  句位   确认   确认率   正确率")
    st = defaultdict(lambda: [0, 0, 0])
    for (sid, p_), v in slot.items():
        st[p_][0] += 1
        if v["fire"]:
            st[p_][1] += 1
            st[p_][2] += 1 if v["ok"] else 0
    for k in sorted(st):
        n, c, ok = st[k]
        if n < 20 and k > 8:
            continue
        print(f"   {k:<6}{n:>6}{c:>7}   {100*c/n:>5.1f}%   "
              f"{100*ok/max(c,1):>6.2f}%")
    # 长尾合并
    tail = [k for k in tot if k > 8]
    if tail:
        n = sum(tot[k][0] for k in tail)
        c = sum(tot[k][1] for k in tail)
        ok = sum(tot[k][2] for k in tail)
        print(f"   >8   {n:>7}{c:>7}   {100*c/n:>5.1f}%   "
              f"{100*ok/max(c,1):>6.2f}%   {c-ok:>2}")


if __name__ == "__main__":
    main()
