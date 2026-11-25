#!/usr/bin/env python3
"""CE **分布**特征扩展（2026-11-25 用户指令「拓展该特征」）。

原特征里「歧义」只有两个点：rank1 的 CE 与 rank2 的 CE（`ce1` / `ce1_gap` /
`ce_w_r2`）。本脚本把**分布**显式给出来：

  ss_*  同 span（lead 覆盖的那段码）内全部同音候选的 CE 分布
        ——「这段码歧义到什么程度、lead 排第几、与最佳同音差多少、质量集中度」
  win_* 整个评分窗的 CE 分布（熵 / lead 质量占比 / 极差 / 窗大小）

两组都只用**评分窗内已有 CE** 的候选 → 插件 `pg_score` 与源码
`PrefixModelScore` 都能算（不引入新数据依赖）。

评估（全部同一 cohort、同 300×15 参数、同 OOF 分半、同 τ@90% 量规则）：
  ① 特征组消融：41 / +ss / +win / +两者
  ② 新增特征重要性 + 与已有关键特征的相关性（确认不是重复信息）
  ③ 单特征门（同量口径）看方向是否如预期

跑法（研究库根目录）：
  python eval/calib/ss_ce_dist.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402

UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
DEP = [c for c in M.FEATS if c not in UNAVAIL]
CACHE = M.L.EVAL / "prefix_model_rows_v2.jsonl"
TARGET_VOL = 0.90


def load_v2():
    """带 g（分布特征）的 cohort 缓存；缺了就从窗口/CE 重建。"""
    if CACHE.exists():
        return [json.loads(l) for l in open(CACHE, encoding="utf-8")]
    cfreq = M.char_freq()
    rows = []
    for tag, out, gf in M.DATA:
        rr, miss = M.build_rows(tag, out, gf, cfreq)
        print(f"[load] {tag}: {len(rr)} 行 miss={miss}")
        rows += rr
    with open(CACHE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] {CACHE}")
    return rows


def mat(rows, feats):
    return np.array([[r["f"][c] if c in r["f"] else r["g"][c] for c in feats]
                     for r in rows], float)


def oof(rows, X, folds):
    import lightgbm as lgb
    P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=60, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
             verbose=-1)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    o = np.zeros(len(rows))
    for A, B in ((0, 1), (1, 0)):
        itr, ite = folds == A, folds == B
        o[ite] = lgb.LGBMClassifier(**P).fit(
            X[itr], y[itr]).predict_proba(X[ite])[:, 1]
    return o


def main():
    rows = load_v2()
    y = np.array([1 if r["ok"] else 0 for r in rows])
    half = np.array([M.half_of(r) for r in rows])
    fold = np.array([int("".join(ch for ch in r["sid"] if ch.isdigit())[-4:]
                         or 0) % 3 for r in rows])
    ntrig = int(sum(r["fire"] for r in rows))
    target = int(round(TARGET_VOL * ntrig))
    fire = np.array([r["fire"] for r in rows], bool)
    print(f"[data] 行 {len(rows)}；触发 {ntrig}（精度 "
          f"{100*y[fire].mean():.2f}%）；目标量 {target}")

    def report(nm, feats, o):
        sel = np.argsort(-o)[:target]
        prec = 100 * y[sel].mean()
        # 同量口径（训练半 → 测试半，各半取本半触发数）
        import lightgbm as lgb
        P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                 min_child_samples=60, subsample=0.9, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
                 verbose=-1)
        X = mat(rows, feats)
        sv = []
        for A, B in ((0, 1), (1, 0)):
            itr, ite = half == A, half == B
            clf = lgb.LGBMClassifier(**P).fit(X[itr], y[itr])
            sc = clf.predict_proba(X[ite])[:, 1]
            k = int(fire[ite].sum())
            sv.append(float(y[ite][np.argsort(-sc)[:k]].mean()))
        f3 = []
        for k in range(3):
            itr, ite = fold != k, fold == k
            clf = lgb.LGBMClassifier(**P).fit(X[itr], y[itr])
            sc = clf.predict_proba(X[ite])[:, 1]
            kk = int(fire[ite].sum())
            f3.append(float(y[ite][np.argsort(-sc)[:kk]].mean()))
        print(f"  {nm:<24} 同量 {100*np.mean(sv):>6.2f}%  "
              f"OOF@90% {prec:>6.2f}%  3 折 "
              f"{'/'.join(f'{100*x:.2f}' for x in f3)}")
        return prec, np.mean(sv)

    print("\n① 特征组消融（41 基线 = 现行落地模型）")
    groups = [("41 特征（现行）", DEP),
              ("41 + ss（同 span 分布 10）", DEP + M.DIST_SS),
              ("41 + win（整窗分布 4）", DEP + M.DIST_WIN),
              ("41 + ss + win（14）", DEP + M.DIST_FEATS)]
    res = {}
    for nm, feats in groups:
        X = mat(rows, feats)
        res[nm] = report(nm, feats, oof(rows, X, half))

    # ② 重要性
    import lightgbm as lgb
    X = mat(rows, DEP + M.DIST_FEATS)
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=60, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
        verbose=-1).fit(X, y)
    imp = sorted(zip(DEP + M.DIST_FEATS, clf.booster_.feature_importance(
        importance_type="split")), key=lambda t: -t[1])
    print("\n② 分裂次数前 14（新特征标 *）")
    for k, v in imp[:14]:
        print(f"  {'*' if k in M.DIST_FEATS else ' '} {k:<18}{v:>5}")
    print("  新特征排名：" + "、".join(
        f"{k}={i+1}" for i, (k, v) in enumerate(imp) if k in M.DIST_FEATS))

    # ③ 单特征方向（触发集里好/坏的中位数）
    print("\n③ 触发集内 好/坏 中位数（方向是否符合直觉）")
    for k in ("ss_n", "ss_lead_rank_w", "ss_wgap", "ss_1gap", "ss_ent",
              "ss_pmass", "ss_top2gap", "win_ent", "win_pmass"):
        v = np.array([r["g"][k] for r in rows], float)
        print(f"  {k:<16} 好 {np.median(v[fire & (y == 1)]):>7.3f}   "
              f"坏 {np.median(v[fire & (y == 0)]):>7.3f}")

    # ④ 与已有关键特征的相关性（防「只是换了个名字」）
    print("\n④ 与现有特征的最大 |ρ|（Spearman；>0.95 = 重复信息）")
    from scipy.stats import spearmanr
    for k in M.DIST_FEATS:
        v = np.array([r["g"][k] for r in rows], float)
        best = max(((abs(spearmanr(v, np.array(
            [r["f"][c] for r in rows], float))[0]), c) for c in DEP),
            key=lambda t: t[0])
        print(f"  {k:<16} vs {best[1]:<16} ρ={best[0]:.3f}")

    # ⑤ 结论
    base = res["41 特征（现行）"]
    print("\n⑤ 结论（相对现行 41 特征）")
    for nm, (p, sv) in res.items():
        if nm.startswith("41 特征"):
            continue
        print(f"  {nm:<24} OOF@90% {p-base[0]:+.2f}pp  "
              f"同量 {100*(sv-base[1]):+.2f}pp")


if __name__ == "__main__":
    main()
