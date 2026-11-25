#!/usr/bin/env python3
"""前缀确认门：LightGBM vs XGBoost 同预算对比（2026-11-25，用户提问）。

问题：把已落地的 LightGBM 300 树 / 15 叶换成 XGBoost，效果如何？

口径与 `export_prefix_gate.py` 完全一致（同一 cohort 缓存、同一 41 个可部署
特征、同一句子奇偶分半 OOF、同一 τ 规则 = OOF 分数在「现行触发量 90%」处
取分位）：

  * 指标 1（落地口径）：OOF 分数排序取前 round(0.90·13482) 个窗的精度；
  * 指标 2（报告口径）：A→B / B→A 两向的「同触发量精度」平均（与
    prefix_confirm_model.txt ② 表一致）；
  * 指标 3：3 折（sid mod 3）同触发量精度，看是否分半特异；
  * 代价：导出成同一平坦 Lua 表的树数 / 节点数 / 体积 / 判门 µs。

跑法（研究库根目录）：
  python eval/calib/xgb_vs_lgb.py [--export eval/lgb/xgb_gate.lua]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402

UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
DEP = [c for c in M.FEATS if c not in UNAVAIL]
TARGET_VOL = 0.90

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=60, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
                  verbose=-1)

# XGBoost 侧：tree_method=hist；正则项对齐（subsample/colsample/lambda）；
# min_child_weight = 最小叶样本的 hessian 当量（二分类 hessian≈p(1-p)≤0.25，
# LGB min_child_samples=60 → 约 15）。
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=15.0,
                tree_method="hist", max_bin=256, random_state=7,
                n_jobs=4, eval_metric="logloss")
XGB_VARIANTS = {
    # 同预算：300 树，深 4 ≈ 16 叶（LGB 15 叶），逐层生长
    "xgb_d4": dict(XGB_BASE, max_depth=4),
    # 同叶预算：leaf-wise 生长 15 叶（与 LightGBM 完全同策略）
    "xgb_leaf15": dict(XGB_BASE, max_depth=0, grow_policy="lossguide",
                       max_leaves=15),
    # 更多容量：深 6（看增容是否过拟合）
    "xgb_d6": dict(XGB_BASE, max_depth=6),
    # 慢学习率 + 双倍树（同等叶预算，更细的加性分解）
    "xgb_d4_slow": dict(XGB_BASE, max_depth=4, n_estimators=600,
                        learning_rate=0.025),
}


def data(rows):
    X = np.array([[r["f"][c] for c in DEP] for r in rows], float)
    y = np.array([1 if r["ok"] else 0 for r in rows], int)
    fire = np.array([r["fire"] for r in rows], bool)
    half = np.array([M.half_of(r) for r in rows], int)
    # 与报告 ②f 同口径：sid 数字部分末 4 位 mod 3
    fold = np.array([int("".join(ch for ch in r["sid"] if ch.isdigit())[-4:]
                         or 0) % 3 for r in rows], int)
    return X, y, fire, half, fold


def fit_lgb(X, y, seed=7):
    import lightgbm as lgb
    p = dict(LGB_PARAMS, random_state=seed)
    return lgb.LGBMClassifier(**p).fit(X, y)


def fit_xgb(X, y, params, seed=7):
    import xgboost as xgb
    p = dict(params, random_state=seed)
    return xgb.XGBClassifier(**p).fit(X, y)


def oof_predict(X, y, half, fit):
    """按句子奇偶分半做 OOF：半 A 训练 → 半 B 预测，再对称一次。"""
    oof = np.zeros(len(y))
    for A, B in ((0, 1), (1, 0)):
        itr = np.where(half == A)[0]
        ite = np.where(half == B)[0]
        clf = fit(X[itr], y[itr])
        oof[ite] = clf.predict_proba(X[ite])[:, 1]
    return oof


def same_vol_prec(rows, X, y, half, fit):
    """报告口径：训练半 → 测试半，取测试半按分数前 k 个（k = 测试半现行触发数）。"""
    out = []
    for A, B in ((0, 1), (1, 0)):
        itr = np.where(half == A)[0]
        ite = np.where(half == B)[0]
        clf = fit(X[itr], y[itr])
        sc = clf.predict_proba(X[ite])[:, 1]
        k = int(np.array([rows[i]["fire"] for i in ite]).sum())
        sel = np.argsort(-sc)[:k]
        out.append(float(y[ite][sel].mean()))
    return float(np.mean(out)), out


def fold3_prec(X, y, fire, fold, fit):
    out = []
    for f in range(3):
        itr = np.where(fold != f)[0]
        ite = np.where(fold == f)[0]
        clf = fit(X[itr], y[itr])
        sc = clf.predict_proba(X[ite])[:, 1]
        k = int(fire[ite].sum())
        sel = np.argsort(-sc)[:k]
        out.append(float(y[ite][sel].mean()))
    return out


def deployed_point(oof, y, ntrig):
    """落地口径：τ = OOF 在「90% 现行触发量」处的分数，报精度与 τ。"""
    target = int(round(TARGET_VOL * ntrig))
    order = np.argsort(-oof)
    tau = float(oof[order[target - 1]])
    sel = order[:target]
    return tau, target, float(y[sel].mean())


# ── XGBoost → 平坦 Lua 表（与 LightGBM 导出同格式，供 lua 侧同 walker 基准）──
def dump_xgb(clf):
    """返回 (trees, base_margin)。trees = [(f,t,l,r,v)]，f/l/r 全 1-based，
    f=0 表示叶。

    ⚠ 阈值按 **float32** 取（XGBoost 的 DMatrix 是 float32，预测时把特征与
    阈值都截成 float32 再比 `x < t`）。导出表要忠实复现，移植侧必须
    `float32(x) < float32(t)`——用 double 直接比会有 ~1/3 的窗走错分支
    （本机实测：double `<` 300 行里 100 行不齐，float32 `<` 0 行不齐）。"""
    booster = clf.get_booster()
    raw = json.loads(booster.save_config())[
        "learner"]["learner_model_param"]["base_score"]
    base_p = float(str(raw).strip("[]"))
    base = float(np.log(base_p / (1.0 - base_p)))
    trees = []
    for js in booster.get_dump(dump_format="json", with_stats=False):
        root = json.loads(js)

        def build(n):
            idx = len(f_) + 1
            f_.append(0), t_.append(0.0), l_.append(0), r_.append(0), \
                v_.append(0.0)
            if "leaf" in n:
                v_[idx - 1] = float(np.float32(n["leaf"]))
                return idx
            f_[idx - 1] = int(n["split"][1:]) + 1   # "f12" → 13（1-based）
            t_[idx - 1] = float(np.float32(n["split_condition"]))
            li = build(next(c for c in n["children"]
                            if c["nodeid"] == n["yes"]))
            ri = build(next(c for c in n["children"]
                            if c["nodeid"] == n["no"]))
            l_[idx - 1], r_[idx - 1] = li, ri
            return idx

        f_, t_, l_, r_, v_ = [], [], [], [], []
        build(root)
        trees.append((f_, t_, l_, r_, v_))
    return trees, base


def write_lua(path, trees, tau, base, feats):
    L = ["-- 前缀确认门（XGBoost 对照导出，%d 树）自动生成，勿手改" % len(trees),
         "-- 生成：eval/calib/xgb_vs_lgb.py（研究库 2026-11-25）",
         "-- base = logit(base_score)；margin = base + Σ leaf",
         "-- ⚠ 分支语义 = float32(x) < float32(t)（XGBoost DMatrix 是 float32）；",
         "--   用 double 直接比会走错分支，移植时必须先截 float32。",
         "return {",
         " n=%d," % len(trees),
         " feats={" + ",".join('"%s"' % c for c in feats) + "},",
         " tau=%.9g," % tau,
         " base=%.17g," % base,
         " cmp=\"lt\", f32=1,",
         " trees={"]
    for f, th, lc, rc, val in trees:
        L.append("  {f={%s},t={%s},l={%s},r={%s},v={%s}}," % (
            ",".join(map(str, f)),
            ",".join("%.17g" % x for x in th),
            ",".join(map(str, lc)), ",".join(map(str, rc)),
            ",".join("%.9g" % x for x in val)))
    L.append(" },")
    L.append("}")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def tune(rows):
    """同协议小网格：两个库各扫一批超参，报 OOF@90% 量的最好点（互相公平）。"""
    X, y, fire, half, fold = data(rows)
    ntrig = int(fire.sum())
    grid_l = []
    for nl in (15, 31):
        for mcs in (20, 60):
            for lr, ne in ((0.05, 300), (0.03, 600)):
                grid_l.append(dict(LGB_PARAMS, num_leaves=nl,
                                   min_child_samples=mcs,
                                   learning_rate=lr, n_estimators=ne))
    grid_x = []
    for d, ml in ((4, 15), (6, 0)):
        for mcw in (5.0, 15.0):
            for lr, ne in ((0.05, 300), (0.03, 600)):
                kw = dict(XGB_BASE, max_depth=d, min_child_weight=mcw,
                          learning_rate=lr, n_estimators=ne)
                if ml:
                    kw.update(max_depth=0, grow_policy="lossguide",
                              max_leaves=ml)
                grid_x.append(kw)
    print(f"[tune] LGB {len(grid_l)} 组 / XGB {len(grid_x)} 组")
    best = {}
    for tag, grid, fit0 in (("LGB", grid_l, "lgb"), ("XGB", grid_x, "xgb")):
        rows_out = []
        for p in grid:
            if fit0 == "lgb":
                import lightgbm as lgb
                fit = (lambda pp: (lambda Xa, ya:
                                   lgb.LGBMClassifier(**pp).fit(Xa, ya)))(p)
            else:
                import xgboost as xgb
                fit = (lambda pp: (lambda Xa, ya:
                                   xgb.XGBClassifier(**pp).fit(Xa, ya)))(p)
            oof = oof_predict(X, y, half, fit)
            tau, keep, prec = deployed_point(oof, y, ntrig)
            sv, _ = same_vol_prec(rows, X, y, half, fit)
            rows_out.append((prec, sv, p))
        rows_out.sort(key=lambda t: (-t[0], -t[1]))
        best[tag] = rows_out
        print(f"\n  {tag} 前 5：")
        for prec, sv, p in rows_out[:5]:
            keep = {k: p[k] for k in ("num_leaves", "max_depth", "max_leaves",
                                      "min_child_samples", "min_child_weight",
                                      "learning_rate", "n_estimators")
                    if k in p}
            print(f"    90%量 {100*prec:.2f}%  同量 {100*sv:.2f}%  {keep}")
    print("\n[结论] 两库最好点差距 = "
          f"{100*(best['XGB'][0][0]-best['LGB'][0][0]):+.2f}pp（XGB − LGB）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="")
    ap.add_argument("--tune", action="store_true",
                    help="两库小网格扫描（只报最好点）")
    a = ap.parse_args()
    rows = M.load_rows()
    if a.tune:
        tune(rows)
        return
    X, y, fire, half, fold = data(rows)
    ntrig = int(fire.sum())
    print(f"[data] 行 {len(rows)}；现行触发 {ntrig}；"
          f"90% 量目标 {int(round(TARGET_VOL * ntrig))}；特征 {len(DEP)}")

    models = [("LightGBM 300×15（现行）", fit_lgb)]
    for nm, p in XGB_VARIANTS.items():
        models.append((nm, (lambda pp: (lambda Xa, ya: fit_xgb(Xa, ya, pp)))(p)))

    res = []
    for nm, fit in models:
        oof = oof_predict(X, y, half, fit)
        tau, keep, prec = deployed_point(oof, y, ntrig)
        sv, both = same_vol_prec(rows, X, y, half, fit)
        f3 = fold3_prec(X, y, fire, fold, fit)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, oof)
        res.append(dict(name=nm, auc=auc, tau=tau, keep=keep, prec=prec,
                        sv=sv, both=both, f3=f3, fit=fit, oof=oof))
        print(f"  {nm:<26} AUC {auc:.4f}  τ@90% {tau:.4f}  "
              f"精度 {100*prec:.2f}%  同量 {100*sv:.2f}%  "
              f"三折 {[f'{100*x:.2f}' for x in f3]}")

    # 曲线（OOF 保留比例 → 精度）
    print("\n[曲线] OOF 保留现行触发量的 x% → 精度")
    hdr = "  变体".ljust(28) + "".join(f"{int(v*100):>8}%" for v in
                                        (1.0, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60))
    print(hdr)
    for r in res:
        oof = r["oof"]
        order = np.argsort(-oof)
        cells = []
        for v in (1.0, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60):
            k = int(round(v * ntrig))
            cells.append(f"{100*y[order[:k]].mean():.2f}")
        print("  " + r["name"].ljust(26) + "".join(f"{c:>9}" for c in cells))

    # 导出 XGB（leaf15 变体）+ 节点统计
    if a.export:
        best = next((r for r in res if r["name"] == "xgb_leaf15"), res[-1])
        clf = best["fit"](X, y)
        trees, base = dump_xgb(clf)
        tau = next(r["tau"] for r in res if r["name"] == best["name"])
        # XGB 侧 τ 是概率域；导出表 score = base + Σv（margin），需换回 margin
        tau_m = float(np.log(tau / (1 - tau)))
        write_lua(a.export, trees, tau_m, base, DEP)
        nodes = sum(len(t[0]) for t in trees)
        import os
        # 平坦表对拍：walker(x) margin vs XGBoost predict(output_margin)
        def walk(x):
            s = base
            for f, th, lc, rc, val in trees:
                i = 1
                while f[i - 1] != 0:
                    xi = np.float32(x[f[i - 1] - 1])
                    i = lc[i - 1] if xi < np.float32(th[i - 1]) \
                        else rc[i - 1]
                s += val[i - 1]
            return s
        ref = clf.predict(X[:400], output_margin=True)
        got = np.array([walk(x) for x in X[:400]])
        print(f"[parity] 400 行平坦表（float32 语义）vs XGBoost margin: "
              f"max|Δ| = {np.abs(ref - got).max():.2e}")
        print(f"\n[export] {a.export} {os.path.getsize(a.export)/1024:.0f} KB"
              f" / {len(trees)} 树 / {nodes} 节点 (base={base:.6f})")


if __name__ == "__main__":
    main()
