#!/usr/bin/env python3
"""前缀确认门：训练 + 导出两版可用的模型产物（2026-11-25，用户定案落地）。

输入 = `prefix_confirm_model.py` 的 cohort 缓存（eval/prefix_model_rows.jsonl）。
输出两件（同一模型的两种序列化）：

  * 插件版：`<out>/prefix_gate_model.lua`
    `{ feats={...}, tau=0.899, trees={{f,t,l,r,v},...}, freq={["的"]=-3.1,...} }`
  * 源码版：`<out>/prefix_gate_model.h`
    `kFeatN / kTreeN / kTau / kF / kT / kL / kR / kV / kFreqText / kFreqLog`

口径（与 2026-11-25 研究一致）：
  特征 = 46 个里的 **41 个可部署项**（去掉 m / n_syl_typed / partial_syl /
  n_syl_total / is_user——前四个要引擎音节切分、is_user 要域信息）
  **+ 12 项 CE 分布特征 `ss_*`**（同 span 同音簇的熵/占比/极差/margin/平局数
  ——2026-11-25 用户指令「落地 ss12」，90% 量 99.09%→99.23%）= **53 维**；
  模型 = LightGBM 300 树 / 15 叶；
  τ = **OOF 分数**在触发集上的 90% 分位（避免 in-sample 高估）。

跑法（研究库根目录）：
  python eval/calib/export_prefix_gate.py [--out eval/lgb]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402
import ss_ce_dist as S  # noqa: E402  （load_v2：带 g 的 cohort 缓存）

UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
# 2026-11-25 起落地 CE 分布特征（ss12）：41 可部署 + 12 分布 = 53
DEP = [c for c in M.FEATS if c not in UNAVAIL] + M.DIST_SS
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
              min_child_samples=60, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=7, verbose=-1)
TARGET_VOL = 0.90


def train(rows, folds=2, target_vol=0.90, target_n=None):
    import lightgbm as lgb
    X = np.array([[r["f"].get(c, r["g"].get(c, r["c"].get(c, 0.0)))
                   for c in DEP] for r in rows], float)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    clf = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    oof = np.zeros(len(rows))
    if folds == 2:
        grp = np.array([M.half_of(r) for r in rows])
    else:
        uniq = sorted({r["sid"] for r in rows})
        gid = {k: kk % folds for kk, k in enumerate(uniq)}
        grp = np.array([gid[r["sid"]] for r in rows])
    for g in range(folds):
        ite = grp == g
        itr = ~ite
        oof[ite] = lgb.LGBMClassifier(**PARAMS).fit(
            X[itr], y[itr]).predict_proba(X[ite])[:, 1]
    fire = np.array([r["fire"] for r in rows])
    yb = y.astype(bool)
    # τ = 覆盖「现行触发量的 90%」所需的分数（P≥0.95 结构触发量；新规则下
    # 含同档新增行）。target_n 可显式覆盖（对齐某个部署量）。
    ntrig = int(fire.sum())
    target = int(target_n or round(target_vol * ntrig))
    order = np.argsort(-oof)
    tau = float(oof[order[target - 1]])
    sel = order[:target]
    add = sum(1 for i in sel if not rows[i]["old_rule"]) \
        if "old_rule" in rows[0] else 0
    print(f"[train] {len(rows)} 行 / {len(DEP)} 特征 / "
          f"{clf.booster_.num_trees()} 树 / {folds} 折 OOF；τ={tau:.9f}"
          f" → 目标 {target}（= P≥0.95 结构触发 {ntrig} 的 "
          f"{100*TARGET_VOL:.0f}%），精度 {100*yb[sel].mean():.2f}%"
          f"（错 {int((~yb[sel]).sum())}；其中 p<0.95 "
          f"{int((~fire[sel]).sum())} 例；新增 {add}）")
    return clf, tau


def dump_trees(clf):
    dm = clf.booster_.dump_model()
    trees = []
    for t in dm["tree_info"]:
        f, th, lc, rc, val = [], [], [], [], []

        def walk(n):
            f.append(0), th.append(0.0), lc.append(0), rc.append(0), \
                val.append(0.0)
            idx = len(f)
            if "split_index" not in n:
                val[idx - 1] = float(n["leaf_value"])
                return idx
            li = walk(n["left_child"])
            ri = walk(n["right_child"])
            f[idx - 1] = n["split_feature"] + 1      # 1-based；0 = 叶
            th[idx - 1] = float(n["threshold"])
            lc[idx - 1] = li
            rc[idx - 1] = ri
            return idx

        walk(t["tree_structure"])
        trees.append((f, th, lc, rc, val))
    return trees


def write_lua(path, trees, tau, freq):
    L = ["-- 前缀确认门（LightGBM 300×15，53 特征 = 41+ss12）自动生成，勿手改",
         "-- 生成：eval/calib/export_prefix_gate.py（研究库；2026-11-26 同档新规则重训）",
         "-- feats = 特征向量顺序；f 下标 1-based（0=叶）；score∈[0,1]",
         "return {",
         " n=%d," % len(trees),
         " feats={" + ",".join('"%s"' % c for c in DEP) + "},",
         " tau=%.9g," % tau,
         " trees={"]
    for f, th, lc, rc, val in trees:
        L.append("  {f={%s},t={%s},l={%s},r={%s},v={%s}}," % (
            ",".join(str(x) for x in f),
            ",".join("%.17g" % x for x in th),
            ",".join(map(str, lc)), ",".join(map(str, rc)),
            ",".join("%.9g" % x for x in val)))
    L.append(" },")
    L.append(" freq={")
    items = ["[%s]=%.6f" % (json.dumps(c, ensure_ascii=False), v)
             for c, v in freq.items()]
    for i in range(0, len(items), 6):
        L.append("  " + ",".join(items[i:i + 6]) + ",")
    L.append(" },")
    L.append("}")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_h(path, trees, tau, freq):
    L = ["// 前缀确认门（LightGBM 300×15，53 特征 = 41+ss12）自动生成，勿手改",
         "// 生成：eval/calib/export_prefix_gate.py（研究库；2026-11-26 同档新规则重训）",
         "// 用法：按 kFeats 顺序装 53 维特征 → 每棵树从根 1 走到 f==0 的叶，",
         "//       sum v[node] → p=1/(1+exp(-sum))；p >= kTau 才确认。",
         "#pragma once",
         "#include <cstdint>",
         ""]
    L.append("// " + ", ".join(DEP))
    L.append("static constexpr int kPrefixFeatN = %d;" % len(DEP))
    L.append("static constexpr int kPrefixTreeN = %d;" % len(trees))
    L.append("static constexpr double kPrefixTau = %.9g;" % tau)
    L.append("static constexpr int kPrefixNodeN = %d;"
             % sum(len(t[0]) for t in trees))
    # 每棵树在全局节点数组里的起点（1-based，0 = 无）
    offs, tot = [], 0
    for t in trees:
        offs.append(tot)
        tot += len(t[0])
    L.append("static constexpr int kPrefixTreeOff[kPrefixTreeN] = {"
             + ",".join(map(str, offs)) + "};")
    for name, idx in (("kPrefixF", 0), ("kPrefixT", 1), ("kPrefixL", 2),
                      ("kPrefixR", 3), ("kPrefixV", 4)):
        vals = []
        for t in trees:
            if idx == 0:
                vals += [str(x) for x in t[0]]
            elif idx == 1:
                vals += ["%.17g" % x for x in t[1]]
            elif idx in (2, 3):
                vals += [str(x) for x in t[idx]]   # 1-based 树内下标；0 = 叶
            else:
                vals += ["%.17g" % x for x in t[idx]]
        ty = "double" if idx in (1, 4) else "int16_t"
        L.append("static const %s %s[kPrefixNodeN] = {" % (ty, name))
        for i in range(0, len(vals), 12):
            L.append("  " + ",".join(vals[i:i + 12]) + ",")
        L.append("};")
    chars = list(freq.keys())
    logs = list(freq.values())
    L.append("// 字频表：字（UTF-8 多字节串按下标拼）与 log 频率，线性查找即可")
    L.append("static const char *kPrefixFreqChar[] = {"
             + ",".join(json.dumps(c, ensure_ascii=False) for c in chars) + "};")
    L.append("static const float kPrefixFreqLog[] = {"
             + ",".join("%.6gf" % v for v in logs) + "};")
    L.append("static constexpr int kPrefixFreqN = %d;" % len(chars))
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(M.L.EVAL / "lgb"))
    ap.add_argument("--rows", default="v2", choices=["v2", "class"],
                    help="v2 = 旧 cohort（2026-11-25 模型）；class = "
                         "同档新规则预算窗行（2026-11-26 重训）")
    ap.add_argument("--min-ctx", type=int, default=0,
                    help="只训练 ctx≥N 字（生产有效上文门；class 建议 4）")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--target-n", type=int, default=0,
                    help="显式目标确认量（0 = 90% × P≥0.95 结构触发量）")
    ap.add_argument("--lua", default="", help="lua 输出路径（空 = out 下）")
    ap.add_argument("--h", default="", help="h 输出路径（空 = out 下）")
    a = ap.parse_args()
    if a.rows == "class":
        import prefix_class_gate as G
        rows = [r for r in G._load_rows(
            G.L.EVAL / "prefix_class_rows_budget.jsonl")
            if r["f"]["ctx_len"] >= a.min_ctx and r["f"]["p"] >= 0.5]
        print(f"[load] class 行 {len(rows)}（budget 窗，p≥0.5，"
              f"ctx≥{a.min_ctx}）")
    else:
        rows = S.load_v2()
        if a.min_ctx:
            rows = [r for r in rows if r["f"]["ctx_len"] >= a.min_ctx]
        print(f"[load] v2 行 {len(rows)}（ctx≥{a.min_ctx}）")
    clf, tau = train(rows, folds=a.folds,
                     target_n=(a.target_n or None))
    trees = dump_trees(clf)
    cfreq = M.char_freq()
    os.makedirs(a.out, exist_ok=True)
    lp = a.lua or os.path.join(a.out, "prefix_gate_model.lua")
    hp = a.h or os.path.join(a.out, "prefix_gate_model.h")
    write_lua(lp, trees, tau, cfreq)
    write_h(hp, trees, tau, cfreq)
    print(f"[export] {lp} {os.path.getsize(lp)/1024:.0f} KB")
    print(f"[export] {hp} {os.path.getsize(hp)/1024:.0f} KB")
    print("[export] 特征顺序：" + ", ".join(DEP))


if __name__ == "__main__":
    main()
