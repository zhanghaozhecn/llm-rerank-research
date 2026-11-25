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
  n_syl_total / is_user——前四个要引擎音节切分、is_user 要域信息）；
  模型 = LightGBM 300 树 / 15 叶（研究最优档，同触发量 98.27%）；
  τ = **OOF 分数**在触发集上的 90% 分位（避免 in-sample 高估；
  90% 量下 OOF 精度 98.56%）。

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

UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
DEP = [c for c in M.FEATS if c not in UNAVAIL]
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
              min_child_samples=60, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=7, verbose=-1)
TARGET_VOL = 0.90


def train(rows):
    import lightgbm as lgb
    X = np.array([[r["f"][c] for c in DEP] for r in rows], float)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    clf = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    oof = np.zeros(len(rows))
    for A, B in ((0, 1), (1, 0)):
        itr = [i for i, r in enumerate(rows) if M.half_of(r) == A]
        ite = [i for i, r in enumerate(rows) if M.half_of(r) == B]
        oof[ite] = lgb.LGBMClassifier(**PARAMS).fit(
            X[itr], y[itr]).predict_proba(X[ite])[:, 1]
    fire = np.array([r["fire"] for r in rows])
    y = np.array([r["ok"] for r in rows], bool)
    # τ = 覆盖「现行触发量的 90%」所需的分数（在全域可触发集合上取分位——
    # 生产侧模型取代 P≥0.95 硬门后，p<0.95 的窗也参与，量按现行触发量折算）
    ntrig = int(fire.sum())
    target = int(round(TARGET_VOL * ntrig))
    order = np.argsort(-oof)
    tau = float(oof[order[target - 1]])
    sel = order[:target]
    print(f"[train] 41 特征 / {clf.booster_.num_trees()} 树；OOF τ={tau:.6f}"
          f" → 保留 {target}（= 现行触发 {ntrig} 的 {100*TARGET_VOL:.0f}%），"
          f"精度 {100*y[sel].mean():.2f}%（其中 p<0.95 的 "
          f"{int((~fire[sel]).sum())} 例）")
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
    L = ["-- 前缀确认门（LightGBM 300×15，41 可部署特征）自动生成，勿手改",
         "-- 生成：eval/calib/export_prefix_gate.py（研究库 2026-11-25）",
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
    L = ["// 前缀确认门（LightGBM 300×15，41 可部署特征）自动生成，勿手改",
         "// 生成：eval/calib/export_prefix_gate.py（研究库 2026-11-25）",
         "// 用法：按 kFeats 顺序装 41 维特征 → 每棵树从根 1 走到 f==0 的叶，",
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
    a = ap.parse_args()
    rows = M.load_rows()
    clf, tau = train(rows)
    trees = dump_trees(clf)
    cfreq = M.char_freq()
    os.makedirs(a.out, exist_ok=True)
    lp = os.path.join(a.out, "prefix_gate_model.lua")
    hp = os.path.join(a.out, "prefix_gate_model.h")
    write_lua(lp, trees, tau, cfreq)
    write_h(hp, trees, tau, cfreq)
    print(f"[export] {lp} {os.path.getsize(lp)/1024:.0f} KB")
    print(f"[export] {hp} {os.path.getsize(hp)/1024:.0f} KB")
    print("[export] 特征顺序：" + ", ".join(DEP))


if __name__ == "__main__":
    main()
