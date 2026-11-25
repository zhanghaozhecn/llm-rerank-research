#!/usr/bin/env python3
"""用户词频（eff / uni）进树模型：wiki 预打 3 万词模拟账本（2026-11-25 用户指令）。

现状：53 特征全是 CE / 组句 / 结构，**没有用户词频**；词频只通过
「层1 准入选窗」和「融合序第 1 必须==lead」间接起作用。本脚本把
`β·log1p(eff)` 与 `log(uni+1)`（含竞争者的同口径值）直接做成特征，
看树模型能否利用。

账本模拟（与既有评估同口径）：按 wiki 句子顺序逐词提交，
`FormulaReplay(tau=400)` 指数衰减 + 原始累计；打满 3 万词。
窗口候选的 CE 序从既有 CE 缓存重算（不产生新 CE）。

特征（5）：
  uf_eff      log1p(eff(lead))
  uf_uni      log(uni(lead)+1)
  uf_eff_r2   log1p(eff(rank2 竞争者))
  uf_uni_r2   log(uni(rank2)+1)
  uf_eff_gap  log1p(eff(lead)) − log1p(eff(rank2))

跑法（研究库根目录）：
  python eval/calib/user_freq_feat.py [--words 30000]
"""
import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402
import luna_mixed_lib as L  # noqa: E402
import ss_ce_dist as S  # noqa: E402
from corpus_std import FormulaReplay  # noqa: E402

UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
DEP = [c for c in M.FEATS if c not in UNAVAIL] + M.DIST_SS
UF = ["uf_eff", "uf_uni", "uf_eff_r2", "uf_uni_r2", "uf_eff_gap"]
TAU_DECAY = 400
TAU = 0.973286164
TA = {"他", "它", "她", "他们", "它们", "她们"}


def build_ledger(n_words):
    """按 wiki 句子顺序预打 n 个词 → **冻结**的账本快照（所有评估句都用它）。

    口径 = 模拟一个"已提交 n 词"的用户；评估窗一律在这个年龄上打分
    （不随评估句推进，避免"评估句本身进了账本"的泄漏）。"""
    seq = []
    for d in (L.OUT, L.EVAL / "prefix_big"):
        for s in L.jsonl_iter(d / "sentences.jsonl"):
            seq.append(s)
    rp = FormulaReplay(TAU_DECAY)
    used = 0
    for s in seq:
        if used >= n_words:
            break
        for w in s["segs"]:
            rp.bump(w)
            used += 1
    snap = dict(tick=rp.tick,
                state={w: list(v) for w, v in rp.state.items()})
    print(f"[ledger] 年龄 {used} 词（tick={rp.tick}），词表 {len(snap['state'])}")
    return snap


def eff_at(snap, word):
    e = snap["state"].get(word)
    if not e:
        return 0.0, 0
    return e[1] * math.exp((e[2] - snap["tick"]) / TAU_DECAY), e[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=30000)
    a = ap.parse_args()
    snaps = build_ledger(a.words)
    rows = S.load_v2()
    wins = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
        for w in L.jsonl_iter(d / "windows.jsonl"):
            wins[w["wid"]] = w
    cands = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
        cands.update(json.load(open(d / "cands.json", encoding="utf-8")))
    ce = L.load_ce()

    extra, miss = [], 0
    for r in rows:
        w = wins.get(r["wid"])
        snap = snaps
        if not w or not snap:
            extra.append({k: 0.0 for k in UF})
            miss += 1
            continue
        pool = cands.get(w["code"], [])[:L.K_COLLECT]
        ced = {}
        for ln, sid in L.ce_lines(w["ctx"], [c[0] for c in pool][:10]):
            rr = ce.get(sid) or {}
            for t in ln["cands"]:
                if t in rr and rr[t][0] is not None:
                    ced[t] = rr[t]
        # CE 序竞者（rank2 = 非 lead 的最佳）
        sc = {}
        for c in pool[:L.K_WINDOW + 1]:
            v = ced.get(c[0])
            if v is None:
                continue
            nt = v[1] if v[1] else sum(1 for x in v[0] if x is not None)
            s = 0.0
            for k in range(min(int(nt), 3)):
                if k < len(v[0]) and v[0][k] is not None:
                    s += M.KCE[k] * v[0][k]
            sc[c[0]] = s
        r2 = None
        for t, _ in sorted(sc.items(), key=lambda kv: kv[1]):
            if t != r["lead"]:
                r2 = t
                break
        e1, u1 = eff_at(snap, r["lead"])
        e2, u2 = eff_at(snap, r2) if r2 else (0.0, 0)
        le, lr2 = math.log1p(e1), math.log1p(e2)
        extra.append(dict(uf_eff=le, uf_uni=math.log(1 + u1),
                          uf_eff_r2=lr2, uf_uni_r2=math.log(1 + u2),
                          uf_eff_gap=le - lr2))
    cov = sum(1 for r, e in zip(rows, extra) if e["uf_uni"] > 0) / len(rows)
    print(f"[cover] lead 在账本里的行占比 {100*cov:.1f}%"
          + (f"（{miss} 行无窗口）" if miss else ""))

    y = np.array([1 if r["ok"] else 0 for r in rows])
    half = np.array([M.half_of(r) for r in rows])
    fire = np.array([r["fire"] for r in rows], bool)
    target = int(round(0.9 * int(fire.sum())))
    import lightgbm as lgb
    P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=60, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=7, verbose=-1)

    def mat(use_uf):
        out = []
        for r, e in zip(rows, extra):
            row = [r["f"][c] if c in r["f"] else r["g"][c] for c in DEP]
            if use_uf:
                row += [e[k] for k in UF]
            out.append(row)
        return np.array(out, float)

    def run(X):
        oof = np.zeros(len(rows))
        sv = []
        for A, B in ((0, 1), (1, 0)):
            itr, ite = half == A, half == B
            clf = lgb.LGBMClassifier(**P).fit(X[itr], y[itr])
            sc = clf.predict_proba(X[ite])[:, 1]
            oof[ite] = sc
            k = int(fire[ite].sum())
            sv.append(float(y[ite][np.argsort(-sc)[:k]].mean()))
        o = np.argsort(-oof)[:target]
        return 100 * y[o].mean(), 100 * np.mean(sv), oof

    print("\n[模型增量] 53 vs 53+用户词频 5 项")
    p0, s0, o0 = run(mat(False))
    print(f"  53（现行）        OOF@90% {p0:.2f}%  同量 {s0:.2f}%")
    p1, s1, o1 = run(mat(True))
    print(f"  53 + 词频 5 项     OOF@90% {p1:.2f}%  同量 {s1:.2f}%  "
          f"Δ {p1-p0:+.2f}/{s1-s0:+.2f}")

    # 消融：只加 eff / 只加 uni
    for nm, ks in (("仅 eff 2 项", ["uf_eff", "uf_eff_r2"]),
                   ("仅 uni 2 项", ["uf_uni", "uf_uni_r2"])):
        X = np.column_stack([
            mat(False),
            np.array([[e[k] for k in ks] for e in extra], float)])
        p, s, _ = run(X)
        print(f"  53 + {nm:<12} OOF@90% {p:.2f}%  同量 {s:.2f}%  "
              f"Δ {p-p0:+.2f}/{s-s0:+.2f}")

    # 词频本身的区分度（触发集里 好/坏 中位数）
    print("\n[方向] 触发集内 好/坏 中位数")
    for k in UF:
        v = np.array([e[k] for e in extra], float)
        print(f"  {k:<12} 好 {np.median(v[fire & (y == 1)]):>6.3f}   "
              f"坏 {np.median(v[fire & (y == 0)]):>6.3f}")


if __name__ == "__main__":
    main()
