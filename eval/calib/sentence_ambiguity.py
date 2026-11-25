#!/usr/bin/env python3
"""前缀确认：组句「最优句 ≠ 实际句」能否当禁用/收紧信号（2026-11-25 用户提问）。

用户假设：beam 最优组句与实际不同 ⇒ 该句「拼音 + 上文」本身信息量不足（歧义）
⇒ 应当禁用/收紧歧义处（音节/编码）的自动确认。

三层验证（同一 cohort 缓存 eval/prefix_model_rows.jsonl，15,746 窗）：

  A  **oracle 上界**：用真值（gen1 句子 == 实际句子）分组看错误率；
     再按「只在相等时确认」算精度/量——这是该思路的理论天花板。
  B  **运行期可得**：句尾（lead 之后）在 top-20 里的稳定性
     （suf_stab / suf_n / suf_gap）——不需要真值，生产可算。
     单门（同量口径）+ 加进 41 特征看是否提升。
  C  **按语料挖禁用名单**（键 = lead 读音码）复现性：
     A 半挖 → B 半评，与新门叠加/替代的效果。

跑法（研究库根目录）：
  python eval/calib/sentence_ambiguity.py
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402

UNAVAIL = {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}
DEP = [c for c in M.FEATS if c not in UNAVAIL]
EXTRA = ["suf_stab", "suf_n", "suf_gap", "lead_frac", "len_var"]
TARGET_VOL = 0.90


def gen_by_wid():
    """gen.tsv 按 wid 归组（两个域分别读）。"""
    out = {}
    for tag, d, fn in (("luna", M.L.OUT, "gen20.tsv"),
                       ("big", M.L.EVAL / "prefix_big", "gen.tsv")):
        p = d / fn
        try:
            fh = open(p, encoding="utf-8")
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) != 7:
                    continue
                try:
                    out[f[0]] = out.get(f[0], []) + [
                        (float(f[2]), f[3], f[6])]
                except ValueError:
                    pass
    for k in out:
        out[k].sort(key=lambda x: -x[0])
    return out


def window_feats(gs, lead):
    """gen 列表 → (oracle 用 gen1 文本, 运行期特征)。"""
    if not gs:
        return None, None
    g1 = gs[0][2]
    suf1 = g1[len(lead):] if g1.startswith(lead) else g1
    sufs = [(s, t[len(lead):] if t.startswith(lead) else t)
            for s, _, t in gs]
    same = [s for s, sf in sufs if sf == suf1]
    other = [s for s, sf in sufs if sf != suf1]
    f = dict(
        suf_stab=len(same) / max(len(gs), 1),
        suf_n=float(len({sf for _, sf in sufs})),
        suf_gap=min(max(same) - max(other), 40.0) if other else 30.0,
        lead_frac=sum(1 for _, ld, _ in gs if ld == lead) / len(gs),
        len_var=float(len({len(t) for _, _, t in gs})),
    )
    return g1, f


def main():
    rows = M.load_rows()
    gmap = gen_by_wid()
    keep, nogen = [], 0
    for r in rows:
        gs = gmap.get(r["wid"])
        g1, f = window_feats(gs, r["lead"])
        if g1 is None:
            nogen += 1
            continue
        keep.append(dict(r, gen1=g1, g=f, truth=r["text"]))
    rows = keep
    n = len(rows)
    fire = [r for r in rows if r["fire"]]
    print(f"[data] 行 {n}（无 gen {nogen}）；现行触发 {len(fire)}、"
          f"精度 {100*sum(r['ok'] for r in fire)/len(fire):.2f}%")

    # ── A oracle：gen1 句子 == 实际句子 ────────────────────────────────
    print("\nA oracle 上界（判断用了真值 → 只能测天花板，不能落地）")
    eq = [r for r in fire if r["gen1"] == r["truth"]]
    ne = [r for r in fire if r["gen1"] != r["truth"]]
    for nm, s in (("gen1 == 实际", eq), ("gen1 != 实际", ne)):
        if not s:
            continue
        print(f"  {nm:<14} 触发 {len(s):>6} 精度 "
              f"{100*sum(r['ok'] for r in s)/len(s):>6.2f}%  "
              f"占现行量 {100*len(s)/len(fire):>5.1f}%")
    if eq:
        print(f"  → 只在相等时确认：量 {100*len(eq)/len(fire):.1f}%、"
              f"精度 {100*sum(r['ok'] for r in eq)/len(eq):.2f}%")

    # ── B 运行期：句尾稳定性 ─────────────────────────────────────────
    print("\nB 运行期信号（不需要真值）：句尾在 top-20 里的稳定性")
    for key in ("suf_stab", "suf_gap", "lead_frac"):
        vals = np.array([r["g"][key] for r in rows], float)
        y = np.array([r["ok"] for r in rows], bool)
        f_ = np.array([r["fire"] for r in rows], bool)
        print(f"  {key}: 触发集 好/坏 中位数 "
              f"{np.median(vals[f_ & y]):.3f} / {np.median(vals[f_ & ~y]):.3f}")
    # 单门：suf_stab 阈值 → 量/精度
    _v = np.array([r["g"]["suf_stab"] for r in fire], float)
    print(f"  suf_stab 在触发集的取值范围：min {_v.min():.2f} / "
          f"p50 {np.median(_v):.2f} / max {_v.max():.2f}"
          "（top-20 从不在句尾上达成一致 → 该信号无区分度）")
    print("  单门（fire ∧ suf_stab ≥ t）：")
    for t in (1.0, 0.95, 0.9, 0.8, 0.6, 0.4):
        s = [r for r in fire if r["g"]["suf_stab"] >= t - 1e-9]
        if s:
            print(f"    t={t:<5} 触发 {len(s):>6}（{100*len(s)/len(fire):>5.1f}%）"
                  f" 精度 {100*sum(r['ok'] for r in s)/len(s):>6.2f}%")

    # ── D 运行期「lead 自己那段码有歧义」= 同 span 竞争 ──────────────
    # 这是用户直觉在**生产可算**侧最近的那条：池里有几个候选与 lead 覆盖
    # 同一段码（同码异字）。它本身就是模型特征 n_same_span。
    print("\nD 运行期「lead 码段本身歧义」= 同 span 候选数（已在模型里）")
    for k in (1, 2, 3, 5, 8):
        s = [r for r in fire if r["f"]["n_same_span"] >= k]
        if s:
            print(f"  n_same_span≥{k}: 触发 {len(s):>6}"
                  f"（{100*len(s)/len(fire):>5.1f}%） 精度 "
                  f"{100*sum(r['ok'] for r in s)/len(s):>6.2f}%")

    # 加进模型（A→B / B→A 平均 + OOF 90% 量）
    def mat(feats):
        return np.array([[r["f"][c] if c in r["f"] else r["g"][c]
                          for c in feats] for r in rows], float)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    half = np.array([M.half_of(r) for r in rows])
    ntrig = int(sum(r["fire"] for r in rows))
    print("\n  加进模型（同 300×15 参数；A→B/B→A 同量精度 + OOF@90% 量）：")
    for nm, feats in (("41 特征（现行）", DEP),
                      ("41 + 句尾稳定 5 项", DEP + EXTRA),
                      ("41 + 仅 suf_stab", DEP + ["suf_stab"])):
        X = mat(feats)
        sv = _same_vol(rows, X, y, half)
        oof = np.zeros(len(y))
        import lightgbm as lgb
        for A, B in ((0, 1), (1, 0)):
            itr, ite = half == A, half == B
            clf = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=60, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
                verbose=-1).fit(X[itr], y[itr])
            oof[ite] = clf.predict_proba(X[ite])[:, 1]
        target = int(round(TARGET_VOL * ntrig))
        o = np.argsort(-oof)[:target]
        print(f"    {nm:<20} 同量 {100*sv:.2f}%  OOF90% "
              f"{100*y[o].mean():.2f}%")

    # ── C 语料挖禁用名单（键 = lead 读音码）── 复现性 ────────────────
    print("\nC 按语料挖「lead 读音码」禁用名单（A 挖 → B 评，再对称）")
    for A, B in ((0, 1), (1, 0)):
        tra = [r for r in rows if M.half_of(r) == A and r["fire"]]
        teb = [r for r in rows if M.half_of(r) == B and r["fire"]]
        off, tight, detail, e_star = M.mine_blacklist(
            tra, lambda r: r["lead_syl"], K=5, min_n=10)
        base = 100 * sum(r["ok"] for r in teb) / max(len(teb), 1)
        for nm, ban in (("仅禁用", off), ("禁用+加强", off | tight)):
            sel = [r for r in teb if r["lead_syl"] not in ban]
            pr = 100 * sum(r["ok"] for r in sel) / max(len(sel), 1)
            print(f"  {A}→{B} {nm:<9} 名单 {len(ban):>2} 项  触发 {len(sel):>6}"
                  f"  精度 {pr:>6.2f}%  基线 {base:>6.2f}%  Δ {pr-base:+.2f}pp")
        names = sorted(off)[:12]
        print(f"       A 半名单（前 12）：{' '.join(names)}")

    # ── C2 挖出的码名单叠在模型上（90% 量）───────────────────────────
    print("\nC2 码名单叠在**模型 90% 量**上（A 挖 → B 半上剔除，再对称）")
    import lightgbm as lgb
    P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=60, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
             verbose=-1)
    X = mat(DEP)
    oof = np.zeros(len(y))
    for A, B in ((0, 1), (1, 0)):
        itr, ite = half == A, half == B
        oof[ite] = lgb.LGBMClassifier(**P).fit(
            X[itr], y[itr]).predict_proba(X[ite])[:, 1]
    target = int(round(TARGET_VOL * ntrig))
    sel = [rows[i] for i in np.argsort(-oof)[:target]]
    print(f"  模型基线（90% 量 {target}）：精度 "
          f"{100*sum(r['ok'] for r in sel)/len(sel):.2f}%")
    for K in (2, 5, 12):
        for A, B in ((0, 1), (1, 0)):
            tra = [r for r in rows if M.half_of(r) == A and r["fire"]]
            off, tight, _d, _e = M.mine_blacklist(
                tra, lambda r: r["lead_syl"], K=K, min_n=10)
            sh = [r for r in sel if M.half_of(r) == B]
            sb = [r for r in sh if r["lead_syl"] not in off]
            st = [r for r in sh if r["lead_syl"] not in (off | tight)]
            pb = 100*sum(r["ok"] for r in sb)/max(len(sb), 1)
            pt = 100*sum(r["ok"] for r in st)/max(len(st), 1)
            print(f"  K={K:<3}{A}→{B} 禁用 {len(off)} 项：{len(sb)}/{len(sh)}"
                  f" 精度 {pb:>6.2f}%（原 "
                  f"{100*sum(r['ok'] for r in sh)/max(len(sh),1):.2f}%）"
                  f"；禁用+加强 {len(off|tight)} 项：{len(st)}/{len(sh)} "
                  f"{pt:>6.2f}%")


def _same_vol(rows, X, y, half, seed=7):
    import lightgbm as lgb
    out = []
    for A, B in ((0, 1), (1, 0)):
        itr, ite = half == A, half == B
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=60, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            verbose=-1).fit(X[itr], y[itr])
        sc = clf.predict_proba(X[ite])[:, 1]
        k = int(sum(rows[i]["fire"] for i in np.where(ite)[0]))
        sel = np.argsort(-sc)[:k]
        out.append(float(y[ite][sel].mean()))
    return float(np.mean(out))


if __name__ == "__main__":
    main()
