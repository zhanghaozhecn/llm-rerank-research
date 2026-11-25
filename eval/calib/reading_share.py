#!/usr/bin/env python3
"""全拼多音字：候选读音占比（连续量）与前缀确认（2026-11-25 用户提问）。

用户提醒：多音字不是「规范/非规范」二分——不同字的读音占比差别很大，
有些接近 1:1（表里 调 tiao 52%/diao 48%、长 chang 58%/zhang 42%）。
所以本脚本用**连续读音占比** P(读音|字) 而不是布尔标志：

  1. 建表：万象 `jichu.dict.yaml`（1,558,937 条，带权重）→ 字→{读音: 权重}，
     落盘缓存 `eval/char_readings.tsv`（7 千字 / 10s 建一次）。
  2. 对齐：候选文本 × 它的 code span（`lead_syl`）→ 用各字读音集合做 DP 切分
     （同一串拼音可能有多种切法，取**联合概率最大**的那种）。
     → 每字一个 share = P(该读音 | 该字)。
  3. 特征（4 项）：`rr_min`（最小 share）/ `rr_mean` / `rr_logp`
     （Σlog share = 该读音组合的对数概率）/ `rr_cnt50`（share ∈[0.35,0.65]
     的近 1:1 字个数），外加 `rr_known`（对齐成功标志）。
  4. 评估：按 share 分桶看精度；再把 5 项加进 53 特征，比 OOF@90% 量
     与同量精度（同 300×15 参数、同一 τ 规则）。

跑法（研究库根目录）：
  python eval/calib/reading_share.py [--rebuild]
"""
import argparse
import math
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import ss_ce_dist as S  # noqa: E402
import prefix_confirm_model as M  # noqa: E402

DICT = (M.L.PROJ.parent / ".." / "Rime").resolve()  # 占位，实际用 env
NG = None
CACHE = M.L.EVAL / "char_readings.tsv"
TONE = {"ā": "a", "á": "a", "ǎ": "a", "à": "a", "ē": "e", "é": "e", "ě": "e",
        "è": "e", "ī": "i", "í": "i", "ǐ": "i", "ì": "i", "ō": "o", "ó": "o",
        "ǒ": "o", "ò": "o", "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
        "ǖ": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v", "ü": "v", "ń": "n",
        "ň": "n", "ǹ": "n", "ḿ": "m"}


def strip_tone(s):
    return "".join(TONE.get(c, c) for c in s)


def build_cache():
    import os
    d = os.path.join(os.environ["APPDATA"], "Rime", "llm_ngram", "dicts")
    cr = defaultdict(Counter)
    for fn in ("jichu.dict.yaml", "shici.dict.yaml"):
        p = os.path.join(d, fn)
        try:
            fh = open(p, encoding="utf-8")
        except FileNotFoundError:
            continue
        with fh:
            for line in fh:
                if line[:1] in "#-" or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 2:
                    continue
                w, py = f[0], f[1]
                syls = [strip_tone(x) for x in py.split()]
                if len(w) != len(syls):
                    continue
                try:
                    wt = float(f[2]) if len(f) > 2 else 1.0
                except ValueError:
                    wt = 1.0
                for c, s in zip(w, syls):
                    if "一" <= c <= "鿿":
                        cr[c][s] += wt
    with open(CACHE, "w", encoding="utf-8") as f:
        for c, v in cr.items():
            tot = sum(v.values())
            for s, wt in v.items():
                f.write(f"{c}\t{s}\t{wt/tot:.6f}\n")
    return cr


def load_cache():
    out = defaultdict(Counter)
    if not CACHE.exists():
        return build_cache()
    for line in open(CACHE, encoding="utf-8"):
        c, s, p = line.rstrip("\n").split("\t")
        out[c][s] = float(p)
    return out


def align(text, code, tab):
    """文本 × 拼音串 → 每字的 share 列表（联合概率最大切分）；失败 None。"""
    n = len(text)
    # dp[i][j] = 处理完 i 个字、消费 j 个字母时的最大 Σlog share
    dp = [[None] * (len(code) + 1) for _ in range(n + 1)]
    dp[0][0] = (0.0, [])
    for i in range(n):
        for j in range(len(code) + 1):
            cur = dp[i][j]
            if cur is None:
                continue
            for s, p in tab.get(text[i], {}).items():
                if p <= 0:
                    continue
                if code.startswith(s, j):
                    nj = j + len(s)
                    val = cur[0] + math.log(max(p, 1e-6))
                    old = dp[i + 1][nj]
                    if old is None or val > old[0]:
                        dp[i + 1][nj] = (val, cur[1] + [(s, p)])
    res = dp[n][len(code)] if code else None
    return res[1] if res else None


def feats_for(rows, tab):
    out = []
    for r in rows:
        al = align(r["lead"], r["lead_syl"], tab)
        if not al:
            out.append(dict(rr_min=1.0, rr_mean=1.0, rr_logp=0.0,
                            rr_cnt50=0.0, rr_known=0.0))
            continue
        ps = [p for _, p in al]
        out.append(dict(
            rr_min=min(ps), rr_mean=sum(ps) / len(ps),
            rr_logp=sum(math.log(max(p, 1e-6)) for p in ps),
            rr_cnt50=float(sum(1 for p in ps if 0.35 <= p <= 0.65)),
            rr_known=1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild:
        CACHE.unlink(missing_ok=True)
    tab = load_cache()
    rows = S.load_v2()
    extra = feats_for(rows, tab)
    y = np.array([1 if r["ok"] else 0 for r in rows])
    fire = [r for r in rows if r["fire"]]
    print(f"[data] 字表 {len(tab)} 字；对齐成功 "
          f"{100*sum(e['rr_known'] for e in extra)/len(rows):.1f}%"
          f"（触发集 {100*np.mean([e['rr_known'] for r,e in zip(rows,extra) if r['fire']]):.1f}%）")

    def prec(s):
        return len(s), 100 * sum(r["ok"] for r in s) / max(len(s), 1)

    print("\n[按最小读音占比分桶] 触发集精度（低占比 = 多音字的罕用读音）")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        sub = [r for r, e in zip(rows, extra)
               if r["fire"] and e["rr_known"] and lo <= e["rr_min"] < hi]
        n, pr = prec(sub)
        print("DEBUG", lo, hi, n)
        if n:
            print(f"  rr_min {lo:.2f}–{hi:.2f}: 触发 {n:>5} 精度 {pr:>6.2f}%")
    for dom in ("wiki", "user"):
        sub = [r for r, e in zip(rows, extra)
               if r["fire"] and r["domain"] == dom and e["rr_min"] < 0.6]
        n, pr = prec(sub)
        tot = prec([r for r in fire if r["domain"] == dom])
        print(f"  {dom}: rr_min<0.6 → {n} 例 / {pr:.2f}%（该域整体 "
              f"{tot[1]:.2f}%）")

    # 模型增量
    import lightgbm as lgb
    FE = M.DIST_SS
    BASE = [c for c in M.FEATS if c not in
            {"m", "n_syl_typed", "partial_syl", "n_syl_total", "is_user"}] + FE
    EX = ["rr_min", "rr_mean", "rr_logp", "rr_cnt50", "rr_known"]
    half = np.array([M.half_of(r) for r in rows])
    ntrig = int(sum(r["fire"] for r in rows))
    target = int(round(0.9 * ntrig))
    P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=60, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
             verbose=-1)

    def mat(with_extra):
        cols = []
        for r, e in zip(rows, extra):
            row = [r["f"][c] if c in r["f"] else r["g"][c] for c in BASE]
            if with_extra:
                row += [e[k] for k in EX]
            cols.append(row)
        return np.array(cols, float)

    print("\n[模型增量] 53 特征 vs 53+读音占比 5 项")
    for nm, we in (("53（现行）", False), ("53 + 读音占比 5", True)):
        X = mat(we)
        oof = np.zeros(len(rows))
        sv = []
        for A, B in ((0, 1), (1, 0)):
            itr, ite = half == A, half == B
            clf = lgb.LGBMClassifier(**P).fit(X[itr], y[itr])
            oof[ite] = clf.predict_proba(X[ite])[:, 1]
            k = int(sum(r["fire"] for r, i in zip(rows, range(len(rows)))
                        if i in np.where(ite)[0]))
            sv.append(float(y[ite][np.argsort(-clf.predict_proba(
                X[ite])[:, 1])[:k]].mean()))
        o = np.argsort(-oof)[:target]
        print(f"  {nm:<18} OOF@90% {100*y[o].mean():>6.2f}%  "
              f"同量 {100*np.mean(sv):>6.2f}%")


if __name__ == "__main__":
    main()
