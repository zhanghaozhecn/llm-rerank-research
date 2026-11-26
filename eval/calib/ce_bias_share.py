#!/usr/bin/env python3
"""字-音节 CE 偏置：读音占比对数概率的转化式标定（2026-11-26）。

问题（用户 2026-11-26）：整句模式里单字候选全部拿到 CE1 之后，**原在引擎
序后面的高频字低频读音**会被 CE 抬上来——CE 只看文本不看读音，而输入码
恰恰条件化在读音上。要补的正是贝叶斯缺的那一项：

    P(字 | 上文, 读音) ∝ P(字 | 上文) · P(读音 | 字)

CE1 = −log P(字|上文) 已经给了第一项；第二项就是字的读音占比
share(读音|字)（`eval/char_readings.tsv`，万象 jichu+shici 累计）。

**转化式**（本脚本要回答"具体怎么转"）：
  T0  基线：score = −CE1（冷账本，无词频项）
  T1  λ·log share                        （贝叶斯原式，λ 为校准系数）
  T2  λ·logit(share) = λ·log(s/(1−s))    （对照：把 share 当独立证据的常见误用）
  T3  λ·max(log share, log ε)            （尾部截断：share 估计在极罕读音上不可靠）
  T4  clip(λ·log share, −C, 0)           （相对 CE 的绝对上限）
  T5  λ·log share，仅 share<1 的字生效    （与 T1 数学等价，作 sanity 对照）

评测口径（冷账本，wiki/user 混合域，句级拆半）：
  窗   = luna 混合窗 cohort（`cohort.jsonl`；扩池 `cohort64.jsonl`）
  子集 = 首字单字候选（候选 span 与真值首字音节一致、读音占比可查）
  任务 = **单字档首位**（这一档正是整句模式的前缀确认结构证据；
         见 prefix_gate 的「同字数档首选」）
  指标 = 单字档 top-1 正确率（= 1 − 选重率）；真值首字不在档内的窗剔除
  拆半 = 按 sid 奇偶，A 选 λ / B 评测 + 对称平均（防同域调参）

输出：eval/ce_bias_share.txt
用法：
  python eval/calib/ce_bias_share.py                 # 16 池（现成）
  python eval/calib/ce_bias_share.py --pool 64       # cohort64 就绪后
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import luna_mixed_lib as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
W2, W3 = 1.10, 0.60          # 生产 CE 权重（单字 ntok=1，只用 ce1）
OUT = L.EVAL / "ce_bias_share.txt"
READINGS = L.EVAL / "char_readings.tsv"

LAMBDAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
EPSILONS = (1e-3, 1e-2, 5e-2)
CAPS = (2.0, 4.0, 8.0)


def load_readings(path=READINGS):
    tab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) == 3:
                try:
                    tab.setdefault(a[0], {})[a[1]] = float(a[2])
                except ValueError:
                    pass
    return tab


def iso_odd(sid):
    try:
        return int(str(sid)[1:]) % 2 == 1
    except ValueError:
        return False


def load_items(pool, cohort=None):
    """→ [dict(sid, correct, cands=[(text, score, share)])]（只留可判窗）。"""
    rows = [json.loads(l) for l in open(cohort or L.COHORT, encoding="utf-8")]
    rows = [r for r in rows if not r["dyn"]]
    tab = load_readings()
    out, stat = [], Counter()
    for r in rows:
        stat["windows"] += 1
        truth = r["text"]
        singles = [e for e in r["eng"][:pool] if len(e[0]) == 1 and e[5]]
        if not singles:
            continue
        stat["has_single"] += 1
        t0 = truth[0]
        hit = [e for e in singles if e[0] == t0]
        if not hit:
            continue
        stat["truth_in_pool"] += 1
        t_end = hit[0][2]                      # 真值首字的音节码末尾
        syl = r["code"][:t_end]
        # 同音节的单字候选（span 相同的才是同一读音的竞争者）
        grp = [e for e in singles if e[2] == t_end]
        cands = []
        for e in grp:
            ce3 = e[5]
            score = -(ce3[0] + (W2 * ce3[1] if len(ce3) > 1 else 0.0)
                      + (W3 * ce3[2] if len(ce3) > 2 else 0.0))
            share = tab.get(e[0], {}).get(syl)
            if share is None or share <= 0:
                share = 1e-4                   # 表外（多音字未收录该读音）
            cands.append((e[0], score, share))
        if len(cands) < 2:
            continue
        out.append(dict(sid=r["sid"], code=r["code"], syl=syl, truth=t0,
                        cands=cands))
    return out, stat


def delta_of(name, share, lam, eps=None, cap=None):
    ls = math.log(max(share, 1e-12))
    if name == "T0":
        return 0.0
    if name == "T1":
        return lam * ls
    if name == "T2":
        s = min(max(share, 1e-6), 1 - 1e-6)
        return lam * math.log(s / (1 - s))
    if name == "T3":
        return lam * max(ls, math.log(eps))
    if name == "T4":
        return max(lam * ls, -cap)
    if name == "T5":
        return lam * ls if share < 1.0 else 0.0
    raise ValueError(name)


def top1(items, name="T0", lam=0.0, eps=None, cap=None):
    ok = 0
    ranks = []
    for it in items:
        scored = sorted(((sc + delta_of(name, sh, lam, eps, cap), t, sh)
                         for (t, sc, sh) in it["cands"]), reverse=True)
        pos = next(i for i, x in enumerate(scored, 1) if x[1] == it["truth"])
        ranks.append(pos)
        ok += (pos == 1)
    n = max(len(items), 1)
    return ok / n, sum(ranks) / n


def eval_variant(items, name, lam, eps=None, cap=None):
    """A 选 λ / B 评测 + 对称平均（λ 也只在 A 上选）。"""
    half = [x for x in items if iso_odd(x["sid"])]
    other = [x for x in items if not iso_odd(x["sid"])]
    accA, rA = top1(half, name, lam, eps, cap)
    accB, rB = top1(other, name, lam, eps, cap)
    return (accA + accB) / 2, (rA + rB) / 2, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=16, choices=[16, 32, 64])
    args = ap.parse_args()
    if args.pool == 64:
        cohort = L.COHORT_EXT
        if not cohort.exists():
            raise SystemExit(f"缺 {cohort}：先跑 eval/calib/luna_mixed_pool64.py assemble")
    else:
        cohort = None
    items, stat = load_items(args.pool, cohort)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 96)
    out(f"字-音节 CE 偏置 · 转化式标定 · 池 {args.pool} · 冷账本 · "
        f"单字档首位")
    out("=" * 96)
    out(f"窗 {stat['windows']} → 有单字候选 {stat['has_single']} → "
        f"真值首字在池 {stat['truth_in_pool']} → 可判 {len(items)}")
    # 基线错因画像
    err_hi_share = err_lo_share = ok_hi = 0
    for it in items:
        scored = sorted(((sc, t, sh) for (t, sc, sh) in it["cands"]),
                        reverse=True)
        win = scored[0]
        truth_share = next(sh for (t, sc, sh) in it["cands"] if t == it["truth"])
        wrong = win[1] != it["truth"]
        if not wrong:
            ok_hi += 1
        elif win[2] >= 0.5 and truth_share < 0.5:
            err_hi_share += 1     # 选中高频读音、丢掉低频读音
        elif win[2] < 0.5 < truth_share:
            err_lo_share += 1     # 反向（罕见读音被抬上来——不该发生的方向）
    out(f"基线：正确 {ok_hi} / 错 {len(items) - ok_hi}"
        f"（其中「赢家读音占比≥0.5 且真值<0.5」{err_hi_share}，"
        f"反向 {err_lo_share}）")

    base = eval_variant(items, "T0", 0.0)
    out(f"\nT0 基线 {'':<12} 单字档 top-1 {100 * base[0]:6.2f}%  "
        f"真值均秩 {base[1]:.2f}")
    rows = [("T0 基线", 0.0, None, None, base)]
    for name in ("T1", "T2", "T5"):
        for lam in LAMBDAS:
            rows.append((name, lam, None, None, eval_variant(items, name, lam)))
    for eps in EPSILONS:
        for lam in (0.5, 0.75, 1.0):
            rows.append(("T3", lam, eps, None,
                         eval_variant(items, "T3", lam, eps=eps)))
    for cap in CAPS:
        for lam in (0.5, 0.75, 1.0):
            rows.append(("T4", lam, None, cap,
                         eval_variant(items, "T4", lam, cap=cap)))
    rows.sort(key=lambda r: -r[4][0])
    out(f"\n{'变体':<10}{'λ':>6}{'ε':>9}{'cap':>6}{'top-1':>9}{'均秩':>7}"
        f"{'vs T0':>9}")
    for name, lam, eps, cap, (acc, mr, n) in rows[:20]:
        e = f"{eps:.0e}" if eps else ""
        c = f"{cap:.0f}" if cap else ""
        out(f"{name:<10}{lam:>6.2f}{e:>9}{c:>6}{100 * acc:>8.2f}%"
            f"{mr:>7.2f}{100 * (acc - base[0]):>+8.2f}pp")
    best = rows[0]
    out(f"\n最优：{best[0]} λ={best[1]:.2f} "
        f"ε={best[2]} cap={best[3]} → {100 * best[4][0]:.2f}% "
        f"（基线 {100 * base[0]:.2f}%，Δ {100 * (best[4][0] - base[0]):+.2f}pp）")
    out("\n注：冷账本（无词频项）；池 %d 只覆盖引擎序前 %d 条，池外单字无 CE。"
        % (args.pool, args.pool))
    out("    真值首字必须在该音节候选组内；读音表缺失的候选按 share=1e-4。")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
