#!/usr/bin/env python3
"""luna 混合窗 cohort 上的排序公式评测（2026-11-24）。

输入：eval/luna_mixed/cohort.jsonl（build_luna_mixed.py assemble 产出）。
口径：
  · 冷账本（层1 分全 0）→ 评分窗 = 引擎序前 max_candidates=6 条非组句候选，
    与生产统一准入一致；β/γ 在本 cohort 上不起作用，网格只扫 CE 位置权重
    (w2, w3) × 长度项（F0 现行 / F1 加权均值 / F2 n^α / F3 λ·n / F4 组合）。
  · 候选正确 ⇔ 文本是该窗真值的前缀；窗内理想 = 窗内最长的正确候选。
    wrong   = 首选不是正确前缀
    nonideal= 首选不是窗内理想（含 wrong；正确但偏短也算）
    short   = 首选正确但比理想短（多付一次确认）
  · 分半诚实：按句子奇偶分 A/B 半，A 选点→B 评测、B→A 取均值。
另报与公式无关的结构量：收集上限（理想候选落在引擎序前 6/16/32 的比例）、
准入余量（同融合分下「引擎序前 6」vs「池内融合序前 6」）、组句置顶后的首选。

输出：eval/luna_mixed_eval.txt / .tsv
用法：python eval/calib/luna_mixed_eval.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import luna_mixed_lib as L  # noqa: E402
from length_formula_sweep import long_score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

W2S = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]
W3S = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
ALPHAS = [0.25, 0.5, 0.75, 1.0]
LAMS = [-0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0]
PROD = dict(w2=1.10, w3=0.60, norm="none", alpha=0.0, lam=0.0)
FAMS = ("F0", "F1", "F2", "F3", "F4")


def load(k_pool=16, k_probe=32, cohort=None, cands_path=None):
    """cohort = 16 池口径（默认）；收集池 64（2026-11-26）传
    cohort=L.COHORT_EXT, cands_path=L.CANDS_EXT, k_pool=k_probe=64。"""
    rows = [json.loads(l) for l in open(cohort or L.COHORT, encoding="utf-8")]
    rows = [r for r in rows if not r["dyn"]]
    cands = json.load(open(cands_path or L.CANDS, encoding="utf-8"))
    n = len(rows)

    def z(shape, fill, dt):
        a = np.empty(shape, dtype=dt)
        a.fill(fill)
        return a

    d = dict(n=n, rows=rows, dom=np.array([r["domain"] for r in rows]),
             sent=np.array([int(r["sid"][1:]) for r in rows]),
             clen=np.array([len(r["code"]) for r in rows]))
    d["c1"], d["c2"], d["c3"] = (z((n, k_probe), -9.0, float) for _ in range(3))
    d["nt"] = z((n, k_probe), 0, int)
    d["pad"] = z((n, k_probe), True, bool)
    d["good"] = z((n, k_probe), False, bool)
    d["tlen"] = z((n, k_probe), 0, int)
    d["endp"] = z((n, k_probe), 0, int)
    d["isgen"] = z((n, k_probe), False, bool)
    d["gshow"] = z((n, k_probe), False, bool)   # 仅显示置顶的组句第 1 条
    d["has_ce"] = z((n, k_probe), False, bool)
    d["etype"] = [[None] * k_probe for _ in range(n)]
    for i, r in enumerate(rows):
        gset = {g[0] for g in r["gen"]}
        gshow_text = r["gen"][0][0] if r["gen"] else None
        ce = {c[0]: c[5:7] for c in r["eng"][:k_pool]}
        for j, c in enumerate(cands.get(r["code"], [])[:k_probe]):
            text, s, e, typ, q = c
            d["pad"][i, j] = False
            d["good"][i, j] = r["text"].startswith(text)
            d["tlen"][i, j] = len(text)
            d["endp"][i, j] = e
            d["isgen"][i, j] = text in gset
            d["gshow"][i, j] = (text == gshow_text)
            d["etype"][i][j] = typ
            ce3, ntok = ce.get(text, (None, None))
            if ce3:
                d["has_ce"][i, j] = True
                for k, key in enumerate(("c1", "c2", "c3")):
                    d[key][i, j] = (ce3[k] if k < len(ce3) and ce3[k] is not None
                                    else -9.0)
                d["nt"][i, j] = ntok or 0
    return d


def metrics(score, mask, d):
    """score/mask [n,W] → 首选及错误指示（masked 处不可选）。"""
    s = np.where(mask & d["has_ce"], score, -1e18)
    top = s.argmax(1)
    ar = np.arange(d["n"])
    g = d["good"][ar, top]
    tl = d["tlen"][ar, top]
    ideal_len = np.where(mask & d["good"], d["tlen"], -1).max(1)
    return dict(wrong=~g, nonideal=(~g) | (tl < ideal_len),
                short=g & (tl < ideal_len), has_good=ideal_len >= 0, top=top,
                ideal_len=ideal_len)


def rates(e):
    return (f"wrong {100 * e['wrong'].mean():6.2f}%  "
            f"nonideal {100 * e['nonideal'].mean():6.2f}%  "
            f"short {100 * e['short'].mean():6.2f}%")


def classes(d, mask):
    """窗内理想候选的 token 类（0/1/2）与覆盖类（0=满覆盖 FM，1=子词）。"""
    ar = np.arange(d["n"])
    ideal = np.where(mask & d["good"], d["tlen"], -1).argmax(1)
    nt = d["nt"][ar, ideal]
    tok = np.where(nt >= 3, 2, np.where(nt >= 2, 1, 0))
    has = np.where(mask & d["good"], d["tlen"], -1).max(1) >= 0
    cov = np.where(d["endp"][ar, ideal] == d["clen"], 0, 1)
    return has, tok, cov


def by_class(e, has, tok, cov):
    parts = []
    for c, nm in ((0, "1tok"), (1, "2tok"), (2, "3+tok")):
        m = has & (tok == c)
        parts.append(f"{nm} {100 * e['nonideal'][m].mean():.2f}%({int(m.sum())})"
                     if m.any() else f"{nm} n/a")
    for c, nm in ((0, "满覆盖"), (1, "子词")):
        m = has & (cov == c)
        parts.append(f"理想{nm} {100 * e['nonideal'][m].mean():.2f}%({int(m.sum())})"
                     if m.any() else f"理想{nm} n/a")
    return "  ".join(parts)


def variants():
    out = [("F0", "none", 0.0, 0.0), ("F1", "wsum", 0.0, 0.0)]
    out += [("F2", "pow", al, 0.0) for al in ALPHAS]
    out += [("F3", "none", 0.0, lm) for lm in LAMS]
    out += [("F4", "pow", al, lm) for al in ALPHAS for lm in LAMS]
    return out


def score_of(d, w2, w3, norm, alpha, lam):
    return long_score(d["c1"], d["c2"], d["c3"], d["nt"], d["pad"], w2, w3,
                      norm, alpha, lam)


def prod_score(d):
    p = PROD
    return score_of(d, p["w2"], p["w3"], p["norm"], p["alpha"], p["lam"])


def fmt_pt(r):
    return (f"w2={r['w2']:.2f} w3={r['w3']:.2f} {r['norm']} "
            f"α={r['alpha']:.2f} λ={r['lam']:+.2f}")


def main():
    d = load()
    n = d["n"]
    ar = np.arange(n)
    half = d["sent"] % 2 == 0
    non_gen = ~d["isgen"]
    rank = np.cumsum(non_gen, axis=1) - 1
    win = non_gen & (rank < L.K_WINDOW)        # 生产统一准入（冷账本）
    pool = non_gen & (rank >= 0) & (rank < L.K_COLLECT) & ~d["pad"]  # 有 CE
    pool32 = non_gen & (rank >= 0) & ~d["pad"]                       # 探针 32 条
    out = ["=" * 110,
           f"luna 混合窗 · 排序公式评测 · 静态窗 {n}"
           f"（wiki {int((d['dom'] == 'wiki').sum())} / "
           f"user {int((d['dom'] == 'user').sum())}）· 冷账本 · 引擎序前 "
           f"{L.K_WINDOW} 进窗 · 前 {L.K_COLLECT} 池内缺 CE "
           f"{int((~d['has_ce'] & pool).sum())}",
           "=" * 110]

    # ── A. 结构 ──
    zero = np.zeros_like(d["c1"])
    e_eng = metrics(zero, win, d)               # 引擎序首选
    has, tok, cov = classes(d, win)
    ideal32 = np.where(pool32 & d["good"], d["tlen"], -1)
    pos32 = ideal32.argmax(1)
    has32 = ideal32.max(1) >= 0
    longest32 = ideal32.max(1)
    out += ["A. 结构（与公式无关）",
            f"  窗内有正确候选 {100 * e_eng['has_good'].mean():.2f}%；"
            f"前 16 池内有 {100 * (np.where(pool & (rank < 16) & d['good'], d['tlen'], -1).max(1) >= 0).mean():.2f}%；"
            f"前 32 池内有 {100 * has32.mean():.2f}%",
            f"  最长正确前缀落在引擎序前 6 / 前 16："
            f"{100 * (has32 & (pos32 < L.K_WINDOW)).mean():.2f}% / "
            f"{100 * (has32 & (pos32 < L.K_COLLECT)).mean():.2f}%"
            f"（32 条内的最长者）",
            f"  引擎序首选：{rates(e_eng)}",
            f"  窗内理想候选的类型分布：token 类 "
            f"{dict(Counter(tok[has].tolist()))}（0=1tok,1=2tok,2=3+tok），"
            f"满覆盖 {int((cov[has] == 0).sum())} / 子词 {int((cov[has] == 1).sum())}",
            f"  按域（引擎序 nonideal）：" + "  ".join(
                f"{dm} {100 * e_eng['nonideal'][d['dom'] == dm].mean():.2f}%"
                for dm in ("wiki", "user"))]
    cnt = Counter((int(pos32[i]) + 1) if has32[i] else 0 for i in range(n))
    out.append("  最长正确前缀的引擎位（1 基，0=32 条内无）："
               + " ".join(f"{k}:{cnt[k]}" for k in sorted(cnt) if k <= 17)
               + f"  >17:{sum(v for k, v in cnt.items() if k > 17)}")
    # 收集上限之外（第 17..32 条）才有正确前缀的窗：现有 16 条全是错的
    only_beyond = (~e_eng["has_good"]) & has32 & (pos32 >= L.K_COLLECT)
    out.append(f"  前 16 条无正确前缀、但第 17–32 条有：{int(only_beyond.sum())} 窗 "
               f"（{100 * only_beyond.mean():.2f}%）——收集上限 16 之外才出现的正确子词")
    within = (~e_eng["has_good"]) & has32 & (pos32 < L.K_COLLECT) & (pos32 >= L.K_WINDOW)
    out.append(f"  窗内（前 6）无正确前缀、但第 7–16 条有：{int(within.sum())} 窗 "
               f"（{100 * within.mean():.2f}%）——在尾窗、不参与 CE")

    # ── B. 生产点 ──
    S = prod_score(d)
    e = metrics(S, win, d)
    e_pool = metrics(S, pool, d)
    top_pool = np.where(pool & d["has_ce"], S, -1e18).argsort(
        axis=1, kind="stable")[:, :L.K_WINDOW]
    win_fused = np.zeros_like(win)
    np.put_along_axis(win_fused, top_pool, True, axis=1)
    e_fused = metrics(S, win_fused, d)
    out += ["", "B. 生产点 (w2=1.10, w3=0.60)",
            f"  窗内融合序首选：{rates(e)}",
            f"  按理想类：{by_class(e, has, tok, cov)}",
            f"  对照·引擎序首选：{rates(e_eng)}",
            f"  准入余量·池 16 按融合序取前 6 为窗：{rates(e_fused)}",
            f"  准入余量·池 16 全排（不设窗）：{rates(e_pool)}"]

    # ── C. 公式族 ──
    out += ["", "C. 公式族（w2×w3×长度项 网格；honest = 分半：A 选点→B、B→A 均值）"]
    best = {}
    for w2 in W2S:
        for w3 in W3S:
            for fam, norm, al, lm in variants():
                ex = metrics(score_of(d, w2, w3, norm, al, lm), win, d)
                rec = dict(err=float(ex["nonideal"].mean()), w2=w2, w3=w3,
                           norm=norm, alpha=al, lam=lm,
                           errA=float(ex["nonideal"][half].mean()),
                           errB=float(ex["nonideal"][~half].mean()))
                for key, val in (((fam, "all"), rec["err"]),
                                 ((fam, "A"), rec["errA"]),
                                 ((fam, "B"), rec["errB"])):
                    if key not in best or val < best[key][0]:
                        best[key] = (val, rec)
    tsv = ["family\terr\thonest\terrA\terrB\tw2\tw3\tnorm\talpha\tlam\t"
           "ntok_err"]
    for fam in FAMS:
        ba, bb = best[(fam, "A")][1], best[(fam, "B")][1]
        honest = 0.5 * (ba["errB"] + bb["errA"])
        val, b = best[(fam, "all")]
        ex = metrics(score_of(d, b["w2"], b["w3"], b["norm"], b["alpha"],
                              b["lam"]), win, d)
        out.append(f"  {fam} 最优 nonideal {100 * val:6.2f}%  wrong "
                   f"{100 * ex['wrong'].mean():6.2f}%  honest "
                   f"{100 * honest:6.2f}%  Δ{100 * (val - e['nonideal'].mean()):+.2f}pp"
                   f"  [{fmt_pt(b)}]")
        out.append(f"       {by_class(ex, has, tok, cov)}")
        tsv.append(f"{fam}\t{100 * val:.3f}\t{100 * honest:.3f}\t"
                   f"{100 * b['errA']:.3f}\t{100 * b['errB']:.3f}\t{b['w2']:.2f}\t"
                   f"{b['w3']:.2f}\t{b['norm']}\t{b['alpha']:.2f}\t"
                   f"{b['lam']:.2f}\t{by_class(ex, has, tok, cov)}")
    # 生产点自身的分半（供 honest 对照）
    out.append(f"  生产点 nonideal {100 * e['nonideal'].mean():6.2f}%  wrong "
               f"{100 * e['wrong'].mean():6.2f}%  "
               f"A半 {100 * e['nonideal'][half].mean():.2f}%  "
               f"B半 {100 * e['nonideal'][~half].mean():.2f}%")

    # ── D. 组句置顶后的首选 ──
    rows = d["rows"]
    disp_good = np.zeros(n, dtype=bool)
    disp_full = np.zeros(n, dtype=bool)
    has_gen = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        gs = [g for g in r["gen"] if g[5] is not None and g[5] > -1e9]
        if not gs:
            continue
        has_gen[i] = True
        bg = max(range(len(gs)), key=lambda k: gs[k][5])
        disp_good[i] = r["text"].startswith(gs[bg][0])
        disp_full[i] = gs[bg][0] == r["text"]
    win_good = d["good"][ar, e["top"]]
    out += ["", "D. 组句置顶（全句 CE 选 1 条置顶，生产逻辑；仅影响首选）",
            f"  有组句的窗 {int(has_gen.sum())} / {n}",
            f"  窗口融合序首选正确：{100 * win_good.mean():.2f}%  "
            f"（满句命中 {100 * (d['tlen'][ar, e['top']] == np.array([len(r['text']) for r in rows])).mean():.2f}%）",
            f"  置顶后首选正确：{100 * disp_good.mean():.2f}%  "
            f"（满句命中 {100 * disp_full.mean():.2f}%）",
            f"  置顶把对的改错：{int((win_good & has_gen & ~disp_good).sum())}；"
            f"把错的改对：{int((~win_good & has_gen & disp_good).sum())}"]

    txt = "\n".join(out)
    (L.EVAL / "luna_mixed_eval.txt").write_text(txt + "\n", encoding="utf-8")
    (L.EVAL / "luna_mixed_eval.tsv").write_text("\n".join(tsv) + "\n",
                                               encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
