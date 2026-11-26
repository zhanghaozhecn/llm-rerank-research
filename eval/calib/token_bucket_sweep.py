#!/usr/bin/env python3
"""按「候选总 token 数」分桶的 CE 系数标定（2026-11-25，用户指令）。

现行公式对所有 token 数用同一组位置权重：
    score = -(1.0·ce1 + 1.10·ce2 + 0.60·ce3)
本脚本改为**按总 token 数分桶**，1-token 桶系数固定 1，其余 5 个系数同时
标定（仍为线性加权求和，不做归一化/长度项）：
    n=1:  -(ce1)
    n=2:  -(a21·ce1 + a22·ce2)
    n≥3:  -(a31·ce1 + a32·ce2 + a33·ce3)     # 4+ token 仍只用前三项
未知参数 = (a21,a22,a31,a32,a33)。目标是修正「3-token 词/句总 CE 更负」
带来的跨桶长度偏置。

数据 = luna 混合窗静态 cohort（21,596 窗；phrase/subword 同池），冷账本 →
频率项为 0，CE 系数可单独观察。标签：候选文本 = 真值前缀；窗内理想 =
最长正确前缀。指标 wrong（首选错）/ nonideal（首选非最长正确）。
分半诚实：按句子奇偶 A/B —— A 选点→B 评测、B 选点→A 评测，取均值；
5 维非光滑目标用「随机采样 + 坐标下降」联合搜索（每次评估都是全窗 argmax）。

输出：eval/token_bucket_sweep.txt / .tsv
用法：python eval/calib/token_bucket_sweep.py [--samples 2000] [--seed 1]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_eval as E  # noqa: E402
import luna_mixed_lib as L  # noqa: E402

BASELINE = np.array([1.00, 1.10, 1.00, 1.10, 0.60])  # a21,a22,a31,a32,a33
GRIDS = [
    np.round(np.arange(0.60, 1.41, 0.10), 2),   # a21
    np.round(np.arange(0.60, 1.51, 0.10), 2),   # a22
    np.round(np.arange(0.60, 1.41, 0.10), 2),   # a31
    np.round(np.arange(0.60, 1.51, 0.10), 2),   # a32
    np.round(np.arange(0.20, 1.01, 0.10), 2),   # a33
]
NAMES = ("a21", "a22", "a31", "a32", "a33")
K_WINDOW = 6


def window_mask(d, mode="top6"):
    if mode == "budget":
        return L.budget_mask(d["nt"], d["gshow"], d["pad"]) & d["has_ce"]
    wm = (~d["pad"]) & (~d["gshow"])
    return wm & (np.cumsum(wm, axis=1) <= K_WINDOW)


def bucket_score(d, p):
    c1, c2, c3, nt = d["c1"], d["c2"], d["c3"], d["nt"]
    c2v = np.maximum(c2, 0.0)
    c3v = np.maximum(c3, 0.0)
    s1 = -c1
    s2 = -(p[0] * c1 + p[1] * c2v)
    s3 = -(p[2] * c1 + p[3] * c2v + p[4] * c3v)
    s = np.where(nt <= 1, s1, np.where(nt == 2, s2, s3))
    bad = ((nt == 0) | ((nt >= 2) & (c2 < 0))
           | ((nt >= 3) & (c3 < 0)) | d["pad"])
    return np.where(bad, -1e9, s)


def evaluate(d, mask, p, sel):
    e = E.metrics(bucket_score(d, p), mask, d)
    return (float(e["wrong"][sel].mean()),
            float(e["nonideal"][sel].mean()),
            e)


def key(d, mask, p, sel):
    w, n, _ = evaluate(d, mask, p, sel)
    return (round(w, 8), round(n, 8))


def search(d, mask, sel, samples, seed):
    """随机采样 + 坐标下降（按 (wrong, nonideal) 字典序）。"""
    rng = np.random.default_rng(seed)
    cand = [BASELINE.copy()]
    for _ in range(samples):
        cand.append(np.array([g[rng.integers(0, len(g))] for g in GRIDS]))
    scored = sorted((key(d, mask, p, sel), i)
                    for i, p in enumerate(cand))[:6]
    best_p, best_k = None, None
    for _, i in scored:
        p = cand[i].copy()
        k = key(d, mask, p, sel)
        for _pass in range(4):
            changed = False
            for j, g in enumerate(GRIDS):
                bv, bv_v = p[j], k
                for v in g:
                    if v == p[j]:
                        continue
                    p2 = p.copy()
                    p2[j] = v
                    k2 = key(d, mask, p2, sel)
                    if k2 < bv_v:
                        bv, bv_v = v, k2
                if bv != p[j]:
                    p[j] = bv
                    k = bv_v
                    changed = True
            if not changed:
                break
        if best_k is None or k < best_k:
            best_p, best_k = p.copy(), k
    return best_p, best_k


def fmt(p):
    return "  ".join(f"{nm}={v:.2f}" for nm, v in zip(NAMES, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", default="top6", choices=["top6", "budget"])
    a = ap.parse_args()
    t0 = time.time()
    d = E.load()
    mask = window_mask(d, a.window)
    sent = d["sent"]
    half = sent % 2 == 0
    halves = (("A(偶)→B(奇)", half, ~half), ("B(奇)→A(偶)", ~half, half))
    print("=" * 96)
    print(f"按 token 数分桶的 CE 系数标定 · luna 混合窗静态 {d['n']} 窗 · "
          f"窗 {int(mask.sum() / max(1, (mask.any(1)).sum())):.0f} 席/窗 · "
          f"{a.window} 口径")
    print(f"基线（现行位置权重）{fmt(BASELINE)}")
    print("=" * 96)
    out = []
    results = []
    for name, tr, te in halves:
        p, k_tr = search(d, mask, tr, a.samples, a.seed)
        w_tr, n_tr, _ = evaluate(d, mask, p, tr)
        w_te, n_te, e_te = evaluate(d, mask, p, te)
        bw_te, bn_te, be_te = evaluate(d, mask, BASELINE, te)
        print(f"\n[{name}]")
        print(f"  选点 {fmt(p)}")
        print(f"  训练: wrong {100*w_tr:.2f}%  nonideal {100*n_tr:.2f}%")
        print(f"  测试: wrong {100*w_te:.2f}%  nonideal {100*n_te:.2f}%")
        print(f"  基线测试: wrong {100*bw_te:.2f}%  nonideal {100*bn_te:.2f}%")
        print(f"  Δ测试: wrong {100*(w_te-bw_te):+.2f}pp  "
              f"nonideal {100*(n_te-bn_te):+.2f}pp")
        # 按域 / 理想 token 类 / 覆盖类
        dom = d["dom"]
        ar = np.arange(d["n"])
        ideal = np.where(mask & d["good"], d["tlen"], -1).argmax(1)
        has = np.where(mask & d["good"], d["tlen"], -1).max(1) >= 0
        nt = d["nt"][ar, ideal]
        cls = np.where(nt >= 3, 2, np.where(nt >= 2, 1, 0))
        cov = np.where(d["endp"][ar, ideal] == d["clen"], 0, 1)
        for tag, m in (("wiki", (dom == "wiki") & te),
                       ("user", (dom == "user") & te),
                       ("理想1tok", has & (cls == 0) & te),
                       ("理想2tok", has & (cls == 1) & te),
                       ("理想3+tok", has & (cls == 2) & te),
                       ("理想满覆盖", has & (cov == 0) & te),
                       ("理想子词", has & (cov == 1) & te)):
            if not m.any():
                continue
            wn, nn = float(e_te["wrong"][m].mean()), float(e_te["nonideal"][m].mean())
            wb, nb = float(be_te["wrong"][m].mean()), float(be_te["nonideal"][m].mean())
            print(f"    {tag:<10}{int(m.sum()):>6}  wrong {100*wb:>5.2f}→"
                  f"{100*wn:>5.2f} ({100*(wn-wb):+.2f})  nonideal "
                  f"{100*nb:>5.2f}→{100*nn:>5.2f} ({100*(nn-nb):+.2f})")
        results.append((name, p, w_te, n_te, bw_te, bn_te))
        out.append(dict(name=name, p=[float(x) for x in p],
                        train_wrong=w_tr, train_nonideal=n_tr,
                        test_wrong=w_te, test_nonideal=n_te,
                        base_test_wrong=bw_te, base_test_nonideal=bn_te))
    # 平均（两次分半的测试侧）
    print("\n两次分半平均（测试侧）")
    aw = np.mean([r[2] for r in results])
    an = np.mean([r[3] for r in results])
    bw = np.mean([r[4] for r in results])
    bn = np.mean([r[5] for r in results])
    print(f"  bucket: wrong {100*aw:.2f}%  nonideal {100*an:.2f}%")
    print(f"  base  : wrong {100*bw:.2f}%  nonideal {100*bn:.2f}%")
    print(f"  Δ     : wrong {100*(aw-bw):+.2f}pp  nonideal {100*(an-bn):+.2f}pp")
    # 参数平均 + 敏感性（测试侧）
    pavg = np.mean([r[1] for r in results], axis=0)
    print(f"\n参数平均: {fmt(pavg)}")
    print("\n敏感性（测试侧 A/B 合并；只动一个参数 ±0.1）")
    for j, nm in enumerate(NAMES):
        parts = []
        for v in (max(GRIDS[j][0], pavg[j] - 0.1), pavg[j],
                  min(GRIDS[j][-1], pavg[j] + 0.1)):
            p = pavg.copy()
            p[j] = v
            w, n, _ = evaluate(d, mask, p, ~half | half)
            parts.append(f"{v:.2f}: {100*w:.2f}/{100*n:.2f}")
        print(f"  {nm}: " + "  ".join(parts))
    suffix = "" if a.window == "top6" else "_" + a.window
    with open(L.EVAL / ("token_bucket_sweep" + suffix + ".tsv"), "w",
              encoding="utf-8") as f:
        f.write("half\tp\ttrain_wrong\ttrain_nonideal\ttest_wrong\ttest_nonideal\t"
                "base_test_wrong\tbase_test_nonideal\n")
        for r in out:
            f.write(f"{r['name']}\t{fmt(np.array(r['p']))}\t"
                    f"{r['train_wrong']:.6f}\t{r['train_nonideal']:.6f}\t"
                    f"{r['test_wrong']:.6f}\t{r['test_nonideal']:.6f}\t"
                    f"{r['base_test_wrong']:.6f}\t{r['base_test_nonideal']:.6f}\n")
    print(f"\n耗时 {time.time()-t0:.0f}s；TSV → "
          f"eval/token_bucket_sweep{suffix}.tsv")


if __name__ == "__main__":
    main()
