#!/usr/bin/env python3
"""（中位延迟, 选重率）帕累托前沿。

输入（可多个文件，按 model 合并）:
  eval/fusion_sweep_<tag>.tsv   model ctx k err honest ... err_prod
  eval/latency_grid_<tag>.tsv   model ctx k device p50 p90 ...

缺少实测延迟的格：按该模型该设备的 T = a + b·ctx + c·k + d·ctx·k 拟合后预测，
输出标 `*`（拟合只用该模型已有实测点）。

用法:
  python tools/pareto_front.py --acc eval/fusion_sweep_05bi.tsv \
      --lat eval/latency_grid_05bi.tsv [--lat eval/latency_grid_08b.tsv ...] \
      --device cpu --thresholds 50,60,70,80,90,100
报告: eval/pareto_report.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

sys.stdout.reconfigure(encoding="utf-8")


def read_tsv(path, cols):
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        idx = {c: head.index(c) for c in cols if c in head}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(head):
                continue
            r = {"model": p[head.index("model")]}
            for dk in ("device", "dev"):
                if dk in head:
                    r["device"] = p[head.index(dk)]
            for c, i in idx.items():
                r[c] = int(float(p[i])) if c in ("ctx", "k") else float(p[i])
            rows.append(r)
    return rows


def fit_latency(cells, model):
    """该模型 T = a + b·ctx + c·k + d·ctx·k（最小二乘）→ 预测函数。"""
    X, y = [], []
    for c in cells:
        if c["model"] != model:
            continue
        X.append([1.0, c["ctx"], c["k"], c["ctx"] * c["k"]])
        y.append(c["p50"])
    if len(X) < 6:
        return None
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)

    def pred(ctx, k):
        return float(coef @ np.array([1.0, ctx, k, ctx * k]))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", action="append", required=True)
    ap.add_argument("--lat", action="append", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresholds", default="50,60,70,80,90,100")
    ap.add_argument("--metric", default="honest",
                    help="前沿用哪个误差列: err（本格最优）/ honest（split-half 留出）/ prod（生产固定参数）")
    ap.add_argument("--pool-ctx", dest="pool_ctx", action="store_true",
                    help="按 (模型,k) 跨 ctx 取中位（仅适用于 ctx 对延迟平坦的口径；"
                         "整窗全程口径下默认关闭）")
    ap.add_argument("--out", default="pareto_report.txt")
    a = ap.parse_args()
    ths = [float(x) for x in a.thresholds.split(",") if x.strip()]

    acc = []
    for p in a.acc:
        acc.extend(read_tsv(p, ["ctx", "k", "err", "honest", "err_prod",
                                "miss"]))
    lat = []
    for p in a.lat:
        for r in read_tsv(p, ["ctx", "k", "p50", "p90"]):
            if r.get("device") != a.device:
                continue
            lat.append(r)
    lat_exact = {(r["model"], int(r["ctx"]), int(r["k"])): r["p50"]
                 for r in lat}
    # ctx 维度实测平坦（<5%）：按 (模型,k) 跨 ctx 取中位数，消掉噪声导致的虚假抖动
    pooled = {}
    if a.pool_ctx:
        byk = {}
        for (m, _c, k), v in lat_exact.items():
            byk.setdefault((m, k), []).append(v)
        pooled = {kk: float(np.median(v)) for kk, v in byk.items()
                  if len(v) >= 3}
    fits = {}
    for m in {r["model"] for r in acc}:
        fits[m] = fit_latency(lat, m)

    def latency(m, ctx, k):
        if a.pool_ctx and (m, k) in pooled:
            return pooled[(m, k)], False
        if (m, ctx, k) in lat_exact:
            return lat_exact[(m, ctx, k)], False
        if fits.get(m):
            return fits[m](ctx, k), True
        return None, True

    pts = []
    for r in acc:
        key = (r["model"], int(r["ctx"]), int(r["k"]))
        p50, pred = latency(*key)
        if p50 is None:
            continue
        pts.append(dict(model=key[0], ctx=key[1], k=key[2], ms=p50,
                        pred=pred, err=r["err"], honest=r["honest"],
                        prod=r.get("err_prod"), miss=r["miss"]))
    if not pts:
        raise SystemExit("没有可配对的 (延迟, 正确率) 点")
    pts.sort(key=lambda p: (p["ms"], p["err"]))

    def score(p):
        return {"err": p["err"], "honest": p["honest"],
                "prod": p["prod"] if p["prod"] is not None else 9e9}[a.metric]

    pts.sort(key=lambda p: (p["ms"], score(p)))
    # 帕累托：按延迟升序，err 严格下降者入选
    front, best = [], 9e9
    for p in pts:
        if score(p) < best - 1e-12:
            front.append(p)
            best = score(p)

    L = ["=" * 104,
         f"帕累托前沿（{a.device} 中位延迟 × 选重率）· 数据 "
         f"{len({(p['model']) for p in pts})} 模型 / {len(pts)} 点 · "
         f"排序口径 = {a.metric}",
         "每点 = (模型, ctx, k)；err/honest/prod 见各列，`*` = 延迟为拟合预测",
         "=" * 104,
         f"{'模型':<8}{'ctx':>4}{'k':>3}{'p50ms':>9}{'选重率':>8}"
         f"{'诚实':>8}{'生产点':>8}{'ms/0.1pp':>10}"]
    for p in front:
        tag = "*" if p["pred"] else " "
        L.append(f"{p['model']:<8}{p['ctx']:>4}{p['k']:>3}"
                 f"{p['ms']:>8.1f}{tag}{p['err']:>7.2f}%"
                 f"{p['honest']:>7.2f}%{p['prod']:>7.2f}%"
                 f"{p['ms']/max(0.1, 100*(p['prod']-p['err'])):>10.1f}")

    L.append("")
    L.append("阈值（中位延迟 ≤ T）下的最优点：")
    L.append(f"{'T(ms)':>6}{'模型':>8}{'ctx':>4}{'k':>3}{'p50':>8}"
             f"{'选重率':>8}{'诚实':>8}")
    for t in ths:
        ok = [p for p in pts if p["ms"] <= t]
        if not ok:
            L.append(f"{t:>6.0f}  （无可行点）")
            continue
        b = min(ok, key=score)
        L.append(f"{t:>6.0f}{b['model']:>8}{b['ctx']:>4}{b['k']:>3}"
                 f"{b['ms']:>8.1f}{b['err']:>7.2f}%{b['honest']:>7.2f}%")

    L.append("")
    L.append("各模型在阈值内的最好点：")
    for m in sorted({p["model"] for p in pts}):
        for t in ths:
            ok = [p for p in pts if p["model"] == m and p["ms"] <= t]
            if not ok:
                continue
            b = min(ok, key=score)
            L.append(f"  {m:<8} T<={t:>5.0f}  ctx={b['ctx']:>2} k={b['k']} "
                     f"p50={b['ms']:>6.1f}ms  选重率 {b['err']:.2f}%"
                     f"（诚实 {b['honest']:.2f}%）")

    txt = "\n".join(L)
    out = corpus_std.EVAL / a.out
    out.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
