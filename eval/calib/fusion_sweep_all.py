#!/usr/bin/env python3
"""逐 (模型, ctx, k) 的融合联合扫参 —— 全网格，取每格最优选重率。

用户定案（2026-11-18）：最终目标是 (中位延迟, 正确率) 的帕累托前沿，阈值可变
（50–100ms），所以**每个 (模型, ctx, k) 都要有自己的最优融合参数**。

分数结构（与生产同一参数化）：
  层1: s1 = b_dec·log1p(eff_τ) + b_cum·log1p(uni)   （稳定降序取前 k）
  层2: s2 = a_ce·L + s1，L = −(ce1 + w2·ce2 + w3·ce3)；a_ce≡1（只比值有意义）
网格：τ × w2 × w3 × b_dec × b_cum。

优化：τ 重放只算一次（与模型/ctx/k 无关）；同一 (τ,b_dec,b_cum) 的层1 序
只排一次、各 k 取前缀；w2/w3 只影响 L。
诚实性：同时报 split-half 留出误差（A 调参→B 评测、B→A，取均值），
量化"按格调参"的选择乐观偏差。

用法:
  python calib/fusion_sweep_all.py --model 0.5Bi=ce_c --ctxs 1-20
  python calib/fusion_sweep_all.py --model 0.8B=ce_08b_c --ctxs 1-7 --ks 5
输出: eval/fusion_sweep_<tag>.tsv / .txt
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from corpus_std import FormulaReplay
from derive_pairfix_ratio import load_w2codes

sys.stdout.reconfigure(encoding="utf-8")
EVAL = corpus_std.EVAL
PROJ = corpus_std.PROJ
W = 10


def num(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def parse_ctxs(s):
    out = []
    for p in s.split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def load_base(chain, arm, lp1_ce):
    """anchors + 累计项/pad/widx（与模型无关，只装一次）。

    **累计项口径（2026-11-18 修正）**：用 `derive_pairfix_ratio.load_domain` 的
    **词码对 (word, code) 计数**，与生产账本 `user_ledger.tsv` 的键一致。
    旧实现走 `summary_4x4.cohort_matrices` 的 **词级**计数，同一 (ctx,k) 会差 ~0.08–0.09pp
    （20k 窗 @ctx10/k5：词级 4.99% vs 词码对 5.07%）。
    """
    from derive_pairfix_ratio import load_domain, load_w2codes
    _, first_code = load_w2codes()          # noqa: F841（load_domain 内部用 first_code）
    d = PROJ / f"cohort_{chain}"
    anch_all = [json.loads(l) for l in open(d / "anchors.jsonl",
                                            encoding="utf-8")]
    keep = [i for i, a in enumerate(anch_all)
            if (a.get("dom") in (None, "big")) and a.get("arm") == arm]
    smp = []
    sp = d / "samples.jsonl"
    if sp.exists():
        smp = [json.loads(l) for l in open(sp, encoding="utf-8")]
    dm = load_domain(f"cohort_{chain}", "big", None, first_code, ce_sub=lp1_ce)
    n = len(keep)
    wins = [anch_all[i]["win"] for i in keep]
    lens = np.array([len(w) for w in wins])
    widx = np.array([w.index(anch_all[i]["w"]) for w, i in zip(wins, keep)])
    pad = np.arange(W)[None, :] >= lens[:, None]
    codes = [(smp[i].get("code") if smp else None)
             or anch_all[i].get("code") or "" for i in keep]
    del smp
    by_ui = {}
    for r, i in enumerate(keep):
        by_ui.setdefault(anch_all[i]["ui"], {}).setdefault(
            anch_all[i]["ci"], []).append(r)
    lp1 = np.where(pad, -1e9, np.log(dm["uni"] + 1.0))
    del dm
    return dict(chain=chain, anch=anch_all, keep=keep, n=n, wins=wins,
                pad=pad, widx=widx, codes=codes, by_ui=by_ui, lp1=lp1)


def eff_by_tau(data, tau):
    _, first_code = load_w2codes()
    n = data["n"]
    du = np.zeros((n, W))
    need = set(data["by_ui"])
    with open(PROJ / ("cohort_" + data["chain"]) / "streams.jsonl",
              encoding="utf-8") as f:
        for ui, line in enumerate(f):
            wmap = data["by_ui"].get(ui)
            if not wmap:
                continue
            rp = FormulaReplay(tau)
            for ci, w in enumerate(json.loads(line)):
                for r in wmap.get(ci, ()):
                    a = data["anch"][data["keep"][r]]
                    code = data["codes"][r]
                    for j, x in enumerate(a["win"]):
                        e = rp.eff_of((x, code))
                        du[r, j] = math.log1p(e) if e else 0.0
                rp.bump((w, first_code.get(w, "")))
    return np.where(data["pad"], 0.0, du)


def load_ce(chain, sub, data):
    """→ c1,c2,c3 [n,W]（缺分 -9）；nt [n,W] 为 0 表示无 CE。"""
    d = PROJ / f"cohort_{chain}" / sub / "results.jsonl"
    if not d.exists():
        return None
    recs = {}
    for l in open(d, encoding="utf-8"):
        try:
            r = json.loads(l)
            recs[r["sid"]] = (r.get("ce") or {}, r.get("ntok") or {})
        except Exception:
            pass
    n = data["n"]
    c1 = np.full((n, W), -9.0); c2 = np.full((n, W), -9.0)
    c3 = np.full((n, W), -9.0); nt = np.zeros((n, W), dtype=int)
    for r, i in enumerate(data["keep"]):
        a = data["anch"][i]
        ce, ntok = recs.get(a["sid"], ({}, {}))
        for j, w in enumerate(a["win"]):
            if w in ce:
                v = ce[w]
                c1[r, j], c2[r, j], c3[r, j] = v[0], v[1], v[2]
                nt[r, j] = ntok.get(w, 0)
    return c1, c2, c3, nt


def ce_shape(c1, c2, c3, nt, pad, w2, w3):
    bad2 = np.where(c2 > 0, c2, 0.0)
    bad3 = np.where(c3 > 0, c3, 0.0)
    s = -c1
    s = np.where(nt >= 2, -(c1 + w2 * bad2), s)
    s = np.where(nt >= 3, -(c1 + w2 * bad2 + w3 * bad3), s)
    bad = (nt == 0) | ((nt >= 2) & (c2 < 0)) | ((nt >= 3) & (c3 < 0))
    return np.where(bad | pad, -1e9, s)


def eval_cell(L, s1, order, widx, k, mask):
    """→ err（mask 内）。order 已按层1 降序（全宽），取前 k。"""
    o = order[:, :k]
    s1k = np.take_along_axis(s1, o, axis=1)
    s2 = s1k + np.take_along_axis(L, o, axis=1)
    win = o[np.arange(len(o)), s2.argmax(1)]
    e = win != widx
    return e if mask is None else e[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="big_ctx10k")
    ap.add_argument("--arm", type=int, default=30000)
    ap.add_argument("--model", action="append", required=True,
                    help="名称=CE 目录前缀（如 0.5Bi=ce_c）；ctx 拼在尾部")
    ap.add_argument("--ctxs", default="1-20")
    ap.add_argument("--ks", default="2,3,4,5,6,7,8,9")
    ap.add_argument("--tau", default="100,200,300,400,600,800,1200")
    ap.add_argument("--w2", default="1.00,1.05,1.10,1.15,1.20,1.30")
    ap.add_argument("--w3", default="0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--w-dec", default="6,10,14,20,28")
    ap.add_argument("--w-cum", default="0.75,1.0,1.25,1.5,1.75,2.0,2.5")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lp1-ce", default="ce",
                    help="任取一个存在的 CE 目录（只借它装载 lp1/pad/widx）")
    a = ap.parse_args()
    ctxs = parse_ctxs(a.ctxs)
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    taus, w2s, w3s = num(a.tau), num(a.w2), num(a.w3)
    decs, cums = num(a.w_dec), num(a.w_cum)

    t0 = time.time()
    data = load_base(a.chain, a.arm, a.lp1_ce)
    print(f"base {data['n']} 窗", flush=True)
    dus = []
    for t in taus:
        dus.append(eff_by_tau(data, t))
        print(f"  τ={t:g} 重放完成 {time.time()-t0:.0f}s", flush=True)

    pad, widx, lp1, n = data["pad"], data["widx"], data["lp1"], data["n"]
    half = np.arange(n) % 2 == 0          # A=偶数, B=奇数
    best, ba, bb = {}, {}, {}             # 全量最优 / A 选参 / B 选参
    prod = {}                             # 生产点（固定参数）对照
    PROD = dict(tau=400.0, w2=1.10, w3=0.60, b_dec=14.0, b_cum=1.0)
    freq_err = None                       # 纯词频（无 LLM）基线
    ncomb = 0
    for spec in a.model:
        name, prefix = spec.split("=", 1)
        for ctx in ctxs:
            sub = f"{prefix}{ctx}"
            ce = load_ce(a.chain, sub, data)
            if ce is None:
                print(f"{name} ctx={ctx}: 缺 {sub}，跳过", flush=True)
                continue
            c1, c2, c3, nt = ce
            Ls = {}
            for w2 in w2s:
                for w3 in w3s:
                    Ls[(w2, w3)] = ce_shape(c1, c2, c3, nt, pad, w2, w3)
            # 每 (τ,b_dec,b_cum) 排一次层1 序，各 k 取前缀
            for ti, tau in enumerate(taus):
                du = dus[ti]
                for bd in decs:
                    for bc in cums:
                        s1 = np.where(pad, -1e9, bd * du + bc * lp1)
                        order = np.argsort(-s1, axis=1, kind="stable")
                        if (freq_err is None and tau == PROD["tau"]
                                and bd == PROD["b_dec"] and bc == PROD["b_cum"]):
                            freq_err = float((order[:, 0] != widx).mean())
                        pres = order == widx[:, None]
                        for k in ks:
                            key = (name, ctx, k)
                            miss = float(1.0 - pres[:, :k].any(1).mean())
                            o = order[:, :k]
                            s1k = np.take_along_axis(s1, o, axis=1)
                            for (w2, w3), L in Ls.items():
                                s2 = s1k + np.take_along_axis(L, o, axis=1)
                                win = o[np.arange(n), s2.argmax(1)]
                                e = win != widx
                                ea = float(e[half].mean())
                                eb = float(e[~half].mean())
                                row = dict(model=name, ctx=ctx, k=k, tau=tau,
                                           w2=w2, w3=w3, b_dec=bd, b_cum=bc,
                                           err=float(e.mean()), errA=ea,
                                           errB=eb, miss=miss)
                                ncomb += 1
                                if (tau == PROD["tau"] and w2 == PROD["w2"]
                                        and w3 == PROD["w3"]
                                        and bd == PROD["b_dec"]
                                        and bc == PROD["b_cum"]):
                                    prod[key] = row["err"]
                                if key not in best or row["err"] < best[key]["err"]:
                                    best[key] = row
                                if key not in ba or ea < ba[key]["errA"]:
                                    ba[key] = row
                                if key not in bb or eb < bb[key]["errB"]:
                                    bb[key] = row
            print(f"{name} ctx={ctx}: {ncomb} 次评估累计 "
                  f"{time.time()-t0:.0f}s", flush=True)

    # 每 (model, ctx, k)：取全量最优 + split-half 诚实误差
    out = ["=" * 108,
           f"逐 (模型, ctx, k) 联合扫参 · cohort_{a.chain}@{a.arm} · "
           f"{data['n']} 窗 · a_ce≡1",
           f"网格 τ{taus} × w2{w2s} × w3{w3s} × b_dec{decs} × b_cum{cums}"
           f" = {len(taus)*len(w2s)*len(w3s)*len(decs)*len(cums)} 组/格",
           "err = 该格最优（同数据调参+评测，略乐观）；honest = split-half 留出",
           f"纯词频（无 LLM）基线 = "
           f"{100*freq_err:.2f}%" if freq_err is not None else "纯词频基线 = n/a",
           "=" * 108,
           f"{'模型':<7}{'ctx':>4}{'k':>3}{'最优err':>9}{'诚实err':>9}"
           f"{'乐观偏差':>9}{'生产点':>9}{'层1miss':>9}   τ    w2    w3  b_dec b_cum"]
    tsv = ["model\tctx\tk\terr\thonest\terr_half_a\terr_half_b\tmiss"
           "\ttau\tw2\tw3\tb_dec\tb_cum\terr_prod"]
    for key in sorted(best, key=lambda x: (x[0], x[1], x[2])):
        b = best[key]
        ha = bb[key]["errA"]      # A 调参（用 B 选）→ 报 A
        hb = ba[key]["errB"]      # B 调参（用 A 选）→ 报 B
        h = 0.5 * (ha + hb)
        ep = prod.get(key, float("nan"))
        out.append(f"{b['model']:<7}{b['ctx']:>4}{b['k']:>3}"
                   f"{100*b['err']:>8.2f}%{100*h:>8.2f}%"
                   f"{100*(h-b['err']):>8.2f}%{100*ep:>8.2f}%"
                   f"{100*b['miss']:>8.2f}%  "
                   f"{b['tau']:>4.0f} {b['w2']:>5.2f} {b['w3']:>5.2f} "
                   f"{b['b_dec']:>5.0f} {b['b_cum']:>5.2f}")
        tsv.append(f"{b['model']}\t{b['ctx']}\t{b['k']}\t{100*b['err']:.3f}"
                   f"\t{100*h:.3f}\t{100*ha:.3f}\t{100*hb:.3f}"
                   f"\t{100*b['miss']:.3f}\t{b['tau']:.0f}\t{b['w2']:.2f}"
                   f"\t{b['w3']:.2f}\t{b['b_dec']:.0f}\t{b['b_cum']:.2f}"
                   f"\t{100*ep:.3f}")
    txt = "\n".join(out)
    tag = a.tag or "_".join(s.split("=", 1)[0] for s in a.model)
    (EVAL / f"fusion_sweep_{tag}.tsv").write_text("\n".join(tsv) + "\n",
                                                 encoding="utf-8")
    (EVAL / f"fusion_sweep_{tag}.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n报告: eval/fusion_sweep_{tag}.txt  ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
