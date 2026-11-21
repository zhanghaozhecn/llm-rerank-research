#!/usr/bin/env python3
"""kCeW×β 常数标定统一入口（2026-11-18 子目录重组合并，2 脚本 → 1）。

吸收并取代（报告文件名与数字口径不变）:
  calibrate_model_05b.py        → --protocol split   网格 argmax + 偶半选点
                                  → 奇半验证 vs 部署点（0.5B 时代标定）
  calibrate_model_05bi_final.py → --protocol 4half   四半区（用户/wiki × 偶/奇）
                                  网格 + 平台交集 + 共识判读（05bi 终审）

共享引擎（模块级, 供 calib/analyze_sel_by_code 等 import）:
  load_raw(dirname, bump_mode, wc_sc, first_code, ce_sub=CE_SUB)  原始矩阵装载
  ce_matrix(ce, ntok, w2, w3)                                     CE 加权矩阵
口径 = 生产两层公式不变: 层1 = β·log1p(eff)+log(uni+1) 选前 K;
层2 = 层1 分 + CE 加权项 −(1·ce1 + w2·ce2 + w3·ce3)（ntok 分类: 1 tok=(1,0,0) /
2 tok=(1,w2,0) / ≥3=(1,w2,w3), 缺位 bad 置 −1e9），无有效 CE 回退层1 分。
词频侧 τ=400/δ 与模型无关不动。数据 = cohort_big_user（34k）+ wiki 池化
（sc + big）。网格: w2 1.00~1.45@0.05 ∪{1.05}, w3 0.40~0.95@0.05 ∪{0.45},
β ∈ {4,6,8,10,12,14,16,20}。

用法:
  python lib/calibrate_model.py                      # split, ce_05b（env CE_SUB 可换）
  python lib/calibrate_model.py --protocol 4half --ce-sub ce_05bi
报告: eval/calibrate_model_05b_report.txt / calibrate_model_05bi_final_report.txt
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_std
from corpus_std import FormulaReplay, MAX_CAND

sys.stdout.reconfigure(encoding="utf-8")
K = 5
CE_SUB = os.environ.get("CE_SUB", "ce_05b")
W2G = np.array(sorted(set(np.round(np.arange(1.00, 1.4501, 0.05), 2)) |
                      {1.05}))
W3G = np.array(sorted(set(np.round(np.arange(0.40, 0.9501, 0.05), 2)) |
                      {0.45}))
BETAS = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]
DEPLOY_05B = (1.05, 0.45, 12.0)       # split 协议参照点（0.5B 时代部署点）
CUR_05BI = (1.10, 0.60, 14.0)         # 4half 协议参照点（05b 共识中点 A）


def load_w2codes_and_first():
    from derive_pairfix_ratio import load_w2codes
    return load_w2codes()


def load_raw(dirname, bump_mode, wc_sc, first_code, ce_sub=CE_SUB):
    """load_domain 同口径, 但保留原始 ce(n,MC,3)/ntok 供重加权。"""
    d = corpus_std.PROJ / dirname
    anchors = [json.loads(l) for l in open(d / "anchors.jsonl",
                                           encoding="utf-8")]
    samples = [json.loads(l) for l in open(d / "samples.jsonl",
                                           encoding="utf-8")]
    n = len(anchors)
    codes = [samples[i]["code"] for i in range(n)]
    wins = [anchors[i]["win"] for i in range(n)]
    widx = np.array([win.index(anchors[i]["w"]) for i, win in
                     enumerate(wins)])
    lens = np.array([len(w) for w in wins])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    del samples
    ce_recs = {}
    with open(d / ce_sub / "results.jsonl", encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
                ce_recs[r["sid"]] = r
            except Exception:
                pass
    ce = np.full((n, MAX_CAND, 3), -9.0)
    ntok = np.zeros((n, MAX_CAND), dtype=np.int32)
    by_ac = defaultdict(list)
    for i, an in enumerate(anchors):
        rec = ce_recs.get(an["sid"])
        for j, x in enumerate(an["win"]):
            if rec and x in rec.get("ce") and x in rec.get("ntok"):
                ce[i, j] = rec["ce"][x]
                ntok[i, j] = rec["ntok"][x]
        by_ac[(an["ui"], an["ci"])].append(i)
    du = np.zeros((n, MAX_CAND))
    uni_pc = np.zeros((n, MAX_CAND))
    with open(d / "streams.jsonl", encoding="utf-8") as f:
        for ui, line in enumerate(f):
            rp = FormulaReplay(400.0)
            uni_p = {}
            for ci, w in enumerate(json.loads(line)):
                for r in by_ac.get((ui, ci), ()):
                    c = codes[r]
                    for j, x in enumerate(wins[r]):
                        du[r, j] = math.log1p(rp.eff_of((x, c)))
                        uni_pc[r, j] = uni_p.get((x, c), 0)
                kc = (w, wc_sc.get(w, "") if bump_mode == "sc"
                      else first_code.get(w, ""))
                rp.bump(kc)
                uni_p[kc] = uni_p.get(kc, 0) + 1
    print(f"{dirname}: {n} windows raw-loaded", flush=True)
    return dict(widx=widx, pad=pad, du=du, uni=uni_pc, ce=ce, ntok=ntok)


def ce_matrix(ce, ntok, w2, w3):
    cls = np.where(ntok <= 1, 0, np.where(ntok == 2, 1, 2))
    ok = ce >= -0.5
    krows = np.stack([np.array([1.0, 0.0, 0.0]),
                      np.array([1.0, w2, 0.0]),
                      np.array([1.0, w2, w3])])[cls]
    ce_f = np.where(ok, ce, 0.0)
    bad = (((krows > 0) & ~ok).any(-1)) | (ntok == 0)
    return np.where(bad, -1e9, -(krows * ce_f).sum(-1))


def load_domains(ce_sub):
    """→ {用户域, wiki池化} 原始矩阵域（两协议共用）。"""
    w2codes, first_code = load_w2codes_and_first()
    _, wc_sc = corpus_std.load_pdsp_dict_single3()
    doms = {}
    doms["用户域"] = load_raw("cohort_big_user", "big", wc_sc, first_code,
                             ce_sub=ce_sub)
    sc = load_raw("cohort_big_sc", "sc", wc_sc, first_code, ce_sub=ce_sub)
    big = load_raw("cohort_big", "big", wc_sc, first_code, ce_sub=ce_sub)
    doms["wiki池化"] = dict(
        widx=np.concatenate([sc["widx"], big["widx"]]),
        pad=np.vstack([sc["pad"], big["pad"]]),
        du=np.vstack([sc["du"], big["du"]]),
        uni=np.vstack([sc["uni"], big["uni"]]),
        ce=np.concatenate([sc["ce"], big["ce"]]),
        ntok=np.concatenate([sc["ntok"], big["ntok"]]))
    del sc, big
    return doms


# ─────────────────── split 协议（原 calibrate_model_05b.main）───────────────

def sweep(dom):
    n = len(dom["widx"])
    pad, du, uni = dom["pad"], dom["du"], dom["uni"]
    widx = dom["widx"]
    logu = np.log(uni + 1.0)
    best = []   # (acc, w2, w3, beta)
    dep_acc = None
    even = (np.arange(n) % 2 == 0)
    for b in BETAS:
        s1 = np.where(pad, -1e9, b * du + logu)
        cols = np.argsort(-s1, axis=1, kind="stable")[:, :K]
        s1w = np.take_along_axis(s1, cols, 1)
        for w2 in W2G:
            for w3 in W3G:
                sce = ce_matrix(dom["ce"], dom["ntok"], w2, w3)
                cew = np.take_along_axis(sce, cols, 1)
                s2 = s1w + cew
                f = cols[np.arange(n), np.argmax(s2, axis=1)]
                acc = (f == widx).mean()
                best.append((acc, float(w2), float(w3), b))
                if (w2, w3, b) == DEPLOY_05B:
                    dep_acc = acc
    best.sort(reverse=True)
    top = best[0][0]
    return best, top, dep_acc, n, widx, even


def run_split(out):
    doms = load_domains(CE_SUB)
    for name, dom in doms.items():
        best, top, dep_acc, n, widx, even = sweep(dom)
        out.append(f"\n== {name}（{n} 窗; 部署点 (1.05,0.45,12) "
                   f"{dep_acc * 100:.4f}%）==")
        out.append(f"  网格 argmax: {top * 100:.4f}%  top6:")
        for acc, w2, w3, b in best[:6]:
            out.append(f"    w2={w2:.2f} w3={w3:.2f} β={b:g}  "
                       f"{acc * 100:.4f}%  (Δ{(acc - top) * 100:+.3f}pp)")
        # 平台: 距 argmax ≤0.05pp 的点
        plat = [(a, w2, w3, b) for a, w2, w3, b in best
                if (top - a) <= 0.0005]
        out.append(f"  平台（≤0.05pp）{len(plat)} 点; 部署点在平台内: "
                   f"{dep_acc >= top - 0.0005}")
        # 诚实分半: 偶半选点 → 奇半验证
        half_best = []
        n_e = int(even.sum())
        for b in BETAS:
            s1 = np.where(dom["pad"], -1e9, b * dom["du"]
                          + np.log(dom["uni"] + 1.0))
            cols = np.argsort(-s1[even], axis=1, kind="stable")[:, :K]
            s1w = np.take_along_axis(s1[even], cols, 1)
            we = widx[even]
            for w2 in W2G:
                for w3 in W3G:
                    sce = ce_matrix(dom["ce"][even], dom["ntok"][even],
                                    w2, w3)
                    cew = np.take_along_axis(sce, cols, 1)
                    f = cols[np.arange(n_e), np.argmax(s1w + cew, axis=1)]
                    half_best.append(((f == we).mean(), float(w2),
                                      float(w3), b))
        half_best.sort(reverse=True)
        ch = half_best[0]          # 偶半 argmax 点
        out.append(f"  偶半选点: w2={ch[1]:.2f} w3={ch[2]:.2f} "
                   f"β={ch[3]:g}（偶半 {ch[0] * 100:.4f}%）")
        # 奇半验证两个点
        odd = ~even
        n_o = int(odd.sum())
        for lbl, pt in [("偶半选点", ch[1:]), ("部署点", DEPLOY_05B)]:
            w2, w3, b = pt
            s1 = np.where(dom["pad"], -1e9, b * dom["du"]
                          + np.log(dom["uni"] + 1.0))
            cols = np.argsort(-s1[odd], axis=1, kind="stable")[:, :K]
            s1w = np.take_along_axis(s1[odd], cols, 1)
            sce = ce_matrix(dom["ce"][odd], dom["ntok"][odd], w2, w3)
            cew = np.take_along_axis(sce, cols, 1)
            f = cols[np.arange(n_o), np.argmax(s1w + cew, axis=1)]
            acc = (f == widx[odd]).mean()
            out.append(f"  奇半验证[{lbl}] (w2={w2:.2f} w3={w3:.2f} "
                       f"β={b:g}): {acc * 100:.4f}%")
    out.append("\n判读: 部署点若在双域平台内 → 常数不变; 否则按两域平台"
               "交集/就近取新点（生产常数 = (kCeW 1.0/1.05/0.45, β=12)）")


# ─────────────────── 4half 协议（原 calibrate_model_05bi_final.main）─────────

def half_grid(dom, msk):
    """半区全网格精度 → dict[(w2,w3,b)] = acc"""
    n_h = int(msk.sum())
    widx = dom["widx"][msk]
    logu = np.log(dom["uni"] + 1.0)
    acc = {}
    s1_by_beta = {}
    for b in BETAS:
        s1 = np.where(dom["pad"], -1e9, b * dom["du"] + logu)
        s1_by_beta[b] = (np.argsort(-s1[msk], axis=1, kind="stable")[:, :K],
                         s1[msk])
    for b in BETAS:
        cols, s1h = s1_by_beta[b]
        s1w = np.take_along_axis(s1h, cols, 1)
        for w2 in W2G:
            for w3 in W3G:
                sce = ce_matrix(dom["ce"][msk], dom["ntok"][msk],
                                float(w2), float(w3))
                cew = np.take_along_axis(sce, cols, 1)
                f = cols[np.arange(n_h), np.argmax(s1w + cew, axis=1)]
                acc[(float(w2), float(w3), b)] = (f == widx).mean()
    return acc


def run_4half(out, ce_sub, cur):
    doms = load_domains(ce_sub)
    grids = {}      # (域, 半) -> acc dict
    for name, dom in doms.items():
        n = len(dom["widx"])
        even = (np.arange(n) % 2 == 0)
        print(f"{name}: 偶半网格...", flush=True)
        grids[(name, "偶")] = half_grid(dom, even)
        print(f"{name}: 奇半网格...", flush=True)
        grids[(name, "奇")] = half_grid(dom, ~even)

    names = ["用户域", "wiki池化"]
    halves = [(nm, h) for nm in names for h in ("偶", "奇")]
    cur_acc = {k: grids[k][cur] for k in grids}

    out.append(f"\n现行部署点 (w2={cur[0]}, w3={cur[1]}, β={cur[2]:g}) 四半区精度:")
    for k in halves:
        d = grids[k]
        top = max(d.values())
        out.append(f"  {k[0]:>5}·{k[1]}半: 现行 {cur_acc[k]*100:.4f}%  "
                   f"半区 argmax {top*100:.4f}%  "
                   f"现行在半区平台内: {cur_acc[k] >= top - 0.0005}")

    # 四半区平台交集（每半区距其 argmax ≤0.05pp）
    plats = []
    for k in halves:
        d = grids[k]
        top = max(d.values())
        plats.append({p for p, a in d.items() if a >= top - 0.0005})
    common = set.intersection(*plats)
    out.append(f"\n四半区平台交集: {len(common)} 点")

    # 一致优于现行点（四半区全部 ≥ 现行）
    better = {p for p in common
              if all(grids[k][p] >= cur_acc[k] for k in halves)}
    out.append(f"四半区全部 ≥ 现行点的交集成员: {len(better)} 个")
    ranked = sorted(better,
                    key=lambda p: -min(grids[k][p] - cur_acc[k]
                                       for k in halves))
    out.append("\n候选点（按 四半区最小增量 降序, top10）:")
    for p in ranked[:10]:
        gains = [f"{grids[k][p]*100:.4f}" for k in halves]
        mins = min(grids[k][p] - cur_acc[k] for k in halves) * 100
        out.append(f"  w2={p[0]:.2f} w3={p[1]:.2f} β={p[2]:g}  "
                   f"四半区 {'/'.join(gains)}  最小增量 {mins:+.3f}pp")

    # 共识中点建议: 候选里取 w2/w3/β 中位附近的点（05b "共识中点 A" 做法）
    verdict = ["\n判读:"]
    if not better:
        verdict.append(f"  无四半区一致优于现行的点 → **常数不变**"
                       f"（({cur[0]},{cur[1]},{cur[2]:g}) 维持）")
    else:
        best = ranked[0]
        w2s = sorted({p[0] for p in better})
        w3s = sorted({p[1] for p in better})
        bs = sorted({p[2] for p in better})
        verdict.append(f"  一致更优点 {len(better)} 个, w2 范围 {w2s[0]:.2f}~"
                       f"{w2s[-1]:.2f}, w3 {w3s[0]:.2f}~{w3s[-1]:.2f}, "
                       f"β {bs[0]:g}~{bs[-1]:g}")
        verdict.append(f"  最大稳健候选 = (w2={best[0]:.2f}, w3={best[1]:.2f}, "
                       f"β={best[2]:g})——若采纳为新常数, 连动两版代码"
                       f"内置值并重跑双域验证")
    out += verdict


def main():
    ap = argparse.ArgumentParser(
        description="kCeW×β 常数标定（split=分半验证 / 4half=四半区终审）")
    ap.add_argument("--protocol", choices=["split", "4half"], default="split")
    ap.add_argument("--ce-sub", default=CE_SUB,
                    help="CE 结果子目录名（4half 05bi 终审传 ce_05bi）")
    ap.add_argument("--report", default=None,
                    help="报告文件名（默认按协议取历史名）")
    a = ap.parse_args()

    if a.protocol == "split":
        head = ["=" * 96,
                "0.5B 模型 kCeW×β 重标定（cohort 用户域 34k + wiki 池化 179,699）",
                "层1/层2 生产公式不变; 词频侧 τ=400/log(uni+1) 与模型无关不动",
                "=" * 96]
        report = a.report or "calibrate_model_05b_report.txt"
        ce_sub = a.ce_sub
    else:
        head = ["=" * 96,
                "05bi 常数终审（instruct CE; 协议 = 05b 四半区一致）",
                "=" * 96]
        report = a.report or "calibrate_model_05bi_final_report.txt"
        ce_sub = "ce_05bi" if a.ce_sub == CE_SUB else a.ce_sub
    out = list(head)
    if a.protocol == "split":
        run_split(out)
    else:
        run_4half(out, ce_sub, CUR_05BI)
    rpt = corpus_std.EVAL / report
    rpt.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n报告: {rpt}")


if __name__ == "__main__":
    main()
