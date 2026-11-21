#!/usr/bin/env python3
"""4×4 汇总表（2026-11-08, 用户指令）：统一口径 E+U1（F1, 无 bigram）、
τ=400、CAP=2万、kW=(1,1.10,0.60)、两层 k=5。

列（4）: 大词典×wiki(默认龄) / 小词典×wiki(默认龄) / 大词典×用户语料 / 小词典×用户语料
行（4）: 固定词典 / 用户词频（无LLM, F1 argmax）/ 纯CE（两层） / LLM+用户词频（F1 argmax）

年龄口径: 2026-11-18 用户定案默认龄 10 万 → **3 万次提交**（corpus_std.DEFAULT_ARM,
COHORT_ARM env 可覆盖）；历史报告 summary_4x4_report.txt 为 10 万龄产物, 不再重跑。

数据源:
  col1 cohort_big     arm=DEFAULT_ARM（25k 窗, CE=GPU 宇宙全量）
  col2 cohort_small6w arm=DEFAULT_ARM（25k 窗, CE=CPU 宇宙）
  col3 user_anchors_std（大词典窗 20,001 窗, CE=user_ce_work 全覆盖）
  col4 user 样本×小词库窗（8,424 窗, CE=复用 97.1% + 缺口 GPU 补算）
用法: python summary_4x4.py
"""
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_std
from corpus_std import MAX_CAND, normalize_ctx, window_key_ctx

PROJ = corpus_std.PROJ
TAU = 400.0
V_CAP = 20000
K = 5
W2, W3 = 1.10, 0.60     # 生产 CE 位置权重（2026-11-14 随 0.5B 重标定；旧 1.05/0.45
                        # 为 0.8B 时代值，历史报告按旧值冻结。derive_pairfix_full 会
                        # 用 --w2/--w3 覆盖本模块值）
B_GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
D_GRID = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]
TRAIN_LOG = corpus_std.CORPUS_PATH   # 常开语料（env RIME_LLM_CORPUS 覆盖）
SMALL_DICT = PROJ / "词库-测评6万" / "pdsp_dict_small6w.yaml"
SIM_GPU = PROJ / "cpp" / "build_sim" / "Release" / "sim_rerank_gpu.exe"


def tl_hit(S1, widx, s_ce):
    order = np.argsort(-S1, axis=1, kind="stable")
    topk = order[:, :K]
    S1k = np.take_along_axis(S1, topk, axis=1)
    sct = s_ce[np.arange(len(topk))[:, None], topk]
    S2k = S1k + np.where(sct > -1e8, sct, 0.0)
    winner = topk[np.arange(len(topk)), S2k.argmax(1)]
    return (winner == widx).mean()


def build_ce_matrix(ce_maps, lens, pad):
    """ce_maps: per-window {word: (ce, ntok) 或 None=不可用} → s_ce 矩阵"""
    n = len(ce_maps)
    ce = np.full((n, MAX_CAND, 3), -9.0)
    ntok = np.zeros((n, MAX_CAND), dtype=np.int32)
    for k, m in enumerate(ce_maps):
        for j, (w, pv) in enumerate(m.items()):
            if pv is None:
                continue
            ce[k, j] = pv[0]
            ntok[k, j] = pv[1]
    cls = np.where(ntok <= 1, 0, np.where(ntok == 2, 1, 2))
    ok = ce >= -0.5
    krows = np.stack([np.array([1.0, 0.0, 0.0]),
                      np.array([1.0, W2, 0.0]),
                      np.array([1.0, W2, W3])])[cls]
    ce_f = np.where(ok, ce, 0.0)
    bad = (((krows > 0) & ~ok).any(-1)) | (ntok == 0)
    return np.where(bad, -1e9, -(krows * ce_f).sum(-1))


def eval_column(name, du, lp1, pad, widx, s_ce):
    w_h = widx
    err_fix = 1 - (w_h == 0).mean()
    rows = [(b, d, err_allwin(b * du + d * lp1, w_h))
            for b in B_GRID for d in D_GRID if b > 0 or d > 0]
    b1, d1, e1 = min(rows, key=lambda t: t[2])
    e_ce = 1 - tl_hit(np.where(pad, -1e9, 0.0), w_h, s_ce)
    rows2 = [(b, d, 1 - tl_hit(b * du + d * lp1, w_h, s_ce))
             for b in B_GRID for d in D_GRID if b > 0 or d > 0]
    b2, d2, e2 = min(rows2, key=lambda t: t[2])
    e_prod = 1 - tl_hit(12 * du + 1 * lp1, w_h, s_ce)
    return dict(name=name, fix=err_fix, freq=(e1, b1, d1), ce=e_ce,
                llm=(e2, b2, d2), prod=e_prod)


def err_allwin(S, widx):
    return 1 - (S.argmax(1) == widx).mean()


def cohort_matrices(chain, arms=None, ce_dir="ce"):
    """→ {arm: dict(du, lp1, s_ce, pad, widx, ui, n)}。
    一次装载 anchors/samples + 一次 streams 重放覆盖全部请求 arm
    （cohort_sigtest 跨 6 档共用；单 arm 数值与旧逐档装载逐位一致）。
    ce_dir: CE 结果子目录（微调模型评测传 "ce_ft" 等, 与基线隔离）。"""
    d = PROJ / f"cohort_{chain}"
    anchors_all = [json.loads(l) for l in open(d / "anchors.jsonl",
                                               encoding="utf-8")]
    groups = defaultdict(list)
    for i, an in enumerate(anchors_all):
        if arms is None or an["arm"] in arms:
            groups[an["arm"]].append(i)
    keep = sorted(i for g in groups.values() for i in g)
    samples_all = [json.loads(l) for l in open(d / "samples.jsonl",
                                               encoding="utf-8")]
    du_full = np.zeros((len(keep), MAX_CAND))
    for r, i in enumerate(keep):
        eff = samples_all[i]["eff_win"]
        for j, w in enumerate(anchors_all[i]["win"]):
            du_full[r, j] = math.log1p(eff.get(w, 0.0))
    del samples_all
    # 单次 streams 重放 → 全部请求 arm 的 lp1
    by_ac = defaultdict(list)
    for r, i in enumerate(keep):
        an = anchors_all[i]
        by_ac[(an["ui"], an["ci"])].append(r)
    uni, n_seen = {}, 0
    lp1_full = np.full((len(keep), MAX_CAND), -1e9)
    with open(d / "streams.jsonl", encoding="utf-8") as f:
        for ui, line in enumerate(f):
            u = {}
            for ci, w in enumerate(json.loads(line)):
                for r in by_ac.get((ui, ci), ()):
                    an = anchors_all[keep[r]]
                    dd = n_seen + V_CAP
                    for j, x in enumerate(an["win"]):
                        lp1_full[r, j] = math.log((u.get(x, 0) + 1) / dd)
                u[w] = u.get(w, 0) + 1
                n_seen += 1
    # CE（只留 ce/ntok 两个字段, scores 不消费）
    ce_recs = {}
    with open(d / ce_dir / "results.jsonl", encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
                ce_recs[r["sid"]] = {"ce": r.get("ce") or {},
                                     "ntok": r.get("ntok") or {}}
            except Exception:
                pass
    keep_arr = np.array(keep)
    out = {}
    for g, idxs in groups.items():
        rows = np.searchsorted(keep_arr, np.array(idxs))
        an_list = [anchors_all[i] for i in idxs]
        lens = np.array([len(an["win"]) for an in an_list])
        pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
        widx = np.array([an["win"].index(an["w"]) for an in an_list])
        du = np.where(pad, -1e9, du_full[rows])
        lp1 = np.where(pad, -1e9, lp1_full[rows])
        ce_maps = []
        for an in an_list:
            rec = ce_recs.get(an["sid"])
            ce_maps.append({w: (rec["ce"][w], rec["ntok"][w])
                            for w in an["win"]
                            if rec and w in rec["ce"] and w in rec["ntok"]})
        s_ce = build_ce_matrix(ce_maps, lens, pad)
        out[g] = dict(du=du, lp1=lp1, s_ce=s_ce, pad=pad, widx=widx,
                      ui=np.array([an["ui"] for an in an_list]),
                      ci=np.array([an["ci"] for an in an_list]),
                      wins=[an["win"] for an in an_list],
                      n=len(idxs))
    return out


def load_cohort_arm(chain, arm):
    m = cohort_matrices(chain, {arm})[arm]
    return eval_column(f"{chain}@{arm}", m["du"], m["lp1"], m["pad"],
                       m["widx"], m["s_ce"])


def user_matrices(chain, ce_dir="user_ce_work"):
    """→ dict(du, lp1, s_ce, pad, widx, ci, n)——真实用户链（单人流, 无用户簇,
    cohort_sigtest 用提交流移动块代替簇）。chain='big': 样本原窗;
    'small6w': 小词库同码窗重建。ce_dir: CE 结果目录（微调评测传
    "user_ce_ft" 等, 与基线隔离; 小词库窗按 (norm_ctx,word) 复用逻辑仅对
    基线 user_ce_work 有效——微调模型的 CE 需全覆盖时用 big 链）。"""
    c2w_s, wc_s = corpus_std.load_pdsp_dict(SMALL_DICT)
    c2w_s, wc_s = corpus_std.load_pdsp_dict(SMALL_DICT)
    samples = [json.loads(l) for l in open(PROJ / "user_samples_std.jsonl",
                                           encoding="utf-8")]
    anchors = [json.loads(l) for l in open(PROJ / "user_anchors_std.jsonl",
                                           encoding="utf-8")]
    words_all = set(corpus_std.load_pdsp_words())
    seq = [w for w in corpus_std.iter_word_stream() if w in words_all]
    keep = []
    for i, (s, an) in enumerate(zip(samples, anchors)):
        if chain == "big":
            win = (s["cands"][:s["pos"]] + [s["word"]]
                   + s["cands"][s["pos"]:])[:MAX_CAND]
        else:
            code = s.get("code")
            if not code or code not in c2w_s:
                continue
            full = c2w_s[code]
            if len(full) < 2 or s["word"] not in full:
                continue
            win = full[:MAX_CAND]
        keep.append((an["ci"], an["ctx"], s["word"], win))
    n = len(keep)
    lens = np.array([len(k3[3]) for k3 in keep])
    pad = np.arange(MAX_CAND)[None, :] >= lens[:, None]
    widx = np.array([k3[3].index(k3[2]) for k3 in keep])
    by_ci = defaultdict(list)
    for k, (ci, ctx, w, win) in enumerate(keep):
        by_ci[ci].append(k)
    rp = corpus_std.FormulaReplay(TAU)
    uni, n_seen = {}, 0
    du = np.zeros((n, MAX_CAND))
    lp1 = np.full((n, MAX_CAND), -1e9)
    for ci, w in enumerate(seq):
        for k in by_ci.get(ci, ()):
            dd = n_seen + V_CAP
            for j, x in enumerate(keep[k][3]):
                du[k, j] = math.log1p(rp.eff_of(x))
                lp1[k, j] = math.log((uni.get(x, 0) + 1) / dd)
        rp.bump(w)
        uni[w] = uni.get(w, 0) + 1
        n_seen += 1
    du = np.where(pad, -1e9, du)
    lp1 = np.where(pad, -1e9, lp1)
    # CE: user_ce_work（大词典窗 sid → 逐词值; 小词库窗按 (norm_ctx,word) 复用）
    ce_recs = {}
    for l in open(PROJ / ce_dir / "results.jsonl", encoding="utf-8"):
        try:
            r = json.loads(l)
            ce_recs[r["sid"]] = r
        except Exception:
            pass
    pair_val = {}
    for an in anchors:
        rec = ce_recs.get(an["sid"])
        if not rec:
            continue
        cn = normalize_ctx(an["ctx"])
        for w, v in rec["ce"].items():
            if w in rec.get("ntok", {}):
                pair_val[(cn, w)] = (v, rec["ntok"][w])
    ce_maps = []
    missing = []
    for k, (ci, ctx, w, win) in enumerate(keep):
        cn = normalize_ctx(ctx)
        m = {}
        miss = []
        for x in win:
            pv = pair_val.get((cn, x))
            if pv:
                m[x] = pv
            else:
                miss.append(x)
        if miss:
            missing.append((k, ctx, miss))
        ce_maps.append(m)
    if missing and chain == "small6w" and ce_dir == "user_ce_work":
        os.environ.pop("SIM_NO_CE_PIPE", None)
        inp = "\n".join(json.dumps({"context": ctx, "cands": miss},
                                   ensure_ascii=False)
                        for k, ctx, miss in missing) + "\n"
        p = subprocess.run([str(SIM_GPU), "--ce", "--ngl", "99"],
                           input=inp.encode("utf-8"), capture_output=True)
        out_lines = p.stdout.decode("utf-8", "replace").strip().split("\n")
        assert len(out_lines) == len(missing)
        for (k, ctx, miss), ol in zip(missing, out_lines):
            r = json.loads(ol)
            for x in miss:
                if x in r.get("ce", {}):
                    ce_maps[k][x] = (r["ce"][x], r["ntok"].get(x, 0))
    s_ce = build_ce_matrix(ce_maps, lens, pad)
    return dict(du=du, lp1=lp1, s_ce=s_ce, pad=pad, widx=widx,
                ci=np.array([k3[0] for k3 in keep]),
                wins=[k3[3] for k3 in keep], n=n)


def load_user_column(chain):
    m = user_matrices(chain)
    return eval_column(f"user×{chain}", m["du"], m["lp1"], m["pad"],
                       m["widx"], m["s_ce"])


def main():
    # 2026-11-18 研究范围收窄：小词库/单字/用户域链已删除，4×4 终表退役
    if not (PROJ / "cohort_small6w").exists():
        raise SystemExit(
            "summary_4x4 的 4×4 终表依赖小词库/用户域链，已随 2026-11-18 "
            "范围收窄删除；历史报告 summary_4x4_report.txt 已冻结。"
            "现行结论见 README「核心结论」与 eval/fusion_2x2_05bi_report.txt。")
    arm = corpus_std.DEFAULT_ARM
    cols = [
        load_cohort_arm("big", arm),
        load_cohort_arm("small6w", arm),
        load_user_column("big"),
        load_user_column("small6w"),
    ]
    hdr = f"{'模型':<26}" + "".join(f"{c['name']:>20}" for c in cols)
    L = ["=" * 106,
         "4×4 汇总 · 统一口径: 用户词频=衰减Unigram+累计Unigram(F1, 无bigram), τ=400,",
         "CAP=2万, kW=(1,1.10,0.60), 两层 k=5; wiki 列=cohort 3万基准龄(空库起步队列)",
         "选重率%（越低越好）; wiki 两列与用户两列绝对值不可比（锚点总体/窗均候选不同）",
         hdr, "-" * 106]
    L.append(f"{'固定词典（无LLM无词频）':<24}"
             + "".join(f"{c['fix'] * 100:>19.2f}%" for c in cols))
    L.append(f"{'用户词频（无LLM）':<24}"
             + "".join(f"{c['freq'][0] * 100:>19.2f}%" for c in cols))
    L.append(f"{'LLM 纯CE（两层k=5）':<24}"
             + "".join(f"{c['ce'] * 100:>19.2f}%" for c in cols))
    L.append(f"{'LLM+用户词频':<24}"
             + "".join(f"{c['llm'][0] * 100:>19.2f}%" for c in cols))
    txt = "\n".join(L)
    (corpus_std.EVAL / "summary_4x4_report.txt").write_text(
        txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
