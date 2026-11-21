#!/usr/bin/env python3
"""多模型同窗对照（2026-11-17 脚本合并）。

原 `compare_model_sizes.py`（用户域 34k）、`compare_model_sizes_wiki.py`
（wiki 池化 179k）、`compare_model_2b.py`（0.8B vs 2B 子集）合并：
**同一批窗、同一套词频态，只换 CE 目录（= 只换模型）**，输出层2 首选率、
相对基线模型的 Δ、同窗配对翻转、纯 CE 行与 CE 尺度诊断。

域通过参数选择：
  --domain user  → cohort_big_user（多字词用户域）
  --domain wiki  → cohort_big_sc（单字 3 码）+ cohort_big（多字词 4 码）池化
  --domain all   → 两域都跑（默认）

用法:
  python compare_models.py
  python compare_models.py --domain user --models 0.5B=ce_05b,0.8B=ce
  python compare_models.py --domain wiki --limit 2000 --k 5 --beta 12
报告: compare_models_report.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from derive_pairfix_ratio import load_domain, load_w2codes

sys.stdout.reconfigure(encoding="utf-8")
OUT = corpus_std.EVAL / "compare_models_report.txt"


def parse_models(text):
    out = []
    for item in (x.strip() for x in text.split(",") if x.strip()):
        if "=" not in item:
            raise SystemExit(f"--models 需要 name=ce_sub 形式: {item!r}")
        name, ce = item.split("=", 1)
        out.append((name.strip(), ce.strip()))
    if not out:
        raise SystemExit("--models 不能为空")
    return out


def load_user(ce_sub, wc_sc3, first_code):
    d = load_domain("cohort_big_user", "big", wc_sc3, first_code,
                    ce_sub=ce_sub)
    return dict(widx=d["widx"], pad=d["pad"], du=d["du"], uni=d["uni"],
                s_ce=d["s_ce"], ce_ok=d["ce_ok"], min_ce=d["min_ce"],
                wins=d["wins"], dom_id=np.zeros(len(d["widx"]), dtype=np.int8))


def load_wiki(ce_sub, wc_sc3, first_code):
    # 2026-11-18 研究范围收窄：单字 3 码链已删除，wiki 域 = 多字词大词库
    big = load_domain("cohort_big", "big", wc_sc3, first_code,
                      ce_sub=ce_sub)
    out = dict(
        widx=big["widx"], pad=big["pad"], du=big["du"], uni=big["uni"],
        s_ce=big["s_ce"], ce_ok=big["ce_ok"], min_ce=big["min_ce"],
        wins=big["wins"],
        dom_id=np.zeros(len(big["widx"]), dtype=np.int8),
    )
    del big
    return out


def evaluate(d, beta, k):
    n = len(d["widx"])
    s1 = np.where(d["pad"], -1e9, beta * d["du"] + np.log(d["uni"] + 1.0))
    cols = np.argsort(-s1, axis=1, kind="stable")[:, :k]
    s1w = np.take_along_axis(s1, cols, 1)
    ce_term = np.where(d["ce_ok"], d["s_ce"], d["min_ce"][:, None])
    cew = np.take_along_axis(ce_term, cols, 1)
    ce_ok_w = np.take_along_axis(d["ce_ok"], cols, 1)
    cew = np.where(ce_ok_w, cew, -1e9)
    s2w = s1w + cew
    f = cols[np.arange(n), np.argmax(s2w, axis=1)]
    c = cols[np.arange(n), np.argmax(cew, axis=1)]
    ce2 = np.sort(cew, axis=1)
    gap = ce2[:, -1] - ce2[:, -2] if k > 1 else np.zeros(n)
    gap = gap[np.isfinite(gap) & (gap > -1e8)]
    return (f == d["widx"]), (c == d["widx"]), (
        cols[:, 0] == d["widx"]), gap


def run_domain(domain, models, base_name, beta, k, limit, wc_sc3, first_code,
               out):
    label = "user（cohort_big_user）" if domain == "user" else \
        "wiki（cohort_big_sc + cohort_big 池化）"
    out.append(f"\n{'=' * 96}\n== {label} ==\n{'=' * 96}")
    res, ref = {}, None
    dom_id = None
    n = 0
    for name, ce in models:
        d = (load_user(ce, wc_sc3, first_code) if domain == "user"
             else load_wiki(ce, wc_sc3, first_code))
        if limit and len(d["widx"]) > limit:
            idx = np.linspace(0, len(d["widx"]) - 1, limit).astype(int)
            d = {key: (v[idx] if isinstance(v, np.ndarray) and
                       len(v) == len(d["widx"]) else v)
                 for key, v in d.items()}
            d["wins"] = [d["wins"][i] for i in idx]
        if ref is None:
            ref = (d["widx"].copy(), d["du"].copy(), d["uni"].copy())
            dom_id = d["dom_id"].copy()
            n = len(d["widx"])
        else:
            assert (d["widx"] == ref[0]).all(), "同窗校验失败: widx"
            assert np.allclose(d["du"], ref[1]), "词频态不一致: du"
            assert np.allclose(d["uni"], ref[2]), "词频态不一致: uni"
            assert (d["dom_id"] == dom_id).all(), "域标记不一致"
        ok, c_ok, _l1_ok, gap = evaluate(d, beta, k)
        res[name] = (ok, c_ok, gap)
        print(f"  eval {name} ({ce}): {n} 窗", flush=True)
        del d
    base = res[base_name][0]
    out.append(f"窗数 {n}; 基线 = {base_name}; 层2 k={k}, β={beta:g}")
    out.append(f"{'模型':<22}{'层2+CE':>10}{'Δvs基线':>10}{'纯CE':>10}")
    for name, _ in models:
        ok, c_ok, _gap = res[name]
        out.append(f"{name:<22}{ok.mean() * 100:>9.3f}%"
                   f"{(ok.mean() - base.mean()) * 100:>+9.3f}pp"
                   f"{c_ok.mean() * 100:>9.3f}%")
    out.append("")
    out.append("同窗配对切换（层2）:")
    for name, _ in models:
        if name == base_name:
            continue
        ok = res[name][0]
        out.append(f"  {name}: 基线错→它对 {int((~base & ok).sum())}, "
                   f"基线对→它错 {int((base & ~ok).sum())}")
    if domain == "wiki":
        out.append("")
        out.append(f"{'域':<8}{'窗数':>8}" +
                   "".join(f"{name.split('-')[-1]:>10}" for name, _ in models))
        # 2026-11-18 起 wiki 域只剩多字词（单字 3 码链已删除）
        mask = np.ones(len(dom_id), dtype=bool)
        out.append(f"{'多字词':<8}{int(mask.sum()):>8}" +
                   "".join(f"{res[name][0][mask].mean() * 100:>9.3f}"
                           for name, _ in models))
    out.append("")
    out.append("CE 尺度诊断（同窗 top1−top2 CE 分差中位）:")
    for name, ce in models:
        gap = res[name][2]
        if len(gap):
            out.append(f"  {name} ({ce}): {np.median(gap):.2f}")
        else:
            out.append(f"  {name} ({ce}): -")


def main():
    ap = argparse.ArgumentParser(description="多模型同窗对照")
    ap.add_argument("--domain", choices=["wiki"], default="wiki")
    ap.add_argument("--models", default="0.5Bi=ce_05bi")
    ap.add_argument("--baseline", default=None,
                    help="基线模型名（默认第一个）")
    ap.add_argument("--beta", type=float, default=14.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时抽样 N 窗（冒烟用）")
    a = ap.parse_args()

    models = parse_models(a.models)
    base_name = a.baseline or models[0][0]
    if base_name not in [n for n, _ in models]:
        raise SystemExit(f"基线 {base_name!r} 不在 --models 中")
    w2codes, first_code = load_w2codes()
    _, wc_sc3 = corpus_std.load_pdsp_dict_single3()

    out = ["=" * 96,
           "多模型同窗对照 · 只换 CE（模型）· 层1 = β·log1p(eff)+log(uni+1)",
           "=" * 96]
    domains = ["user", "wiki"] if a.domain == "all" else [a.domain]
    for domain in domains:
        run_domain(domain, models, base_name, a.beta, a.k, a.limit,
                   wc_sc3, first_code, out)
    txt = "\n".join(out)
    OUT.write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    print(f"\n报告: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
