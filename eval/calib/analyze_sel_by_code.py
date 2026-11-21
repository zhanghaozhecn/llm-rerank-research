#!/usr/bin/env python3
"""按码统计选重数（2026-11-17, 用户口径）。

选重 = **融合重排后的最终首选 ≠ 正确选项（实际应打的词）**。
选重数 = 某个码下发生这种事件的窗数。与"改对/改错"无关——只看最终
有没有把正确词放到首位。

数据（全 CE 覆盖, CE=ce_05bi / 现行 instruct 模型）:
  wiki 合并 = cohort_big_sc（单字 3 码, 59,681 窗, bump=sc）
             + cohort_big（多字词 4 码, 120,018 窗, bump=big）
  user      = cohort_big_user（用户语料多字词 4 码, 34,000 窗, bump=big）
生产两层公式: 层1 β14·log1p(eff)+log(uni+1) 稳定前 K=5;
层2 + kCeW(1,1.10,0.60) 加权 CE, 无 CE 用窗内最低有效 CE。

产出:
  eval/sel_by_code_report.txt
  eval/sel_by_code_wiki_full.tsv / sel_by_code_top100_wiki.tsv    （wiki 合并）
  eval/sel_by_code_user_full.tsv / sel_by_code_top100_user.tsv    （用户）
  eval/sel_by_code_wiki_sc_full.tsv / _top100_wiki_sc.tsv          （子域）
  eval/sel_by_code_wiki_big_full.tsv / _top100_wiki_big.tsv        （子域）
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
import calibrate_model as cm
from derive_pairfix_ratio import load_w2codes

sys.stdout.reconfigure(encoding="utf-8")

BETA, W2, W3, K = 14.0, 1.10, 0.60, 5
CE_SUB = "ce_05bi"
OUT = corpus_std.EVAL / "sel_by_code_report.txt"
TOPN = 100


def load_codes_targets(dirname):
    d = corpus_std.PROJ / dirname
    anchors = [json.loads(l) for l in open(d / "anchors.jsonl",
                                           encoding="utf-8")]
    samples = [json.loads(l) for l in open(d / "samples.jsonl",
                                           encoding="utf-8")]
    if len(anchors) != len(samples):
        raise SystemExit(f"{dirname}: anchors {len(anchors)} != "
                         f"samples {len(samples)}")
    return ([s["code"] for s in samples],
            [a["w"] for a in anchors])


def winners(dom):
    n = len(dom["widx"])
    s1 = np.where(dom["pad"], -1e9,
                  BETA * dom["du"] + np.log(dom["uni"] + 1.0))
    cols = np.argsort(-s1, axis=1, kind="stable")[:, :K]
    s1w = np.take_along_axis(s1, cols, 1)
    sce = cm.ce_matrix(dom["ce"], dom["ntok"], W2, W3)
    cew = np.take_along_axis(sce, cols, 1)
    return cols[np.arange(n), np.argmax(s1w + cew, axis=1)]


def compute(dirname, bump_mode):
    w2codes, first_code = load_w2codes()
    _, wc_sc = corpus_std.load_pdsp_dict_single3()
    codes, targets = load_codes_targets(dirname)
    dom = cm.load_raw(dirname, bump_mode, wc_sc, first_code, ce_sub=CE_SUB)
    f = winners(dom)
    widx = dom["widx"]
    n = len(codes)
    if len(widx) != n:
        raise SystemExit(f"{dirname}: dom {len(widx)} != codes {n}")
    miss = f != widx
    per = defaultdict(lambda: dict(n=0, sel=0, missed=Counter(),
                                   all=Counter()))
    for i in range(n):
        d = per[codes[i]]
        d["n"] += 1
        d["all"][targets[i]] += 1
        if miss[i]:
            d["sel"] += 1
            d["missed"][targets[i]] += 1
    print(f"{dirname} ({bump_mode}): {n} windows, "
          f"{int(miss.sum())} selections", flush=True)
    return per, n, int(miss.sum())


def merge(pers):
    out = defaultdict(lambda: dict(n=0, sel=0, missed=Counter(),
                                   all=Counter()))
    for per in pers:
        for c, d in per.items():
            o = out[c]
            o["n"] += d["n"]
            o["sel"] += d["sel"]
            o["missed"].update(d["missed"])
            o["all"].update(d["all"])
    return out


def gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    if not len(v) or v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def hist_rows(counts):
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20),
            (21, 50), (51, 100), (101, 10 ** 9)]
    rows = []
    for lo, hi in bins:
        codes = [c for c, v in counts.items() if lo <= v <= hi]
        if not codes:
            continue
        nsel = sum(counts[c] for c in codes)
        rows.append((f"{lo}" if lo == hi else
                     (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}"),
                     len(codes), nsel))
    return rows


def fmt_top(cnt, k=4):
    return " ".join(f"{w}:{v}" for w, v in cnt.most_common(k))


def write_section(name, per, total_win, total_sel, full_name, top_name,
                  out, print_top=TOPN):
    order = sorted(per.items(), key=lambda kv: (-kv[1]["sel"],
                                               -kv[1]["n"], kv[0]))
    n_codes = len(per)
    codes_with = sum(1 for d in per.values() if d["sel"] > 0)
    counts = {c: d["sel"] for c, d in per.items() if d["sel"] > 0}

    cum_by_rank = []
    cum = 0
    for r, (c, d) in enumerate(order, 1):
        cum += d["sel"]
        cum_by_rank.append((r, cum))

    def rank_for(frac):
        tgt = total_sel * frac
        for r, cc in cum_by_rank:
            if cc >= tgt:
                return r
        return len(cum_by_rank)

    top_shares = {}
    for t in (1, 5, 10, 20, 50, 100, 200, 500):
        top_shares[t] = sum(d["sel"] for _, d in order[:t])

    full = corpus_std.EVAL / full_name
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("code\tn_sel\tn_win\tsel_rate_pct\tshare_pct\t"
                 "n_distinct_targets\ttop_missed_words\ttop_all_words\n")
        for c, d in order:
            fh.write(f"{c}\t{d['sel']}\t{d['n']}\t"
                     f"{100.0 * d['sel'] / d['n']:.3f}\t"
                     f"{100.0 * d['sel'] / total_sel if total_sel else 0:.4f}\t"
                     f"{len(d['all'])}\t{fmt_top(d['missed'])}\t"
                     f"{fmt_top(d['all'])}\n")
    top = corpus_std.EVAL / top_name
    with open(top, "w", encoding="utf-8") as fh:
        fh.write("rank\tcode\tn_sel\tn_win\tsel_rate_pct\tshare_pct\t"
                 "cum_pct\tn_distinct_targets\ttop_missed_words\t"
                 "top_all_words\n")
        cum = 0
        for r, (c, d) in enumerate(order[:TOPN], 1):
            cum += d["sel"]
            fh.write(f"{r}\t{c}\t{d['sel']}\t{d['n']}\t"
                     f"{100.0 * d['sel'] / d['n']:.3f}\t"
                     f"{100.0 * d['sel'] / total_sel if total_sel else 0:.4f}\t"
                     f"{100.0 * cum / total_sel if total_sel else 0:.2f}\t"
                     f"{len(d['all'])}\t{fmt_top(d['missed'])}\t"
                     f"{fmt_top(d['all'])}\n")

    out.append("")
    out.append("=" * 108)
    out.append(f"== {name} ==")
    out.append(f"窗 {total_win:,}; 选重（最终首选≠正确词）{total_sel:,}; "
               f"整体选重率 {100.0 * total_sel / total_win:.3f}%")
    out.append(f"出现码 {n_codes:,}; 有选重的码 {codes_with:,} "
               f"({100.0 * codes_with / n_codes:.1f}% of codes)")
    out.append(f"每码选重数 Gini = {gini([d['sel'] for _, d in order]):.3f}; "
               f"达到总选重 50%/80%/90% 所需码数 = "
               f"{rank_for(0.5)}/{rank_for(0.8)}/{rank_for(0.9)} "
               f"（占全部码 {100.0 * rank_for(0.5) / n_codes:.1f}% / "
               f"{100.0 * rank_for(0.8) / n_codes:.1f}% / "
               f"{100.0 * rank_for(0.9) / n_codes:.1f}%）")
    out.append("前 N 码覆盖的选重占比: " + "  ".join(
        f"top{t} {100.0 * v / total_sel:.1f}%" for t, v in
        top_shares.items() if v and t <= 200))
    # 参考：把码的首选固定为「全局最高频正确词」的理论收益上限。
    # 只作可行性参考——固定后该码剩余选重 = 窗数 − 该词总出现数。
    gain_all = 0
    gain_top = 0
    for r, (c, d) in enumerate(order, 1):
        modal_n = d["all"].most_common(1)[0][1]
        g = max(0, d["sel"] - (d["n"] - modal_n))
        gain_all += g
        if r <= TOPN:
            gain_top += g
    out.append(f"参考：固定为全局最高频正确词的理论可消选重（仅正收益）"
               f" = 全码 {gain_all:,} ({100.0 * gain_all / total_sel:.1f}%), "
               f"前100码 {gain_top:,} "
               f"({100.0 * gain_top / total_sel:.1f}%)")
    out.append("")
    out.append("每码选重数分布（仅选重>0 的码）:")
    out.append("  选重数区间\t码数\t该组选重数\t占全部选重%")
    for lbl, ncode, nsel in hist_rows(counts):
        out.append(f"  {lbl}\t{ncode}\t{nsel}\t"
                   f"{100.0 * nsel / total_sel:.1f}%")
    out.append("")
    out.append(f"前 {print_top} 码（按选重数降序; 同名次按窗数降序）:")
    out.append("  #  码      选重  窗数  码选重率  占全部  累计%  "
               "漏选词Top（失败窗正确词）      全局正确词Top")
    cum = 0
    for r, (c, d) in enumerate(order[:print_top], 1):
        cum += d["sel"]
        out.append(f"  {r:>3} {c:<6} {d['sel']:>5} {d['n']:>5} "
                   f"{100.0 * d['sel'] / d['n']:>7.2f}% "
                   f"{100.0 * d['sel'] / total_sel:>7.2f}% "
                   f"{100.0 * cum / total_sel:>7.2f}%  "
                   f"{fmt_top(d['missed'], 3):<30} "
                   f"{fmt_top(d['all'], 3)}")
    return order


def topcodes(per, k=TOPN):
    return [c for c, d in sorted(per.items(),
            key=lambda kv: (-kv[1]["sel"], -kv[1]["n"], kv[0]))[:k]]


def main():
    out = ["=" * 108,
           "按码统计选重数（wiki 语料 vs 用户语料）",
           "选重 = 融合重排后的最终首选 ≠ 正确选项（实际应打的词）; "
           "按码计数, 选重数降序",
           "两层生产公式 β14 / kCeW(1,1.10,0.60) / k=5; CE=ce_05bi",
           "=" * 108]

    sc_per, sc_win, sc_sel = compute("cohort_big_sc", "sc")
    big_per, big_win, big_sel = compute("cohort_big", "big")
    wiki_per = merge([sc_per, big_per])
    wiki_win, wiki_sel = sc_win + big_win, sc_sel + big_sel
    write_section("wiki 语料（单字3码 + 多字词4码）", wiki_per,
                  wiki_win, wiki_sel, "sel_by_code_wiki_full.tsv",
                  "sel_by_code_top100_wiki.tsv", out)

    out.append("")
    out.append("wiki 子域拆分（同一口径, 便于看集中度来源）:")
    write_section("wiki 单字3码（子域）", sc_per, sc_win, sc_sel,
                  "sel_by_code_wiki_sc_full.tsv",
                  "sel_by_code_top100_wiki_sc.tsv", out, print_top=10)
    write_section("wiki 多字词4码（子域）", big_per, big_win, big_sel,
                  "sel_by_code_wiki_big_full.tsv",
                  "sel_by_code_top100_wiki_big.tsv", out, print_top=10)

    user_per, user_win, user_sel = compute("cohort_big_user", "big")
    write_section("用户语料（多字词4码）", user_per, user_win, user_sel,
                  "sel_by_code_user_full.tsv",
                  "sel_by_code_top100_user.tsv", out)

    tw, tu = topcodes(wiki_per), topcodes(user_per)
    inter = [c for c in tw if c in set(tu)]
    out.append("")
    out.append("== wiki / 用户 前100码交集 ==")
    out.append(f"交集 {len(inter)} 个: {' '.join(inter) if inter else '(空)'}")
    Path(OUT).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n报告: {OUT}")


if __name__ == "__main__":
    main()
