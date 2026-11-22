#!/usr/bin/env python3
"""precommit_conf_report.py — 「确定前缀预提交」效果研究 · 汇总

输入（precommit_conf_collect.py + precommit_conf_score.lua 的产物）：
  <prefix>.jobs.jsonl   每决策点：ctx/rem/头候选（引擎序+rank）/真值
  <prefix>.scores.tsv   sid \\t i \\t 候选 \\t score（LLM CE 分，负值域）

输出（stdout）：
  · 数据规模与可行性（真值在头候选 / 生产窗 / 打分集的比例）
  · 引擎基线与 LLM 的词级首选正确率
  · margin 分桶准确率（置信度曲线）
  · 阈值扫描：确认覆盖率、确认正确率、每句自动确认错误数、每句按键成本
  · 对照：逐词全选（词模式）与整句一选（现行整句模式）的按键成本

成本模型：一次选词 = 1 键 + 翻页 floor(rank/5)（page_size=5）；自动确认
= 0 键；自动确认错误 = 1 退格 + 重选真值。尾部词不参与决策（标点顶掉）。
"""
import argparse
import json
import sys
from collections import defaultdict

PAGE = 5
THETAS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0,
          20.0]


def load_jobs(path):
    jobs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def load_scores(path):
    sc = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            sc[(int(p[0]), int(p[1]))][p[2]] = float(p[3])
    return sc


def pct(vals, p):
    if not vals:
        return 0
    return sorted(vals)[int(round((len(vals) - 1) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="collect 的 --out 前缀")
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--score-cap", type=int, default=12)
    a = ap.parse_args()

    jobs = load_jobs(a.prefix + ".jobs.jsonl")
    scores = load_scores(a.prefix + ".scores.tsv")

    n_sent = len({j["sid"] for j in jobs})
    n_dec = len(jobs)
    in_head = sum(1 for j in jobs if j["truth_in_head"])
    in_dict = sum(1 for j in jobs if j.get("truth_in_dict"))
    in_win = 0
    in_sub = 0
    for j in jobs:
        win = [h for h in j["head_cands"] if h["rank"] < a.win]
        if j["truth"] in [h["text"] for h in win]:
            in_win += 1
        sub = [h["text"] for h in win][:a.score_cap]
        if j["truth"] in sub:
            in_sub += 1

    # 每个决策点的 LLM 打分结果
    view = []
    for j in jobs:
        key = (j["sid"], j["i"])
        win = [h for h in j["head_cands"] if h["rank"] < a.win]
        sub = [h["text"] for h in win][:a.score_cap]
        d = {
            "job": j, "win": win, "sub": sub,
            "truth": j["truth"],
            "engine_rank": (next((k for k, h in enumerate(j["head_cands"])
                                  if h["text"] == j["truth"]), -1)),
            "scored": [],
            "truth_submitted": j["truth"] in sub,
        }
        s = scores.get(key, {})
        cand = [(t, s[t]) for t in sub
                if t in s and s[t] > -1e9 and s[t] == s[t]]
        cand.sort(key=lambda x: -x[1])
        d["scored"] = cand
        d["top1"] = cand[0][0] if cand else None
        d["margin"] = (cand[0][1] - cand[1][1]) if len(cand) >= 2 else None
        d["truth_llm_rank"] = (next((k for k, (t, _) in enumerate(cand)
                                     if t == j["truth"]), -1))
        view.append(d)

    def rank_label(d):
        """用户看到的序中的真值名次（LLM 有分则 LLM 序，否则引擎窗序）。"""
        if d["scored"] and d["truth_llm_rank"] >= 0:
            return d["truth_llm_rank"]
        win = [h["text"] for h in d["win"]]
        return win.index(d["truth"]) if d["truth"] in win else 0

    def sel_cost(d):
        return 1 + rank_label(d) // PAGE

    # 句子可行性：所有决策点真值都在打分集里
    sent_ok = defaultdict(bool)
    for d in view:
        sent_ok[d["job"]["sid"]] = True
    for d in view:
        if not d["truth_submitted"]:
            sent_ok[d["job"]["sid"]] = False
    good_sents = {s for s, ok in sent_ok.items() if ok}

    print("== 预提交模式 · 词级置信度研究 ==")
    print("sentences=%d decisions=%d（尾词不决策）" % (n_sent, n_dec))
    print("真值在词典（引擎可产出）: %d (%.1f%%)"
          % (in_dict, 100.0 * in_dict / max(n_dec, 1)))
    print("真值在头候选 all/win%d/cap%d: %d (%.1f%%) / %d (%.1f%%) / "
          "%d (%.1f%%)" % (a.win, a.score_cap, in_head,
                           100.0 * in_head / n_dec, in_win,
                           100.0 * in_win / n_dec, in_sub,
                           100.0 * in_sub / n_dec))
    print("全决策可行的句子: %d / %d (%.1f%%)"
          % (len(good_sents), n_sent, 100.0 * len(good_sents) / max(n_sent, 1)))

    # 引擎 vs LLM 首选（分母统一 = 真值在打分集里的可行决策）
    feas = [d for d in view if d["truth_submitted"]]
    eng_ok = [d for d in feas if d["engine_rank"] == 0]
    scorable = [d for d in feas if d["margin"] is not None]
    llm_ok = [d for d in scorable if d["top1"] == d["truth"]]
    print("引擎词级首选正确率（可行决策）: %.1f%% (%d/%d)"
          % (100.0 * len(eng_ok) / max(len(feas), 1), len(eng_ok), len(feas)))
    print("LLM 词级首选正确率（可行且≥2 候选可打分）: %.1f%% (%d/%d)"
          % (100.0 * len(llm_ok) / max(len(scorable), 1),
             len(llm_ok), len(scorable)))

    margins = [d["margin"] for d in scorable]
    if margins:
        print("margin 分位: p10=%.2f p50=%.2f p90=%.2f"
              % (pct(margins, .1), pct(margins, .5), pct(margins, .9)))

    # margin 分桶准确率
    print("\nmargin 桶（=> 该桶内 LLM 首选正确率 / 占比）:")
    edges = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 1e9]
    for lo, hi in zip(edges, edges[1:]):
        bucket = [d for d in scorable if lo <= d["margin"] < hi]
        if not bucket:
            continue
        ok = sum(1 for d in bucket if d["top1"] == d["truth"])
        print("  [%.1f, %.1f)  acc=%.1f%%  n=%d (%.0f%%)"
              % (lo, min(hi, 999), 100.0 * ok / len(bucket), len(bucket),
                 100.0 * len(bucket) / len(scorable)))

    # 整句一选基线（i=-1：全句覆盖候选的引擎/LLM 首选是否 == 真值句）
    sent_jobs = {j["sid"]: j for j in jobs if j["i"] == 0}
    eng_n = eng_ok_s = 0
    llm_n = llm_ok_s = 0
    for sid, j in sent_jobs.items():
        fc = [h["text"] for h in j.get("full_cands", [])]
        if not fc:
            continue
        truth = j.get("full_truth", "")
        eng_n += 1
        if fc[0] == truth:
            eng_ok_s += 1
        s = scores.get((sid, -1), {})
        cand = [(t, s[t]) for t in fc
                if t in s and s[t] > -1e9 and s[t] == s[t]]
        if len(cand) >= 2:
            cand.sort(key=lambda x: -x[1])
            llm_n += 1
            if cand[0][0] == truth:
                llm_ok_s += 1
    print("\n整句一选基线（全句覆盖候选，text == 原句为正确）:")
    print("  引擎首选整句正确率: %.1f%% (%d/%d)"
          % (100.0 * eng_ok_s / max(eng_n, 1), eng_ok_s, eng_n))
    print("  LLM  首选整句正确率: %.1f%% (%d/%d)"
          % (100.0 * llm_ok_s / max(llm_n, 1), llm_ok_s, llm_n))

    # 阈值扫描：分母 = 可行决策（真值在打分集里；否则用户也选不到，句子已废）
    feasible = [d for d in view if d["truth_submitted"]]
    print("\n阈值扫描（确认条件 = margin >= θ；分母 = %d 个可行决策）:"
          % len(feasible))
    print("  θ     覆盖率   确认正确率   自动错误/百决策  选词/百决策  按键/百决策")
    n_f = max(len(feasible), 1)
    for th in THETAS:
        conf = [d for d in feasible
                if d["margin"] is not None and d["margin"] >= th]
        wrong = [d for d in conf if d["top1"] != d["truth"]]
        acc = 100.0 * (len(conf) - len(wrong)) / len(conf) if conf else 0.0
        cov = 100.0 * len(conf) / len(feasible) if feasible else 0.0
        sel_cnt = 0
        key_cnt = 0
        for d in feasible:
            if d["margin"] is not None and d["margin"] >= th:
                if d["top1"] != d["truth"]:
                    sel_cnt += 2              # 退格 + 重选
                    key_cnt += 1 + sel_cost(d)
            else:
                sel_cnt += 1
                key_cnt += sel_cost(d)
        print("  %-5.1f %6.1f%% %8.1f%% %14.2f %12.2f %12.2f"
              % (th, cov, acc, 100.0 * len(wrong) / n_f,
                 100.0 * sel_cnt / n_f, 100.0 * key_cnt / n_f))
    # 句子级成本（仅全可行句子；wiki 技术文本 OOV 多时可能为 0）
    if good_sents:
        print("\n句子级成本（全决策可行句子 n=%d）:" % len(good_sents))
        print("  θ     自动错误/句   选词/句   按键/句")
        for th in THETAS:
            sel_cnt = 0
            key_cnt = 0
            errs = 0
            for d in view:
                if d["job"]["sid"] not in good_sents:
                    continue
                if d["margin"] is not None and d["margin"] >= th:
                    if d["top1"] != d["truth"]:
                        errs += 1
                        sel_cnt += 2
                        key_cnt += 1 + sel_cost(d)
                else:
                    sel_cnt += 1
                    key_cnt += sel_cost(d)
            print("  %-5.1f %10.2f %9.2f %9.2f"
                  % (th, errs / len(good_sents), sel_cnt / len(good_sents),
                     key_cnt / len(good_sents)))
        word_sel = sum(1 for d in view if d["job"]["sid"] in good_sents)
        word_key = sum(sel_cost(d) for d in view
                       if d["job"]["sid"] in good_sents)
        print("  -- 对照：逐词全选（词模式） %.2f 选词/句，%.2f 按键/句"
              % (word_sel / len(good_sents), word_key / len(good_sents)))
        print("  -- 对照：整句一选（现行整句模式） 1.00 选词/句（准确率不在本研究口径内）")
    else:
        print("\n句子级成本: n/a（本样本无全决策可行句子——wiki 技术文本 OOV 占比高）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
