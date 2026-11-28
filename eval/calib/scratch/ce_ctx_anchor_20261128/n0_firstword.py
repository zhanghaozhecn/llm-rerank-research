#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wiki 分条正文首句首词的真 n=0 测试（用户 2026-11-28 指定人群）。

人群 = segments_hanlp_cache.jsonl 每条 〖标题〗**之后的正文首句首词**（纯
CJK token；无 〗 退回首个 CJK token）。最纯的文档起始场景：ctx 严格为空，
truth = 首词本身，码 = 全拼。959,461 条 → 538,302 唯一首词，取频次
Top 10,000（= 生产条目权重；去重防同码窗重复计数）。

臂（冷账本）：
  A  freq_only·引擎原序 top1 = 池[0]（词模式口径 = 现行 freq_only）；
  A2 freq_only·整句显示序：span（候选覆盖字母数）降序分组 → 组内引擎序；
  C2 锚·整句显示序：同上分组，组内改 CE 加权和序——**生产口径的反事实**。
  首版无分组的「纯 CE 序」会把跨 span 长度偏差放大成灾难（隙 11.94 vs
  西班牙语 17.26：1-token 候选只付一笔 token 钱）——那是显示序不存在的
  比较方式，只留一行「全体·纯CE序」作反面教材。
  CE = 裸「。」（ctx="。"，SIM_CTX_ANCHOR=0 裸跑 = 守卫扩到 n≤1 的等效
  上下文），池前 10，[10:] 无 CE 沉底（生产压制规则），落临时目录（主缓存
  同 sid =「。。」口径不可混写）。
选重率 = P(首选 ≠ 真词)，条件在真词∈池（生产语义：只能选显示过的）。

跑法（研究库根目录）：
  python eval/calib/scratch/ce_ctx_anchor_20261128/n0_firstword.py          # 干跑
  ... --apply    # 抽词 → 引擎池探针 → 裸「。」CE 补算 → 三臂评测
"""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # eval/calib
sys.path.insert(0, str(HERE.parents[2] / "lib"))  # eval/lib
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L  # noqa: E402

KCE = (1.0, 1.10, 0.60)
SCRATCH = HERE / "n0_bare_punct_ce"        # 裸「。」CE 共享临时目录
TOP_N = 10000
CACHE = HERE / "n0_firstword_top.json"
SEG = L.PROJ / "segments_hanlp_cache.jsonl"


def log(msg):
    print(f"[fw] {msg}", flush=True)


def fused_of(c3):
    if not c3 or not c3[0]:
        return None
    s = 0.0
    for k in range(min(3, len(c3[0]))):
        x = c3[0][k]
        if x is not None and x > -1e8:
            s += KCE[k] * x
    return -s


def mcnemar_p(b, c):
    k = b + c
    if not k:
        return 1.0
    return math.erfc(abs(b - c) / math.sqrt(k) / math.sqrt(2))


def boot_ci(pa, pb, it=2000):
    n = len(pa)
    if not n:
        return (0.0, 0.0)
    diffs = [y - x for x, y in zip(pa, pb)]
    rng = random.Random(20261128)
    ds = []
    for _ in range(it):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        ds.append(100.0 * s / n)
    ds.sort()
    return (ds[int(0.025 * it)], ds[int(0.975 * it)])


def extract():
    if CACHE.exists():
        d = json.load(open(CACHE, encoding="utf-8"))
        log(f"首词表缓存 {len(d)} 词 ← {CACHE.name}")
        return Counter(d)
    cnt = Counter()
    arts = no_word = 0
    with open(SEG, encoding="utf-8") as fh:
        for line in fh:
            arts += 1
            try:
                toks = json.loads(line)
            except ValueError:
                continue
            start = 0
            if toks and toks[0] == "〖":
                try:
                    start = toks.index("〗") + 1
                except ValueError:
                    start = 1
            for t in toks[start:]:
                if isinstance(t, str) and t and all("一" <= c <= "鿿" for c in t):
                    cnt[t] += 1
                    break
                if isinstance(t, str) and any(c in t for c in "。！？"):
                    break
            else:
                no_word += 1
    log(f"扫描 {arts} 条，无首词 {no_word}，唯一首词 {len(cnt)}")
    top = dict(cnt.most_common(TOP_N))
    CACHE.write_text(json.dumps(top, ensure_ascii=False), encoding="utf-8")
    log(f"Top{TOP_N} → {CACHE.name}（条目覆盖 "
        f"{100*sum(top.values())/sum(cnt.values()):.1f}%）")
    return Counter(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    cnt = extract()
    from pypinyin import lazy_pinyin
    words, codes = [], []
    seen = set()
    for w, c in cnt.most_common():
        code = "".join(lazy_pinyin(w))
        if not code.isascii() or not code.isalpha() or code in seen:
            continue
        seen.add(code)
        words.append((w, c))
        codes.append(code)
    log(f"唯一首词 {len(words)}（拼音码去重后）")

    probe_work = HERE / "n0_firstword_probe"
    cands_all = L.run_probe(codes, work=probe_work)
    pop = []                      # (w, cnt, code)
    for (w, c), code in zip(words, codes):
        texts = [x[0] for x in cands_all.get(code, [])[:L.K_COLLECT]]
        if len(texts) < 2 or w not in texts:
            continue              # 真词不在池（生产语义：不可提交）
        pop.append((w, c, code))
    log(f"有效窗 {len(pop)} / {len(words)}（真词不在池或池<2 剔除）")

    # ── 裸「。」CE 缺口 ──
    bak = {**L.load_ce(L.CE_RESULTS_EXT.with_suffix(".jsonl.bak_preanchor")),
           **L.load_ce(L.CE_RESULTS.with_suffix(".jsonl.bak_preanchor"))}
    have = {**bak, **L.load_ce(SCRATCH / "results.jsonl")}
    todo = []
    for w, c, code in pop:
        texts = [x[0] for x in cands_all[code][:L.K_COLLECT]]
        todo += [(ln, sid) for ln, sid in L.ce_lines("。", texts[:10])
                 if sid not in have]
    log(f"裸「。」需补算 {len(todo)} 行（去重 sid {len({s for _, s in todo})}）")
    if a.apply and todo:
        import os
        os.environ["SIM_CTX_ANCHOR"] = "0"
        SCRATCH.mkdir(parents=True, exist_ok=True)
        L.run_ce(todo, workers=8, out_dir=SCRATCH)
        log("补算完成")
    punct = {**bak, **L.load_ce(SCRATCH / "results.jsonl")}
    if a.apply and todo:
        want = sum(len(L.ce_lines("。", [x[0] for x in cands_all[cd][:L.K_COLLECT]][:10]))
                   for _, _, cd in pop)
        got = sum(1 for w, c, code in pop
                  for ln, sid in L.ce_lines("。", [x[0] for x in cands_all[code][:L.K_COLLECT]][:10])
                  if sid in punct)
        log(f"覆盖核验 {got}/{want}")
        if got != want:
            log("FATAL: 覆盖不全，断点续跑重试 --apply")
            return 1
    if not a.apply:
        log("dry-run（--apply 探针+补算+评测）")
        return 0

    # ── 三臂评测（A 引擎原序 / A2·C2 整句显示序 span 分组）──
    stat = defaultdict(lambda: dict(n=0, okA=0, okA2=0, okC2=0, okC1=0,
                                    arts=0, okA2_w=0, okC2_w=0))
    rec = defaultdict(list)
    miss2 = defaultdict(int)
    for w, c, code in pop:
        sl = "1字" if len(w) == 1 else "2字" if len(w) == 2 else "3+字"
        entries = cands_all[code][:L.K_COLLECT]
        texts = [e[0] for e in entries]
        ends = [e[2] for e in entries]
        lns = L.ce_lines("。", texts[:10])
        rows = {}
        for ln, sid in lns:
            rr = punct.get(sid)
            if rr:
                for t in ln["cands"]:
                    if t in rr and rr[t][0] is not None:
                        rows[t] = rr[t]
        f10 = {t: fused_of(rows.get(t)) for t in texts[:10]}
        if any(v is None for v in f10.values()):
            miss2[f"{sl}_ce_missing"] += 1
            continue
        min_valid = min(f10.values())

        def ce_of(i):
            return f10[texts[i]] if i < 10 else min_valid

        # A：引擎原序（词模式 freq_only 口径）
        okA = texts[0] == w
        # 整句显示序：span 降序分组，取最长 span 组（组内序分臂）
        groups = defaultdict(list)
        for i, e_ in enumerate(ends):
            groups[e_].append(i)          # 引擎序（稳定）
        top_span = max(groups)
        grp = groups[top_span]
        okA2 = texts[grp[0]] == w
        # C2：组内 CE 加权和序（并列保引擎序）
        best = grp[0]
        for i in grp[1:]:
            if ce_of(i) > ce_of(best):
                best = i
        okC2 = texts[best] == w
        # 反面教材：纯 CE 全池序（无 span 分组）
        b1 = 0
        for i in range(1, len(texts)):
            if ce_of(i) > ce_of(b1):
                b1 = i
        okC1 = texts[b1] == w

        e = stat[sl]
        e["n"] += 1
        e["okA"] += okA
        e["okA2"] += okA2
        e["okC2"] += okC2
        e["okC1"] += okC1
        e["arts"] += c
        e["okA2_w"] += c * okA2
        e["okC2_w"] += c * okC2
        rec[sl].append((okA2, okC2, okA, okC1))

    # ── 汇总 ──
    out = []

    def emit(x=""):
        print(x, flush=True)
        out.append(x)

    emit("wiki 分条正文首句首词 · 真 n=0 选重率（Top10000 唯一首词；真词∈池；冷账本）")
    emit("评测中剔除（CE 缺值）：" +
         ("；".join(f"{k}={v}" for k, v in sorted(miss2.items())) or "0"))
    emit("")
    hdr = (f"{'词长':<6}{'词N':>7}{'条目覆盖':>9}{'引擎原序':>10}{'整句显示序':>10}"
           f"{'锚·整句显示序':>12}{'Δ(A2→C2)':>10}  95%CI              翻转      "
           f"{'纯CE序(反面)':>12}")
    emit(hdr)
    sums = defaultdict(int)

    def fmt_row(name, e, pa=None, pb=None, extra=""):
        n = e["n"]
        r = (f"{name:<6}{n:>7}{100.0*e['arts']/sum(x['arts'] for x in stat.values()):>8.1f}%"
             f"{100.0*(n-e['okA'])/n:>10.2f}"
             f"{100.0*(n-e['okA2'])/n:>10.2f}"
             f"{100.0*(n-e['okC2'])/n:>12.2f}")
        if pa:
            lo, hi = boot_ci(pa, pb)
            b = sum(1 for x, y in zip(pa, pb) if not x and y)
            cc = sum(1 for x, y in zip(pa, pb) if x and not y)
            d = 100.0 * (sum(pb) - sum(pa)) / len(pa)
            r += f"{100.0*(sum(pa)-sum(pb))/len(pa):>+10.2f}  [{-(hi):+6.2f},{-(lo):+6.2f}]  {b:>4}/{cc:<4}"
        return r + extra

    for sl in ("1字", "2字", "3+字"):
        e = stat.get(sl)
        if not e or not e["n"]:
            continue
        pa = [r[0] for r in rec[sl]]
        pb = [r[1] for r in rec[sl]]
        b = sum(1 for x, y in zip(pa, pb) if not x and y)
        cc = sum(1 for x, y in zip(pa, pb) if x and not y)
        emit(fmt_row(sl, e, pa, pb,
                     f"  {100.0*(e['n']-e['okC1'])/e['n']:>11.2f}"
                     f"  p={mcnemar_p(b, cc):.1e}"))
    N = sum(e["n"] for e in stat.values())
    allE = defaultdict(int)
    for e in stat.values():
        for k in ("n", "okA", "okA2", "okC2", "okC1", "arts", "okA2_w", "okC2_w"):
            allE[k] += e[k]
    allpa = [r[0] for rs in rec.values() for r in rs]
    allpb = [r[1] for rs in rec.values() for r in rs]
    b = sum(1 for x, y in zip(allpa, allpb) if not x and y)
    cc = sum(1 for x, y in zip(allpa, allpb) if x and not y)
    emit(fmt_row("全体", allE, allpa, allpb,
                 f"  {100.0*(allE['n']-allE['okC1'])/allE['n']:>11.2f}"
                 f"  p={mcnemar_p(b, cc):.1e}"))
    AW, CW = allE["okA2_w"], allE["okC2_w"]
    ART = allE["arts"]
    emit(f"{'加权':<6}{ART:>7}{'':>9}{'':>10}{'':>10}{'':>12}"
         f"  （按条目频次加权：整句显示序选重率 "
         f"{100.0*(ART-AW)/ART:.2f}% → 锚 {100.0*(ART-CW)/ART:.2f}%，"
         f"Δ {100.0*(AW-CW)/ART:+.2f}pp）")
    emit("")
    emit("列说明：引擎原序 = 词模式 freq_only 口径；整句显示序 = span 降序分组"
         "（生产显示序①，组内引擎序）；锚·整句显示序 = 组内改 CE 加权和序"
         "（显示序③的 CE 接管）；Δ/CI 为选重率变化（负 = 锚改善）；"
         "纯CE序 = 无 span 分组的全局 CE 序——显示序不存在这种比较，"
         "其崩溃（首版观察）= 1-token 候选只付一笔 token 钱的长度偏差。")
    (HERE / "n0_firstword.txt").write_text("\n".join(out) + "\n",
                                           encoding="utf-8")
    log("报告 → n0_firstword.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
