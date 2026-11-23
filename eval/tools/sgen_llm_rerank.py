#!/usr/bin/env python3
"""sgen_llm_rerank.py — N-gram 组句候选 → LLM 句级似然重排（第二段实验）

输入 = sentence_gen_gram.py 的输出 JSONL（每句带 top-k 候选与 before 上文）。
对每个候选算 Qwen3-0.6B 的整句对数似然（cpp/sent_score.exe，完整候选 token
的 CE 和），按 sum_ce 升序重排，测：
  · ceiling = 真值在候选集里（hit@k，N-gram 生成的上限）
  · llm_hit@1 = 重排后真值排第一
  · ngram_hit@1 = 重排前（N-gram 分）真值排第一
  · 修复/破坏 = LLM 重排把 N-gram 首选改对/改错的数量
另报按字符数归一（sum_ce / chars）与按 token 平均（mean_ce）两种口径对照。

用法：
  python sgen_llm_rerank.py --jobs eval/sgen_gram_100.test.jsonl \
      --exe cpp/sent_score.exe [--topk 5] [--model path.gguf] \
      [--out eval/sgen_llm_100]
"""
import argparse
import json
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def run_scorer(exe, model, jobs):
    """jobs = [(sid, idx, ctx, cand)] → {(sid, idx): (sum, mean, ntok)}。"""
    payload = "".join("%d\t%d\t%s\t%s\n" % (sid, idx, ctx, cand)
                      for sid, idx, ctx, cand in jobs)
    cmd = [exe] + ([model] if model else [])
    p = subprocess.run(cmd, input=payload.encode("utf-8"),
                       capture_output=True, timeout=7200)
    if p.returncode != 0:
        sys.stderr.write("scorer rc=%d\n%s\n" %
                         (p.returncode, p.stderr.decode("utf-8", "replace")[-1500:]))
        raise SystemExit(1)
    out = {}
    for ln in p.stdout.decode("utf-8", "replace").splitlines():
        f = ln.split("\t")
        if len(f) < 5:
            continue
        sid, idx = int(f[0]), int(f[1])
        out[(sid, idx)] = (float(f[2]), float(f[3]), int(f[4]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True,
                    help="sentence_gen_gram.py 的 .test.jsonl")
    ap.add_argument("--exe", default=os.path.join(PROJ, "cpp", "sent_score.exe"))
    ap.add_argument("--model", default=None,
                    help="GGUF 路径（缺省 = sent_score.exe 内置默认）")
    ap.add_argument("--topk", type=int, default=5,
                    help="参与重排的候选数（= 生成器 top-k）")
    ap.add_argument("--out", default=os.path.join(PROJ, "eval", "sgen_llm"))
    a = ap.parse_args()

    rows = [json.loads(ln) for ln in open(a.jobs, encoding="utf-8")]
    jobs = []
    for r in rows:
        for i, c in enumerate(r["cands"][:a.topk]):
            jobs.append((r["sid"], i, r["before"], c["text"]))
    print("scoring %d candidates (%d sentences) ..." % (len(jobs), len(rows)),
          flush=True)
    scores = run_scorer(a.exe, a.model, jobs)
    print("scored %d" % len(scores), flush=True)

    def pick(r, key):
        cands = r["cands"][:a.topk]
        vals = []
        for i, c in enumerate(cands):
            s = scores.get((r["sid"], i))
            if s is None:
                continue
            vals.append((key(s, c), i))
        if not vals:
            return None
        vals.sort(key=lambda x: x[0])
        return cands[vals[0][1]]["text"]

    def summarize(name, key):
        n = len(rows)
        ceil = sum(1 for r in rows if r["truth"] in
                   [c["text"] for c in r["cands"][:a.topk]])
        llm1 = sum(1 for r in rows if pick(r, key) == r["truth"])
        ng1 = sum(1 for r in rows if 0 <= r["rank"] < 1)
        fixed = broken = 0
        for r in rows:
            ng_top = r["cands"][0]["text"] if r["cands"] else None
            new_top = pick(r, key)
            if new_top == r["truth"] and ng_top != r["truth"]:
                fixed += 1
            elif new_top != r["truth"] and ng_top == r["truth"]:
                broken += 1
        print("%-10s n=%d ceiling(hit@%d)=%d(%.1f%%)  ngram_hit@1=%d(%.1f%%)"
              "  llm_hit@1=%d(%.1f%%)  fixed=%d broken=%d"
              % (name, n, a.topk, ceil, 100.0 * ceil / n, ng1,
                 100.0 * ng1 / n, llm1, 100.0 * llm1 / n, fixed, broken))
        return ceil, llm1

    print("\n=== 重排口径对照 ===")
    summarize("sum_ce", lambda s, c: s[0])
    summarize("sum/char", lambda s, c: s[0] / max(len(c["text"]), 1))
    summarize("mean_ce", lambda s, c: s[1])

    # 词表内子集（组句器可以完整拼出的句子，生产主域）
    inv = [r for r in rows if r["all_in_vocab"]]
    if inv and len(inv) != len(rows):
        rows_all = rows
        rows = inv
        print("\n=== 词表内子集 (n=%d) ===" % len(inv))
        summarize("sum_ce", lambda s, c: s[0])
        rows = rows_all

    # 明细
    base = os.path.abspath(a.out)
    with open(base + ".rerank.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            cands = []
            for i, c in enumerate(r["cands"][:a.topk]):
                s = scores.get((r["sid"], i))
                cands.append({"text": c["text"], "ngram_score": c["score"],
                              "sum_ce": s[0] if s else None,
                              "mean_ce": s[1] if s else None})
            cands.sort(key=lambda x: (x["sum_ce"] if x["sum_ce"] is not None
                                      else 1e30))
            out = {"sid": r["sid"], "truth": r["truth"],
                   "before": r["before"], "all_in_vocab": r["all_in_vocab"],
                   "ngram_rank": r["rank"],
                   "llm_rank": next((k for k, c in enumerate(cands)
                                     if c["text"] == r["truth"]), -1),
                   "cands": cands}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("\nout: %s.rerank.jsonl" % base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
