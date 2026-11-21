#!/usr/bin/env python3
"""cohort 队列模拟的 DP 先验：全文语料 unigram（2026-11-07）。

与旧链（前缀 n 篇 unigram）不同, cohort 各年龄档共用一个**固定先验**
（≈ 生产词典先验的地位, 不随档位/用户变化, 消除"DP 先验随重放长度变"
的附带失真）。只统计 vocab 内词的计数 + 全语料 CJK token 总数。

用法: python cohort_unigram.py --dict small6w|big [--out PATH]
产物: cohort_unigram_<dict>.json = {"total": N, "logp": {w: log p}, "unseen": ...}
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

PROJ = corpus_std.PROJ
SHUFFLED = PROJ / "article_corpus_shuffled.txt"
CACHE = PROJ / "segments_hanlp_cache.jsonl"
SMALL_DICT = PROJ / "词库-测评6万" / "pdsp_dict_small6w.yaml"
IDEO_RE = re.compile(r"[一-鿿]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", choices=["small6w", "big", "r2"], default="small6w")
    a = ap.parse_args()
    out = corpus_std.EVAL / f"cohort_unigram_{a.dict}.json"

    vocab = set(corpus_std.load_pdsp_words(
        SMALL_DICT if a.dict == "small6w" else None))
    if a.dict == "small6w":
        big_words = corpus_std.load_pdsp_words()
        vocab |= {w for w in big_words if len(w) == 1 and IDEO_RE.search(w)}
    print(f"vocab {len(vocab)}", flush=True)

    freq = Counter()
    total = 0
    with open(CACHE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            for t in json.loads(line):
                if IDEO_RE.search(t):
                    total += 1
                    if t in vocab:
                        freq[t] += 1
            if (i + 1) % 200000 == 0:
                print(f"  {i + 1} articles, ideo {total}", flush=True)
    payload = {
        "total": total,
        "unseen_logp": math.log(1.0 / total),
        "logp": {w: math.log(c / total) for w, c in freq.items()},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"total {total}, covered types {len(freq)}/{len(vocab)} -> {out}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
