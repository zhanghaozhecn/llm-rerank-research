#!/usr/bin/env python3
"""
一次性构建（2026-11-05 新口径, 详见 口径标准-corpus-standards.md）:
  1) 打乱语料: 全部通过 ASCII≤10% 过滤的文章**整体随机打乱**（seed 固定）
     → article_corpus_shuffled.txt；打字模拟按该文件顺序取前缀（快检取前 1%/10%）
  2) HanLP 分段缓存: 前 N 篇的自然切分结果 → segments_hanlp_cache.jsonl
     （增量: 已就绪前缀复用, 只补缺; 行号与打乱语料一一对齐; **方案无关**——
      词典相关的 OOV 拆词在重放时实时做, 见 prep_samples_std.py）

用法: python eval/build_segment_cache.py [--articles N | --prefix F] [--rebuild]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

PROJ = corpus_std.PROJ
SHUFFLED = PROJ / "article_corpus_shuffled.txt"
CACHE = PROJ / "segments_hanlp_cache.jsonl"
COUNT_SIDECAR = PROJ / "article_corpus_shuffled.count"


def shuffled_count():
    if COUNT_SIDECAR.exists():
        return int(COUNT_SIDECAR.read_text().strip())
    n = 0
    with open(SHUFFLED, encoding="utf-8") as f:
        for _ in f:
            n += 1
    COUNT_SIDECAR.write_text(str(n))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=None, help="缓存前 N 篇")
    ap.add_argument("--prefix", type=float, default=None,
                    help="缓存前 F 比例（0.01=快检, 0.1=中档, 1.0=全量）")
    ap.add_argument("--rebuild", action="store_true", help="强制重打乱（慎用）")
    a = ap.parse_args()

    if not SHUFFLED.exists() or a.rebuild:
        print(f"shuffling whole corpus ({corpus_std.WIKI_ARTICLES}) ...")
        n_total = corpus_std.build_shuffled_corpus(corpus_std.WIKI_ARTICLES,
                                                   SHUFFLED)
        COUNT_SIDECAR.write_text(str(n_total))
        print(f"  shuffled {n_total} articles → {SHUFFLED.name}")
    else:
        n_total = shuffled_count()
        print(f"shuffled corpus exists: {n_total} articles")

    if a.articles is None and a.prefix is None:
        return
    n = a.articles if a.articles is not None else max(1, int(n_total * a.prefix))
    done = 0
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            done = sum(1 for _ in f)
    print(f"segment cache: ready {done}, target {n}")
    if done < n:
        corpus_std.ensure_segment_cache(SHUFFLED, CACHE, n)
        print("done")
    else:
        print("already complete")


if __name__ == "__main__":
    main()
