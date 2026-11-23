#!/usr/bin/env python3
"""sentence_gen_gram.py — 组句候选生成 · 万象 LTS gram 词级 beam（第一版）

研究问题（2026-11-23 用户指示）：整句模式 stage ② 的句候选来自引擎 poet
（max_sentences=1 → 通常只有 1 个整句）。若改用**外部 N-gram 模型**（万象
LTS gram，librime-octagram 格式）做词级 beam 搜索生成 top-5 句子候选，
真值句能否落在这 5 个里？命中率达标后再接 LLM 句级评分。

与 Rime poet 同构的计分（`grammar.h` Grammar::Evaluate）：
  edge_score = [log(词典频次) − log(1e8)] + gram.Query(上文, 词, is_rear)
  · 词表 = 万象 zi/jichu/duoyin/diming/mingren/yiren（读音去声调、ü→v）
  · gram = 万象 LTS 模型（Rime::DartsGrammar/3，纯 Python 读取见 eval/lib/wxgram.py）
  · 上文 = 前句尾部 ≤N 字 + 本句已生成文字；is_rear = 该词结束整句（poet 语义）

流程：
  1) 词表：词 → 读音（含词频）；码反查表 code2words。
  2) 测试句：wiki 分段缓存抽样 3..12 词 CJK 句；码 = 逐词读音（唯一读音优先，
     pypinyin 常用读法在其读音集合内时优先；否则最高词频读音）→ 真值路径总在格中。
  3) 格 + beam：码串每个起点枚举所有以该码开头的词；逐位置保留 B 个状态
     （按输出文本去重），末尾取 top-5。
  4) 命中率：候选文本 == 真值文本（汉字级、不要求同分词）；分母同时报
     ① 词级可得（全部真值词在词表）② 格内可达（真值文本可由词表词拼出，DP 判定）。

用法：
  python sentence_gen_gram.py --gram <model.gram> --dict-dir <wanxiang dicts dir> \
      [--test-skip 400000] [--sentences 200] [--beam 64]
"""
import argparse
import json
import math
import os
import re
import sys
import time

from pypinyin import lazy_pinyin

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # 研究库根（eval/tools/ 上三级）
EVAL = os.path.join(PROJ, "eval")
sys.path.insert(0, os.path.join(EVAL, "lib"))
from wxgram import WanxiangGram, normalize_pinyin  # noqa: E402

CJK = re.compile(r"[\u3400-\u9fff]")
DICT_FILES = ("zi", "jichu", "duoyin", "diming", "mingren", "yiren")
K_LOG_SCALE = 18.420680743952367  # log(1e8)：librime 词典权重归一常数


# ── 词表 ────────────────────────────────────────────────────────────────
def load_lexicon(dict_dir):
    """万象词库 → {词: [(码, 词频)]}（同码取最大词频；读音去声调、ü→v）。"""
    readings = {}
    for name in DICT_FILES:
        path = os.path.join(dict_dir, name + ".dict.yaml")
        if not os.path.exists(path):
            continue
        started = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not started:
                    if line.strip() == "...":
                        started = True
                    continue
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2:
                    continue
                w = parts[0]
                code = normalize_pinyin(parts[1])
                if not w or not code or not code.isascii() or not code.isalpha():
                    continue
                try:
                    wt = float(parts[2]) if len(parts) > 2 else 1.0
                except ValueError:
                    wt = 1.0
                if wt <= 0:
                    wt = 1.0
                readings.setdefault(w, []).append((code, wt))
    # 同 (词, 码) 去重取最大词频
    out = {}
    for w, rs in readings.items():
        best = {}
        for code, wt in rs:
            if wt > best.get(code, 0.0):
                best[code] = wt
        out[w] = [(c, wt) for c, wt in best.items()]
    return out


def build_code_index(readings):
    code2words = {}
    for w, rs in readings.items():
        for code, wt in rs:
            code2words.setdefault(code, []).append((w, wt))
    return code2words


def edge_score(wt):
    return math.log(max(wt, 1e-6)) - K_LOG_SCALE


# ── 格与 beam ──────────────────────────────────────────────────────────
def build_lattice(code, code2words, min_weight=10.0):
    """edges[i] = [(j, word, weight)]，word 的读音 == code[i:j]。
    min_weight：词频下限（默认 10；本数据集上不影响真值词存活，见 README）。
    词频越低的词先验分越低（log(wt)−18.4），几乎不可能进 top-5；
    截断只影响 beam 的算力，不改变候选质量口径。"""
    n = len(code)
    edges = []
    for i in range(n):
        cur = []
        for j in range(i + 1, n + 1):
            ws = code2words.get(code[i:j])
            if not ws:
                continue
            for w, wt in ws:
                if wt < min_weight:
                    continue
                cur.append((j, w, edge_score(wt)))
        edges.append(cur)
    return edges


class GramScorer:
    """gram.Query + (上下文尾部, 词, is_rear) 记忆化（上下文只影响末 5 字）。"""

    def __init__(self, gram, max_len=6):
        self.gram = gram
        self.n = max_len - 1
        self.cache = {}

    def score(self, context, word, is_rear=False):
        tail = context[-self.n:] if self.n > 0 else ""
        key = (tail, word, is_rear)
        v = self.cache.get(key)
        if v is None:
            v = self.gram.query(context, word, is_rear)
            self.cache[key] = v
        return v


def beam_decode(code, edges, scorer, prev_tail, beam=64, topk=5):
    """states[pos] = [(score, out_text)]；文本去重、逐位置截断 beam。"""
    n = len(code)
    states = [[] for _ in range(n + 1)]
    states[0] = [(0.0, "")]
    for i in range(n):
        cur = states[i]
        if not cur:
            continue
        best = {}
        for sc, txt in cur:
            old = best.get(txt)
            if old is None or sc > old:
                best[txt] = sc
        cur = sorted(best.items(), key=lambda kv: -kv[1])[:beam]
        for txt, sc in cur:
            ctx = prev_tail + txt
            for (j, w, wscore) in edges[i]:
                s = sc + wscore + scorer.score(ctx, w, j == n)
                states[j].append((s, txt + w))
    out = {}
    for sc, txt in states[n]:
        if txt not in out or sc > out[txt]:
            out[txt] = sc
    ranked = sorted(out.items(), key=lambda kv: -kv[1])
    return [(t, s) for t, s in ranked[:topk]]


def truth_reachable(code, edges, truth):
    """DP：码串能否由格内词拼出与 truth 完全相同的汉字串。"""
    n = len(code)
    states = [set() for _ in range(n + 1)]
    states[0].add(0)
    lt = len(truth)
    for i in range(n):
        if not states[i]:
            continue
        for ti in states[i]:
            for (j, w, ws) in edges[i]:
                if ti + len(w) <= lt and truth.startswith(w, ti):
                    states[j].add(ti + len(w))
    return lt in states[n]


# ── 数据集 ─────────────────────────────────────────────────────────────
def pick_sentences(path, skip, articles, min_words, max_words, limit,
                   ctx_chars=10):
    """取 3..max_words 词 CJK 句；同时给句前上文尾部（同文章内 ≤ctx_chars 字）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= skip + articles:
                break
            if idx < skip:
                continue
            try:
                toks = json.loads(line)
            except json.JSONDecodeError:
                continue
            run = []
            before = ""
            for t in toks:
                cjk = (isinstance(t, str) and t
                       and all(CJK.match(ch) for ch in t))
                if cjk:
                    if not run:
                        before = before[-ctx_chars:]
                    run.append(t)
                    continue
                if run:
                    if min_words <= len(run) <= max_words:
                        out.append((run, before[-ctx_chars:]))
                        if len(out) >= limit:
                            return out
                    before = (before + "".join(run))[-ctx_chars:]
                    run = []
            if run and min_words <= len(run) <= max_words:
                out.append((run, before[-ctx_chars:]))
                if len(out) >= limit:
                    return out
    return out


def encode_sentence(words, readings):
    """词 → (码串, 每词码, 是否全部词在词表)。
    唯一读音优先；pypinyin 读音在集合内则用它（用户常用读法）；否则最高词频读音。"""
    codes = []
    all_in = True
    for w in words:
        rs = readings.get(w)
        if not rs:
            all_in = False
            c = "".join(lazy_pinyin(w))
            if not (c and c.isascii() and c.isalpha()):
                return None
            codes.append(c)
            continue
        if len(rs) == 1:
            codes.append(rs[0][0])
            continue
        py = "".join(lazy_pinyin(w))
        cand = {c for c, wt in rs}
        if py in cand:
            codes.append(py)
        else:
            codes.append(max(rs, key=lambda x: x[1])[0])
    return "".join(codes), codes, all_in


def prep_sentences(sents, readings, code2words, min_weight=10.0):
    items = []
    for si, (words, before) in enumerate(sents):
        enc = encode_sentence(words, readings)
        if enc is None:
            continue
        code, codes, all_in = enc
        truth = "".join(words)
        edges = build_lattice(code, code2words, min_weight)
        items.append({
            "sid": si, "words": words, "codes": codes, "code": code,
            "truth": truth, "all_in_vocab": all_in, "before": before,
            "reachable": truth_reachable(code, edges, truth), "edges": edges,
        })
    return items


def decode_items(items, scorer, beam, topk, ctx_tail=10, progress=0):
    rows = []
    t0 = time.time()
    for k, it in enumerate(items):
        cands = beam_decode(it["code"], it["edges"], scorer,
                            it["before"][-ctx_tail:], beam=beam, topk=topk) \
            if it["reachable"] else []
        rank = -1
        for r, (txt, sc) in enumerate(cands):
            if txt == it["truth"]:
                rank = r
                break
        rows.append({
            "sid": it["sid"], "words": it["words"], "codes": it["codes"],
            "code": it["code"], "truth": it["truth"], "before": it["before"],
            "all_in_vocab": it["all_in_vocab"], "reachable": it["reachable"],
            "rank": rank, "cands": [{"text": t, "score": s} for t, s in cands],
            "chars": len(it["truth"]), "n_words": len(it["words"]),
        })
        if progress and (k + 1) % progress == 0:
            print("  %d/%d %.1fs" % (k + 1, len(items), time.time() - t0),
                  flush=True)
    return rows


def summarize(rows, topk):
    def hit(r, k):
        return 0 <= r["rank"] < k

    feas = [r for r in rows if r["reachable"]]
    return {
        "n": len(rows),
        "in_vocab": sum(1 for r in rows if r["all_in_vocab"]),
        "reachable": len(feas),
        "hit1": sum(1 for r in rows if hit(r, 1)),
        "hit5": sum(1 for r in rows if hit(r, 5)),
        "hit10": sum(1 for r in rows if hit(r, 10)),
        "hitk": sum(1 for r in rows if hit(r, topk)),
        "feas_hit1": sum(1 for r in feas if hit(r, 1)),
        "feas_hit5": sum(1 for r in feas if hit(r, 5)),
        "feas_hit10": sum(1 for r in feas if hit(r, 10)),
        "feas_hitk": sum(1 for r in feas if hit(r, topk)),
    }


def fmt_summary(s, topk):
    n = max(s["n"], 1)
    f = max(s["reachable"], 1)
    return ("sentences=%d  in_vocab=%d(%.1f%%)  reachable=%d(%.1f%%)\n"
            "hit@1=%d(%.1f%%)  hit@5=%d(%.1f%%)  hit@10=%d(%.1f%%)  "
            "hit@%d=%d(%.1f%%)\n"
            "reachable-only: hit@1=%.1f%%  hit@5=%.1f%%  hit@10=%.1f%%  "
            "hit@%d=%.1f%%"
            % (s["n"], s["in_vocab"], 100.0 * s["in_vocab"] / n,
               s["reachable"], 100.0 * s["reachable"] / n,
               s["hit1"], 100.0 * s["hit1"] / n,
               s["hit5"], 100.0 * s["hit5"] / n,
               s["hit10"], 100.0 * s["hit10"] / n,
               topk, s["hitk"], 100.0 * s["hitk"] / n,
               100.0 * s["feas_hit1"] / f, 100.0 * s["feas_hit5"] / f,
               100.0 * s["feas_hit10"] / f, topk,
               100.0 * s["feas_hitk"] / f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gram", required=True, help="万象 LTS .gram 模型")
    ap.add_argument("--dict-dir", required=True,
                    help="万象 dicts 目录（zi/jichu/... .dict.yaml）")
    ap.add_argument("--segments", default=os.path.join(
        PROJ, "segments_hanlp_cache.jsonl"))
    ap.add_argument("--test-skip", type=int, default=400000)
    ap.add_argument("--test-articles", type=int, default=2000)
    ap.add_argument("--sentences", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-weight", type=float, default=10.0,
                    help="格边词频下限（默认 10；0 = 全量）")
    ap.add_argument("--ctx-tail", type=int, default=10,
                    help="句前上文尾部字数（poet preceding_text 语义）")
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--coll-penalty", type=float, default=-14.0)
    ap.add_argument("--non-coll-penalty", type=float, default=-6.0)
    ap.add_argument("--weak-penalty", type=float, default=-100.0)
    ap.add_argument("--rear-penalty", type=float, default=-18.0)
    ap.add_argument("--out", default=os.path.join(EVAL, "sgen_gram"))
    a = ap.parse_args()

    t0 = time.time()
    print("[1/4] grammar model ...", flush=True)
    gram = WanxiangGram(a.gram, max_len=a.max_len, min_len=a.min_len,
                        collocation_penalty=a.coll_penalty,
                        non_collocation_penalty=a.non_coll_penalty,
                        weak_collocation_penalty=a.weak_penalty,
                        rear_penalty=a.rear_penalty)
    print("  sections=%s  %.1fs"
          % ([(s[0], s[3]) for s in gram.sections], time.time() - t0),
          flush=True)

    print("[2/4] lexicon ...", flush=True)
    readings = load_lexicon(a.dict_dir)
    code2words = build_code_index(readings)
    print("  words=%d codes=%d  %.1fs"
          % (len(readings), len(code2words), time.time() - t0), flush=True)

    print("[3/4] sentences ...", flush=True)
    sents = pick_sentences(a.segments, a.test_skip, a.test_articles,
                           a.min_words, a.max_words, a.sentences,
                           ctx_chars=a.ctx_tail)
    items = prep_sentences(sents, readings, code2words, a.min_weight)
    scorer = GramScorer(gram, max_len=a.max_len)
    print("  sentences=%d (encoded %d)  %.1fs"
          % (len(sents), len(items), time.time() - t0), flush=True)

    print("[4/4] beam decode ...", flush=True)
    rows = decode_items(items, scorer, a.beam, a.topk, a.ctx_tail,
                        progress=5)
    s = summarize(rows, a.topk)
    print(fmt_summary(s, a.topk))

    print("\nby length (words):")
    for lo, hi in ((3, 5), (6, 8), (9, 12)):
        sub = [r for r in rows if lo <= r["n_words"] <= hi]
        if not sub:
            continue
        ss = summarize(sub, a.topk)
        print("  %d-%d: n=%d reachable=%.0f%% hit@%d=%.1f%% (reachable %.1f%%)"
              % (lo, hi, ss["n"], 100.0 * ss["reachable"] / max(ss["n"], 1),
                 a.topk, 100.0 * ss["hit5"] / max(ss["n"], 1),
                 100.0 * ss["feas_hit5"] / max(ss["reachable"], 1)))
    miss = [r for r in rows if r["reachable"] and not (0 <= r["rank"] < a.topk)]
    print("\nreachable misses: %d" % len(miss))
    for r in miss[:8]:
        print("  truth=%s" % r["truth"])
        for c in r["cands"][:5]:
            print("    %.2f %s" % (c["score"], c["text"]))

    base = os.path.abspath(a.out)
    with open(base + ".test.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(base + ".report.txt", "w", encoding="utf-8") as f:
        f.write("gram=%s dict=%s beam=%d topk=%d ctx_tail=%d params=%s\n"
                % (a.gram, a.dict_dir, a.beam, a.topk, a.ctx_tail,
                   [a.max_len, a.min_len, a.coll_penalty, a.non_coll_penalty,
                    a.weak_penalty, a.rear_penalty]))
        f.write(fmt_summary(s, a.topk) + "\n")
        f.write("elapsed %.1fs\n" % (time.time() - t0))
    print("\nout: %s.test.jsonl / .report.txt  (%.1fs)"
          % (base, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
