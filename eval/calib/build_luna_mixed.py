#!/usr/bin/env python3
"""luna 混合窗 cohort 构建（2026-11-24；取代 71 窗真实语料版）。

研究问题：统一准入后窗口里「满覆盖句 / 等长词 / 子词」同池比较，现行
-(1.0·ce1 + 1.10·ce2 + 0.60·ce3) 随 token 数变负；词域 cohort 全是等长窗，
回答不了「子词 vs 整段谁该在前」。本脚本用真实 luna 引擎（无头探针，带 span）
重建混合窗，标签 = 句子真值的前缀。

窗口 = (m, L)：句首 m 个字已确认（进上文）、其后音节已打 L 个字母。
  ctx  = before + 真值[:m]
  code = 真值[m:] 音节串的前 L 个字母
  候选 c 正确 ⇔ c.text 是 真值[m:] 的前缀；理想候选 = 其中最长者。
冷口径：userdb 全新（无用户词典联想）、LLM 账本为空（层1 序 = 引擎序——
生产词频键 = (候选, 整段码)，整句模式下只有同码重现才命中，近似成立）。

子命令（产物落 eval/luna_mixed/）：
  sentences  语料 → 句子 → 逐字音节（luna 词典读音优先，OOV 回退 pypinyin）
             → sentences.jsonl
  windows    句子 → 窗口网格；静态 cohort 每句抽 per_sent 格，dyn 子集整格
             保留（供前缀确认 2×2 动态仿真）；全部唯一码串跑探针 → cands.json
  ce_input   静态窗引擎池前 16 条 → sim_rerank 输入（≤10 候选/行）→ ce_input.jsonl
  gen        静态窗（≥4 码）N-gram 组句（生产参数，插件 DLL）→ gen.tsv
  genfull    5 条组句 lead 不一致的窗补全句 CE（生产 score_full）→ genfull.tsv
  assemble   合并窗口 / CE / 组句 → cohort.jsonl（分析脚本的唯一输入）

用法（环境变量 GGUF_MODEL_06B 可覆盖模型路径，默认 Qwen3-0.6B-Base-Q4_K_M）:
  python eval/calib/build_luna_mixed.py sentences --per-domain 1500 --dyn-per-domain 150
  python eval/calib/build_luna_mixed.py windows --per-sent 8
  python eval/calib/build_luna_mixed.py ce_input
  python eval/ce/run_ce_cohort.py --input eval/luna_mixed/ce_input.jsonl \
      --out eval/luna_mixed/ce --workers 3 --ctx 20 --model <gguf>
  python eval/calib/build_luna_mixed.py gen
  python eval/calib/build_luna_mixed.py genfull
  python eval/calib/build_luna_mixed.py assemble
"""
import argparse
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "tools"))
import luna_mixed_lib as L  # noqa: E402
import corpus_std  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CJK = re.compile(r"[一-鿿]")
MIN_CHARS, MAX_CHARS = 4, 16
CTX_CHARS = 20
SEED = 20261124


def log(msg):
    print(f"[luna_mixed {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 音节 ──────────────────────────────────────────────────────────────
def norm_py(s):
    return s.replace("ü", "v").replace("u:", "v")


class Readings:
    """引擎可见词表 = luna_pinyin.dict.yaml（带读音）∪ essay.txt（八股文词表，
    `use_preset_vocabulary: true` 编译期并入；只有词频、读音取 pypinyin）。
    只认 .dict.yaml 会把「确定/需要/上下文」这类引擎词当 OOV。"""

    def __init__(self, lazy_pinyin):
        self.py = lazy_pinyin
        self.rd = self._load_dict()
        self.essay = self._load_essay()
        self.cache = {}

    def _load_dict(self):
        path = Path(L.WEASEL_DATA) / "luna_pinyin.dict.yaml"
        rd = defaultdict(dict)
        started = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not started:
                    started = line.strip() == "..."
                    continue
                fl = line.rstrip("\n").split("\t")
                if len(fl) < 2:
                    continue
                w, syl = fl[0], fl[1].split()
                if len(w) != len(syl) or not all(s.isascii() and s.isalpha()
                                                 for s in syl):
                    continue
                try:
                    wt = float(fl[2]) if len(fl) > 2 and fl[2] else 1.0
                except ValueError:
                    wt = 1.0
                key = tuple(syl)
                rd[w][key] = max(rd[w].get(key, 0.0), wt)
        return rd

    def _load_essay(self):
        path = Path(L.WEASEL_DATA) / "essay.txt"
        out = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.split("\t", 1)[0].strip()
                if w and len(w) <= 8 and CJK.fullmatch(w):
                    out.add(w)
        return out

    def __contains__(self, w):
        return w in self.rd or w in self.essay

    def reading(self, w):
        """→ (音节 list, 是否引擎词)。词典词：唯一读音优先，多读音取 pypinyin
        常用读法（若在读音集内），否则最高词频读音。"""
        rs = self.rd.get(w)
        if rs:
            if len(rs) == 1:
                return list(next(iter(rs))), True
            py = tuple(norm_py(s) for s in self.py(w))
            pick = py if py in rs else max(rs, key=lambda k: rs[k])
            return list(pick), True
        hit = self.cache.get(w)
        if hit is None:
            s = [norm_py(x) for x in self.py(w)]
            if len(s) != len(w) or not all(x.isascii() and x.isalpha()
                                           for x in s):
                s = None
            hit = (s, w in self.essay)
            self.cache[w] = hit
        return hit


def split_user_text(text, vocab, maxlen=6):
    """无分词文本 → 最长匹配引擎词（未命中退单字）。"""
    segs, i = [], 0
    while i < len(text):
        j = min(len(text), i + maxlen)
        while j > i + 1 and text[i:j] not in vocab:
            j -= 1
        segs.append(text[i:j])
        i = j
    return segs


# ── 句子 ──────────────────────────────────────────────────────────────
def sentence_record(sid, domain, text, before, segs, vocab):
    syls, inv = [], True
    for w in segs:
        s, ok = vocab.reading(w)
        if s is None:
            return None
        syls += s
        inv = inv and ok
    return dict(sid=sid, domain=domain, text=text, before=before, segs=segs,
                syls=syls, in_vocab=inv, dyn=False)


def is_punct(tok):
    """标点/符号 token（Unicode P* / S*）。句子的左右界都必须是它——被数字、
    英文、空白切断的 CJK 段不是自然的输入单元（如「由创立于｜1901年」）。"""
    return (isinstance(tok, str) and bool(tok)
            and all(unicodedata.category(c)[0] in "PS" for c in tok))


def is_cjk_tok(tok):
    return (isinstance(tok, str) and bool(tok)
            and all(CJK.fullmatch(c) for c in tok))


def wiki_sentences(vocab, limit, skip, articles, per_article=3):
    """HanLP 分词缓存 → 两侧被标点封口的 3..12 词 CJK 句，带句前上文。"""
    path = L.PROJ / "segments_hanlp_cache.jsonl"
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip:
                continue
            if idx >= skip + articles:
                break
            try:
                toks = json.loads(line)
            except ValueError:
                continue
            taken = 0
            i = 0
            while i < len(toks) and taken < per_article:
                if not is_cjk_tok(toks[i]):
                    i += 1
                    continue
                j = i
                while j < len(toks) and is_cjk_tok(toks[j]):
                    j += 1
                words = toks[i:j]
                text = "".join(words)
                if (i > 0 and is_punct(toks[i - 1]) and j < len(toks)
                        and is_punct(toks[j]) and 3 <= len(words) <= 12
                        and MIN_CHARS <= len(text) <= MAX_CHARS
                        and text not in seen):
                    before = "".join(toks[max(0, i - 40):i])[-CTX_CHARS:]
                    rec = sentence_record("w%06d" % len(out), "wiki", text,
                                          before, words, vocab)
                    if rec:
                        seen.add(text)
                        out.append(rec)
                        taken += 1
                i = j
            if len(out) >= limit:
                break
    return out


def user_sentences(vocab, limit):
    """真实打字 scene（一次聚焦内的连续提交）按提交序拼成全文，取两侧被
    标点封口的完整 CJK 段（4..16 字）；落在全文首尾或被数字/英文切断的段
    丢弃，句前上文取段前 ≤20 字。"""
    out, seen = [], set()
    for p in corpus_std.CORPUS_PATHS:
        for sc in corpus_std.iter_scenes(str(p)):
            if len(sc) < 2:
                continue
            t = "".join(c["word"] for c in sc)
            for mm in re.finditer(r"[一-鿿]+", t):
                a, b = mm.start(), mm.end()
                s = mm.group(0)
                if (a == 0 or b >= len(t) or not is_punct(t[a - 1])
                        or not is_punct(t[b])
                        or not (MIN_CHARS <= len(s) <= MAX_CHARS)):
                    continue
                before = t[max(0, a - CTX_CHARS):a]
                if not before.strip() or s in seen:
                    continue
                rec = sentence_record("u%06d" % len(out), "user", s, before,
                                      split_user_text(s, vocab), vocab)
                if rec:
                    seen.add(s)
                    out.append(rec)
                    if len(out) >= limit:
                        return out
    return out


def cmd_sentences(a):
    from pypinyin import lazy_pinyin
    vocab = Readings(lazy_pinyin)
    log(f"luna dict words={len(vocab.rd)} essay words={len(vocab.essay)}")
    wiki = wiki_sentences(vocab, a.per_domain, a.skip, a.articles)
    log(f"wiki sentences={len(wiki)}")
    user = user_sentences(vocab, a.per_domain)
    log(f"user sentences={len(user)}")
    rng = random.Random(SEED)
    for dom, sents in (("wiki", wiki), ("user", user)):
        cands = [s for s in sents if 5 <= len(s["text"]) <= 12]
        rng.shuffle(cands)
        for s in cands[:a.dyn_per_domain]:
            s["dyn"] = True
    L.OUT.mkdir(parents=True, exist_ok=True)
    with open(L.SENTENCES, "w", encoding="utf-8") as f:
        for s in wiki + user:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    stat = Counter((s["domain"], s["in_vocab"]) for s in wiki + user)
    log(f"written {L.SENTENCES}: {dict(stat)}")
    for s in (wiki[:3] + user[:3]):
        log(f"  {s['sid']} {'/'.join(s['segs'])} → {' '.join(s['syls'])}")


# ── 窗口 ──────────────────────────────────────────────────────────────
def grid_cells(s):
    """句子的全部 (m, L, ctx, code) 格。"""
    text, syl = s["text"], s["syls"]
    cells = []
    for m in range(len(text) - 1):
        code_m = "".join(syl[m:])
        ctx = s["before"] + text[:m]
        for Lg in range(2, len(code_m) + 1):
            cells.append((m, Lg, ctx, code_m[:Lg]))
    return cells


def cmd_windows(a):
    sents = list(L.jsonl_iter(L.SENTENCES))
    rng = random.Random(SEED + 1)
    wins, by_key, grid = [], {}, {}
    for s in sents:
        cells = grid_cells(s)
        take = cells if s["dyn"] else rng.sample(cells, min(a.per_sent, len(cells)))
        g = grid.setdefault(s["sid"], {})
        for (m, Lg, ctx, code) in take:
            key = (ctx, code)
            wid = by_key.get(key)
            if wid is None:
                wid = "W%07d" % len(wins)
                by_key[key] = wid
                wins.append(dict(wid=wid, sid=s["sid"], domain=s["domain"],
                                 m=m, L=Lg, ctx=ctx, code=code,
                                 text=s["text"][m:], dyn=bool(s["dyn"])))
            g[(m, Lg)] = wid
    log(f"windows={len(wins)} (dyn {sum(1 for w in wins if w['dyn'])})")
    codes = sorted({w["code"] for w in wins})
    log(f"unique codes={len(codes)}; running engine probe ...")
    t0 = time.time()
    cands = L.run_probe(codes)
    log(f"probe done {len(cands)} codes in {time.time() - t0:.0f}s")
    with open(L.CANDS, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    with open(L.WINDOWS, "w", encoding="utf-8") as f:
        for w in wins:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    with open(L.OUT / "grid.json", "w", encoding="utf-8") as f:
        json.dump({sid: {f"{m}:{Lg}": wid for (m, Lg), wid in g.items()}
                   for sid, g in grid.items()}, f)
    n = [len(v) for v in cands.values()]
    log(f"candidates/code: mean={sum(n) / max(len(n), 1):.1f} "
        f"ge16={sum(1 for x in n if x >= 16) / max(len(n), 1):.3f}")
    cl = Counter(min(w["L"] // 4 * 4, 20) for w in wins if not w["dyn"])
    log(f"static typed-length buckets: {sorted(cl.items())}")


def cmd_ce_input(a):
    """静态窗引擎池前 16 条 → sim 输入。dyn 格的 CE 由分析脚本按需补算。"""
    cands = json.load(open(L.CANDS, encoding="utf-8"))
    seen, n_line, n_win = set(), 0, 0
    with open(L.OUT / "ce_input.jsonl", "w", encoding="utf-8") as f:
        for w in L.jsonl_iter(L.WINDOWS):
            if w["dyn"]:
                continue
            pool = [c[0] for c in cands.get(w["code"], [])[:L.K_COLLECT]]
            if len(pool) < 2:
                continue
            n_win += 1
            for ln, sid in L.ce_lines(w["ctx"], pool):
                if sid in seen:
                    continue
                seen.add(sid)
                f.write(json.dumps(ln, ensure_ascii=False) + "\n")
                n_line += 1
    log(f"ce_input: {n_win} static windows → {n_line} lines")


def static_wins(cands):
    return [w for w in L.jsonl_iter(L.WINDOWS)
            if not w["dyn"] and cands.get(w["code"])]


def cmd_gen(a):
    """静态窗（≥4 码）的 N-gram 组句（生产参数）。"""
    cands = json.load(open(L.CANDS, encoding="utf-8"))
    jobs = [(w["wid"], w["ctx"], w["code"]) for w in static_wins(cands)
            if len(w["code"]) >= 4]
    log(f"n-gram jobs={len(jobs)}")
    n = L.run_gen(jobs)
    log(f"n-gram generated={n}")


def cmd_genfull(a):
    """全句 CE：只在 5 条组句 lead 不一致时参与句级前缀判定（一致则无论分数
    都触发），故只给这些窗算。"""
    cands = json.load(open(L.CANDS, encoding="utf-8"))
    wins = static_wins(cands)
    gen = L.load_gen()
    full = [(w["wid"], w["ctx"], [x["text"] for x in gen[w["wid"]]])
            for w in wins if w["wid"] in gen
            and len({x["lead"] for x in gen[w["wid"]]}) > 1]
    log(f"full-CE jobs (lead differs)={len(full)} of {len(wins)} static windows")
    n = L.run_full(full)
    log(f"full CE scored={n}")


def cmd_assemble(a):
    cands = json.load(open(L.CANDS, encoding="utf-8"))
    ce, gen, full = L.load_ce(), L.load_gen(), L.load_genfull()
    n_win = n_miss_ce = 0
    with open(L.COHORT, "w", encoding="utf-8") as f:
        for w in L.jsonl_iter(L.WINDOWS):
            pool = cands.get(w["code"], [])[:L.K_COLLECT]
            sids = [sid for (_, sid) in L.ce_lines(w["ctx"], [c[0] for c in pool])]
            rows = []
            for i, c in enumerate(pool):
                res = ce.get(sids[i // 10], {}).get(c[0])
                ce3, ntok = (res if res else (None, None))
                if ce3 is None:
                    n_miss_ce += 1
                rows.append([c[0], c[1], c[2], c[4], c[3], ce3, ntok])
            gs = []
            for g in gen.get(w["wid"], []):
                r = full.get(w["wid"], {}).get(g["text"])
                gs.append([g["text"], g["nscore"], g["lead"], g["lead_end"],
                           g["words"], None if r is None else -r[0],
                           None if r is None else r[1]])
            f.write(json.dumps(dict(w, eng=rows, gen=gs), ensure_ascii=False)
                    + "\n")
            n_win += 1
    log(f"cohort: {n_win} windows, engine cands missing CE={n_miss_ce}, "
        f"gen windows={sum(1 for v in gen.values() if v)}, "
        f"full-CE windows={len(full)} → {L.COHORT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sentences")
    p.add_argument("--per-domain", type=int, default=1500)
    p.add_argument("--dyn-per-domain", type=int, default=150)
    p.add_argument("--skip", type=int, default=300000,
                   help="HanLP 缓存起始文章号（与既有 sgen 测试集错开）")
    p.add_argument("--articles", type=int, default=12000)
    p.set_defaults(fn=cmd_sentences)
    p = sub.add_parser("windows")
    p.add_argument("--per-sent", type=int, default=8)
    p.set_defaults(fn=cmd_windows)
    sub.add_parser("ce_input").set_defaults(fn=cmd_ce_input)
    sub.add_parser("gen").set_defaults(fn=cmd_gen)
    sub.add_parser("genfull").set_defaults(fn=cmd_genfull)
    sub.add_parser("assemble").set_defaults(fn=cmd_assemble)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
