#!/usr/bin/env python3
"""前缀确认「大 cohort」构建（2026-11-25）。

动机：现行 luna 混合窗 cohort 的静态窗可触发集合只有 3,075 个、错误 125 个
——不足以训练表模型（树/神经）也不足以挖掘「结构性错判」（他/它/她 这类
同音不可判项）。本脚本按同一口径扩到 wiki 12,000 句（约 8×），并且只对
**能进入前缀确认判定区**的窗算 CE（P≥pmin ∧ lead 在引擎池同 span ∧ 剩余
≥2 音节），把 CE 量从「全部窗」压到 ~20%。

口径（与 luna_mixed 完全一致，只是句子更多）：
  窗口 = (m, L)：句首 m 字已确认（进上文），其后音节打 L 个字母。
  ctx  = before + 真值[:m]；code = 真值[m:] 音节串前 L 个字母
  候选正确 ⇔ 候选文本是 真值[m:] 的前缀
  冷账本 → 层1 序 = 引擎序；窗口 = 生产「只排除显示置顶的组句第 1 条」
  CE 缓存与 luna_mixed 共用（content-keyed sid），不重复算。

子命令（产物落 eval/prefix_big/）：
  sentences   wiki 句子（skip 与既有 3,000 句错开）→ sentences.jsonl
  windows     窗口网格（每句抽 --per-sent 格）→ windows.jsonl + cands.json
  gen         静态窗（≥4 码）N-gram 组句 → gen.tsv（--procs 并行分片）
  ce          判定区预筛 → 只对这些窗补 CE（追加到 luna_mixed 共用缓存）
  assemble    窗口 + CE + 组句 → cohort.jsonl

用法（研究库根目录）：
  python eval/calib/build_prefix_big.py sentences --per-domain 12000 \
      --skip 320000 --articles 96000
  python eval/calib/build_prefix_big.py windows --per-sent 8
  python eval/calib/build_prefix_big.py gen --procs 4
  python eval/calib/build_prefix_big.py ce --pmin 0.5 --workers 8
  python eval/calib/build_prefix_big.py assemble
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import pathlib
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import build_luna_mixed as B  # noqa: E402
import corpus_std  # noqa: E402
import luna_mixed_lib as L  # noqa: E402

_sys.stdout.reconfigure(encoding="utf-8")

OUT2 = L.EVAL / "prefix_big"
WORK2 = OUT2 / "work"
SENTS2 = OUT2 / "sentences.jsonl"
WINS2 = OUT2 / "windows.jsonl"
CANDS2 = OUT2 / "cands.json"
GEN2 = OUT2 / "gen.tsv"
TRIED2 = WORK2 / "gen_tried.txt"
COHORT2 = OUT2 / "cohort.jsonl"
SEED = 20261125


def log(msg):
    print(f"[prefix_big {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 句子 ──────────────────────────────────────────────────────────────
def cmd_sentences(a):
    from pypinyin import lazy_pinyin
    OUT2.mkdir(parents=True, exist_ok=True)
    vocab = B.Readings(lazy_pinyin)
    log(f"vocab words={len(vocab.rd)} essay={len(vocab.essay)}")
    t0 = time.time()
    sents = B.wiki_sentences(vocab, a.per_domain, a.skip, a.articles)
    log(f"wiki sentences={len(sents)} in {time.time() - t0:.0f}s")
    # sid 前缀 v，避免与 luna_mixed 的 w/u 命名空间冲突
    for i, s in enumerate(sents):
        s["sid"] = "v%06d" % i
        s["dyn"] = False
    with open(SENTS2, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log(f"written {SENTS2}: {len(sents)} sentences, "
        f"len {min(len(s['text']) for s in sents)}.."
        f"{max(len(s['text']) for s in sents)}")


# ── 窗口 + 探针 ───────────────────────────────────────────────────────
def cmd_windows(a):
    import random
    sents = list(L.jsonl_iter(SENTS2))
    rng = random.Random(SEED + 1)
    wins, by_key = [], {}
    for s in sents:
        cells = B.grid_cells(s)
        take = rng.sample(cells, min(a.per_sent, len(cells)))
        for (m, Lg, ctx, code) in take:
            key = (ctx, code)
            wid = by_key.get(key)
            if wid is None:
                wid = "B%07d" % len(wins)
                by_key[key] = wid
                wins.append(dict(wid=wid, sid=s["sid"], domain=s["domain"],
                                 m=m, L=Lg, ctx=ctx, code=code,
                                 text=s["text"][m:], dyn=False))
    log(f"windows={len(wins)} (unique (ctx,code))")
    codes = sorted({w["code"] for w in wins})
    log(f"unique codes={len(codes)}; engine probe ...")
    t0 = time.time()
    cands = L.run_probe(codes, work=WORK2)
    log(f"probe done {len(cands)} codes in {time.time() - t0:.0f}s")
    CANDS2.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDS2, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    with open(WINS2, "w", encoding="utf-8") as f:
        for w in wins:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    n = [len(v) for v in cands.values()]
    log(f"candidates/code: mean={sum(n) / max(len(n), 1):.1f} "
        f"ge16={sum(1 for x in n if x >= 16) / max(len(n), 1):.3f}")


# ── 组句（并行分片）────────────────────────────────────────────────
def _gen_launch(script, jobs, out, cache):
    jf = WORK2 / (out.stem + "_jobs.tsv")
    with open(jf, "w", encoding="utf-8", newline="\n") as f:
        for w, c, k in jobs:
            ctx = c.replace("\t", " ").replace("\n", " ")
            f.write(f"{w}\t{ctx}\t{k}\n")
    e = dict(os.environ)
    e.update({
        "LM_MODE": "gen", "LM_JOBS": str(jf), "LM_OUT": str(out),
        "LM_MODEL": L.MODEL, "LM_THREADS": "1",
        "LM_GRAM": str(L.NGRAM_DIR / "wanxiang-lts-zh-hans.gram"),
        "LM_DICT": str(L.NGRAM_DIR / "dicts"),
        "LM_FILES": ",".join(L.NGRAM_FILES),
        "LM_CACHE": str(cache), "LM_PARAMS": L.NGRAM_PARAMS})
    logf = WORK2 / (out.stem + ".log")
    err = open(logf, "ab")
    p = subprocess.Popen([str(L.LUA_HOST), str(script)],
                         cwd=str(L.LUA_HOST.parent), env=e,
                         stdout=err, stderr=subprocess.STDOUT)
    return p, err


def cmd_gen(a):
    cands = json.load(open(CANDS2, encoding="utf-8"))
    tried = set(TRIED2.read_text(encoding="ascii").split()) \
        if TRIED2.exists() else set()
    done = {r[0] for r in L.load_gen_raw(GEN2)}
    jobs = [(w["wid"], w["ctx"], w["code"]) for w in L.jsonl_iter(WINS2)
            if len(w["code"]) >= 4 and cands.get(w["code"])
            and w["wid"] not in done and w["wid"] not in tried]
    log(f"n-gram jobs={len(jobs)} (done={len(done)} tried={len(tried)})")
    if not jobs:
        return
    WORK2.mkdir(parents=True, exist_ok=True)
    cache = WORK2 / "ngram_lex.bin"
    if not cache.exists():
        import shutil
        src = L.WORK / "ngram_lex.bin"
        if not src.exists():
            src = L.RIME_USER / "llm_rerank" / "ngram_lex" / "luna_pinyin.bin"
        shutil.copy2(src, cache)
    procs = max(1, a.procs)
    # 交错分片：各片码长分布相近
    chunks = [jobs[i::procs] for i in range(procs)]
    script = pathlib.Path(L.HERE) / "luna_mixed_gen.lua"
    running = []
    t0 = time.time()
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        out = WORK2 / f"gen_part{i}.tsv"
        out.unlink(missing_ok=True)
        p, err = _gen_launch(script, ch, out, cache)
        running.append((i, ch, out, p, err))
    log(f"launched {len(running)} gen processes, jobs={len(jobs)}")
    for i, ch, out, p, err in running:
        rc = p.wait()
        err.close()
        log(f"  part{i} rc={rc}")
    n_rows = 0
    with open(GEN2, "a", encoding="utf-8") as fo, \
            open(TRIED2, "a", encoding="ascii") as ft:
        for i, ch, out, p, err in running:
            if out.exists():
                with open(out, encoding="utf-8") as fi:
                    for line in fi:
                        fo.write(line)
                        n_rows += 1
            for w, _, _ in ch:
                ft.write(w + "\n")
    log(f"gen appended rows={n_rows} in {time.time() - t0:.0f}s → {GEN2}")


# ── 判定区预筛 + CE ───────────────────────────────────────────────────
def lead_stats(gs):
    import math
    mx = max(x["score"] for x in gs)
    mass, le_of, txt_of = defaultdict(float), {}, {}
    tot = 0.0
    for x in gs:
        wt = math.exp(x["score"] - mx)
        tot += wt
        mass[x["lead"]] += wt
        le_of.setdefault(x["lead"], x["lead_end"])
        txt_of.setdefault(x["lead"], x["text"])
    lead, best = None, -1.0
    for t, m in mass.items():
        if m > best or (m == best and lead is not None and t < lead):
            lead, best = t, m
    top = gs[0]
    other = next((x for x in gs[1:] if x["lead"] != top["lead"]), None)
    d = (top["score"] - other["score"]) if other else 30.0
    return lead, best / tot if tot else 0.0, le_of[lead], txt_of[lead], d


def load_gen_map(path=GEN2):
    out = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) != 7:
            continue
        try:
            out[f[0]].append(dict(score=float(f[2]), lead=f[3],
                                  lead_end=int(f[4]), text=f[6]))
        except ValueError:
            pass
    for v in out.values():
        v.sort(key=lambda x: -x["score"])
    return out


def cmd_ce(a):
    cands = json.load(open(CANDS2, encoding="utf-8"))
    g20 = load_gen_map()
    ce = L.load_ce_multi()
    lines, n_win, n_cached = [], 0, 0
    for w in L.jsonl_iter(WINS2):
        if len(w["code"]) < 4:
            continue
        gs = g20.get(w["wid"])
        if not gs:
            continue
        lead, p, le, sent, d = lead_stats(gs)
        if p < a.pmin:
            continue
        sl, syl = len(lead), len(sent)
        pool = cands.get(w["code"], [])[:L.K_COLLECT]
        if not any(c[0] == lead and c[2] == le for c in pool):
            continue
        if not (le < len(w["code"]) and syl - sl >= 2):
            continue
        n_win += 1
        for ln, sid in L.ce_lines(w["ctx"], [c[0] for c in pool]):
            if sid in ce:
                n_cached += 1
                continue
            lines.append((ln, sid))
    log(f"prefilter (P≥{a.pmin}) windows={n_win}; CE lines new={len(lines)} "
        f"cached={n_cached}")
    if lines:
        t0 = time.time()
        n = L.run_ce(lines, workers=a.workers)
        log(f"CE computed {n} lines in {time.time() - t0:.0f}s")


def cmd_assemble(a):
    cands = json.load(open(CANDS2, encoding="utf-8"))
    ce, gen = L.load_ce_multi(), load_gen_map()
    n_win = n_miss = n_gen = 0
    with open(COHORT2, "w", encoding="utf-8") as f:
        for w in L.jsonl_iter(WINS2):
            pool = cands.get(w["code"], [])[:L.K_COLLECT]
            sids = [sid for (_, sid) in L.ce_lines(w["ctx"], [c[0] for c in pool])]
            rows = []
            for i, c in enumerate(pool):
                res = ce.get(sids[i // 10], {}).get(c[0])
                ce3, ntok = (res if res else (None, None))
                if ce3 is None:
                    n_miss += 1
                rows.append([c[0], c[1], c[2], c[4], c[3], ce3, ntok])
            gs = [[g["text"], g["score"], g["lead"], g["lead_end"]]
                  for g in gen.get(w["wid"], [])]
            if gs:
                n_gen += 1
            f.write(json.dumps(dict(w, eng=rows, gen=gs), ensure_ascii=False)
                    + "\n")
            n_win += 1
    log(f"cohort: {n_win} windows, gen windows={n_gen}, "
        f"engine cands missing CE={n_miss} → {COHORT2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sentences")
    p.add_argument("--per-domain", type=int, default=12000)
    p.add_argument("--skip", type=int, default=320000)
    p.add_argument("--articles", type=int, default=96000)
    p.set_defaults(fn=cmd_sentences)
    p = sub.add_parser("windows")
    p.add_argument("--per-sent", type=int, default=8)
    p.set_defaults(fn=cmd_windows)
    p = sub.add_parser("gen")
    p.add_argument("--procs", type=int, default=4)
    p.set_defaults(fn=cmd_gen)
    p = sub.add_parser("ce")
    p.add_argument("--pmin", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_ce)
    sub.add_parser("assemble").set_defaults(fn=cmd_assemble)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
