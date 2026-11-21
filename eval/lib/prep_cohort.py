#!/usr/bin/env python3
"""cohort 队列模拟重放（2026-11-07，取代"全文/前缀重放"的量级失真）。

背景: 旧链在 2.84 亿提交（wiki 全文, ≈真实用户 #tick 的 1 万倍）的特征分布上
标定 β; 真实生产用户 user_freq.tsv #tick≈2.8 万。本脚本模拟**一群独立用户**:
- 语料按顺序切成互不重叠的连续文章段, 每段 ≈H 次提交 = 一个虚拟用户完整打字史;
- 每用户**空库起步**: eff/n_seen/uni/c1/bi/h2 全部逐用户重置（h2 跨文章但不跨用户,
  重置语义落在 streams.jsonl 行边界——每行一个用户, eval 端逐行重置即可）;
- 测试窗从每个用户自己的段内**水库采样** K 个 → 每窗特征历史深度 ∈ [0,H],
  全体窗口的特征分布 = 对应龄段真实用户群的分布;
- 每窗记录 depth（该用户第几次提交, 0 基）→ 支持按历史深度切片分析 β(H)。

年龄档（用户定案）: H ∈ {3千,1万,3万,10万,30万,100万} 提交/用户。
DP 先验 = 全文语料 unigram（cohort_unigram.py, 各档共用固定先验）。
产物（cohort_<chain>/ 目录, 追加式）:
  anchors.jsonl {sid,arm,ui,ai,ci,w,win,ctx,depth}   — ai=ui=streams 行号(用户)
  samples.jsonl {sid,arm,ui,ctx,word,code,pos,cands,eff_win,winsize,depth}
  streams.jsonl 每行 = 一个用户的完整词流
  meta.json     各档配置与实际产量、文章占用区间（断点续跑: done 档跳过）
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_std
from corpus_std import MAX_CAND, CTX_TAIL, SEED, TAU

PROJ = corpus_std.PROJ
SHUFFLED = PROJ / "article_corpus_shuffled.txt"
CACHE = PROJ / "segments_hanlp_cache.jsonl"
SMALL_DICT = PROJ / "词库-测评6万" / "pdsp_dict_small6w.yaml"
IDEO_RE = re.compile(r"[一-鿿]")


def build_chain(chain, single=False):
    """→ (splitter, word_code_mc, code_to_words, V_cap,
    word_code_sc, code_to_words_sc)（后两个仅 --single/--single2 非 None）"""
    uni_path = corpus_std.EVAL / f"cohort_unigram_{chain}.json"
    payload = json.loads(uni_path.read_text(encoding="utf-8"))
    logp = {w: float(v) for w, v in payload["logp"].items()}
    unseen = float(payload["unseen_logp"])
    if chain == "small6w":
        words_small = set(corpus_std.load_pdsp_words(SMALL_DICT))
        big_words = corpus_std.load_pdsp_words()
        chars = {w for w in big_words if len(w) == 1 and IDEO_RE.search(w)}
        vocab = words_small | chars
        code_to_words, word_code_mc = corpus_std.load_pdsp_dict(SMALL_DICT)
        v_cap = len(words_small)
    else:
        vocab = set(corpus_std.load_pdsp_words())
        code_to_words, word_code_mc = corpus_std.load_pdsp_dict()
        v_cap = len(vocab)
    splitter = corpus_std.DictSplitter(vocab, logp, unseen)
    if single == "sc2":
        code_to_words_sc, word_code_sc = corpus_std.load_pdsp_dict_single2()
    elif single:
        code_to_words_sc, word_code_sc = corpus_std.load_pdsp_dict_single3()
    else:
        code_to_words_sc, word_code_sc = None, None
    return (splitter, word_code_mc, code_to_words, v_cap,
            word_code_sc, code_to_words_sc)


def win_gate(w, single, word_code_mc, code_to_words, word_code_sc,
             code_to_words_sc):
    """窗口资格门（2026-11-12 一字词研究）：多字词 4 码口径 / --single
    单字 3 码口径二选一。→ (code, win, pos, winsize) 或 None。"""
    if single:
        if len(w) != 1:
            return None
        code = word_code_sc.get(w)
        full = code_to_words_sc.get(code) if code else None
    else:
        if len(w) < 2:
            return None
        code = word_code_mc.get(w)
        full = code_to_words.get(code) if code else None
    if not full or len(full) < 2 or full.index(w) >= MAX_CAND:
        return None
    return code, full[:MAX_CAND], full.index(w), len(full)


def map_article(art, tok_line, splitter):
    """HanLP token → (词, pos) 映射（与 prep_samples_std/small6w 逐字同语义）。"""
    mapped = []
    char_pos = 0
    for t in json.loads(tok_line):
        if not IDEO_RE.search(t):
            continue
        pos = art.find(t, char_pos)
        if pos < 0:
            char_pos += len(t)
            continue
        char_pos = pos + len(t)
        subs = splitter.split(t)
        if subs is None:
            continue
        mapped.extend((w, pos) for w in subs)
    return mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", choices=["small6w", "big", "r2"], default="small6w")
    ap.add_argument("--single", action="store_true",
                    help="单字 3 码重排域（2026-11-12）：窗口资格 = CJK 单字"
                         "且 3 码重码（2 码固定顺序不重排，部署语义用户口授）；"
                         "streams/eff 重放不变（单字提交推进 tick 与部署一致）")
    ap.add_argument("--single2", action="store_true",
                    help="单字 2 码重排域（2026-11-13 扩样研究）：窗口资格 = "
                         "CJK 单字且 2 码（声韵）重码——评估 2 码窗放开重排的"
                         "可行性；其余语义与 --single 相同")
    ap.add_argument("--arms", default=(
        "30000:60:420"),
        help="H:n_users:k_per（2026-11-18 定案：多年龄档/多用户重放退役，"
             "默认只跑 3 万龄单档 ≈2.5 万窗）")
    ap.add_argument("--pilot", type=int, default=None,
                    help="只测前 N 篇的 commits/hits 产量, 不产出文件")
    ap.add_argument("--tau", type=float, default=400.0,
                    help="eff 衰减 τ（生产 v3 = 400, 两版 llm_processor/filter 同值;"
                         "corpus_std.TAU=800 是 2026-11-02 旧值, 勿用）")
    ap.add_argument("--redo-eff", action="store_true",
                    help="已有 meta 全 done 时: 同 seed 同窗重放, 只重写 "
                         "samples.jsonl 的 eff_win（anchors/streams/CE 不动）")
    ap.add_argument("--tag", default="",
                    help="目录后缀: cohort_<dict><tag>/（训练数据专用链, 与评测链隔离）")
    ap.add_argument("--start-article", type=int, default=None,
                    help="新链目录的初始文章游标（接续已有链的 next_article, "
                         "保证训练/评测文章不相交; 仅对空 meta 生效）")
    ap.add_argument("--articles", default=None,
                    help="文章语料路径覆盖（默认 article_corpus_shuffled.txt; "
                         "用户域链: user_articles.txt, 配套缓存 --cache）")
    ap.add_argument("--cache", default=None,
                    help="HanLP 分段缓存路径覆盖（默认 segments_hanlp_cache.jsonl）")
    a = ap.parse_args()

    global SHUFFLED, CACHE
    if a.articles:
        SHUFFLED = Path(a.articles)
    if a.cache:
        CACHE = Path(a.cache)

    if (a.single or a.single2) and a.redo_eff:
        print("--redo-eff 仅多字词链（单字链无既有 eff 需重写）", flush=True)
        return 1
    if a.single2 and not a.tag:
        a.tag = "_sc2"
    elif a.single and not a.tag:
        a.tag = "_sc"
    splitter, word_code_mc, code_to_words, v_cap, word_code_sc, \
        code_to_words_sc = build_chain(
            a.dict, single=("sc2" if a.single2 else a.single))
    print(f"chain={a.dict}{a.tag} V_cap={v_cap} "
          f"single={a.single}", flush=True)

    # ── 试点: 测产量 ──
    if a.pilot is not None:
        n_com = n_hit = 0
        with open(SHUFFLED, encoding="utf-8") as cf, \
                open(CACHE, encoding="utf-8") as tf:
            for i, (art_line, tok_line) in enumerate(zip(cf, tf)):
                if i >= a.pilot:
                    break
                mapped = map_article(art_line.rstrip("\n\r"), tok_line, splitter)
                n_com += len(mapped)
                for w, _ in mapped:
                    if win_gate(w, a.single or a.single2, word_code_mc,
                                code_to_words, word_code_sc,
                                code_to_words_sc):
                        n_hit += 1
        print(f"pilot {a.pilot} arts: commits {n_com} "
              f"({n_com / a.pilot:.1f}/art), hits {n_hit} "
              f"(r={n_hit / max(1, n_com):.4f})", flush=True)
        for h in (30000,):
            per_user = h * n_hit / max(1, n_com)
            print(f"  H={h:>7}: hits/user≈{per_user:7.0f}  "
                  f"25k 窗需 ≈{25000 / max(1, per_user):5.0f} 用户", flush=True)
        return 0

    # ── 正式重放 ──
    out_dir = PROJ / f"cohort_{a.dict}{a.tag}"
    out_dir.mkdir(exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = {"dict": a.dict, "v_cap": v_cap, "arms": [], "next_article": 0,
            "tag": a.tag}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif a.start_article is not None:
        meta["next_article"] = a.start_article
        print(f"fresh chain with --start-article {a.start_article}",
              flush=True)
    done_hs = {m["h"] for m in meta["arms"] if m.get("done")}
    arms = [tuple(int(x) for x in s.split(":")) for s in a.arms.split(",")]
    if a.redo_eff and done_hs:
        # ── 只重算 eff: 重放同样选择（同 seed 同序 → 同窗）, 重写 samples ──
        samples_path = out_dir / "samples.jsonl"
        rng2 = random.Random(SEED)
        n_re = 0
        ui_c = 0
        with open(samples_path, "w", encoding="utf-8") as sf, \
                open(SHUFFLED, encoding="utf-8") as cf, \
                open(CACHE, encoding="utf-8") as tf:
            arts = ((i, al.rstrip("\n\r"), tl)
                    for i, (al, tl) in enumerate(zip(cf, tf)))
            for h, n_users, k_per in arms:
                if h not in done_hs:
                    continue
                nu = next(m["n_users"] for m in meta["arms"] if m["h"] == h)
                for _ in range(nu):
                    replay = corpus_std.FormulaReplay(a.tau)
                    sel, hits_seen, ci = [], 0, 0
                    while ci < h:
                        try:
                            _, art, tok_line = next(arts)
                        except StopIteration:
                            break
                        for w, pos in map_article(art, tok_line, splitter):
                            if len(w) >= 2:
                                code = word_code_mc.get(w)
                                full = code_to_words.get(code) if code else None
                                if full and len(full) >= 2 \
                                        and full.index(w) < MAX_CAND:
                                    win = full[:MAX_CAND]
                                    hits_seen += 1
                                    rec = (ci, w, win, art[:pos][-CTX_TAIL:],
                                           full.index(w), code, len(full),
                                           {x: replay.eff_of(x) for x in win})
                                    if hits_seen <= k_per:
                                        sel.append(rec)
                                    else:
                                        j = rng2.randrange(hits_seen)
                                        if j < k_per:
                                            sel[j] = rec
                            replay.bump(w)
                            ci += 1
                        if ci >= h:
                            break
                    for ci0, w, win, ctx, pos, code, winsize, eff_win in sel:
                        sf.write(json.dumps(
                            {"sid": corpus_std.window_key_ctx(ctx, win),
                             "arm": h, "ui": ui_c, "ctx": ctx, "word": w,
                             "code": code, "pos": pos,
                             "cands": [x for x in win if x != w]
                             [:MAX_CAND - 1],
                             "eff_win": eff_win, "winsize": winsize,
                             "depth": ci0}, ensure_ascii=False) + "\n")
                        n_re += 1
                    ui_c += 1
        print(f"redo-eff (τ={a.tau}): rewrote {n_re} samples", flush=True)
        return 0
    arms = [tuple(int(x) for x in s.split(":")) for s in a.arms.split(",")]

    streams_path = out_dir / "streams.jsonl"  # noqa: E305（redo 块在上方已返回）
    ui = 0
    if streams_path.exists():
        with open(streams_path, encoding="utf-8") as f:
            for _ in f:
                ui += 1

    rng = random.Random(SEED)
    af = open(out_dir / "anchors.jsonl", "a", encoding="utf-8")
    sf = open(out_dir / "samples.jsonl", "a", encoding="utf-8")
    stf = open(streams_path, "a", encoding="utf-8")

    with open(SHUFFLED, encoding="utf-8") as cf, \
            open(CACHE, encoding="utf-8") as tf:

        def article_stream(start):
            for i, (al, tl) in enumerate(zip(cf, tf)):
                if i < start:
                    continue
                yield i, al.rstrip("\n\r"), tl

        arts = article_stream(meta["next_article"])
        for h, n_users, k_per in arms:
            if h in done_hs:
                continue
            art_start = meta["next_article"]
            arm_users = arm_windows = 0
            for _u in range(n_users):
                replay = corpus_std.FormulaReplay(a.tau)  # 空库起步
                # （2026-11-09 修: 原误用 corpus_std.TAU=800 旧值; 评测链靠
                #   --redo-eff 补救, 新链直接生成就用生产 τ=400）
                sel = []                                  # 水池（≤K 个样本）
                hits_seen = 0
                ci = 0
                words_line = []
                while ci < h:
                    try:
                        ai_, art, tok_line = next(arts)
                    except StopIteration:
                        break
                    meta["next_article"] = ai_ + 1
                    for w, pos in map_article(art, tok_line, splitter):
                        g = win_gate(w, a.single or a.single2,
                                     word_code_mc, code_to_words,
                                     word_code_sc, code_to_words_sc)
                        if g:
                            code, win, wpos, winsize = g
                            hits_seen += 1
                            if hits_seen <= k_per:
                                sel.append((ci, w, win,
                                            art[:pos][-CTX_TAIL:],
                                            wpos, code, winsize,
                                            {x: replay.eff_of(x)
                                             for x in win}))
                            else:
                                j = rng.randrange(hits_seen)
                                if j < k_per:
                                    sel[j] = (ci, w, win,
                                              art[:pos][-CTX_TAIL:],
                                              wpos, code, winsize,
                                              {x: replay.eff_of(x)
                                               for x in win})
                        replay.bump(w)          # 全部提交 bump（含单字）
                        ci += 1
                        words_line.append(w)
                    if (ai_ + 1) % 5000 == 0:
                        print(f"  [H={h}] art {ai_ + 1}, user "
                              f"{arm_users + 1}/{n_users}, commits {ci}/{h}",
                              flush=True)
                    if ci >= h:
                        break
                if not words_line:
                    continue                     # 语料耗尽且无内容
                stf.write(json.dumps(words_line, ensure_ascii=False) + "\n")
                for ci0, w, win, ctx, pos, code, winsize, eff_win in sel:
                    sid = corpus_std.window_key_ctx(ctx, win)
                    af.write(json.dumps(
                        {"sid": sid, "arm": h, "ui": ui, "ai": ui, "ci": ci0,
                         "w": w, "win": win, "ctx": ctx, "depth": ci0},
                        ensure_ascii=False) + "\n")
                    sf.write(json.dumps(
                        {"sid": sid, "arm": h, "ui": ui, "ctx": ctx,
                         "word": w, "code": code, "pos": pos,
                         "cands": [x for x in win if x != w][:MAX_CAND - 1],
                         "eff_win": eff_win, "winsize": winsize,
                         "depth": ci0}, ensure_ascii=False) + "\n")
                ui += 1
                arm_users += 1
                arm_windows += len(sel)
                if arm_users % 10 == 0:
                    print(f"  [H={h}] users {arm_users}/{n_users}, "
                          f"windows {arm_windows}", flush=True)
            meta["arms"] = [m for m in meta["arms"] if m["h"] != h] + [dict(
                h=h, n_users=arm_users, k_per=k_per, windows=arm_windows,
                art_start=art_start, art_end=meta["next_article"], done=True)]
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=1),
                encoding="utf-8")
            af.flush(); sf.flush(); stf.flush()
            print(f"arm H={h} done: users {arm_users}, windows "
                  f"{arm_windows}, articles [{art_start}, "
                  f"{meta['next_article']})", flush=True)

    af.close(); sf.close(); stf.close()
    print(f"done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
