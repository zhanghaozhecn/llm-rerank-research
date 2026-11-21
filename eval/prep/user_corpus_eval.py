#!/usr/bin/env python3
"""用户真实语料（rime_llm_corpus.tsv v2）跨域评测 — "没见过的语料" 泛化测试。

原料 = 打字提交日志（词\t码 时序对, 含标点直上屏）, **不需要 HanLP/DP**——
用户实际提交词即词流。模拟 = 按真实时序从零重放 eff（新用户冷启动）, 窗口
形成与 prep 同口径（多字 4 码组, 目标词文件位次 <10, ctx = 词连接文本尾
120 字）。三模型:
  - 固定词典 / 纯 eff: 本脚本直出
  - trigram: wiki 全量计数（过滤重扫 streams, 用户提供键）——跨域泛化
  - LLM CE / LSTM: gen-input → sim（GPU 排队） / lstm_lm --score-user（待训完）

用法: python eval/user_corpus_eval.py [--ce]（--ce = 产推理输入, GPU 空时跑）
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
os.environ.setdefault("RIME_LLM_CE_CACHE",
                      str(corpus_std.PROJ / "std_ce_work" / "results.jsonl"))
import numpy as np
from corpus_std import CTX_TAIL, TAU

PROJ = corpus_std.PROJ
TRAIN_LOG = Path(os.environ.get(
    "RIME_LLM_USER_CORPUS", str(corpus_std.CORPUS_PATH)))
SAMPLES = PROJ / "user_samples_std.jsonl"
ANCHORS = PROJ / "user_anchors_std.jsonl"
CE_INPUT = PROJ / "user_ce_input.jsonl"
CE_DUMP = PROJ / "user_ce_work/results.jsonl"
TRI_REPORT = corpus_std.EVAL / "user_corpus_report.txt"
W_TRI, W_BI, W_UNI = 0.6, 0.3, 0.1


def io_open_lines(p):
    with open(p, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l:
                yield l


def parse_pairs():
    # 旧口径：全部提交（含被回撤），忽略 ←；v2 由 iter_commits 提供。
    return [(c["word"], c["code"])
            for c in corpus_std.iter_commits(TRAIN_LOG,
                                             include_retracted=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", action="store_true")
    a = ap.parse_args()

    code_to_words, word_code_mc = corpus_std.load_pdsp_dict()
    words_all = corpus_std.load_pdsp_words()

    pairs = parse_pairs()
    print(f"parsed {len(pairs)} commits ({TRAIN_LOG.name})", flush=True)

    # ── 时序重放 + 窗口 ──
    replay = corpus_std.FormulaReplay(TAU)
    samples = []
    anchors = []
    seq = []                       # dict 词序（LSTM/trigram 上下文用）
    hist = []                      # (idx, h1, h2)
    h1 = h2 = None
    n_deep = 0
    for w, logged_code in pairs:
        in_dict = w in words_all
        ctx = ''.join(seq)[-CTX_TAIL:] if seq else ''
        code = word_code_mc.get(w) if len(w) >= 2 else None
        full = code_to_words.get(code) if code else None
        if full and len(full) >= 2 and full.index(w) < corpus_std.MAX_CAND:
            win = full[:corpus_std.MAX_CAND]
            eff_win = {x: replay.eff_of(x) for x in win}
            sid = corpus_std.window_key_ctx(ctx, win)
            samples.append(dict(ctx=ctx, word=w, code=code,
                                pos=full.index(w),
                                cands=[x for x in win if x != w]
                                [:corpus_std.MAX_CAND - 1],
                                eff_win=eff_win, winsize=len(full)))
            anchors.append({"sid": sid, "ai": len(anchors), "ci": len(seq),
                            "w": w, "win": win, "ctx": ctx})
            hist.append((h1, h2))
        elif full and len(full) >= 2:
            n_deep += 1
        if in_dict:
            replay.bump(w)
            seq.append(w)
            h1, h2 = h2, w
    with open(SAMPLES, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(ANCHORS, "w", encoding="utf-8") as f:
        for an in anchors:
            f.write(json.dumps(an, ensure_ascii=False) + "\n")
    print(f"windows {len(samples)} (pos>=10 剔除 {n_deep}), "
          f"dict commits {len(seq)}/{len(pairs)}", flush=True)

    # ── 固定词典 / 纯 eff 行 ──
    nn = len(samples)
    pos = np.array([s["pos"] for s in samples])
    eff = np.zeros((nn, corpus_std.MAX_CAND))
    widx = np.zeros(nn, dtype=int)
    for i, s in enumerate(samples):
        win = corpus_std.window_of(s)
        for j, w in enumerate(win):
            eff[i, j] = s["eff_win"].get(w, 0.0)
            if w == s["word"]:
                widx[i] = j
    dict_hit = (pos == 0).mean()
    eff_hit = (eff.argmax(1) == widx).mean()

    # ── trigram: wiki 计数（过滤重扫 streams, 用户键）, 跨域打分 ──
    need_tri, need_bi, need_uni = set(), set(), set()
    need_c1, need_c2 = set(), set()
    for idx, (h1, h2) in enumerate(hist):
        win = anchors[idx]["win"]
        for x in win:
            need_uni.add(x)
            if h2 is not None:
                need_bi.add((h2, x))
                need_c1.add(h2)
                if h1 is not None:
                    need_tri.add((h1, h2, x))
                    need_c2.add((h1, h2))
    uni, bi, tri = defaultdict(int), defaultdict(int), defaultdict(int)
    c1, c2 = defaultdict(int), defaultdict(int)
    wiki_tok = 0
    with open(corpus_std.PROJ / "streams_std.jsonl", encoding="utf-8") as f:
        for line in f:
            pw1 = pw2 = None
            for w in json.loads(line):
                wiki_tok += 1
                if w in need_uni:
                    uni[w] += 1
                if pw2 is not None:
                    if pw2 in need_c1:
                        c1[pw2] += 1
                    if (pw2, w) in need_bi:
                        bi[(pw2, w)] += 1
                    if pw1 is not None and (pw1, pw2) in need_c2:
                        c2[(pw1, pw2)] += 1
                        if (pw1, pw2, w) in need_tri:
                            tri[(pw1, pw2, w)] += 1
                pw1, pw2 = pw2, w
    print(f"wiki counts: tokens {wiki_tok}, tri {len(tri)} types",
          flush=True)

    # 留一不适用（用户语料 ≠ wiki 语料, 无自计数泄漏）; v/n_tok 用 wiki 域
    vv = len(set().union(need_uni))  # 下界; 精确词表数影响 floor 级别, 可忽略
    v_wiki = 301363                  # streams 词表类型数（口径标准实测）

    def logp(h1, h2, x):
        p1 = (uni.get(x, 0) + 1) / (wiki_tok + v_wiki)
        if h2 is None:
            return math.log(max(W_UNI * p1, 1e-12))
        p2 = bi.get((h2, x), 0) / c1[h2] if c1.get(h2) else 0.0
        if h1 is None:
            return math.log(max(W_BI * p2 + W_UNI * p1, 1e-12))
        p3 = (tri.get((h1, h2, x), 0) / c2[(h1, h2)]
              if c2.get((h1, h2)) else 0.0)
        return math.log(max(W_TRI * p3 + W_BI * p2 + W_UNI * p1, 1e-12))

    ng_r = np.full((nn, corpus_std.MAX_CAND), -1e9)
    for i, s in enumerate(samples):
        h1, h2 = hist[i]
        win = corpus_std.window_of(s)
        for j, x in enumerate(win):
            if h1 is not None:
                ng_r[i, j] = logp(h1, h2, x)
            elif h2 is not None:
                ng_r[i, j] = logp(None, h2, x)
            else:
                ng_r[i, j] = math.log(W_UNI * 1.0 / (wiki_tok + v_wiki))

    l1 = np.log1p(eff)
    tri10 = (ng_r.argmax(1) == widx).mean()
    tri5 = ((ng_r[:, :5].argmax(1) == widx) & (widx < 5)).mean()
    tri_eff10 = (ng_r + 8.0 * l1).argmax(1) == widx
    tri_eff10 = tri_eff10.mean()
    tri_eff5 = ((ng_r[:, :5] + 8.0 * l1[:, :5]).argmax(1) == widx) & (widx < 5)
    tri_eff5 = tri_eff5.mean()

    # bigram-only 变体（同一份过滤计数, 权重 0/0.75/0.25）—— 2gram vs 3gram
    ng_r_bi = np.full_like(ng_r, -1e9)
    for i, s in enumerate(samples):
        h1, h2 = hist[i]
        win = corpus_std.window_of(s)
        for j, x in enumerate(win):
            p1 = (uni.get(x, 0) + 1) / (wiki_tok + v_wiki)
            if h2 is None:
                ng_r_bi[i, j] = math.log(max(W_UNI * p1, 1e-12))
            else:
                p2 = bi.get((h2, x), 0) / c1[h2] if c1.get(h2) else 0.0
                if h1 is None:
                    lp = math.log(max(W_BI * p2 + W_UNI * p1, 1e-12))
                else:
                    p3 = (tri.get((h1, h2, x), 0) / c2[(h1, h2)]
                          if c2.get((h1, h2)) else 0.0)
                    lp = math.log(max(W_TRI * p3 + W_BI * p2 + W_UNI * p1,
                                      1e-12))
                    # bi-only: 去掉 tri 分量
                    lp = math.log(max(W_BI / 0.3 * p2 + W_UNI / 0.1 * p1
                                      if False else 0.75 * p2 + 0.25 * p1,
                                      1e-12))
            ng_r_bi[i, j] = lp if h2 is not None else ng_r_bi[i, j]
    # 上面的内联分支冗余, 直接重算 bi-only
    for i, s in enumerate(samples):
        h1, h2 = hist[i]
        win = corpus_std.window_of(s)
        for j, x in enumerate(win):
            p1 = (uni.get(x, 0) + 1) / (wiki_tok + v_wiki)
            if h2 is None:
                lp = math.log(max(0.25 * p1, 1e-12))
            else:
                p2 = bi.get((h2, x), 0) / c1[h2] if c1.get(h2) else 0.0
                lp = math.log(max(0.75 * p2 + 0.25 * p1, 1e-12))
            ng_r_bi[i, j] = lp
    bi10 = (ng_r_bi.argmax(1) == widx).mean()
    bi_eff10 = (ng_r_bi + 8.0 * l1).argmax(1) == widx
    bi_eff10 = bi_eff10.mean()

    L = ["=" * 66,
         "用户真实语料（llm_training, 未见过的语料）跨域评测",
         f"提交 {len(pairs)}, dict 词 {len(seq)} ({len(seq)/len(pairs)*100:.1f}%), "
         f"窗口 {nn}（pos>=10 剔除 {n_deep}）",
         "trigram/LSTM/CE 均未见过此语料; eff 从零重放（新用户冷启动）",
         "",
         f"[固定词典] hit@10 = {dict_hit:.4f}",
         f"[纯 eff]   hit@10 = {eff_hit:.4f}",
         f"[trigram(跨域)] @10={tri10:.4f} @5={tri5:.4f} "
         f"+β8·eff @10={tri_eff10:.4f} @5={tri_eff5:.4f}",
         f"[bigram(跨域)] @10={bi10:.4f} +β8·eff @10={bi_eff10:.4f}",
         ""]
    if a.ce:
        with open(CE_INPUT, "w", encoding="utf-8") as f:
            for an in anchors:
                f.write(json.dumps({"context": an["ctx"],
                                    "cands": list(an["win"])},
                                   ensure_ascii=False) + "\n")
        L.append(f"CE 推理输入 → {CE_INPUT} ({len(anchors)} 窗)")
    out = "\n".join(L)
    TRI_REPORT.write_text(out + "\n", encoding="utf-8")
    print(out, flush=True)
    print(f"report → {TRI_REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
