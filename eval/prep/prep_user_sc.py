#!/usr/bin/env python3
"""用户语料单字 3 码测试窗（2026-11-12 一字词重排研究）。

输入: %APPDATA%/Rime/llm_rerank/rime_llm_corpus.tsv（v2；iter_scenes strict 已处理 M 回撤）
      + 词典单字 3 码口径（corpus_std.load_pdsp_dict_single3，正典映射）
窗资格: CJK 单字 + 提交码长==3（用户实际打的全码）+ 同码窗 ≥2 + pos<10。
候选窗 = 词典序 code_to_words_sc[code][:MAX_CAND]（与百科标定域同构）；
日志里的真实候选窗快照另存 real_win 字段做对照（不改变窗构造）。
eff/uni 重放: FormulaReplay(τ=400) 单人流，事前 eff；ctx = 提交词典词拼接
的 CTX_TAIL 尾（与 user_anchors_std 同法；非词典提交不入 ctx/bump——
与 wiki 域 map_article 只产词典词同口径）。
输出: user_sc_samples.jsonl {ci, ctx, word, code, pos, cands, eff_win,
winsize, real_win, depth}——供 GPU CE 与标定脚本消费。

用法: python eval/prep_user_sc.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from corpus_std import CTX_TAIL

PROJ = corpus_std.PROJ
# 2026-11-17 记录重构: 统一语料 v2（迭代回撤已在 iter_scenes 内按 M.kind 处理）
TRAIN_LOG = corpus_std.CORPUS_PATH
OUT = PROJ / "user_sc_samples.jsonl"


def main():
    c2w_sc, wc_sc = corpus_std.load_pdsp_dict_single3()
    words_all = set(corpus_std.load_pdsp_words())
    print(f"dict: {len(wc_sc)} 正典单字词, {len(c2w_sc)} 码组", flush=True)

    scenes = []
    for scene in corpus_std.iter_scenes(TRAIN_LOG, reset_mode="strict"):
        scenes.append([(c["word"], c["code"], c.get("real_win"))
                       for c in scene])
    print(f"scenes: {len(scenes)}", flush=True)

    rp = corpus_std.FormulaReplay(400.0)          # 生产 τ（v3 定案）
    uni, n_seen, ci = {}, 0, 0                    # ci = 词典词提交序（与
    ctx_str = ""                                  #   user_matrices seq 同口径）
    stats = {"commits": 0, "cjk1": 0, "code3": 0, "kept": 0,
             "no_canon": 0, "pos_ge": 0, "real_win": 0}
    rows = []
    for scene in scenes:
        words = scene
        for w, c, cw in words:
            stats["commits"] += 1
            is_dict = w in words_all
            if len(w) == 1 and "\u4e00" <= w <= "\u9fff":
                stats["cjk1"] += 1
                if len(c) == 3 and not c.startswith("\\"):
                    stats["code3"] += 1
                    full = c2w_sc.get(c)
                    if full and w in full:
                        pos = full.index(w)
                        if pos < corpus_std.MAX_CAND and len(full) >= 2:
                            win = full[:corpus_std.MAX_CAND]
                            eff = {x: rp.eff_of(x) for x in win}
                            rows.append({
                                "ci": ci, "ctx": ctx_str[-CTX_TAIL:],
                                "word": w, "code": c, "pos": pos,
                                "cands": [x for x in win if x != w]
                                [:corpus_std.MAX_CAND - 1],
                                "eff_win": eff, "winsize": len(full),
                                "real_win": cw, "uni_win":
                                {x: uni.get(x, 0) for x in win},
                                "depth": ci, "n_seen": n_seen})
                            stats["kept"] += 1
                            if cw:
                                stats["real_win"] += 1
                        else:
                            stats["pos_ge"] += 1
                    elif full is not None:
                        stats["pos_ge"] += 1
                    else:
                        stats["no_canon"] += 1
            if is_dict:
                # 词典词：窗已在 bump 前取（事前 eff 口径）；此处落账
                rp.bump(w)
                uni[w] = uni.get(w, 0) + 1
                n_seen += 1
                ci += 1
                ctx_str += w
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"stats: {stats}", flush=True)
    print(f"done -> {OUT} ({len(rows)} windows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
