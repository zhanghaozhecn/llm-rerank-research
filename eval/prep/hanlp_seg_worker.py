#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hanlp_seg_worker.py — HanLP 分词常驻子进程（corpus_std.ensure_segment_cache 专用）

协议（stdin/stdout 逐行 JSON，UTF-8，双向 flush）:
  收 {"texts": ["文章1", "文章2", ...]}
  回 {"tokens": [["词", ...], ["词", ...]]}          与输入 texts 一一对应

分词口径（= eval_rerank 同款，2026-11-05 定为语料线标准）:
  HanLP COARSE_ELECTRA_SMALL_ZH（bert_seg venv）；混排 token 按 CJK/非CJK
  边界拆开（"450毫米" → "450" + "毫米"），其余整词保留。
常驻进程 = 模型只加载一次（加载约 30-60s，逐批起进程不可接受）。
运行: RIME_LLM_VENV_PY（bert_seg venv）python hanlp_seg_worker.py
"""
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 显存碎片化缓解（批量按字符预算组批后注意力峰值受控, 2026-11-05 OOM 教训:
# 64 篇长文整批 electra 注意力矩阵要 6.12GiB → CUDA OOM 杀死常驻 worker）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def is_cjk(ch):
    return "一" <= ch <= "鿿"


def split_mixed(w):
    """混排 token 按 CJK/非CJK 边界拆成极大段（全 CJK 或全非 CJK 段）"""
    parts, buf, in_cjk = [], [], None
    for ch in w:
        c = is_cjk(ch)
        if in_cjk is None or c == in_cjk:
            buf.append(ch)
        else:
            parts.append("".join(buf))
            buf = [ch]
        in_cjk = c
    parts.append("".join(buf))
    return parts


def tok_budgeted(tok, texts, char_budget=16000, max_batch=64):
    """按字符预算组批推理（批内按长度降序贪心装填, 单篇超预算切成若干片——
    切片边界处的跨片词会被拆开, 概率极低且对重放无害）。
    → 与 texts 等长的 token 列表"""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    results = [None] * len(texts)
    i = 0
    while i < len(order):
        idxs, chars = [], 0
        while i < len(order):
            L = len(texts[order[i]])
            if idxs and (chars + L > char_budget or len(idxs) >= max_batch):
                break
            idxs.append(order[i])
            chars += L
            i += 1
        seg_texts, spans = [], []
        for k in idxs:
            pieces = [texts[k][o:o + char_budget]
                      for o in range(0, len(texts[k]), char_budget)] or [texts[k]]
            for pi, p in enumerate(pieces):
                seg_texts.append(p)
                spans.append(k)
        out = tok(seg_texts)
        if out and isinstance(out[0], str):
            out = [out]
        for k, toks in zip(spans, out):
            if results[k] is None:
                results[k] = []
            results[k].extend(toks)
    return results


def main():
    # 协议通道 = 复制出的真 stdout; sys.stdout 改道 stderr —— hanlp/torch 的加载
    # 进度条与警告若落 stdout 会污染逐行 JSON 协议（2026-11-05 实测: 父进程把
    # "Building model..." 进度行当 JSON 解析报 JSONDecodeError）
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr

    import hanlp
    devices = os.environ.get("HANLP_DEVICES", "")   # 如 "-1"=CPU（默认 hanlp 自动=GPU）
    kw = {"devices": int(devices)} if devices else {}
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH, **kw)
    # 批参数 env 可调（2026-11-05 吞吐基准: 16k/64 是保守初值, GPU 上 64k/512 更快）
    char_budget = int(os.environ.get("HANLP_CHAR_BUDGET", "16000"))
    max_batch = int(os.environ.get("HANLP_MAX_BATCH", "64"))
    print(f"worker ready (budget={char_budget}, max_batch={max_batch}, "
          f"devices={devices or 'auto'})", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        texts = json.loads(line)["texts"]
        out = []
        for toks in tok_budgeted(tok, texts, char_budget, max_batch):
            words = []
            for w in toks:
                if is_cjk(w[0]) and is_cjk(w[-1]):
                    words.append(w)          # 全 CJK 整词
                elif not any(is_cjk(c) for c in w):
                    words.append(w)          # 全非 CJK 整词
                else:
                    words.extend(split_mixed(w))
            out.append(words)
        proto.write(json.dumps({"tokens": out}, ensure_ascii=False) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
