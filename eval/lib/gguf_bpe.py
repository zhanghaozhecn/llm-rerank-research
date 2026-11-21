#!/usr/bin/env python3
"""GGUF 词表复刻的字节级 BPE（llama.cpp 口径），用于分词相关分析。

2026-11-18 起用于：
  - 比较不同模型分词器的上下文覆盖长度（ctx=10 各覆盖多少汉字）；
  - 构造"每候选恰好 3 token 且首 token 互不相同"的合成候选（延迟最坏情况）。
校验：CE 结果里的 `ntok` 与 `ntok()` 输出 302/302 逐词一致（0.5B/0.8B 各一次）。

用法:
  from gguf_bpe import BPE
  t = BPE("d:/gguf_models/Qwen3.5-0.8B-Q4_K_M.gguf")
  t.ntok("饰演")   # 3 token 数
  t.pieces("...")  # 每个 token 的字节级串
"""
import regex

# llama.cpp src/llama-vocab.cpp 的预分词正则
RE_QWEN2 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
            r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
            r"|\s*[\r\n]+|\s+(?!\S)|\s+")
RE_QWEN35 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
             r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
             r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


B2U = bytes_to_unicode()


class BPE:
    def __init__(self, path):
        from gguf import GGUFReader
        r = GGUFReader(path, "r")
        toks = list(r.get_field("tokenizer.ggml.tokens").contents())
        mg = r.get_field("tokenizer.ggml.merges")
        merges = list(mg.contents()) if mg is not None else []
        pre = r.get_field("tokenizer.ggml.pre")
        pre_s = pre.contents() if pre is not None else "qwen2"
        self.vocab = {t: i for i, t in enumerate(toks)}
        self.ranks = {}
        for i, m in enumerate(merges):
            a, b = m.split(" ", 1)
            self.ranks[(a, b)] = i
        self.re = regex.compile(RE_QWEN35 if "qwen35" in str(pre_s) else RE_QWEN2)

    def _bpe(self, piece):
        word = list(piece)
        if len(word) < 2:
            return word
        while True:
            cand = [(self.ranks[p], i) for i, p in enumerate(zip(word, word[1:]))
                    if p in self.ranks]
            if not cand:
                return word
            _, i = min(cand)
            word = word[:i] + [word[i] + word[i + 1]] + word[i + 2:]

    def pieces(self, text):
        out = []
        for m in self.re.finditer(text):
            by = "".join(B2U[b] for b in m.group(0).encode("utf-8"))
            out.extend(self._bpe(by))
        return out

    def ntok(self, text):
        return len(self.pieces(text))
