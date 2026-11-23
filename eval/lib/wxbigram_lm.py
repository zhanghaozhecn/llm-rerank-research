#!/usr/bin/env python3
"""wxbigram_lm.py — 微信输入法 lmcnmac.bin 词级 bigram 离线查询适配器

格式（2026-11-23 逆向实证，纯静态分析；报告见研究库
《商业输入法 ngram 逆向-微信豆包-2026-11-23.md》）：
  HashMapReader（libwxhld.dll `wxime::data::HashMapReader`）：
  [u32 magic 0x80000001][u32 nb][u32 bucket_off[nb]]
  bucket: [u32 n][u32 check[n] 排序][u16 cum_val_off[n]][values 背靠背]
  键 = CityHash64WithSeed(word1 + b"\\x00" + word2 (UTF-8), 0x1dbba5) 低 32 位
  —— 文件只存哈希不存词串（键不可枚举、任意词对可离线查询）
  值 = 16-bit 量化对数分：2B [score]（1.12M 条 bigram，实测分数域 [-5.4,-1.0]）
       4B [score, backoff]（0.76M 条，键型未破——非词对/三词/字级/拼音/标记变体
       均未命中；q0 域 [-9.5,-3.0] 疑 unigram，开放问题）
  量化解码（bit 语义，常数 0x3F000000=0.5f / 0x07800000=2^-112 取自 .rdata）：
    m = q & 0x7FFF
    m < 0x400:  f = bits(0x3F000000|m) - 0.5        （近零正域 = 引擎 miss 哨兵）
    否则:       f = bits(0x70000000 | m<<13) * 2^-112
    bit15 = 符号（负 = 真分数，引擎侧 `score > 0` 判 miss）

计分口径（对齐 wxgram.WanxiangGram.query，供 sentence_gen_gram 复用）：
  query(context_text, word, is_rear)：
    取上文尾做**逆向最大匹配**切出上一词 w_prev（词表 = 评测词表），
    查 bigram(w_prev, word)；miss 再试末单字 bigram；再 miss 返回 miss_penalty。
  is_rear 不参与（微信模型无句尾标记证据）。
实现注：纯 dict 索引（~300MB）——本机 numpy 2.x 的 np.searchsorted 标量循环
有病态性能（10 万次 >100s），不用 numpy 查询路径。
依赖：`pip install cityhash`（键长 >32B 的纯 Python CityHash 移植未验证，硬依赖）。
"""
import mmap
import struct

try:
    import cityhash as _city
except ImportError as _e:  # pragma: no cover
    raise ImportError("wxbigram_lm 需要 cityhash 包：pip install cityhash") from _e

_I32 = struct.Struct("<I")
_F32 = struct.Struct("<f")
_H = struct.Struct("<H")
C2_POW = _F32.unpack(_I32.pack(0x07800000))[0]   # 2^-112
CHECK_SEED = 0x1DBBA5
MAGIC = 0x80000001
KEY_MAX_BYTES = 64


def _bits_f(bits):
    return _F32.unpack(_I32.pack(bits & 0xFFFFFFFF))[0]


def dequant(q):
    """16-bit 量化 → float（负 = 有效分数；近零正 = 引擎 miss 哨兵）。"""
    m = q & 0x7FFF
    if m * 0x20000 <= 0x7FFFFFF:
        f = _bits_f(0x3F000000 | m) - 0.5
    else:
        f = _bits_f(0x70000000 | (m << 13)) * C2_POW
    return -f if (q & 0x8000) else f


def check_hash(key_bytes):
    return _city.CityHash64WithSeed(key_bytes, CHECK_SEED) & 0xFFFFFFFF


class WxBigramLM:
    """lmcnmac.bin 离线查询器 + sentence_gen_gram 计分适配。"""

    def __init__(self, path, vocab=None, miss_penalty=-6.0, max_word_len=8):
        self.path = path
        self.miss_penalty = miss_penalty
        self.vocab = vocab or set()
        self.max_word_len = max_word_len
        self.n_query = self.n_hit1 = self.n_hit2 = 0
        with open(path, "rb") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, nb = struct.unpack_from("<II", self.mm, 0)
        if magic != MAGIC:
            raise ValueError("not lmcnmac HashMapReader: magic=%#x" % magic)
        size = len(self.mm)
        offs = struct.unpack_from("<%dI" % nb, self.mm, 8)
        # 纯 dict 平面索引：check_u32 -> off*4 + (1 if 4B else 0)
        index = {}
        for b in range(nb):
            st = offs[b]
            n = struct.unpack_from("<I", self.mm, st)[0]
            p = st + 4
            checks = struct.unpack_from("<%dI" % n, self.mm, p)
            p += 4 * n
            cums = struct.unpack_from("<%dH" % n, self.mm, p)
            base = p + 2 * n
            prev = 0
            for i in range(n):
                c = checks[i]
                cur = cums[i]
                index[c] = (base + prev) * 4 + (1 if cur - prev == 4 else 0)
                prev = cur
        self.index = index
        self.nb = nb
        self.n_items = len(index)
        self._seg_cache = {}
        self._score_cache = {}

    def version_note(self):
        return "wxbigram items=%d buckets=%d" % (self.n_items, self.nb)

    # ── 底层查询 ──
    def pair_score(self, w1, w2):
        key = w1.encode("utf-8") + b"\x00" + w2.encode("utf-8")
        if not key or len(key) > KEY_MAX_BYTES:
            return None
        v = self.index.get(_city.CityHash64WithSeed(key, CHECK_SEED) & 0xFFFFFFFF)
        if v is None:
            return None
        off, four = v >> 2, v & 1
        q0 = _H.unpack_from(self.mm, off)[0]
        return dequant(q0)

    # ── 上文切分（逆向最大匹配，词表 = 评测词表）──
    def _ctx_word(self, tail):
        w = self._seg_cache.get(tail)
        if w is not None:
            return w
        best = ""
        n = len(tail)
        for k in range(min(self.max_word_len, n), 0, -1):
            cand = tail[n - k:]
            if cand in self.vocab:
                best = cand
                break
        if not best and n:
            best = tail[-1]  # 兜底：末单字（可能不在词表，查哈希无妨）
        self._seg_cache[tail] = best
        return best

    # ── sentence_gen_gram 兼容接口 ──
    def query(self, context, word, is_rear=False):
        if not context:
            return self.miss_penalty
        tail = context[-12:]
        key = (tail, word)
        v = self._score_cache.get(key)
        if v is None:
            self.n_query += 1
            w_prev = self._ctx_word(tail)
            v = None
            if w_prev:
                v = self.pair_score(w_prev, word)
                if v is not None:
                    self.n_hit1 += 1
            if v is None and tail and tail[-1] != w_prev:
                v = self.pair_score(tail[-1], word)
                if v is not None:
                    self.n_hit2 += 1
            if v is None:
                v = self.miss_penalty
            self._score_cache[key] = v
        return v

    def close(self):
        try:
            self.mm.close()
        except Exception:
            pass
