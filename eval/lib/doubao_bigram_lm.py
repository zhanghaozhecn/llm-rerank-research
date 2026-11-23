#!/usr/bin/env python3
"""doubao_bigram_lm.py — 豆包输入法 sys_bigram + sys.dat 词级 bigram 离线适配器

格式（2026-11-23 逆向实证；报告见研究库
《商业输入法 ngram 逆向-微信豆包-2026-11-23.md》§豆包）：

**sys_bigram**（本机 `files/data/dict/9f9ec5e462c2`，26,090,856 B）
  头 u32：`[0]=0x2AFB 魔数 [1]=版本戳(20260313) [2]=索引偏移(108) [3]=文件大小
  [4]=记录数(275460) [5]=108 [6]=索引字节(275460*24)`，其后 6 组
  `(段基址, 段字节, 条目数)`：S1 995832×u32 / S2 148379×6B / S3 149863×10B /
  S4 596033×u32 / S5 400633×8B / S6 1879570×u32，末段尾 == 文件大小。
  索引：每词条 24B = 6×u32，各字段 = 对应段内的**起始条目下标**
  （终点取下一记录；末记录恰好 = 六段条目数，即哨兵，实测逐段精确闭合）。
  **LM = S6**（唯一同时满足「按 w2 id 升序」= 引擎二分前提、且语义正确）：
    条目 u32 = (cost12 << 20) | w2_id20，cost12 = -lnP(w2|w1)×100
    实证：的→怎么 cost 364 / 我→你就 285 / 今天→多 212 / 中国→这 43。
  S1 分数饱和(≈362)且语义乱、S3/S4/S5 非 id 排序 → 非 LM 库（S2 类同）。

**sys.dat**（`991bca749d55`，22,507,... B）提供 id ↔ 词：
  魔数 0x2AFA；每词条切片（S2 + A[wid]）= `{u16 cost, u16 wid, u16 _, u16 nch2,
  u16 cid[nch2/2]}`，字符 = u16 @ (D + cid*4)（variant 0，子代理用 今天/我们/的
  校准）；id 3..6 = ，。！？；1/2 占位。
  **u16 cost = -lnP(w)×100**（的 218 / 我 246 / 中国 610 / 香蕉 783）——与 bigram
  成本字段同尺度，天然构成 Katz 退避。

计分（统一到 ln p 量级，与 octagram/微信适配器可比；higher = better）：
  命中：  score = -(cost12_field) / 100
  miss：  score = -(unigram_cost(w2) + K) / 100      （K = backoff_K，默认 250）
"""
import struct

SYS_BIGRAM_MAGIC = 0x2AFB
SYS_MAGIC = 0x2AFA
SEC_FIELDS = 6          # 索引每记录 6 个段起始
LM_SEC = 5              # S6


class DoubaoVocab:
    """sys.dat：word <-> id + unigram 成本。"""

    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        self.data = data
        hdr = struct.unpack_from("<12I", data, 0)
        if hdr[0] != SYS_MAGIC:
            raise ValueError("sys.dat magic %#x (expect %#x)" % (hdr[0], SYS_MAGIC))
        self.S2 = hdr[8]
        s3 = data[hdr[10]:]
        self.count = struct.unpack_from("<I", s3, 0x18)[0]
        a_off = struct.unpack_from("<I", s3, 0x1c)[0]
        d_off = struct.unpack_from("<I", s3, 0x34)[0]
        self.A = struct.unpack_from("<%dI" % (self.count + 1), s3, a_off)
        self.D = s3[d_off:]
        self.word2id = {}
        self.id2word = {}
        self.cost = {}
        punct = {3: "，", 4: "。", 5: "！", 6: "？"}
        for k, v in punct.items():
            self.word2id[v] = k
            self.id2word[k] = v
            self.cost[k] = 300
        for wid in range(7, self.count + 1):
            off = self.A[wid]
            if off == 0:
                continue
            cost, _wid, _x, nch2 = struct.unpack_from("<HHHH", data, self.S2 + off)
            n = nch2 >> 1
            if n == 0 or n > 40:
                continue
            cids = struct.unpack_from("<%dH" % n, data, self.S2 + off + 8)
            w = "".join(chr(struct.unpack_from("<H", self.D, c * 4)[0])
                        for c in cids)
            self.id2word[wid] = w
            self.cost[wid] = cost
            if w not in self.word2id or wid < self.word2id[w]:
                self.word2id[w] = wid

    def get_id(self, w):
        return self.word2id.get(w, 0)

    def get_word(self, i):
        return self.id2word.get(i, "<%d>" % i)

    def unigram_cost(self, i):
        return self.cost.get(i)


class DoubaoSysBigram:
    """sys_bigram 的 S6（真 bigram）查询。"""

    def __init__(self, path, vocab, backoff_K=0.0, bias=1000.0):
        with open(path, "rb") as f:
            self.data = f.read()
        h = struct.unpack_from("<32I", self.data, 0)
        if h[0] != SYS_BIGRAM_MAGIC:
            raise ValueError("sys_bigram magic %#x" % h[0])
        if h[3] != len(self.data):
            raise ValueError("size field %d != %d" % (h[3], len(self.data)))
        self.N = h[4]
        self.IDX = h[2]
        self.secs = [(h[8], h[7], 4), (h[13], h[12], 6), (h[16], h[15], 10),
                     (h[19], h[18], 4), (h[22], h[21], 8), (h[25], h[24], 4)]
        self.vocab = vocab
        self.K = backoff_K
        self.bias = bias
        base, cnt, sz = self.secs[LM_SEC]
        self.base, self.cnt, self.sz = base, cnt, sz
        self._cache = {}

    def _range(self, w1):
        o = self.IDX + w1 * 24 + LM_SEC * 4
        r0 = struct.unpack_from("<I", self.data, o)[0]
        r1 = struct.unpack_from("<I", self.data, o + 24)[0]
        return r0, min(r1, self.cnt)

    def bigram_cost(self, w1, w2):
        """引擎同口径二分：条目按 w2 id 升序，键 = u32 & 0xFFFFF。"""
        if not (1 <= w1 < self.N) or not w2:
            return None
        r0, r1 = self._range(w1)
        if r1 <= r0:
            return None
        base, sz = self.base, self.sz
        lo, hi = r0, r1
        while lo < hi:
            mid = (lo + hi) >> 1
            u = struct.unpack_from("<I", self.data, base + mid * sz)[0]
            k = u & 0xFFFFF
            if k == w2:
                return u >> 20
            if w2 <= k:
                hi = mid
            else:
                lo = mid + 1
        return None

    def unigram_cost(self, w2):
        c = self.vocab.unigram_cost(w2)
        return c if c is not None else 700

    def pair_score(self, w1, w2):
        """→ 对数分（higher = better）；w1 不在词表则退单字先验。

        引擎同口径 `score = (u >> 20) - 1000`（FUN_14115aa90，存 short）；
        bias 即那个 1000——它是「命中/未命中」相对尺度的标定旋钮。"""
        i1 = self.vocab.get_id(w1)
        i2 = self.vocab.get_id(w2)
        if i2 and i1:
            c = self.bigram_cost(i1, i2)
            if c is not None:
                return -(c - self.bias) / 100.0
        if i2:
            return -(self.unigram_cost(i2) + self.K) / 100.0
        return None


class DoubaoBigramLM:
    """sentence_gen_gram 计分适配（query 口径同 wxbigram_lm.WxBigramLM）。"""

    def __init__(self, bigram_path, sys_path, vocab=None, miss_penalty=-9.0,
                 max_word_len=8, backoff_K=0.0, bias=1000.0):
        self.vocab_set = vocab or set()
        self.miss_penalty = miss_penalty
        self.max_word_len = max_word_len
        self.dv = DoubaoVocab(sys_path)
        self.ds = DoubaoSysBigram(bigram_path, self.dv, backoff_K=backoff_K,
                                  bias=bias)
        self.n_query = self.n_hit1 = self.n_hit2 = 0
        self._seg_cache = {}
        self._score_cache = {}

    def version_note(self):
        return ("doubao bigram: vocab=%d bigram_records=%d S6_entries=%d "
                "K=%.0f bias=%.0f"
                % (len(self.dv.word2id), self.ds.N, self.ds.cnt, self.ds.K,
                   self.ds.bias))

    def _ctx_word(self, tail):
        w = self._seg_cache.get(tail)
        if w is not None:
            return w
        best = ""
        n = len(tail)
        for k in range(min(self.max_word_len, n), 0, -1):
            cand = tail[n - k:]
            if cand in self.vocab_set:
                best = cand
                break
        if not best and n:
            best = tail[-1]
        self._seg_cache[tail] = best
        return best

    def query(self, context, word, is_rear=False):
        key = (context[-12:], word) if context else ("", word)
        v = self._score_cache.get(key)
        if v is not None:
            return v
        self.n_query += 1
        v = None
        if context:
            tail = context[-12:]
            w_prev = self._ctx_word(tail)
            if w_prev:
                v = self.ds.pair_score(w_prev, word)
                if v is not None:
                    self.n_hit1 += 1
            if v is None and tail and tail[-1] != w_prev:
                v = self.ds.pair_score(tail[-1], word)
                if v is not None:
                    self.n_hit2 += 1
        if v is None:
            v = self.miss_penalty
        self._score_cache[key] = v
        return v

    def close(self):
        pass
