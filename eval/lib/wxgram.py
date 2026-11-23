#!/usr/bin/env python3
"""wxgram.py — 万象 LTS `.gram`（Rime::DartsGrammar/3）纯 Python 读取器

模型 = `librime-evo-octagram` 的分区 Darts 双数组格式（万象 400M LTS 模型）：
  Metadata(48B): char format[32] = "Rime::DartsGrammar/3"; u64 section_count;
                 u64 names_bytes
  Section[count] (24B each): u64 array_bytes; i64 constant_value; u64 name_bytes
  names: 各分区名顺序拼接（补零到 4 字节对齐）
  images: 各分区 Darts 双数组顺序拼接，每段 array_bytes 字节

计分与 `librime-evo-octagram/src/octagram.cc` 的 `Octagram::Query` 一致：
  score = max over 上下文后缀 × 词前缀匹配（log 频次 + collocation/weak 罚分）
  默认参数取自万象 schema：[6, 2, -14, -6, -100, -18]（max_len, min_len,
  collocation_penalty, non_collocation_penalty, weak_collocation_penalty,
  rear_penalty）。整句内部造句用全部分区（poet 语义）。

用法（调试）：
  python wxgram.py <model.gram> 今天天气 很好
"""
import mmap
import struct
import sys

MAGIC_V3 = b"Rime::DartsGrammar/"
MAGIC_V1 = b"Rime::Grammar/"
META = struct.Struct("<32sQQ")      # v3
META_V1 = struct.Struct("<32sIIi")  # format, checksum, units, offset
SECT = struct.Struct("<QqQ")
UNIT = struct.Struct("<I")
K_MAX_RESULTS = 8
K_VALUE_SCALE = 10000.0

# 万象 algebra 的 xlit：声调 → 无调字母，ü → v
_TONE = str.maketrans({
    "ā": "a", "á": "a", "ǎ": "a", "à": "a",
    "ē": "e", "é": "e", "ě": "e", "è": "e",
    "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
    "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
    "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
    "ü": "v", "ǖ": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v",
    "ń": "n", "ň": "n", "ǹ": "n", "ḿ": "m",
})


def normalize_pinyin(code):
    """词典读音 → 用户实际键入的无调码（空格分隔音节 → 连写）。"""
    return code.translate(_TONE).replace(" ", "").replace("'", "").lower()


def encode(text):
    """grammar::encode：UTF-8 → 压缩编码（librime-evo-octagram/gram_encoding）。"""
    out = bytearray()
    for ch in text:
        u = ord(ch)
        if u < 0x80:
            out.append(0xE0 if u == 0 else u)
        elif 0x4000 <= u < 0xA000:
            hi = u >> 8
            lo = u & 0xFF
            if lo == 0:
                out.append(0xE1)
                out.append(hi + 0x40)
            else:
                out.append(hi + 0x40)
                out.append(lo)
        else:
            bits = 32
            v = u
            while bits > 0 and (v & 0xFE000000) == 0:
                bits -= 7
                v = (v << 7) & 0xFFFFFFFF
            n = (bits + 6) // 7
            out.append(0xE0 | n)
            while n > 0:
                n -= 1
                out.append(((v >> 25) & 0x7F) | 0x80)
                v = (v << 7) & 0xFFFFFFFF
    return bytes(out)


def _advance(enc, pos):
    b = enc[pos]
    if not (b & 0x80):
        return pos + 1
    if (b & 0xF0) == 0xE0:
        return pos + (b & 0x0F) + 1
    return pos + 2


def unicode_length(enc, length):
    n = 0
    p = 0
    while p < length:
        p = _advance(enc, p)
        n += 1
    return n


class WanxiangGram:
    def __init__(self, path, max_len=6, min_len=2, collocation_penalty=-14.0,
                 non_collocation_penalty=-6.0, weak_collocation_penalty=-100.0,
                 rear_penalty=-18.0):
        self.max_len = max_len
        self.min_len = min_len
        self.collocation_penalty = collocation_penalty
        self.non_collocation_penalty = non_collocation_penalty
        self.weak_collocation_penalty = weak_collocation_penalty
        self.rear_penalty = rear_penalty
        with open(path, "rb") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.n = len(self.mm)
        fmt = self.mm[:32]
        if fmt.startswith(MAGIC_V3):
            self._load_v3()
        elif fmt.startswith(MAGIC_V1):
            self._load_v1()
        else:
            raise ValueError("unknown gram format: %r" % fmt)
        self._unit = UNIT
        self._mm = self.mm
        self.version = "v3" if fmt.startswith(MAGIC_V3) else "v1"

    def _load_v3(self):
        fmt, count, names_bytes = META.unpack_from(self.mm, 0)
        if count == 0:
            raise ValueError("empty section table")
        table_end = META.size + count * SECT.size
        padded = (names_bytes + 3) & ~3
        if table_end + padded > self.n:
            raise ValueError("truncated gram file")
        names = self.mm[table_end:table_end + names_bytes]
        self.sections = []  # (name, image_offset, constant_value, units)
        off = table_end + padded
        pos = 0
        for i in range(count):
            array_bytes, const_val, name_bytes = SECT.unpack_from(
                self.mm, META.size + i * SECT.size)
            name = names[pos:pos + name_bytes].decode("utf-8", "replace")
            pos += name_bytes
            if off + array_bytes > self.n:
                raise ValueError("truncated section image: " + name)
            self.sections.append((name, off, const_val, array_bytes // 4))
            off += array_bytes

    def _load_v1(self):
        """旧版单 trie：Metadata = format[32] + checksum + units + OffsetPtr。"""
        fmt, checksum, units, rel = META_V1.unpack_from(self.mm, 0)
        image = META_V1.size - 4 + rel  # OffsetPtr 相对自身字段地址
        avail = (self.n - image) // 4
        # 发布文件曾按 ShrinkToFit 截掉尾部未用单元（声明 units 略大于实际）
        self.declared_units = units
        if units > avail:
            units = avail
        self.sections = [("default", image, -1, units)]

    @property
    def version_note(self):
        return "%s units=%d/%d" % (self.version, self.sections[-1][3],
                                   getattr(self, "declared_units",
                                           self.sections[-1][3]))

    # ── Darts 双数组 ────────────────────────────────────────────────
    def _u(self, image, pos):
        return self._unit.unpack_from(self._mm, image + pos * 4)[0]

    @staticmethod
    def _offset(u):
        return (u >> 10) << ((u & 0x200) >> 6)

    @staticmethod
    def _label(u):
        return (u & 0x80000000) | (u & 0xFF)

    def _traverse(self, image, key):
        node = 0
        unit = self._u(image, node)
        for b in key:
            node ^= self._offset(unit) ^ b
            if image + node * 4 + 4 > self.n:
                return -1
            unit = self._u(image, node)
            if self._label(unit) != b:
                return -1
        return node

    def _common_prefix(self, image, key, node):
        """返回 [(匹配字节数, 值)]，最多 K_MAX_RESULTS 个。"""
        out = []
        unit = self._u(image, node)
        node ^= self._offset(unit)
        for i, b in enumerate(key):
            node ^= b
            if image + node * 4 + 4 > self.n:
                break
            unit = self._u(image, node)
            if self._label(unit) != b:
                break
            node ^= self._offset(unit)
            if (unit >> 8) & 1:
                out.append((i + 1, self._u(image, node) & 0x7FFFFFFF))
                if len(out) >= K_MAX_RESULTS:
                    break
        return out

    def _lookup_section(self, name, image, const_val, ctx_enc, word_enc):
        node = self._traverse(image, ctx_enc)
        if node < 0:
            return []
        matches = self._common_prefix(image, word_enc, node)
        if const_val >= 0:
            return [(ln, const_val) for ln, _ in matches]
        return matches

    def lookup(self, ctx_enc, word_enc):
        """全部分区合并（同匹配长度时后面的分区覆盖）。"""
        if not word_enc or not self.sections:
            return []
        merged = []
        for name, off, const_val, _units in self.sections:
            for ln, val in self._lookup_section(name, off, const_val,
                                                ctx_enc, word_enc):
                pos = 0
                while pos < len(merged) and merged[pos][0] < ln:
                    pos += 1
                if pos < len(merged) and merged[pos][0] == ln:
                    merged[pos] = (ln, val)
                elif len(merged) < K_MAX_RESULTS:
                    merged.insert(pos, (ln, val))
                    merged = merged[:K_MAX_RESULTS]
        return merged

    # ── Octagram::Query ───────────────────────────────────────────
    def query(self, context, word, is_rear=False):
        if not context:
            return self.non_collocation_penalty
        n = min(8, self.max_len - 1)
        ctx_tail = context[-n:]
        ctx_enc = encode(ctx_tail)
        word_enc = encode(word[:n])
        result = self.non_collocation_penalty
        pos = 0
        context_len = len(ctx_tail)
        while context_len > 0:
            matches = self.lookup(ctx_enc[pos:], word_enc)
            for enc_len, val in matches:
                match_len = unicode_length(word_enc, enc_len)
                colloc_len = context_len + match_len
                whole = (pos == 0 and enc_len == len(word_enc))
                score = val / K_VALUE_SCALE + (
                    self.collocation_penalty
                    if (colloc_len >= self.min_len or whole)
                    else self.weak_collocation_penalty)
                if score > result:
                    result = score
            pos = _advance(ctx_enc, pos)
            context_len -= 1
        if is_rear and len(word) <= n:
            m = self.lookup(word_enc, encode("$"))
            if m and m[0][1] / K_VALUE_SCALE + self.rear_penalty > result:
                result = m[0][1] / K_VALUE_SCALE + self.rear_penalty
        return result


def main():
    if len(sys.argv) < 4:
        print("usage: python wxgram.py <model.gram> <context> <word>")
        return 2
    g = WanxiangGram(sys.argv[1])
    print("sections:", [(s[0], s[3]) for s in g.sections])
    print("query(%r, %r) = %.3f"
          % (sys.argv[2], sys.argv[3],
             g.query(sys.argv[2], sys.argv[3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
