# -*- coding: utf-8 -*-
"""解码 scope_*.txt: top-5 的 gpt2 byte-unicode 词表串 → 可读文本, 输出对比表."""
import io, sys, re, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 标准 GPT-2 bytes_to_unicode 逆映射 (已用 改革/开放/天国/之日起/今天天气 锚点验证)
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
c2b = {}
for b, c in zip(bs, cs):
    c2b[chr(c)] = b

def gpt2_decode(s):
    out = bytearray()
    for ch in s:
        if ch in c2b:
            out.append(c2b[ch])
        else:
            out.extend(ch.encode("utf-8"))
    try:
        return out.decode("utf-8", errors="replace")
    except Exception:
        return repr(out)

for path in sorted(glob.glob("scope_*.txt")):
    print("=" * 30, path, "=" * 30)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("ctx="):
                continue
            m = re.match(r'ctx="([^"]+)" ntok=(\d+) \| (.*?) \| ce:(.*)', line)
            if not m:
                print(line)
                continue
            ctx, ntok, top5, ce = m.groups()
            items = re.findall(r" (\S+)\((-?\d+\.\d+)\)", top5)
            tops = " ".join("%s(%s)" % (gpt2_decode(t), v) for t, v in items)
            print('ctx="%s" ntok=%s | top5: %s |%s' % (ctx, ntok, tops, ce))
    print()
