#!/usr/bin/env python3
"""llm_gen_a_index.py — A 路线索引构建（研究工具，2026-11-24）

从模型词表（llm_gen_a.exe dump-vocab 产物）生成「拼音字母串 → token id」索引。
这是 A 路线替代万象 .gram（419MB）+ 1.7M 词库的全部依赖：只剩 2 万多个汉字
token 的读音表。

三个关键口径：
① 读音来源 = **万象词典的单字条**（带调，权威、与输入法音节表一致），
   去调、ü→v 归一；字典里没有的字回退 pypinyin（同样归一、过滤野读音）。
② **野读音过滤**：`嗯→n/ng`、`呣→m` 这类只作叹词的读音会让 `yun` 被拆成
   `预+嗯`，必须剔除（黑名单 BLACKLIST）。
③ **只保留简体 token**：繁体变体（t2s 会变字）直接从索引里删掉——否则
   同一字母串在繁/简两份 token 上分摊概率，模型还会优先吐繁体。

用法：
  python llm_gen_a_index.py --vocab eval/llm_gen_a/vocab.tsv \
      --out eval/llm_gen_a/index.tsv [--dict-dir D:/gguf_models/wanxiang/dicts] \
      [--max-combos 6] [--max-chars 6] [--check jobs.jsonl ...]
"""
import argparse
import glob
import io
import json
import os
import re
import unicodedata
from collections import defaultdict

CJK_LO, CJK_HI = 0x4E00, 0x9FFF
BLACKLIST = {"n", "ng", "m", "hm", "hng"}  # 叹词读音，会破坏音节切分
SYL_RE = re.compile(r"^[a-z]{2,6}$")


def is_cjk(ch):
    return CJK_LO <= ord(ch) <= CJK_HI


def norm_reading(s):
    """带调拼音 → 无调字母串（ü→v），并剔除非法/叹词读音。"""
    if not s:
        return None
    s = s.strip().lower()
    s = s.replace("ü", "v").replace("ǖ", "v").replace("ǘ", "v").replace("ǚ", "v").replace("ǜ", "v")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ü", "v").replace("u:", "v")
    if s in BLACKLIST or not SYL_RE.match(s):
        return None
    return s


def load_dict_readings(dict_dir):
    """万象词典单字条 → {字: {无调读音}}。"""
    c2r = defaultdict(set)
    for p in sorted(glob.glob(os.path.join(dict_dir, "*.dict.yaml"))):
        with io.open(p, encoding="utf-8") as f:
            body = False
            for ln in f:
                if ln.startswith("..."):
                    body = True
                    continue
                if not body or ln.startswith("#"):
                    continue
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                w, code = parts[0], parts[1].strip()
                if len(w) == 1 and code and " " not in code:
                    r = norm_reading(code)
                    if r:
                        c2r[w].add(r)
    return c2r


def load_vocab(path):
    """读 dump-vocab 产物；BPE 里有很多非 UTF-8 的字节片段，逐行跳过。"""
    out = []
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.rstrip(b"\r\n")
            if not raw:
                continue
            tid_s, _, text_b = raw.partition(b"\t")
            try:
                out.append((int(tid_s), text_b.decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict-dir", default="D:/gguf_models/wanxiang/dicts")
    ap.add_argument("--max-combos", type=int, default=6)
    ap.add_argument("--max-chars", type=int, default=6)
    ap.add_argument("--check", nargs="*", default=None)
    ap.add_argument("--allow-traditional", action="store_true")
    a = ap.parse_args()

    from pypinyin import pinyin, Style
    try:
        import opencc
        t2s = opencc.OpenCC("t2s")
    except Exception:
        t2s = None
        print("[warn] opencc 不可用，跳过简体过滤")

    c2r = load_dict_readings(a.dict_dir)
    print("词典单字读音: %d 字" % len(c2r))

    def readings(ch):
        rs = set(c2r.get(ch, ()))
        if not rs:  # 回退 pypinyin
            got = pinyin(ch, style=Style.NORMAL, heteronym=True,
                         errors=lambda x: [None])
            if got and got[0]:
                for r in got[0]:
                    n = norm_reading(r)
                    if n:
                        rs.add(n)
        return sorted(rs)

    vocab = load_vocab(a.vocab)
    entries = defaultdict(list)
    lens = defaultdict(int)
    n_cjk = n_trad = 0
    for tid, text in vocab:
        if not (1 <= len(text) <= a.max_chars) or not all(is_cjk(c) for c in text):
            continue
        n_cjk += 1
        if t2s and not a.allow_traditional and t2s.convert(text) != text:
            n_trad += 1
            continue
        per = [readings(c) for c in text]
        if any(not p for p in per):
            continue
        combos = [""]
        for opts in per:
            combos = [pre + o for pre in combos for o in opts][:a.max_combos]
            if not combos:
                break
        if not combos:
            continue
        lens[len(text)] += 1
        for ls in combos:
            entries[ls].append((tid, text))

    for k in entries:
        uniq = {}
        for tid, text in entries[k]:
            uniq.setdefault(tid, text)
        entries[k] = sorted(uniq.items())

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for ls in sorted(entries):
            f.write("%s\t%s\n" % (ls, ";".join(str(t) for t, _ in entries[ls])))

    print("vocab tokens: %d（纯汉字 %d，其中繁体 %d 已剔除）" % (
        len(vocab), n_cjk, n_trad))
    print("入索引 token 字长分布: %s" % dict(sorted(lens.items())))
    print("字母串: %d 条，最长 %d" % (
        len(entries), max((len(k) for k in entries), default=0)))
    sizes = sorted(len(v) for v in entries.values())
    print("同串 token 数: 中位 %d / p95 %d / max %d" % (
        sizes[len(sizes) // 2], sizes[int(len(sizes) * 0.95)], sizes[-1]))

    if a.check:
        check_reach(a.check, entries)


def reachable(code, truth, entries, maxlen):
    """真值可达：存在 token 序列，文本 == truth 且字母串逐段拼成 code。"""
    L, T = len(code), len(truth)
    seen = {(0, 0)}
    cur = [(0, 0)]
    while cur:
        nxt = []
        for cp, tp in cur:
            if cp == L and tp == T:
                return True
            if cp >= L or tp >= T:
                continue
            for k in range(1, min(maxlen, L - cp) + 1):
                for _tid, txt in entries.get(code[cp:cp + k], ()):
                    if truth.startswith(txt, tp):
                        st = (cp + k, tp + len(txt))
                        if st not in seen:
                            seen.add(st)
                            nxt.append(st)
        cur = nxt
    return False


def check_reach(paths, entries):
    maxlen = max((len(k) for k in entries), default=1)
    tot = ok = 0
    per = []
    for p in paths:
        n = m = 0
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                n += 1
                if reachable(r["code"], r["truth"], entries, maxlen):
                    m += 1
        per.append((os.path.basename(p), m, n))
        tot += n
        ok += m
    print("\n真值可达率（约束下能拼出真值 = A 路线准确率天花板）")
    for name, m, n in per:
        print("  %-28s %3d/%3d = %.1f%%" % (name, m, n, 100.0 * m / max(n, 1)))
    print("  %-28s %3d/%3d = %.1f%%" % ("ALL", ok, tot, 100.0 * ok / max(tot, 1)))


if __name__ == "__main__":
    main()
