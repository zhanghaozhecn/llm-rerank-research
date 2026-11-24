#!/usr/bin/env python3
"""llm_gen_a_eval.py — A 路线质量评估（研究工具，2026-11-24）

读 llm_gen_a.exe decode 的产物，与 N-gram 组句、留档 LLM 贪心同口径对比：
  · 整句全对 hit@1/@5/@10
  · 首词正确率（= 前缀确认真正依赖的量）
  · 字级准确率（1 - 字编辑距离/长度）
  · 约束校验：生成文本的拼音是否逐字母等于输入码（必须 100%）

用法：
  python llm_gen_a_eval.py --out eval/llm_gen_a/out_A.tsv \
      --jobs eval/llm_gen_a/jobs_A.tsv \
      --sets eval/sgen_gram_100.test.jsonl eval/sgen_gram_600k.test.jsonl
"""
import argparse
import glob
import io
import json
import os
import unicodedata
from collections import defaultdict

CJK_LO, CJK_HI = 0x4E00, 0x9FFF
BLACKLIST = {"n", "ng", "m", "hm", "hng"}


def is_cjk(ch):
    return CJK_LO <= ord(ch) <= CJK_HI


def norm_reading(s):
    if not s:
        return None
    s = s.strip().lower()
    for v in "üǖǘǚǜ":
        s = s.replace(v, "v")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    if s in BLACKLIST or not s.isalpha():
        return None
    return s


def load_readings(dict_dir):
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


def pinyin_of(text, c2r):
    """逐字取读音；多音字取能与 code 对齐的那组（这里只做校验，取第一读也可）。"""
    out = []
    for ch in text:
        rs = c2r.get(ch)
        if not rs:
            return None
        out.append(sorted(rs))
    return out


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--dict-dir", default="D:/gguf_models/wanxiang/dicts")
    a = ap.parse_args()

    meta = {}
    for p in a.sets:
        tag = "A" if "600k" not in p else "B"
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                meta["%s%d" % (tag, r["sid"])] = r

    code_of = {}
    with io.open(a.jobs, encoding="utf-8") as f:
        for ln in f:
            sid, ctx, code = ln.rstrip("\n").split("\t")
            code_of[sid] = code

    gens = defaultdict(list)  # sid -> [(rank, text, logp, lead, lead_p, ntok)]
    with io.open(a.out, encoding="utf-8") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            gens[p[0]].append((int(p[1]), p[2], float(p[3]), p[4], float(p[5]),
                               int(p[6])))

    c2r = load_readings(a.dict_dir)

    def code_ok(text, code):
        """生成文本能否用某组读音逐字拼成 code。"""
        opts = pinyin_of(text, c2r)
        if opts is None:
            return None
        # DP：多音字组合里是否存在一条 == code
        cur = {0}
        for rs in opts:
            nxt = set()
            for pos in cur:
                for r in rs:
                    if code.startswith(r, pos) and pos + len(r) in (set() or {}):
                        pass
                    if code.startswith(r, pos):
                        nxt.add(pos + len(r))
            cur = nxt
            if not cur:
                return False
        return len(code) in cur

    stats = defaultdict(lambda: defaultdict(int))
    bad_ct = 0
    tot_ct = 0
    rows = []
    for sid, items in sorted(gens.items()):
        r = meta.get(sid)
        if not r:
            continue
        tag = sid[0]
        items.sort()
        texts = [t for _, t, *_ in items]
        truth = r["truth"]
        w1 = r["words"][0]
        code = code_of.get(sid, "")
        if texts and texts[0]:
            tot_ct += 1
            ok = code_ok(texts[0], code)
            if ok is False:
                bad_ct += 1
        st = stats[tag]
        st["n"] += 1
        if texts:
            if texts[0] == truth:
                st["hit1"] += 1
            if truth in texts[:5]:
                st["hit5"] += 1
            if truth in texts[:10]:
                st["hit10"] += 1
            if texts[0].startswith(w1):
                st["lead1"] += 1
            d = levenshtein(texts[0], truth)
            st["characc_x1000"] += int(round(
                1000.0 * (1.0 - d / float(max(len(texts[0]), len(truth))))))
        rows.append((sid, truth, texts[0] if texts else "", len(texts)))

    def report(tag):
        st = stats[tag]
        n = max(st["n"], 1)
        print("%s  n=%3d  整句 hit@1 %5.1f%%  hit@5 %5.1f%%  hit@10 %5.1f%%"
              "  首词 %5.1f%%  字级 %5.1f%%" % (
                  tag, st["n"], 100.0 * st["hit1"] / n, 100.0 * st["hit5"] / n,
                  100.0 * st["hit10"] / n, 100.0 * st["lead1"] / n,
                  st["characc_x1000"] / 10.0 / n))

    print("A 路线（拼音约束 LLM 直接组句）")
    report("A")
    report("B")
    allst = defaultdict(int)
    for tag in ("A", "B"):
        for k, v in stats[tag].items():
            allst[k] += v
    n = max(allst["n"], 1)
    print("ALL n=%3d  整句 hit@1 %5.1f%%  hit@5 %5.1f%%  hit@10 %5.1f%%"
          "  首词 %5.1f%%  字级 %5.1f%%" % (
              allst["n"], 100.0 * allst["hit1"] / n, 100.0 * allst["hit5"] / n,
              100.0 * allst["hit10"] / n, 100.0 * allst["lead1"] / n,
              allst["characc_x1000"] / 10.0 / n))
    print("\n约束校验（生成文本拼音 == 输入码）：%d/%d 合规，%d 违规" % (
        tot_ct - bad_ct, tot_ct, bad_ct))
    print("\n样例（truth | 生成）：")
    for sid, truth, gen, k in rows[:6]:
        print("  %s %s | %s  (%d 候选)" % (sid, truth, gen, k))


if __name__ == "__main__":
    main()
