#!/usr/bin/env python3
"""precommit_conf_collect.py — 「确定前缀预提交」效果研究 · 数据采集

背景：预提交模式（2026-11-23 设计讨论）在每一决策点（第 i 个词）要判断
「头部词是否足够确定」：确定则自动确认该词、继续；不确定才让用户选。
本工具不改输入法，只做离线测量——用真实引擎枚举「剩余码」的候选，取出与
头部词同码的候选集（词级候选），交给 LLM 打分（precommit_conf_score.lua），
由 precommit_conf_report.py 汇总 margin → 准确率 / 覆盖率 / 按键成本曲线。

判定口径：
  · 头候选 = 候选词的词典码恰等于第 i 个词的码（覆盖头部词的全部音节）；
  · 生产窗 = 该候选在引擎原序中的整体位置 < 16（= 现行 Collect 上限，
    超出则生产上用户也看不到/选不到）；
  · 句子可行性 = 每个非尾词的真值都在其头候选里（否则预提交模式在这一步
    根本无法选对，句子另计 infeasible）。

流程：
  wiki 分段缓存 → 选句（3..12 词、全 CJK、码可得）→ 逐决策点构造
  （ctx = 前句尾部 + 已确认前缀；rem = 第 i 词起的连写码）→ 探针批量 Q →
  取头候选 → 写 out.jobs.jsonl + out.score_in.tsv（供 lua 批量打分）。

用法：
  python precommit_conf_collect.py --shared "<rime shared data dir>" \
      [--sentences 100] [--articles 3000] [--out precommit_conf]
  --shared 为该机 Weasel 安装目录的 data（含 luna_pinyin 等方案与词典）；
  --user 默认 %APPDATA%\\Rime（luna_probe 方案副本所在处）。
"""
import argparse
import json
import os
import re
import subprocess
import sys

import opencc
from pypinyin import lazy_pinyin

CJK = re.compile(r"[\u3400-\u9fff]")
WORD_SEP = "\x1f"


def default_paths():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))  # research 根
    repo = os.path.dirname(proj)
    user = os.path.join(os.environ.get("APPDATA", ""), "Rime")
    return {
        "segments": os.path.join(proj, "segments_hanlp_cache.jsonl"),
        "probe": os.path.join(repo, "rime-llm-ime", "bin",
                              "rime_cand_probe.exe"),
        "bin": os.path.join(repo, "rime-llm-ime", "bin"),
        "user": user,
        "dict": os.path.join(user, "luna_pinyin.dict.yaml"),
    }


def load_dict(path):
    """词 → 码集合（繁体条目；同一词多读音 = 多行 = 多码）。"""
    word_codes = {}
    started = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not started:
                if line.strip() == "...":
                    started = True
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            w = parts[0]
            # 码字段 = 一个读音（音节空格分隔）——整行拼成一个码；多读音 =
            # 多行 = 集合里多个码
            c = parts[1].replace(" ", "")
            if c:
                word_codes.setdefault(w, set()).add(c)
    return word_codes


def codes_of(word, word_codes, cc):
    """词 → 码集合（繁简都试）。不在词典 = 空集（OOV 不是引擎词）。"""
    s = word_codes.get(word)
    if s:
        return s
    return word_codes.get(cc.convert(word), set())


def pick_sentences(path, skip, articles, min_words, max_words, limit):
    """逐行读 HanLP 分段缓存（每行 = 一段的 token 列表）；按标点/非 CJK
    token 切句，取 3..max_words 词的 CJK 句子。skip = 跳过前 N 行（换域）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= articles:
                break
            if idx < skip:
                continue
            try:
                toks = json.loads(line)
            except json.JSONDecodeError:
                continue
            cur = []
            for t in toks:
                cjk = (isinstance(t, str) and t
                       and all(CJK.match(ch) for ch in t))
                if cjk:
                    cur.append(t)
                    continue
                if cur:
                    if min_words <= len(cur) <= max_words:
                        out.append(cur)
                        if len(out) >= limit:
                            return out
                    cur = []
            if cur and min_words <= len(cur) <= max_words:
                out.append(cur)
                if len(out) >= limit:
                    return out
    return out


def main():
    d = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default=d["segments"])
    ap.add_argument("--probe", default=d["probe"])
    ap.add_argument("--bin", default=d["bin"])
    ap.add_argument("--shared", required=True,
                    help="rime shared data dir（含方案/词典）")
    ap.add_argument("--user", default=d["user"])
    ap.add_argument("--schema", default="luna_probe")
    ap.add_argument("--dict", default=None,
                    help="词典路径（缺省 = <shared>/luna_pinyin.dict.yaml）")
    ap.add_argument("--sentences", type=int, default=100)
    ap.add_argument("--articles", type=int, default=3000)
    ap.add_argument("--skip-articles", type=int, default=0,
                    help="跳过前 N 行（换域采样）")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--max-head", type=int, default=40,
                    help="每个决策点记录的头候选上限（引擎序）")
    ap.add_argument("--score-cap", type=int, default=12,
                    help="交给 LLM 打分的头候选上限（= DLL n_seq_max 12）")
    ap.add_argument("--win", type=int, default=16,
                    help="生产收集窗（引擎整体序 < 该值才可见）")
    ap.add_argument("--out", default="precommit_conf")
    ap.add_argument("--require-in-dict", action="store_true",
                    help="只保留所有词都在 luna 词典里的句子（模拟用户常用词域，"
                         "wiki 技术文本 OOV 多时用）")
    a = ap.parse_args()

    cc = opencc.OpenCC("s2t")
    st2s = opencc.OpenCC("t2s")
    dict_path = a.dict or os.path.join(a.shared, "luna_pinyin.dict.yaml")
    word_code = load_dict(dict_path)

    sents = pick_sentences(a.segments, a.skip_articles, a.articles,
                           a.min_words, a.max_words, a.sentences * 30)
    # 句 → 词序列 + 码序列。码优先取词典读音（多读音用 pypinyin 常用读法 =
    # 用户实际会打的码），OOV 词用 pypinyin 拼出"用户会打的码"——OOV 词引擎
    # 产不出，其决策点会在可行性统计里自然暴露（truth_in_head=0），不硬凑。
    prepared = []
    for toks in sents:
        codes = []
        ok = True
        for t in toks:
            cs = codes_of(t, word_code, cc)
            if a.require_in_dict and not cs:
                ok = False
                break
            if len(cs) == 1:
                codes.append(next(iter(cs)))
                continue
            c = "".join(lazy_pinyin(t))
            if not (c and c.isascii() and c.isalpha()):
                ok = False
                break
            codes.append(c)
        if ok:
            prepared.append((toks, codes))
            if len(prepared) >= a.sentences:
                break
    if not prepared:
        print("no usable sentence", file=sys.stderr)
        return 1

    # 决策点：i = 0..n-2（尾词不自动确认）；ctx = 前句尾部 + 已确认前缀
    jobs = []
    for si, (words, codes) in enumerate(prepared):
        prev_tail = ""
        if si > 0:
            prev_tail = "".join(prepared[si - 1][0][-3:])
        for i in range(len(words) - 1):
            jobs.append({
                "sid": si,
                "i": i,
                "n": len(words),
                "words": words,
                "codes": codes,
                "truth": words[i],
                "head_code": codes[i],
                "ctx": prev_tail + "".join(words[:i]),
                "rem": "".join(codes[i:]),
            })

    # 一次探针批量枚举全部「剩余码」
    payload = "".join(j["rem"] + "\n" for j in jobs)
    p = subprocess.run([a.probe, "rime.dll", a.shared, a.user, a.schema],
                       cwd=a.bin, input=payload.encode("utf-8"),
                       capture_output=True, timeout=7200)
    lines = [ln for ln in p.stdout.decode("utf-8", "replace").splitlines()
             if ln.startswith("Q\t")]
    if len(lines) != len(jobs):
        print("WARN probe lines %d != jobs %d" % (len(lines), len(jobs)),
              file=sys.stderr)
    if p.returncode != 0:
        print("probe rc=%d stderr tail:\n%s" % (
            p.returncode, p.stderr.decode("utf-8", "replace")[-800:]),
            file=sys.stderr)
        return 1

    code_by_text = {}
    for j, ln in zip(jobs, lines):
        f = ln.split("\t")
        if len(f) < 7:
            j["head_cands"] = []
            continue
        cands = [t for t in f[6].split(WORD_SEP) if t]
        heads, seen = [], set()
        for rank, t in enumerate(cands):
            if t in seen:
                continue
            seen.add(t)
            if j["head_code"] in codes_of(t, word_code, cc):
                heads.append({"text": t, "rank": rank})
                if len(heads) >= a.max_head:
                    break
        j["head_cands"] = heads
        # 整句基线（i=0 时顺带取）：覆盖整句的候选 = 词长（汉字数）恰等于
        # 音节数的候选（公开 API 无 span；luna 全拼 1 音节 ↔ 1 汉字，
        # 仅少数儿化例外）。text == 原句 视为整句首选正确。
        if j["i"] == 0:
            j["full_truth"] = "".join(j["words"])
            n_syl = len(j["full_truth"])   # 汉字数 = 音节数（luna 全拼 1:1）
            full, seen2 = [], set()
            for rank, t in enumerate(cands):
                if t in seen2 or len(t) != n_syl:
                    continue
                seen2.add(t)
                full.append({"text": t, "rank": rank})
                if len(full) >= a.score_cap:
                    break
            j["full_cands"] = full

    # 输出
    n_feasible_decisions = 0
    n_truth_in_head = 0
    for j in jobs:
        texts = [h["text"] for h in j["head_cands"]]
        j["truth_in_head"] = j["truth"] in texts
        j["truth_rank"] = (texts.index(j["truth"])
                           if j["truth_in_head"] else -1)
        j["truth_in_dict"] = bool(codes_of(j["truth"], word_code, cc))
        if j["truth_in_head"]:
            n_truth_in_head += 1
        n_feasible_decisions += 1

    with open(a.out + ".jobs.jsonl", "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # 打分输入：头候选限生产窗（整体 rank < win），引擎序取前 score_cap
    n_score_jobs = 0
    with open(a.out + ".score_in.tsv", "w", encoding="utf-8") as f:
        for j in jobs:
            cands = [h["text"] for h in j["head_cands"]
                     if h["rank"] < a.win][:a.score_cap]
            if len(cands) >= 2:
                n_score_jobs += 1
                f.write("%d\t%d\t%s\t%s\n" % (j["sid"], j["i"], j["ctx"],
                                              WORD_SEP.join(cands)))
            # 整句基线（i=-1）：同一 ctx（前句尾部）
            if j["i"] == 0 and len(j.get("full_cands", [])) >= 2:
                fc = [h["text"] for h in j["full_cands"]]
                n_score_jobs += 1
                f.write("%d\t%d\t%s\t%s\n" % (j["sid"], -1, j["ctx"],
                                              WORD_SEP.join(fc)))

    print("sentences=%d decisions=%d truth_in_head=%d (%.1f%%) "
          "score_jobs=%d" % (len(prepared), len(jobs), n_truth_in_head,
                             100.0 * n_truth_in_head / max(len(jobs), 1),
                             n_score_jobs))
    print("out: %s.jobs.jsonl / %s.score_in.tsv" % (a.out, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
