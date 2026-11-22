#!/usr/bin/env python3
"""wiki_fp_pos.py — wiki 语料 + 真实引擎（全拼 luna）候选位置统计（2026-11-23）

用途：为 llm_filter 的 Collect 上限取值提供依据——统计「正确候选在引擎候选
序中的位置」分布（全拼方案、真实 librime 引擎、简体候选）。

流程：HanLP 分段缓存 → 多字 CJK 词频 → 全拼码（luna 词典繁体词条，OOV 用
pypinyin）→ rime_cand_probe 逐码查目标位置（找到即停，探针侧上限 400）。

环境要求：
  · rime_cand_probe.exe（源码仓 rime-llm-ime/scripts/rime_cand_probe.cpp 编译，
    放到 rime.dll 同目录运行——opencc/vcomp140 依赖按 exe 目录解析）
  · 探针用的干净全拼方案（无 llm_filter、用户词典为空）。做法：把安装版
    luna_pinyin.schema.yaml 复制成 luna_probe.schema.yaml（改 schema_id/
    translator.prism），加入 schema_list 后部署一次（分析完移除）。
  · 用户全局 simplification: true 时引擎候选为简体（探针输出即简体，直接
    用原简体词匹配）；繁体环境下把目标改为 opencc s2t。

用法：
  python wiki_fp_pos.py --probe <exe> --shared <rime data dir> \
      --user <Rime user dir> [--articles 1500] [--minfreq 3] [--out report.txt]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

import opencc
from pypinyin import lazy_pinyin

CJK = re.compile(r"[\u3400-\u9fff]")


def default_paths():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))  # research 根
    user = os.path.join(os.environ.get("APPDATA", ""), "Rime")
    return {
        "segs": os.path.join(proj, "segments_hanlp_cache.jsonl"),
        "probe": os.path.join(os.path.dirname(proj), "rime-llm-ime", "bin",
                              "rime_cand_probe.exe"),
        "bin": os.path.join(os.path.dirname(proj), "rime-llm-ime", "bin"),
        "user": user,
        "dict": os.path.join(user, "build", "..", "luna_pinyin.dict.yaml"),
    }


def load_dict(path):
    """词 → 首码（音节连写）。luna 词典为繁体条目。"""
    word_code = {}
    started = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not started:
                if line.strip() == "...":
                    started = True
                continue
            if not line or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            w, c = p[0], p[1].replace(" ", "")
            if w not in word_code:
                word_code[w] = c
    return word_code


def collect_words(segs, articles, minfreq):
    wf = Counter()
    with open(segs, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= articles:
                break
            try:
                toks = json.loads(line)
            except json.JSONDecodeError:
                continue
            for t in toks:
                if len(t) >= 2 and CJK.search(t):
                    wf[t] += 1
    return Counter({w: c for w, c in wf.items() if c >= minfreq})


def pct(vals, p):
    if not vals:
        return 0
    return sorted(vals)[int(round((len(vals) - 1) * p))]


def main():
    d = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default=d["segs"])
    ap.add_argument("--probe", default=d["probe"])
    ap.add_argument("--bin", default=d["bin"])
    ap.add_argument("--shared", required=True, help="rime shared data dir")
    ap.add_argument("--user", default=d["user"])
    ap.add_argument("--schema", default="luna_probe")
    ap.add_argument("--dict", default=d["dict"])
    ap.add_argument("--articles", type=int, default=1500)
    ap.add_argument("--minfreq", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cc = opencc.OpenCC("s2t")
    word_code = load_dict(a.dict)
    wf = collect_words(a.segs, a.articles, a.minfreq)

    items = []
    no_code = 0
    for w, fr in wf.items():
        code = word_code.get(cc.convert(w))
        if not code:
            code = "".join(lazy_pinyin(w))
            if not (code.isascii() and code.isalpha()):
                no_code += 1
                continue
        items.append((w, code, fr))     # 目标用简体（候选已被 simplifier 简化）

    payload = "".join("%s>%s\n" % (code, w) for w, code, _ in items)
    p = subprocess.run([a.probe, "rime.dll", a.shared, a.user, a.schema],
                       cwd=a.bin, input=payload.encode("utf-8"),
                       capture_output=True, timeout=7200)
    plines = [ln.split("\t") for ln in
              p.stdout.decode("utf-8", "replace").splitlines()
              if ln.startswith("P\t")]
    if len(plines) != len(items):
        print("WARN probe lines %d != items %d" % (len(plines), len(items)),
              file=sys.stderr)

    results = []
    for (w, code, fr), f in zip(items, plines):
        results.append((w, code, fr, int(f[2])))
    found = [r for r in results if r[3] >= 0]
    occ = sum(r[2] for r in results)
    occ_found = sum(r[2] for r in found)

    pos_t = [r[3] for r in found]
    pos_w = []
    for r in found:
        pos_w.extend([r[3]] * r[2])

    out = []
    out.append("== wiki 全拼（真实引擎）候选位置 ==")
    out.append("articles=%d 目标词型=%d（freq>=%d） 命中=%d 未命中=%d OOV码=%d"
               % (a.articles, len(results), a.minfreq, len(found),
                  len(results) - len(found), no_code))
    out.append("位置分位（词型）: max=%d p99.9=%d p99=%d p95=%d p50=%d"
               % (max(pos_t), pct(pos_t, .999), pct(pos_t, .99),
                  pct(pos_t, .95), pct(pos_t, .5)))
    out.append("位置分位（词频加权）: max=%d p99.9=%d p99=%d p95=%d p50=%d"
               % (max(pos_w), pct(pos_w, .999), pct(pos_w, .99),
                  pct(pos_w, .95), pct(pos_w, .5)))
    out.append("")
    out.append("上限 k 覆盖（词型 / 词频加权）:")
    for k in (8, 10, 12, 16, 20, 24, 32):
        c1 = sum(1 for r in found if r[3] < k)
        c2 = sum(r[2] for r in found if r[3] < k)
        out.append("  k=%2d  词型 %6.2f%%  加权 %6.2f%%"
                   % (k, 100.0 * c1 / max(len(found), 1),
                      100.0 * c2 / max(occ_found, 1)))
    out.append("")
    out.append("最深 20（词型）:")
    for r in sorted(found, key=lambda x: -x[3])[:20]:
        out.append("  pos=%3d freq=%5d code=%-18s %s"
                   % (r[3], r[2], r[1], r[0]))
    text = "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
