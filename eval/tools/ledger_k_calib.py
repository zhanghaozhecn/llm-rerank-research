#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 账本权重标度 K 标定（2026-11-27 README「后续方向」P1-2）。

数据：语料 C 行的真值 (code, word) 对（用户实际选重的词）+ 探针全枚举。
方法：rime_cand_probe --cls 全枚举每个 code 的引擎原序（RIME_LLM_LEDGER_K
环境变量扫 K），对每对真值算：
  - span 位次：选中词在**同 span**（cls start-end 相同）候选里的名次（0 基）
  - 顶格：span 位次 = 0（该 span 的首选就是用户要的词）
  - 账本位移害：非账本词的选中词上方同 span 出现 ledger/completion 型候选
真值冲突（同码不同历史选重）对任意 K 等同影响，比较仍成立。

用法（研究库根目录）：
  python eval/tools/ledger_k_calib.py --ks 5,15,30,60 \
      [--dll ../rime-llm-ime/librime/build/bin/Release/rime.dll] \
      [--probe ../rime-llm-ime/bin/rime_cand_probe.exe] \
      [--shared ../rime-llm-ime/bin/data] [--user D:/llm-rerank/probe-scratch/user-1128] \
      [--schema luna_pinyin] [--limit N] [--codes-file P]
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ledger_double_weight import load_ledger, load_corpus  # noqa: E402


def ground_truth(schema, limit=None, codes_file=None):
    base = os.path.join(os.environ["APPDATA"], "Rime", "llm_rerank", schema)
    w_rows, c_by_ref, now = load_corpus(os.path.join(base, "rime_llm_corpus.tsv"))
    eff_of = load_ledger(os.path.join(base, "user_ledger.tsv"), now)
    pairs = Counter()
    for _ref, commits in c_by_ref.items():
        for c in commits:
            code = c["code"]
            if code.isalpha() and code.islower() and 2 <= len(code) <= 20 and c["word"]:
                pairs[(code, c["word"])] += 1
    if codes_file:
        keep = set()
        with open(codes_file, encoding="utf-8") as f:
            for line in f:
                keep.add(line.strip())
        pairs = Counter({k: v for k, v in pairs.items() if k[0] in keep})
    if limit:
        pairs = Counter(dict(pairs.most_common(limit)))
    return pairs, eff_of


def run_probe(probe, dll, shared, user, schema, ks, codes, timeout=600):
    """每个 K 一次探针进程，返回 {K: {code: (texts, cls)}}。"""
    out = {}
    env = dict(os.environ)
    for k in ks:
        env["RIME_LLM_LEDGER_K"] = str(k)
        proc = subprocess.Popen(
            [probe, "--cls", dll, shared, user, schema],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env)
        lines = "\n".join(codes) + "\n\n"
        stdout, _ = proc.communicate(lines.encode("utf-8"), timeout=timeout)
        res = {}
        for row in stdout.decode("utf-8", "replace").splitlines():
            col = row.split("\t")
            if col[0] != "Q" or len(col) < 7 or not col[1]:
                continue
            code = col[1]
            texts = col[6].split("\x1f") if col[6] else []
            cls = col[7].split("\x1f") if len(col) > 7 and col[7] else []
            res[code] = (texts, cls)
        out[k] = res
        print("  K=%s: %d codes enumerated" % (k, len(res)), file=sys.stderr)
    return out


def span_of(cls_item):
    # type/start-end/comp → (type, start, end)
    part = cls_item.split("/")
    t = part[0]
    se = part[1].split("-") if len(part) > 1 else ("", "")
    try:
        return t, int(se[0]), int(se[1])
    except (ValueError, IndexError):
        return t, -1, -1


def evaluate(kv, pairs, eff_of):
    """一个 K 的指标。"""
    n = top1 = 0
    rank_sum = 0
    led_n = led_top1 = 0
    non_n = non_top1 = non_led_disp = 0
    examples = []
    for (code, word), _cnt in pairs.items():
        rec = kv.get(code)
        if not rec:
            continue
        texts, cls = rec
        idx = next((i for i, t in enumerate(texts) if t == word), -1)
        if idx < 0 or idx >= len(cls):
            continue
        wt, ws, we = span_of(cls[idx])
        span_rank = 0
        led_above = False
        for i in range(idx):
            t2, s2, e2 = span_of(cls[i])
            if s2 == ws and e2 == we:
                span_rank += 1
                if t2 in ("ledger", "completion") and wt not in ("ledger",
                                                                 "completion"):
                    led_above = True
        n += 1
        rank_sum += span_rank
        if span_rank == 0:
            top1 += 1
        is_led = word in eff_of
        if is_led:
            led_n += 1
            if span_rank == 0:
                led_top1 += 1
        else:
            non_n += 1
            if span_rank == 0:
                non_top1 += 1
            if led_above:
                non_led_disp += 1
                if len(examples) < 8:
                    examples.append("%s>%s(账本%s占上)" %
                                    (code, word, cls[idx].split("/")[0]))
    return {
        "n": n, "top1": top1, "mean_span_rank": rank_sum / n if n else 0,
        "led_top1": "%d/%d" % (led_top1, led_n),
        "non_top1": "%d/%d" % (non_top1, non_n),
        "non_led_disp": non_led_disp, "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="luna_pinyin")
    ap.add_argument("--ks", default="5,15,30,60")
    ap.add_argument("--dll",
                    default="../rime-llm-ime/librime/build/bin/Release/rime.dll")
    ap.add_argument("--probe", default="../rime-llm-ime/bin/rime_cand_probe.exe")
    ap.add_argument("--shared", default="../rime-llm-ime/bin/data")
    ap.add_argument("--user", default="D:/llm-rerank/probe-scratch/user-1128")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--codes-file", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pairs, eff_of = ground_truth(args.schema, args.limit, args.codes_file)
    codes = sorted({c for (c, _w) in pairs})
    print("真值对=%d（唯一码=%d，账本词对=%d）" %
          (len(pairs), len(codes), sum(1 for (c, w) in pairs if w in eff_of)))
    ks = [float(x) for x in args.ks.split(",")]
    print("扫 K=%s …" % ks)
    enum = run_probe(args.probe, args.dll, args.shared, args.user,
                     args.schema, ks, codes)
    print("\nK    顶格率           账本词顶格      非账本顶格      非账本被账本位移  平均span位次")
    for k in ks:
        m = evaluate(enum[k], pairs, eff_of)
        print("K=%-4.0f %s/%d (%.1f%%)  %-12s  %-12s  %-14d  %.3f" %
              (k, m["top1"], m["n"], 100.0 * m["top1"] / m["n"] if m["n"] else 0,
               m["led_top1"], m["non_top1"], m["non_led_disp"],
               m["mean_span_rank"]))
        if m["examples"]:
            for e in m["examples"][:4]:
                print("      %s" % e)
    if args.json:
        out = {str(k): evaluate(enum[k], pairs, eff_of) for k in ks}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("JSON -> %s" % args.json)


if __name__ == "__main__":
    main()
