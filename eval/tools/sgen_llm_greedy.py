#!/usr/bin/env python3
"""sgen_llm_greedy.py — LLM 首 token CE 贪心组句（替代 N-gram beam 的可行性验证）

对每个句子，从码串起点贪心：每步取该位置的出边（等长词 + 子词，词频
top-N），用生产 `llm.score(ctx, cands)`（3-token 加权 CE，1.0/1.10/0.60）
选最高分，前进到词尾。与「N-gram top-5 + 全句 CE 重排 top-1 = 64%/57%」
对比。

口径说明：每位置只取词频 top-N（默认 12 = 生产 n_seq_max 上限）——这是
本实验的已知限制（真值边可能被截断），结果应视为贪心上界/下界参考。

用法：
  python sgen_llm_greedy.py --jobs eval/sgen_gram_100.test.jsonl \
      --jobs2 eval/sgen_gram_600k.test.jsonl [--topn 12] [--threads 8]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sentence_gen_gram import (load_lexicon, build_code_index,
                               build_lattice)  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REPO = os.path.dirname(PROJ)
HOST_DEFAULT = os.path.join(REPO, "rime-llm-rerank", "cpp", "lua_host.exe")
LUA = os.path.join(HERE, "sgen_llm_greedy.lua")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--jobs2", default=None)
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--min-weight", type=float, default=10.0)
    ap.add_argument("--dict-dir", default="D:/gguf_models/wanxiang/dicts")
    ap.add_argument("--lua-host", default=HOST_DEFAULT)
    ap.add_argument("--model", default="D:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(PROJ, "eval", "sgen_llm_greedy"))
    a = ap.parse_args()

    print("loading lexicon ...", flush=True)
    readings = load_lexicon(a.dict_dir)
    code2words = build_code_index(readings)
    print("lexicon: %d words, %d codes" % (len(readings), len(code2words)),
          flush=True)

    rows = []
    for path in [p for p in (a.jobs, a.jobs2) if p]:
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            code = r["code"]
            edges = build_lattice(code, code2words, a.min_weight)
            by_pos = {}
            for i, outs in enumerate(edges):
                if not outs:
                    continue
                # 按词频先验分降序（= 词频），取 topn；同分按词长/词序稳定
                outs = sorted(outs, key=lambda e: (-e[2], e[1]))
                by_pos[i] = outs[:a.topn]
            enc = "\x1e".join("%d|%d|%s|%.4f" % (i, j, w, s)
                              for i, outs in by_pos.items()
                              for (j, w, s) in outs)
            rows.append((len(rows), r["before"], enc, r["truth"], path))

    base = os.path.abspath(a.out)
    jobs_path = base + ".jobs.tsv"
    out_path = base + ".raw.tsv"
    with open(jobs_path, "w", encoding="utf-8", newline="\n") as f:
        for sid, before, enc, _truth, _p in rows:
            f.write("%d\t%s\t%s\n" % (sid, before, enc))
    print("jobs=%d -> %s" % (len(rows), jobs_path), flush=True)

    env = dict(os.environ)
    env["GRD_JOBS"] = jobs_path
    env["GRD_OUT"] = out_path
    env["GRD_MODEL"] = a.model
    env["GRD_THREADS"] = str(a.threads)
    p = subprocess.run([a.lua_host, LUA],
                       cwd=os.path.dirname(os.path.abspath(a.lua_host)),
                       env=env, capture_output=True, timeout=7200)
    if p.returncode != 0 and not (os.path.exists(out_path)
                                  and os.path.getsize(out_path) > 0):
        sys.stderr.write("lua_host rc=%d\n%s\n%s\n"
                         % (p.returncode,
                            p.stdout.decode("utf-8", "replace")[-800:],
                            p.stderr.decode("utf-8", "replace")[-800:]))
        return 1

    got = {}
    with open(out_path, encoding="utf-8") as f:
        for ln in f:
            sid, text = ln.rstrip("\r\n").split("\t", 1)
            got[int(sid)] = text

    stats = {}
    for sid, before, _enc, truth, path in rows:
        reg = "A" if path == a.jobs else "B"
        d = stats.setdefault(reg, [0, 0])
        d[0] += 1
        if got.get(sid) == truth:
            d[1] += 1
    print("\n=== LLM 贪心组句 top-1（每位置 top-%d）===" % a.topn)
    for reg in sorted(stats):
        n, ok = stats[reg]
        print("%s: %d/%d = %.1f%%" % (reg, ok, n, 100.0 * ok / max(1, n)))
    tot_n = sum(v[0] for v in stats.values())
    tot_ok = sum(v[1] for v in stats.values())
    print("ALL: %d/%d = %.1f%%" % (tot_ok, tot_n, 100.0 * tot_ok / max(1, tot_n)))
    with open(base + ".report.txt", "w", encoding="utf-8") as f:
        f.write("topn=%d min_weight=%g\n" % (a.topn, a.min_weight))
        for reg in sorted(stats):
            n, ok = stats[reg]
            f.write("%s: %d/%d = %.1f%%\n" % (reg, ok, n, 100.0 * ok / n))
        f.write("ALL: %d/%d = %.1f%%\n"
                % (tot_ok, tot_n, 100.0 * tot_ok / max(1, tot_n)))
    print("\nout: %s.report.txt / .raw.tsv" % base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
