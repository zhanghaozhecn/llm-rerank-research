#!/usr/bin/env python3
"""sgen_latency.py — 5 个组句候选的评分延迟基准（生产 3-token vs 全句 CE）

语料 = `sentence_gen_gram.py` 的输出 JSONL（A/B 两区各 100 句，每句 top-5
候选 + 上文）；模拟「整句模式一次 query：ctx + 5 个句子候选」：
  · 3tok = 生产 `llm.score`（score_batch：前 3 token 加权 CE）
  · full = `llm.score_full`（新增：完整候选 token 的 CE 和，独立 g_full_ctx）

跑一个 lua_host 进程（模型加载一次），逐句记录墙钟毫秒；输出均值 /
p50 / p95 / max，并按候选 token 数分桶。

用法：
  python sgen_latency.py --jobs eval/sgen_gram_100.test.jsonl \
      --jobs2 eval/sgen_gram_600k.test.jsonl \
      [--lua-host <plugin>/cpp/lua_host.exe] [--model gguf] \
      [--out eval/sgen_latency]
"""
import argparse
import json
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REPO = os.path.dirname(PROJ)
HOST_DEFAULT = os.path.join(REPO, "rime-llm-rerank", "cpp", "lua_host.exe")
LUA = os.path.join(PROJ, "eval", "tools", "sgen_latency.lua")


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def stat(name, vals):
    if not vals:
        print("%-6s n=0" % name)
        return
    print("%-6s n=%d mean=%.1fms p50=%.1f p95=%.1f max=%.1f"
          % (name, len(vals), sum(vals) / len(vals), pct(vals, 50),
             pct(vals, 95), max(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--jobs2", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lua-host", default=HOST_DEFAULT)
    ap.add_argument("--model", default="D:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf")
    ap.add_argument("--threads", type=int, default=None,
                    help="推理线程数（缺省 = DLL 默认 4）")
    ap.add_argument("--out", default=os.path.join(PROJ, "eval", "sgen_latency"))
    a = ap.parse_args()

    rows = []
    for path in [p for p in (a.jobs, a.jobs2) if p]:
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            cands = [c["text"] for c in r["cands"][:a.topk]]
            if len(cands) >= 2 and r["before"]:
                rows.append((len(rows), r["before"], cands))
    base = os.path.abspath(a.out)
    jobs_path = base + ".jobs.tsv"
    out_path = base + ".raw.tsv"
    with open(jobs_path, "w", encoding="utf-8", newline="\n") as f:
        for sid, ctx, cands in rows:
            f.write("%d\t%s\t%s\n" % (sid, ctx, "\x1f".join(cands)))
    print("jobs=%d -> %s" % (len(rows), jobs_path), flush=True)

    env = dict(os.environ)
    env["LAT_JOBS"] = jobs_path
    env["LAT_OUT"] = out_path
    env["LAT_MODEL"] = a.model
    if a.threads:
        env["LAT_THREADS"] = str(a.threads)
    p = subprocess.run([a.lua_host, LUA],
                       cwd=os.path.dirname(os.path.abspath(a.lua_host)),
                       env=env, capture_output=True, timeout=7200)
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    for ln in out.splitlines():
        if ln.startswith("[lat]"):
            print(ln)
    if p.returncode != 0 and not (os.path.exists(out_path)
                                  and os.path.getsize(out_path) > 0):
        sys.stderr.write("lua_host rc=%d\n%s\n%s\n"
                         % (p.returncode, out[-1000:], err[-1000:]))
        return 1
    if p.returncode != 0:
        print("WARN lua_host rc=%d（输出已写出，按文件继续）" % p.returncode)

    v3, vf, ntok_by = [], [], {}
    with open(out_path, encoding="utf-8") as f:
        for ln in f:
            sid, mode, ms, nt = ln.rstrip("\r\n").split("\t")
            ms = float(ms)
            nt = int(nt)
            if nt <= 0:
                continue
            if mode == "3tok":
                v3.append(ms)
            else:
                vf.append(ms)
                ntok_by.setdefault(nt, []).append(ms)
    print("\n=== 每句（ctx + 5 候选）===")
    stat("3tok", v3)
    stat("full", vf)
    print("\n=== 全句按候选最大 token 数 ===")
    for nt in sorted(ntok_by):
        stat("tok=%d" % nt, ntok_by[nt])
    with open(base + ".report.txt", "w", encoding="utf-8") as f:
        f.write("jobs=%d\n" % len(rows))
        for name, vals in (("3tok", v3), ("full", vf)):
            if vals:
                f.write("%s n=%d mean=%.1f p50=%.1f p95=%.1f max=%.1f\n"
                        % (name, len(vals), sum(vals) / len(vals),
                           pct(vals, 50), pct(vals, 95), max(vals)))
    print("\nout: %s.report.txt / .raw.tsv" % base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
