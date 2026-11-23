#!/usr/bin/env python3
"""sgen_llm3_rerank.py — 组句候选的生产 3-token 打分重排（对照全句 CE）

生产 `rime_llm.dll` 的 `score_batch` 只对候选前 3 个 token 算加权 CE
（权重 1.0 / 1.10 / 0.60）——词级重排的安全基线。本工具把 5 个**句子候选**
原样交给 `llm.score(ctx, cands)`（lua_host + 生产 DLL），取回 score 后重排，
与 `sgen_llm_rerank.py`（全句 sum_ce）同表对照。

流程：
  1) 读 `sentence_gen_gram.py` 的 .test.jsonl（top-5 候选 + before 上文）；
  2) 每句一行 jobs：sid \t 0 \t ctx \t 候选1\x1f…\x1f候选5；
  3) 跑 `cpp/lua_host.exe` + `build/Release/rime_llm.dll`（cwd = 插件 cpp\）；
  4) 解析 `sid \t i \t 候选 \t score`，按 score 降序重排；
  5) 报 ceiling / ngram@1 / llm3@1 / fixed / broken，并与全句 rerank
     （`sgen_llm_rerank.py` 输出）并排。

用法：
  python sgen_llm3_rerank.py --jobs eval/sgen_gram_100.test.jsonl \
      [--full eval/sgen_llm_100.rerank.jsonl] [--topk 5] \
      [--lua-host <plugin>/cpp/lua_host.exe] [--model gguf] \
      [--out eval/sgen_llm3_100]
"""
import argparse
import json
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REPO = os.path.dirname(PROJ)  # D:\llm-rerank（三子目录的父目录）
HOST_DEFAULT = os.path.join(REPO, "rime-llm-rerank", "cpp", "lua_host.exe")
SCORE_LUA = os.path.join(PROJ, "eval", "tools", "precommit_conf_score.lua")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--full", default=None,
                    help="全句 rerank JSONL（sgen_llm_rerank.py 输出）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lua-host", default=HOST_DEFAULT)
    ap.add_argument("--model", default="D:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf")
    ap.add_argument("--out", default=os.path.join(PROJ, "eval", "sgen_llm3"))
    a = ap.parse_args()

    rows = [json.loads(ln) for ln in open(a.jobs, encoding="utf-8")]
    base = os.path.abspath(a.out)
    jobs_path = base + ".jobs.tsv"
    scores_path = base + ".scores.tsv"
    n_jobs = 0
    with open(jobs_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            cands = [c["text"] for c in r["cands"][:a.topk]]
            if len(cands) < 2:
                continue
            f.write("%d\t0\t%s\t%s\n"
                    % (r["sid"], r["before"], "\x1f".join(cands)))
            n_jobs += 1
    print("jobs=%d -> %s" % (n_jobs, jobs_path), flush=True)

    env = dict(os.environ)
    env["PM_JOBS"] = jobs_path
    env["PM_OUT"] = scores_path
    env["PM_MODEL"] = a.model
    p = subprocess.run([a.lua_host, SCORE_LUA],
                       cwd=os.path.dirname(os.path.abspath(a.lua_host)),
                       env=env, capture_output=True, timeout=7200)
    out = p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    for ln in out.splitlines():
        if ln.startswith("[score]"):
            print(ln)
    if p.returncode != 0:
        # DLL 在进程退出阶段偶发 0xC0000005（评分与落盘已完成）；只要输出文件
        # 有内容就按成功处理，避免把已完成的结果判死。
        if os.path.exists(scores_path) and os.path.getsize(scores_path) > 0:
            print("WARN lua_host rc=%d（评分已写出，按文件继续）"
                  % p.returncode, flush=True)
        else:
            sys.stderr.write("lua_host rc=%d\n%s\n%s\n"
                             % (p.returncode, out[-1000:], err[-1000:]))
            return 1

    scores = {}
    with open(scores_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\r\n")
            sid, i, text, sc = ln.split("\t")
            scores.setdefault(int(sid), {})[text] = float(sc)
    print("scored sentences=%d" % len(scores), flush=True)

    def summarize(name, rows_sub):
        n = len(rows_sub)
        ceil = sum(1 for r in rows_sub if r["truth"] in
                   [c["text"] for c in r["cands"][:a.topk]])
        ng1 = sum(1 for r in rows_sub if 0 <= r["rank"] < 1)
        llm1 = fixed = broken = 0
        for r in rows_sub:
            sc = scores.get(r["sid"])
            cands = r["cands"][:a.topk]
            if not sc:
                continue
            ranked = sorted(cands, key=lambda c: -sc.get(c["text"], -1e30))
            top = ranked[0]["text"] if ranked else None
            if top == r["truth"]:
                llm1 += 1
            ng_top = cands[0]["text"] if cands else None
            if top == r["truth"] and ng_top != r["truth"]:
                fixed += 1
            elif top != r["truth"] and ng_top == r["truth"]:
                broken += 1
        print("%-12s n=%d ceiling@%d=%d(%.1f%%) ngram@1=%d(%.1f%%) "
              "llm3@1=%d(%.1f%%) fixed=%d broken=%d"
              % (name, n, a.topk, ceil, 100.0 * ceil / max(n, 1),
                 ng1, 100.0 * ng1 / max(n, 1), llm1, 100.0 * llm1 / max(n, 1),
                 fixed, broken))
        return llm1

    print("\n=== 前 3 token（生产 DLL 加权 CE）===")
    summarize("all", rows)
    inv = [r for r in rows if r["all_in_vocab"]]
    if inv and len(inv) != len(rows):
        print("--- in_vocab ---")
        summarize("in_vocab", inv)

    if a.full:
        full = {r["sid"]: r for r in
                (json.loads(ln) for ln in open(a.full, encoding="utf-8"))}
        n = len(rows)
        ceil = sum(1 for r in rows if r["truth"] in
                   [c["text"] for c in r["cands"][:a.topk]])
        f1 = sum(1 for r in rows
                 if full.get(r["sid"], {}).get("llm_rank") == 0)
        print("\n=== 全句 CE 对照（%s）===" % a.full)
        print("ceiling@%d=%d(%.1f%%)  full_llm@1=%d(%.1f%%)"
              % (a.topk, ceil, 100.0 * ceil / max(n, 1), f1,
                 100.0 * f1 / max(n, 1)))

    # 明细
    with open(base + ".rerank.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            sc = scores.get(r["sid"], {})
            cands = []
            for c in r["cands"][:a.topk]:
                cands.append({"text": c["text"], "ngram_score": c["score"],
                              "score3": sc.get(c["text"])})
            cands.sort(key=lambda x: -(x["score3"] if x["score3"] is not None
                                       else -1e30))
            f.write(json.dumps({
                "sid": r["sid"], "truth": r["truth"], "before": r["before"],
                "all_in_vocab": r["all_in_vocab"], "ngram_rank": r["rank"],
                "llm3_rank": next((k for k, c in enumerate(cands)
                                   if c["text"] == r["truth"]), -1),
                "cands": cands}, ensure_ascii=False) + "\n")
    print("\nout: %s.rerank.jsonl" % base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
