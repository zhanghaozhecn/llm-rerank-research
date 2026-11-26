# -*- coding: utf-8 -*-
"""prefix_big 64 池按需回填（只算「能成为前缀确认训练行」的窗）。

子命令：
  probe     eligible_codes.txt → 引擎 64 池探针 → eval/prefix_big/cands64.json
  bench N   抽 N 行试跑 sim，测行速率（外推总时长用）
  ce --batch B   按 B 行一批补算缺口（断点续跑；反复跑至剩余 0）
  assemble  eval/prefix_big/cohort64.jsonl（只含 eligible 窗，eng 内嵌 CE）

CE 写入与 luna 共享的 ce64/results.jsonl（sid 内容键控，幂等）。
注意：probe 的 stamp 在 luna work 下按 tag 命名，跑完会覆盖 luna 的
probe_out64.md5 —— assemble 前后不依赖它；如需复跑 luna 探针会重枚举。
"""
import argparse
import json
import sys
import time
from pathlib import Path

EVAL = Path(r"D:\llm-rerank\llm-rerank-research\eval")
sys.path.insert(0, str(EVAL / "calib"))
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L          # noqa: E402
import prefix_confirm_model as M    # noqa: E402

BIG = EVAL / "prefix_big"
W64 = BIG / "work64"


def eligible_windows():
    wids = set(json.load(open(W64 / "eligible_wids.json", encoding="utf-8")))
    return [w for w in L.jsonl_iter(BIG / "windows.jsonl")
            if w["wid"] in wids]


def load_cands64():
    with open(BIG / "cands64.json", encoding="utf-8") as f:
        return json.load(f)


def window_lines(wins, cands, ce):
    """→ (per_win lines, todo 未算行列表)。"""
    per_win, todo = [], []
    for w in wins:
        pool = [c[0] for c in cands.get(w["code"], [])[:L.K_COLLECT_EXT]]
        lns = L.ce_lines(w["ctx"], pool, k_pool=L.K_COLLECT_EXT)
        per_win.append((w, lns))
        todo += [(ln, sid) for ln, sid in lns if sid not in ce]
    return per_win, todo


def cmd_probe(a):
    codes = [l.strip() for l in
             open(W64 / "eligible_codes.txt", encoding="ascii") if l.strip()]
    print(f"[big64] probe codes={len(codes)} k=64 ...", flush=True)
    t0 = time.time()
    cands = L.run_probe(codes, work=W64, k_probe=L.K_PROBE_EXT, tag="64")
    with open(BIG / "cands64.json", "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    ns = [len(v) for v in cands.values()]
    print(f"[big64] probe done {time.time()-t0:.0f}s → cands64.json；"
          f"池均值 {sum(ns)/max(len(ns),1):.1f}、=64(截断) "
          f"{sum(1 for x in ns if x >= L.K_PROBE_EXT)}")


def cmd_bench(a):
    wins = eligible_windows()
    cands = load_cands64()
    ce = L.load_ce_multi()
    per_win, todo = window_lines(wins, cands, ce)
    print(f"[big64] eligible 窗 {len(per_win)}；缺口行 {len(todo)}"
          f"（估速：{(len(todo))} 行）")
    if a.n == 0:
        return 0
    t0 = time.time()
    n = L.run_ce(todo[:a.n], out_dir=L.CE_DIR_EXT)
    dt = time.time() - t0
    print(f"[big64] bench：{n} 行 / {dt:.0f}s = {n/max(dt,1e-9):.2f} 行/s；"
          f"外推剩余 {len(todo)-n} 行 ≈ {(len(todo)-n)/max(n/max(dt,1e-9),1e-9)/3600:.1f} h")
    return 0


def cmd_ce(a):
    wins = eligible_windows()
    cands = load_cands64()
    ce = L.load_ce_multi()
    _, todo = window_lines(wins, cands, ce)
    print(f"[big64] 剩余缺口 {len(todo)} 行；本批 {min(a.batch, len(todo))}",
          flush=True)
    if not todo:
        print("[big64] CE ALL DONE")
        return 0
    t0 = time.time()
    n = L.run_ce(todo[:a.batch], out_dir=L.CE_DIR_EXT)
    dt = time.time() - t0
    print(f"[big64] 本批 {n} 行 / {dt:.0f}s（{n/max(dt,1e-9):.2f} 行/s）",
          flush=True)
    return 10 if len(todo) > a.batch else 0   # 10 = 还有剩余，循环继续


def cmd_assemble(a):
    cands = load_cands64()
    ce = L.load_ce_multi()
    gen = M.load_gen_map(BIG / "gen.tsv")
    wins = eligible_windows()
    n_miss = n_pad = 0
    out = BIG / "cohort64.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for w in wins:
            pool = cands.get(w["code"], [])[:L.K_COLLECT_EXT]
            lns = L.ce_lines(w["ctx"], [c[0] for c in pool],
                             k_pool=L.K_COLLECT_EXT)
            rows = []
            for i, c in enumerate(pool):
                res = ce.get(lns[i // 10][1], {}).get(c[0])
                ce3, ntok = (res if res else (None, None))
                if ce3 is None:
                    if len(pool) == 1:
                        n_pad += 1
                    else:
                        n_miss += 1
                rows.append([c[0], c[1], c[2], c[4], c[3], ce3, ntok])
            gs = gen.get(w["wid"], [])
            genrows = [[g["text"], g["score"], g["lead"], g["lead_end"]]
                       for g in sorted(gs, key=lambda x: -x["score"])]
            f.write(json.dumps(dict(w, eng=rows, gen=genrows),
                               ensure_ascii=False) + "\n")
    print(f"[big64] assemble：{len(wins)} 窗 → {out}；真缺口 {n_miss}；"
          f"单候选「〇」补位 {n_pad}")
    return 0 if n_miss == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "bench", "ce", "assemble"])
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--n", type=int, default=30, help="bench 试跑行数")
    a = ap.parse_args()
    rc = {"probe": cmd_probe, "bench": cmd_bench, "ce": cmd_ce,
          "assemble": cmd_assemble}[a.cmd](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
