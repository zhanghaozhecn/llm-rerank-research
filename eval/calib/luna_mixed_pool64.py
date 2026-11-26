#!/usr/bin/env python3
"""luna 混合窗：**收集池 16 → 64**（2026-11-26）。

背景（迁移审计 C 节）：线上 Collect 已是 64（30ms 预算不变），但 cohort 的
候选只在引擎序前 16 条上算过 CE、探针也只取 32——所有「预算窗」结果都是
16 池模拟，超出部分（多为 1-token 免费候选）没有 CE，覆盖/窗宽被低估。
本脚本把同 cohort（同一批 sentences / windows）的口径补齐：

  1. `probe`  —— 探针上限 32 → 64（写 `work/probe_out64.tsv`，旧
                  `probe_out.tsv`/缓存戳不动）→ 解码成 `cands64.json`；
                 候选数不足 64 的码是**引擎真的只给这么多**（探针已触
                 Translation 耗尽），不是截断。
  2. `gap`    —— 池 64 的 CE 输入行，按 sid 与旧 16 池结果对账：旧行
                 sid 完全复用的占比（应 100%）、需要新算的行数 / 窗数。
  3. `ce`     —— 分批跑缺口（`ce64/results.jsonl`，按 sid 断点续跑；
                 单进程 sim、批大小可调，避免长任务一次占满）。
  4. `assemble` —— `cohort64.jsonl`（eng 池 64；gen/full 同旧 cohort）。
  5. `compare`  —— 调用 luna_mixed_budget 的 report()，16 / 32 / 64 三档并排。

口径提示：**扩大的是 CE 池，不是窗**。预算准入（`budget_mask`，B=12）
扫描深度随池自动变深；每个码最多 64 条与线上 collect 一致。

产物（全部新增，旧 16 池产物一字不动）：
  eval/luna_mixed/cands64.json
  eval/luna_mixed/cohort64.jsonl
  eval/luna_mixed/ce64/results.jsonl
  eval/luna_mixed_pool64.txt

用法：
  python eval/calib/luna_mixed_pool64.py gap
  python eval/calib/luna_mixed_pool64.py ce --batch 2500      # 反复跑至 done=0
  python eval/calib/luna_mixed_pool64.py probe assemble compare
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import luna_mixed_eval as E  # noqa: E402
import luna_mixed_lib as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
REPORT = L.EVAL / "luna_mixed_pool64.txt"
POOL = L.K_COLLECT_EXT


def log(msg):
    print(f"[pool64] {msg}", flush=True)


def static_windows():
    return [w for w in L.jsonl_iter(L.WINDOWS) if not w["dyn"]]


def load_cands64():
    if not L.CANDS_EXT.exists():
        raise SystemExit(f"缺 {L.CANDS_EXT}：先跑 probe")
    with open(L.CANDS_EXT, encoding="utf-8") as f:
        return json.load(f)


# ── 1. 探针 64 ───────────────────────────────────────────────────────
def cmd_probe(a):
    wins = static_windows()
    codes = sorted({w["code"] for w in wins})
    log(f"codes={len(codes)}; probe max={L.K_PROBE_EXT} ...")
    t0 = time.time()
    cands = L.run_probe(codes, k_probe=L.K_PROBE_EXT, tag=str(L.K_PROBE_EXT))
    with open(L.CANDS_EXT, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    log(f"probe done {time.time() - t0:.0f}s → {L.CANDS_EXT}")
    ns = [len(v) for v in cands.values()]
    log(f"candidates/code: mean={sum(ns) / max(len(ns), 1):.1f} "
        f">32={sum(1 for x in ns if x > 32)} codes, "
        f"=64(截断){sum(1 for x in ns if x >= L.K_PROBE_EXT)} codes")


# ── 2. 缺口对账 ──────────────────────────────────────────────────────
def gap_plan():
    """→ (lines [[(line, sid)] per window], cands64, 统计 dict)。"""
    cands = load_cands64()
    wins = [w for w in static_windows() if len(cands.get(w["code"], [])) >= 2]
    old = L.load_ce()
    have = L.load_ce_multi()      # 旧 16 池 + 已回填的 ce64
    per_win, n_lines, reuse, done = [], 0, 0, 0
    for w in wins:
        pool = [c[0] for c in cands[w["code"]][:POOL]]
        lns = L.ce_lines(w["ctx"], pool, k_pool=POOL)
        per_win.append(lns)
        n_lines += len(lns)
        reuse += sum(1 for _, sid in lns if sid in old)
        done += sum(1 for _, sid in lns if sid in have)
    stat = dict(wins=len(wins), lines=n_lines, reuse=reuse,
                have=done, todo=n_lines - done)
    return per_win, cands, stat


def cmd_gap(a):
    per_win, cands, st = gap_plan()
    # 15% ≠ 异常：sid 含整块词表（10 词/块），池 16 只有首块能按 sid 复用；
    # 块 2–7 全要新算。首块复用率应 100%（同 ctx 同首 10 词）。
    old = L.load_ce()
    first_hit = sum(1 for lns in per_win if lns and lns[0][1] in old)
    n = sum(1 for lns in per_win if lns)
    log(f"池 {POOL}：窗 {st['wins']} · CE 行 {st['lines']}"
        f"（旧 16 池整块命中 {st['reuse']} = "
        f"{100 * st['reuse'] / max(st['lines'], 1):.2f}%，首块 "
        f"{first_hit}/{n}；累计已算 {st['have']}）· **待算 {st['todo']} 行**")
    if st["todo"] == 0:
        log("CE ALL DONE")
        return 0
    return 1 if first_hit != n else 2


# ── 3. CE（分批）───────────────────────────────────────────────────────
def cmd_ce(a):
    per_win, _, st = gap_plan()
    have = L.load_ce_multi()
    todo = [(ln, sid) for lns in per_win for (ln, sid) in lns
            if sid not in have]
    log(f"待算 {len(todo)} 行（已完成 {st['have']}/{st['lines']}）；"
        f"本批 {min(a.batch, len(todo))}")
    if st["todo"] == 0:
        log("CE ALL DONE")
        return 0
    t0 = time.time()
    n = L.run_ce(todo[:a.batch], workers=a.workers, out_dir=L.CE_DIR_EXT)
    dt = time.time() - t0
    log(f"本批完成 {n} 行 / {dt:.0f}s（{n / max(dt, 1e-9):.1f} 行/s）；"
        f"新算累计 {len(L.load_ce(L.CE_RESULTS_EXT))} 行 → {L.CE_RESULTS_EXT}")
    return 0


# ── 4. 组装 cohort64 ──────────────────────────────────────────────────
def cmd_assemble(a):
    cands = load_cands64()
    ce = L.load_ce_multi()
    # gen / full-CE 与窗口代码无关，按 wid 直接沿用旧 cohort 行
    legacy = {r["wid"]: r for r in L.jsonl_iter(L.COHORT)}
    n_win = n_miss = n_pad = 0
    with open(L.COHORT_EXT, "w", encoding="utf-8") as f:
        for w in static_windows():
            pool = cands.get(w["code"], [])[:POOL]
            lns = L.ce_lines(w["ctx"], [c[0] for c in pool], k_pool=POOL)
            rows = []
            for i, c in enumerate(pool):
                res = ce.get(lns[i // 10][1], {}).get(c[0])
                ce3, ntok = (res if res else (None, None))
                if ce3 is None:
                    if len(pool) == 1:
                        # 池内只有 1 个候选（den/din 的生僻字）→ 该块按约定
                        # 补「〇」，sim 只算真候选 → 无 CE。**不是回填缺口**：
                        # 窗成员 <2 时生产本就不重排（few_candidates）。
                        n_pad += 1
                    else:
                        n_miss += 1
                rows.append([c[0], c[1], c[2], c[4], c[3], ce3, ntok])
            old = legacy.get(w["wid"], {})
            f.write(json.dumps(dict(w, eng=rows, gen=old.get("gen", [])),
                               ensure_ascii=False) + "\n")
            n_win += 1
    log(f"cohort64: {n_win} 窗 · 单候选窗「〇」补位无 CE {n_pad} 条"
        f"（生产不重排，评测掩掉）· 真缺口 {n_miss} 条 → {L.COHORT_EXT}")
    return 0 if n_miss == 0 else 1


# ── 5. 三档并排 ────────────────────────────────────────────────────────
def cmd_compare(a):
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    d64 = E.load(k_pool=POOL, k_probe=POOL, cohort=L.COHORT_EXT,
                 cands_path=L.CANDS_EXT)
    import luna_mixed_budget as B  # noqa: E402

    hdr = (f"luna 混合窗 · 收集池 16/32/64 · 静态窗 {d64['n']}"
           f"（wiki {int((d64['dom'] == 'wiki').sum())} / user "
           f"{int((d64['dom'] == 'user').sum())}）· 冷账本 · CE 预算 12")
    out("=" * 100)
    out(hdr)
    out("=" * 100)
    old_stdout = sys.stdout

    class Tee:
        def write(self, s):
            old_stdout.write(s)
            lines.append(s)

        def flush(self):
            old_stdout.flush()

    masks64 = (
        ("16 池（旧口径）", B.count_mask(d64, 16)),
        ("32 池（旧探针上限）", B.count_mask(d64, 32)),
        (f"{POOL} 池（线上 collect）", B.count_mask(d64, POOL)),
        (f"CE 预算 12 @ {POOL} 池",
         L.budget_mask(d64["nt"], d64["gshow"], d64["pad"]) & d64["has_ce"]),
    )
    sys.stdout = Tee()
    try:
        for name, m in masks64:
            B.report(name, d64, m)
        out("\n注：窗成员取自池内全部候选（1-token 候选数与探针返回数一致），"
            "预算准入只按预算花销截断；has_ce 掩掉无 CE 的（本表应为 0）。")
    finally:
        sys.stdout = old_stdout
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"→ {REPORT}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="luna 混合窗收集池 16→64")
    ap.add_argument("steps", nargs="+",
                    choices=["probe", "gap", "ce", "assemble", "compare"])
    ap.add_argument("--batch", type=int, default=2500,
                    help="ce 单批行数（每批一个 sim 进程）")
    ap.add_argument("--workers", type=int, default=1,
                    help="sim 进程数（>1 线程池；默认单进程）")
    a = ap.parse_args()
    for s in a.steps:
        rc = {"probe": cmd_probe, "gap": cmd_gap, "ce": cmd_ce,
              "assemble": cmd_assemble, "compare": cmd_compare}[s](a)
        if rc:
            raise SystemExit(f"step {s} rc={rc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
