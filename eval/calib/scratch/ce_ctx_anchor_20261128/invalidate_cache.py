# -*- coding: utf-8 -*-
"""CE 缓存 1-token 上文条目失效工具（2026-11-28 CE 上文锚口径切换）。

锚规则：推理上文恰 1 token（ids.size()==1）垫「。」后算 CE（生产两版与
sim_rerank 同口径；n=0 不垫、n≥2 零效应）。既有 CE 缓存里 ntok(ctx)==1 的
条目全部过期（锚使 CE 系统性下降 ~6 nats），必须失效重放；n=0/n≥2 条目
不受影响（重放它们只会得到逐位相同的值，白花机时）。

缓存行键 = corpus_std.window_key_ctx(ctx, chunk) = md5(去空白ctx+词表)，
**缓存行内无 ctx 原文** → 从各生产线构造源超集复算 sid，再用 gguf_bpe
（llama.cpp 字节级 BPE 复刻，302/302 校验过）精确判 ntok(ctx)==1。

构造源（与写侧逐位同构；过滤子集不影响 sid）：
  luna 16 池    eval/luna_mixed/ce_input.jsonl（原始构建的精确输入行存档）
  prefix_big 16 eval/prefix_big/windows.jsonl × cands.json[:16]
                （写侧 build_prefix_big cmd_ce 是其过滤子集，同 ctx 同池）
  short_ctx     复刻 short_ctx.py 选句（--n 12000）× n∈{0..7} 截断 ctx ×
                合并 cands 池前 10（写侧同式；仅 n==1 进过期集）
  luna 64 池    luna windows(static) × cands64.json[:64]
  prefix_big 64 eval/prefix_big/work64/eligible_wids.json × cands64.json[:64]

用法（研究库根目录）：
  python eval/calib/scratch/ce_ctx_anchor_20261128/invalidate_cache.py
      默认 dry-run：报告两缓存的总行数/过期命中数/重放行数。
  ... --apply  备份原缓存为 *.bak_preanchor、重写为无过期条目版本，并把
      重放输入落 eval/luna_mixed/work/refill_preanchor_{16,64}.jsonl
      （{"context","cands"} 行，run_ce_cohort --input 自算 sid 补算）。
  ... --scan-big  额外扫描大库缓存（cohort_big_*/instruct_ce_probe）的
      anchors ntok==1 窗数——只报告不失效（当期无消费方；这些链下次
      重跑实验前应先对本工具逻辑同款失效）。

sim_rerank.exe 须为锚版（≥2026-11-28 14:47，SIM_CTX_ANCHOR 默认开）。
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                 # eval/calib
sys.path.insert(0, str(HERE.parents[2] / "lib"))         # eval/lib
sys.stdout.reconfigure(encoding="utf-8")
import corpus_std                        # noqa: E402
import luna_mixed_lib as L               # noqa: E402
from gguf_bpe import BPE                 # noqa: E402

BIG = L.EVAL / "prefix_big"
CACHE16 = L.CE_RESULTS                   # eval/luna_mixed/ce/results.jsonl
CACHE64 = L.CE_RESULTS_EXT               # eval/luna_mixed/ce64/results.jsonl
REPORT = HERE / "invalidate_report.txt"
NTOK = {}                                # ctx → token 数（进程内缓存）
bpe = None


def log(msg):
    print(f"[inv] {msg}", flush=True)


def ntok1(ctx):
    if not ctx:
        return False
    n = NTOK.get(ctx)
    if n is None:
        n = NTOK[ctx] = bpe.ntok(ctx)
    return n == 1


def short_ctx_picked(n_target=12000):
    """复刻 short_ctx.py 选句（同迭代序同过滤同采样），→ [(wid, w, s)]。"""
    sents = {}
    for d in (L.OUT, BIG):
        for x in L.jsonl_iter(d / "sentences.jsonl"):
            sents[x["sid"]] = x
    wins = {}
    for d in (L.OUT, BIG):
        for w in L.jsonl_iter(d / "windows.jsonl"):
            wins[w["wid"]] = w
    picked = []
    for wid, w in wins.items():
        if w["m"] != 0 or len(w["code"]) < 6 or w.get("dyn"):
            continue
        s = sents.get(w["sid"])
        if not s or len(s.get("before") or "") < 20:
            continue
        picked.append((wid, w, s))
    if len(picked) > n_target:
        picked = picked[::max(1, len(picked) // n_target)][:n_target]
    return picked


def collect_lines():
    """→ {缓存目录: {sid: line}}（过期行；line = {"context","cands"}）。"""

    stale = {L.CE_DIR: {}, L.CE_DIR_EXT: {}}

    def add(target, ctx, cands):
        if not ntok1(ctx):
            return
        sid = corpus_std.window_key_ctx(ctx, cands)
        stale[target].setdefault(sid, {"context": ctx, "cands": list(cands)})

    # ── luna 16 池：ce_input.jsonl 原始记录 ──
    n_in = 0
    for l in open(L.OUT / "ce_input.jsonl", encoding="utf-8"):
        r = json.loads(l)
        n_in += 1
        add(L.CE_DIR, r["context"], r["cands"])
    log(f"luna16 ce_input {n_in} 行")

    # ── prefix_big 16 池（写侧 cmd_ce 为过滤子集）──
    with open(BIG / "cands.json", encoding="utf-8") as f:
        big16 = json.load(f)
    n_in = 0
    for w in L.jsonl_iter(BIG / "windows.jsonl"):
        pool = [c[0] for c in big16.get(w["code"], [])[:L.K_COLLECT]]
        if len(pool) < 2:
            continue
        for ln, _sid in L.ce_lines(w["ctx"], pool):
            n_in += 1
            add(L.CE_DIR, w["ctx"], ln["cands"])
    log(f"big16 windows → {n_in} 行")

    # ── short_ctx n=1（复刻选句；cands 合并序与写侧一致）──
    merged = {}
    for d in (L.OUT, BIG):
        merged.update(json.load(open(d / "cands.json", encoding="utf-8")))
    n_in = 0
    for wid, w, s in short_ctx_picked():
        pool = merged.get(w["code"], [])[:L.K_COLLECT]
        need = [c[0] for c in pool][:10]
        if len(need) < 2:
            continue
        ctx = (s.get("before") or "")[-1:]
        for ln, _sid in L.ce_lines(ctx, need):
            n_in += 1
            add(L.CE_DIR, ctx, ln["cands"])
    log(f"short_ctx n=1 → {n_in} 行")

    # ── luna 64 池（static 窗）──
    with open(L.CANDS_EXT, encoding="utf-8") as f:
        luna64 = json.load(f)
    n_in = 0
    for w in L.jsonl_iter(L.WINDOWS):
        if w.get("dyn"):
            continue
        pool = [c[0] for c in luna64.get(w["code"], [])[:L.K_COLLECT_EXT]]
        if len(pool) < 2:
            continue
        for ln, _sid in L.ce_lines(w["ctx"], pool, k_pool=L.K_COLLECT_EXT):
            n_in += 1
            add(L.CE_DIR_EXT, w["ctx"], ln["cands"])
    log(f"luna64 windows → {n_in} 行")

    # ── prefix_big 64 池（eligible 窗）──
    with open(BIG / "cands64.json", encoding="utf-8") as f:
        big64 = json.load(f)
    wids = set(json.load(open(BIG / "work64" / "eligible_wids.json",
                              encoding="utf-8")))
    n_in = 0
    for w in L.jsonl_iter(BIG / "windows.jsonl"):
        if w["wid"] not in wids:
            continue
        pool = [c[0] for c in big64.get(w["code"], [])[:L.K_COLLECT_EXT]]
        if len(pool) < 2:
            continue
        for ln, _sid in L.ce_lines(w["ctx"], pool, k_pool=L.K_COLLECT_EXT):
            n_in += 1
            add(L.CE_DIR_EXT, w["ctx"], ln["cands"])
    log(f"big64 eligible windows → {n_in} 行")
    return stale


def scan_cache(path, stale_all):
    """→ (总行数, 过期命中数)。stale_all = 两目录过期 sid 并集。"""
    total = hit = 0
    with open(path, encoding="utf-8") as f:
        for l in f:
            try:
                sid = json.loads(l)["sid"]
            except ValueError:
                continue
            total += 1
            if sid in stale_all:
                hit += 1
    return total, hit


def main():
    global bpe
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--scan-big", action="store_true")
    a = ap.parse_args()
    bpe = BPE(str(L.MODEL))
    log(f"tokenizer ← {L.MODEL}")

    stale = collect_lines()
    u = set(stale[L.CE_DIR]) | set(stale[L.CE_DIR_EXT])
    log(f"过期 sid：ce 目录 {len(stale[L.CE_DIR])} · ce64 目录 "
        f"{len(stale[L.CE_DIR_EXT])} · 并集 {len(u)}")

    rep = []
    for path in (CACHE16, CACHE64):
        total, hit = scan_cache(path, u)
        rep.append((str(path), total, hit))
        log(f"{path}: 总 {total} 行，过期命中 {hit}")

    if a.scan_big:
        import glob
        pats = [str(L.PROJ / "cohort_big_*" / "anchors.jsonl")]
        for pat in pats:
            for f in sorted(glob.glob(pat)):
                wins = one = 0
                for l in open(f, encoding="utf-8"):
                    r = json.loads(l)
                    ctx = r.get("ctx")
                    if ctx is None:
                        continue
                    wins += 1
                    if ctx and bpe.ntok(ctx) == 1:
                        one += 1
                log(f"[big] {f}: 窗 {wins}，ntok==1 {one}")
        log("[big] instruct_ce_probe: 无 anchors 存档（只有 results.jsonl，"
            "行内无 ctx）→ 无法扫；其输入窗来自 instruct 语料，下次重跑前"
            "应按同款逻辑对输入失效")

    lines = [f"CE 缓存锚口径失效报告（2026-11-28）",
             f"过期 sid：ce {len(stale[L.CE_DIR])} / ce64 "
             f"{len(stale[L.CE_DIR_EXT])} / 并集 {len(u)}"]
    for p, t, h in rep:
        lines.append(f"{p}: 总 {t} 行，过期命中 {h}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not a.apply:
        log("dry-run（--apply 落失效与重放输入）")
        return 0
    work = L.WORK
    work.mkdir(parents=True, exist_ok=True)
    for path in (CACHE16, CACHE64):
        target = L.CE_DIR if path == CACHE16 else L.CE_DIR_EXT
        bak = path.with_suffix(".jsonl.bak_preanchor")
        if not bak.exists():
            path.rename(bak)
        tmp = path.with_suffix(".jsonl.new")
        kept = dropped = 0
        with open(bak, encoding="utf-8") as f, \
                open(tmp, "w", encoding="utf-8") as g:
            for l in f:
                try:
                    sid = json.loads(l)["sid"]
                except ValueError:
                    g.write(l)
                    continue
                if sid in u:
                    dropped += 1
                    continue
                g.write(l)
                kept += 1
        tmp.rename(path)
        log(f"{path.name}: 保留 {kept}，失效 {dropped}（备份 {bak.name}）")
        out = work / (f"refill_preanchor_{'16' if target == L.CE_DIR else '64'}.jsonl")
        # 重放输入 = 该目录实际被删的 sid 对应行（并集行对两个目录同构）
        with open(out, "w", encoding="utf-8") as g:
            for sid, ln in sorted(stale[target].items()):
                g.write(json.dumps(ln, ensure_ascii=False) + "\n")
        log(f"重放输入 {out}（{len(stale[target])} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
