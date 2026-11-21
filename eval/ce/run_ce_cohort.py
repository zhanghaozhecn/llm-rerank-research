#!/usr/bin/env python3
"""cohort CE 跑批统一入口（2026-11-18 子目录重组合并，3 脚本 → 1）。

吸收并取代（CLI 主路径兼容原 run_ce_cohort）:
  run_ce_cohort.py（本体）  : cohort_<dict><tag>/anchors.jsonl 或 --samples
                              → ce<tag>/results.jsonl（分块断点续跑、分片贴核）
  sim_gpu_parallel.py       : --input 通用 sim 格式 JSONL + --workers N 线程池
                              （exe 内跨窗批处理已判负回滚——并行走多进程, 精度零风险）
  seed_ce_full.py           : --seed-from 旧链目录列表——CE 是纯文本推理, 与账本
                              状态无关, 同 sid 同 CE; 新链先按 sid 播种再补缺口

结果缓存: {sid, scores[, ce, ntok]}（--scores-only 只写 scores）; sid =
window_key_ctx(ctx, win)（md5, 全链去重, 断点续跑重跑幂等）。
CPU: SIM_NO_CE_PIPE=1 + --ngl 0, 多分片/多 worker 贴核; GPU: --ngl >0 必须
--nshards 1 --workers 1（单 CUDA 进程铁律, 2026-11-08 用户定案）。
GPU 模式必须用 sim_rerank_gpu.exe（专用 CUDA 链接; sim_rerank.exe 是 CPU 静态链,
其 --ngl 99 实际走 CPU——2026-11-08 实证）。

用法:
  python ce/run_ce_cohort.py --dict big --tag _user --ngl 0 --shard 0 --nshards 5
  python ce/run_ce_cohort.py --samples ../user_sc_samples.jsonl --out ce_samples
  python ce/run_ce_cohort.py --input input.jsonl -o results.jsonl --workers 3 [--scores-only]
  python ce/run_ce_cohort.py --dict big --tag _full --ce-tag _05b --seed-from "cohort_big_sc,cohort_big_sc_x2,cohort_big,cohort_big_x2,cohort_big_user"
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
SIM_CPU = PROJ / "cpp" / "build_sim" / "Release" / "sim_rerank.exe"
SIM_GPU = PROJ / "cpp" / "build_sim" / "Release" / "sim_rerank_gpu.exe"


def resolve(p):
    """路径解析: 绝对路径原样; 相对路径优先按 cwd（存在才用）, 否则按研究库根。"""
    p = Path(p)
    if p.is_absolute():
        return p
    return p if p.exists() else PROJ / p


def load_items(a, d):
    """→ items [{sid, ctx, win, arm}]（--samples > --input > anchors）。"""
    if a.samples:
        rows = [json.loads(l) for l in open(resolve(a.samples),
                                            encoding="utf-8")]
        items = []
        for r in rows:
            win = (r["cands"][:r["pos"]] + [r["word"]] +
                   r["cands"][r["pos"]:])[:corpus_std.MAX_CAND]
            items.append({"sid": corpus_std.window_key_ctx(r["ctx"], win),
                          "ctx": r["ctx"], "win": win, "arm": None})
        return items
    if a.input:
        ip = resolve(a.input)
        items, seen = [], set()
        for l in open(ip, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            d2 = json.loads(l)
            cands = list(d2.get("cands", []))
            sid = corpus_std.window_key_ctx(d2["context"], cands)
            if sid in seen:
                continue
            seen.add(sid)
            items.append({"sid": sid, "ctx": d2["context"],
                          "win": cands, "arm": None})
        return items
    apath = resolve(a.anchors) if a.anchors else d / "anchors.jsonl"
    return [json.loads(l) for l in open(apath, encoding="utf-8")]


def seed_from(results, items, old_dirs):
    """旧链 results.jsonl 按 sid 播种（只应在首次 CE 前跑一次）。"""
    old = {}
    for name in old_dirs:
        p = resolve(name) / "results.jsonl"
        n = 0
        if not p.exists():
            print(f"seed skip (缺) {p}", flush=True)
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                sid = r.get("sid")
                if sid and sid not in old:
                    old[sid] = line if line.endswith("\n") else line + "\n"
                n += 1
        print(f"seed {name}: {n} records", flush=True)
    sids = {x["sid"] for x in items}
    n_hit = n_total = 0
    with open(results, "w", encoding="utf-8") as out:
        for sid in sids:
            n_total += 1
            rec = old.get(sid)
            if rec is not None:
                out.write(rec)
                n_hit += 1
    print(f"seed: items {n_total}, seeded {n_hit} "
          f"({n_hit / max(1, n_total) * 100:.1f}%) → {results}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="cohort CE 跑批统一入口（anchors/samples/input 三源, "
                    "分片/线程池并行, 断点续跑, 旧链播种）")
    ap.add_argument("--dict", choices=["small6w", "big"], default="small6w")
    ap.add_argument("--anchors", default=None,
                    help="锚点 JSONL（默认 cohort_<dict><tag>/anchors.jsonl）")
    ap.add_argument("--samples", default=None,
                    help="样本 JSONL（word/pos/cands）: 按 pos 插回目标词后"
                         "作为候选窗跑 CE")
    ap.add_argument("--input", default=None,
                    help="通用 sim 输入 JSONL（context/cands 行, 吸收原"
                         " sim_gpu_parallel）")
    ap.add_argument("--scores-only", action="store_true",
                    help="不透传 --ce: 只写 scores（缓存行无 ce/ntok）")
    ap.add_argument("--out", default=None,
                    help="结果目录（默认 cohort_<dict><tag>/ce<tag>；"
                         "--samples 模式默认 ce_samples）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 = 线程池并行跑块（GPU 版实测最优 3 进程; CPU 版"
                         " 进程数 × 6 内置线程 ≤ 物理核）")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--ngl", type=int, default=0,
                    help="0=CPU（SIM_NO_CE_PIPE=1, 多分片贴核）; >0=GPU 层数"
                         "（单 CUDA 进程铁律: nshards/workers 必须 1）")
    ap.add_argument("--tag", default="",
                    help="目录后缀 cohort_<dict><tag>/（训练数据链）")
    ap.add_argument("--model", default=None,
                    help="GGUF 路径覆盖（微调模型评测）; 默认 sim 内置路径")
    ap.add_argument("--ce-tag", default="",
                    help="CE 结果子目录后缀: ce<tag>/（微调模型评测与基线隔离）")
    ap.add_argument("--ctx", type=int, default=None,
                    help="sim --ctx 上文截断 token 数（§四十一; 须配 --ce-tag 隔离）")
    ap.add_argument("--arm", type=int, default=None,
                    help="只跑该年龄档的锚点（微调评测只需基准档）")
    ap.add_argument("--seed-from", default=None,
                    help="逗号分隔旧链目录: 首次 CE 前按 sid 播种（吸收原"
                         " seed_ce_full; cohort_big_full 标准五链见文档）")
    a = ap.parse_args()
    if a.ngl > 0 and (a.nshards != 1 or a.workers != 1):
        raise SystemExit("GPU 模式必须 --nshards 1 --workers 1（单 CUDA 进程铁律）")
    SIM = SIM_GPU if a.ngl > 0 else SIM_CPU
    if a.ngl > 0:
        os.environ.pop("SIM_NO_CE_PIPE", None)
    else:
        os.environ["SIM_NO_CE_PIPE"] = "1"

    d = PROJ / f"cohort_{a.dict}{a.tag}"
    items = load_items(a, d)
    out_dir = resolve(a.out) if a.out else (
        PROJ / "ce_samples" if a.samples else d / ("ce" + a.ce_tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    if a.seed_from and not results.exists():
        seed_from(results, items,
                  [s.strip() for s in a.seed_from.split(",") if s.strip()])
    done_sids = set()
    if results.exists():
        with open(results, encoding="utf-8") as f:
            for l in f:
                try:
                    done_sids.add(json.loads(l)["sid"])
                except Exception:
                    pass
    todo = [x for x in items if x["sid"] not in done_sids
            and (a.arm is None or x.get("arm") == a.arm)][
        a.shard::a.nshards]
    print(f"[shard {a.shard}/{a.nshards} workers {a.workers}] "
          f"items {len(items)}, done {len(done_sids)}, todo {len(todo)}",
          flush=True)
    if not todo:
        return 0

    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    lock = threading.Lock()
    t0 = time.time()
    n_ok = [0]

    def do_chunk(part):
        inp = "\n".join(json.dumps({"context": x["ctx"],
                                    "cands": list(x["win"])},
                                   ensure_ascii=False) for x in part) + "\n"
        cmd = [str(SIM)]
        if a.model:
            cmd.append(a.model)             # sim argv[1] = 模型路径
        if a.scores_only:
            cmd.append("--scores")
        else:
            cmd.append("--ce")
        cmd += ["--ngl", str(a.ngl)]
        if a.ctx is not None:
            cmd += ["--ctx", str(a.ctx)]
        p = subprocess.run(cmd, input=inp.encode("utf-8"),
                           capture_output=True, timeout=a.timeout)
        out_lines = p.stdout.decode("utf-8", "replace").strip().split("\n")
        if len(out_lines) != len(part):
            raise RuntimeError(
                f"输出行数 {len(out_lines)} != 输入 {len(part)}"
                f"\nstderr: {p.stderr.decode('utf-8', 'replace')[-500:]}")
        with lock, open(results, "a", encoding="utf-8") as outf:
            for x, ol in zip(part, out_lines):
                r = json.loads(ol)
                row = {"sid": x["sid"], "scores": r.get("scores", {})}
                if not a.scores_only:
                    row["ce"] = r.get("ce", {})
                    row["ntok"] = r.get("ntok", {})
                outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            outf.flush()
        with lock:
            n_ok[0] += len(part)
            rate = n_ok[0] / max(time.time() - t0, 1e-9)
            print(f"[shard {a.shard}] ce done {n_ok[0]}/{len(todo)} "
                  f"({rate:.1f} 窗/s)", flush=True)

    if a.workers > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for _ in pool.map(do_chunk, chunks):
                pass
    else:
        for part in chunks:
            do_chunk(part)
    print(f"[shard {a.shard}] 完成 → {results}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
