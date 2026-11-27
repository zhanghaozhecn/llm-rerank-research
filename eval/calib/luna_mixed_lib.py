#!/usr/bin/env python3
"""luna 混合窗 cohort 的公共件（2026-11-24）。

路径、无头引擎探针、插件 DLL 批跑（N-gram 组句 / 全句 CE）、窗口 3-token CE
批跑，以及三个可追加的落盘存储（gen.tsv / genfull.tsv / ce/results.jsonl）。
构建脚本 luna_mixed_build.py 与分析脚本 luna_mixed_eval.py /
prefix_confirm_2x2.py 共用——动态仿真按需补算时也走这里的入口，产物幂等追加。

口径常量与生产 llm_filter.cc 对齐：
  K_COLLECT = 16   Collect 上限（kMaxCollect；30ms 预算不模拟）——**历史口径**，
                   2026-11-26 线上已是 64，见 K_COLLECT_EXT
  K_WINDOW  = 6    评分窗 max_candidates
  MAX_CTX   = 20   推理上文 token 上限（max_tokens）

**收集池 16 → 64（2026-11-26）**：线上 Collect=64（30ms 预算不变），但本 cohort
的 CE 只算到 16 条（`ce_lines`/`assemble` 截 `K_COLLECT`），探针上限也只 32
（`K_PROBE`）——「预算窗」结果全部是 16 池模拟，超出的 1-token 免费候选没有 CE。
扩池走 `K_COLLECT_EXT = 64`：探针上限、CE 输入、cohort 组装三处统一用它；
`K_COLLECT` 保持 16，遗留脚本（build_prefix_big / prefix_* / user_freq_feat /
short_ctx）继续按 16 池口径读旧产物（那些产物里 eng 只有 16 条）。扩池后写
`cands64.json` / `cohort64.jsonl` / `ce64/results.jsonl`，旧产物原样保留。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import corpus_std  # noqa: E402

PROJ = corpus_std.PROJ
EVAL = corpus_std.EVAL
OUT = EVAL / "luna_mixed"
WORK = OUT / "work"
RIME_IME = PROJ.parent / "rime-llm-ime"       # 源码版（rime.dll / 探针头文件）
PLUGIN = PROJ.parent / "rime-llm-rerank"      # 插件版（rime_llm.dll + lua_host）
PROBE_EXE = PROJ / "cpp" / "luna_mixed_probe.exe"
PROBE_SRC = EVAL / "tools" / "luna_mixed"
LUA_HOST = PLUGIN / "cpp" / "lua_host.exe"
RIME_DLL = RIME_IME / "bin" / "rime.dll"
SIM_CPU = PROJ / "cpp" / "build_sim" / "Release" / "sim_rerank.exe"
WEASEL_DATA = os.environ.get(
    "RIME_LLM_WEASEL_DATA", r"C:\Program Files\Rime\weasel-0.17.4\data")
RIME_USER = Path(os.environ.get("APPDATA", "")) / "Rime"
NGRAM_DIR = Path(os.environ.get("RIME_LLM_NGRAM_DIR",
                                str(RIME_USER / "llm_ngram")))
MODEL = os.environ.get("GGUF_MODEL_06B",
                       "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf")
LUA_THREADS = os.environ.get("LUNA_MIXED_THREADS", "8")

K_COLLECT = 16
K_COLLECT_EXT = 64        # 线上 Collect 上限（2026-11-26）；扩池产物用这个
K_WINDOW = 6
K_PROBE = 32              # **旧**探针上限（probe_out.tsv 的缓存键含此值）
K_PROBE_EXT = 64          # 扩池探针上限（写 probe_out64.tsv）
MAX_CTX = 20
PROBE_SCHEMA = "luna_mixed"
NGRAM_FILES = ("zi", "jichu", "duoyin", "diming", "mingren", "yiren")
# 生产 ngram_generate 参数（llm_filter.cc）：topk 5 / beam 32 / min_weight 10 /
# max_edges 8 / ctx_tail 10
NGRAM_PARAMS = "topk=5,beam=32,min_weight=10,max_edges=8,ctx_tail=10"

SENTENCES = OUT / "sentences.jsonl"
CANDS = OUT / "cands.json"
WINDOWS = OUT / "windows.jsonl"
COHORT = OUT / "cohort.jsonl"
GEN = OUT / "gen.tsv"
GENFULL = OUT / "genfull.tsv"
GEN_TRIED = WORK / "gen_tried.txt"
CE_DIR = OUT / "ce"
CE_RESULTS = CE_DIR / "results.jsonl"
# ── 扩池产物（收集池 64，2026-11-26）──────────────────────────────────
# 与旧产物完全隔离：cands.json / cohort.jsonl / ce/results.jsonl 一字不动
# （eng 只有 16 条，遗留脚本照旧按 16 池口径读）。
CANDS_EXT = OUT / "cands64.json"
COHORT_EXT = OUT / "cohort64.jsonl"
CE_DIR_EXT = OUT / ("ce" + str(K_COLLECT_EXT))
CE_RESULTS_EXT = CE_DIR_EXT / "results.jsonl"


def jsonl_iter(path):
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_probe_user(work=None):
    """探针用户目录 = 研究方案 + lua 滤镜 + default.custom（全新 userdb）。
    2026-11-28 基线重采：另放真实账本快照（→ llm_rerank/luna_mixed/
    user_ledger.tsv）——账本有序源在 engine 层按方案目录读账本，研究池
    须与生产同源（B2 起引擎序含账本热词；无账本 = 池退回旧行为）。
    快照每次调用刷新（真机账本演进后重跑 probe 自动跟进）。"""
    user = Path(work or WORK) / "user"
    (user / "lua").mkdir(parents=True, exist_ok=True)
    for name in ("luna_mixed.schema.yaml", "default.custom.yaml"):
        shutil.copy2(PROBE_SRC / name, user / name)
    shutil.copy2(PROBE_SRC / "lua" / "dump_span.lua", user / "lua" / "dump_span.lua")
    try:
        led = RIME_USER / "llm_rerank" / "luna_pinyin" / "user_ledger.tsv"
        if led.exists():
            # 两处都放：luna_mixed 无 llm_filter → g_schema_dir 空 → 引擎侧
            # 账本读取退回 llm_rerank **根**（llm_file_path_ex migrate 回退）；
            # 方案目录那份留给未来研究方案挂 llm_filter 时用。
            root_dir = user / "llm_rerank"
            root_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(led, root_dir / "user_ledger.tsv")
            dst = root_dir / "luna_mixed"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(led, dst / "user_ledger.tsv")
    except OSError:
        pass
    return user


def run_probe(codes, work=None, k_probe=None, tag=""):
    """码串 → 引擎原序候选 [[text, start, end, quality, type], ...]（≤k_probe）。
    结果按（码串清单 md5 + 条数上限）缓存，重跑构建不重复枚举（45k 码 ≈ 5 分钟）。
    work 可指定独立工作目录（多 cohort 共用入口时不互相踩缓存）。
    k_probe / tag 缺省 = 旧口径（32 / 无后缀 → probe_out.tsv）；扩池用
    k_probe=K_PROBE_EXT + tag="64" → probe_out64.tsv（缓存戳同带后缀，
    与旧戳互不覆盖——2026-11-26 收集池 16→64）。"""
    work = Path(work or WORK)
    user = prepare_probe_user(work)
    work.mkdir(parents=True, exist_ok=True)
    kp = K_PROBE if k_probe is None else int(k_probe)
    fin = work / "probe_codes.txt"
    fout = work / f"probe_out{tag}.tsv"
    stamp = WORK / f"probe_out{tag}.md5"
    with open(fin, "w", encoding="ascii", newline="\n") as f:
        for c in codes:
            f.write(c + "\n")
    key = hashlib.md5(fin.read_bytes()).hexdigest() + f":{kp}"
    cached = (fout.exists() and stamp.exists()
              and stamp.read_text(encoding="ascii").strip() == key)
    if not cached:
        with open(fin, "rb") as stdin, open(fout, "wb") as stdout, \
                open(WORK / f"probe_err{tag}.log", "wb") as stderr:
            p = subprocess.run([str(PROBE_EXE), str(RIME_DLL), WEASEL_DATA,
                                str(user), PROBE_SCHEMA, str(kp)],
                               stdin=stdin, stdout=stdout, stderr=stderr,
                               timeout=3600)
        if p.returncode != 0:
            raise SystemExit(
                f"probe rc={p.returncode}; see {WORK / f'probe_err{tag}.log'}")
        stamp.write_text(key, encoding="ascii")
    out = {}
    with open(fout, encoding="utf-8") as f:
        for line in f:
            fl = line.rstrip("\n").split("\t")
            if len(fl) < 5 or fl[0] != "Q":
                continue
            cands = []
            if fl[4]:
                for item in fl[4].split("\x1f"):
                    text, _, info = item.partition("\x1e")
                    s, e, q, typ = (info.split("|") + ["", "", "", ""])[:4]
                    try:
                        cands.append([text, int(s), int(e), float(q), typ])
                    except ValueError:
                        continue
            out[fl[1]] = cands
    return out


def run_lua(script, env):
    """跑插件 cpp\\lua_host.exe（cwd 固定在其目录；路径一律绝对）。
    rime_llm.dll 在进程退出的静态析构期会访问冲突（0xC0000005，模型释放
    时序），输出文件在 "[lm] … done jobs=" 之前已全部落盘——以该标记为准，
    有标记即视为成功。"""
    e = dict(os.environ)
    e.update({k: str(v) for k, v in env.items()})
    log = WORK / (Path(script).stem + ".log")
    WORK.mkdir(parents=True, exist_ok=True)
    base = log.stat().st_size if log.exists() else 0
    with open(log, "ab") as err:
        p = subprocess.run([str(LUA_HOST), str(script)], cwd=str(LUA_HOST.parent),
                           env=e, stdout=err, stderr=subprocess.STDOUT,
                           timeout=7 * 86400)
    if p.returncode != 0:
        # rime_llm.dll 在进程退出的静态析构期常报 0xC0000005（模型释放时序），
        # 输出早在收尾前落盘——调用方按"还缺哪些 wid"重试即可收敛，故这里
        # 只告警不抛（真失败会表现为"无进展"，由调用方的轮次上限兜住）。
        log_msg(f"lua_host rc={p.returncode}（忽略，按缺口重试）：{log}")
    return p.returncode


def log_msg(msg):
    print("[lm]", msg, flush=True)


def run_gen(jobs, out=None, tried=None):
    """N-gram 组句（生产参数）。jobs = [(wid, ctx, code)] → gen.tsv 追加。
    行：wid \t rank \t ngram 分 \t lead \t lead_end \t 词表(end:word;…) \t text

    组句要求码串从 0 连续覆盖到末尾；L 切在音节中间（如 `shangl`）时生成器
    返回空 ⇒ 这些窗在 gen.tsv 里没有任何行。**"没有行"≠"没跑过"**，故另记
    `gen_tried.txt`：跑过但为空的窗，分析侧必须当成 gen=[]（句级路径不可用），
    而不是当成数据缺口。"""
    out = Path(out or GEN)
    tried = Path(tried or GEN_TRIED)
    tried_set = set(tried.read_text(encoding="ascii").split()) \
        if tried.exists() else set()
    done = {r[0] for r in load_gen_raw(out)}
    todo = [(w, c, k) for (w, c, k) in jobs
            if w not in done and w not in tried_set]
    if not todo:
        return 0
    tried.parent.mkdir(parents=True, exist_ok=True)
    jf = tried.parent / (tried.stem + "_jobs.tsv")
    with open(jf, "w", encoding="utf-8", newline="\n") as f:
        for w, c, k in todo:
            ctx = c.replace("\t", " ").replace("\n", " ")
            f.write(f"{w}\t{ctx}\t{k}\n")
    cache = Path(tried).parent / "ngram_lex.bin"
    if not cache.exists() and (RIME_USER / "llm_rerank" / "ngram_lex" /
                               "luna_pinyin.bin").exists():
        shutil.copy2(RIME_USER / "llm_rerank" / "ngram_lex" / "luna_pinyin.bin",
                     cache)
    run_lua(HERE / "luna_mixed_gen.lua", {
        "LM_MODE": "gen", "LM_JOBS": jf, "LM_OUT": out,
        "LM_MODEL": MODEL, "LM_THREADS": LUA_THREADS,
        "LM_GRAM": NGRAM_DIR / "wanxiang-lts-zh-hans.gram",
        "LM_DICT": NGRAM_DIR / "dicts", "LM_FILES": ",".join(NGRAM_FILES),
        "LM_CACHE": cache, "LM_PARAMS": NGRAM_PARAMS})
    with open(tried, "a", encoding="ascii") as f:
        for w, _, _ in todo:
            f.write(w + "\n")
    return len(todo)


def load_gen_tried():
    """跑过组句的窗（含生成结果为空的）。"""
    if not GEN_TRIED.exists():
        return set()
    return set(GEN_TRIED.read_text(encoding="ascii").split())


def run_full(jobs):
    """全句 CE（生产 score_full，≤5 条/窗）。jobs = [(wid, ctx, [text...])] →
    genfull.tsv 追加。行：wid \t text \t sum_ce \t ntok"""
    have = load_genfull()
    todo = [(w, c, ts) for (w, c, ts) in jobs
            if any(t not in have.get(w, {}) for t in ts)]
    if not todo:
        return 0
    jf = WORK / "full_jobs.tsv"
    with open(jf, "w", encoding="utf-8", newline="\n") as f:
        for w, c, ts in todo:
            ctx = c.replace("\t", " ").replace("\n", " ")
            f.write(f"{w}\t{ctx}\t" + "\x1f".join(ts[:5]) + "\n")
    run_lua(HERE / "luna_mixed_gen.lua", {
        "LM_MODE": "full", "LM_JOBS": jf, "LM_OUT": GENFULL,
        "LM_MODEL": MODEL, "LM_THREADS": LUA_THREADS, "LM_MAXCTX": MAX_CTX})
    return len(todo)


def load_gen_raw(path=None):
    """gen.tsv 原始行 [(wid, rank, nscore, lead, lead_end, words, text)]。"""
    path = Path(path or GEN)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fl = line.rstrip("\n").split("\t")
            if len(fl) != 7:
                continue
            rows.append((fl[0], int(fl[1]), float(fl[2]), fl[3], int(fl[4]),
                         fl[5], fl[6]))
    return rows


def load_gen():
    """wid → [ {text, nscore, lead, lead_end, words:[[end, word],…]}, … ]（ngram 序）。"""
    out = {}
    for (wid, rank, ns, lead, le, words, text) in load_gen_raw():
        wl = []
        for it in words.split(";"):
            if not it:
                continue
            e, _, w = it.partition(":")
            wl.append([int(e), w])
        out.setdefault(wid, []).append(
            dict(text=text, nscore=ns, lead=lead, lead_end=le, words=wl))
    for v in out.values():
        v.sort(key=lambda g: g["nscore"], reverse=True)
    return out


def load_genfull():
    """wid → {text: (sum_ce, ntok)}。"""
    out = {}
    if not GENFULL.exists():
        return out
    with open(GENFULL, encoding="utf-8") as f:
        for line in f:
            fl = line.rstrip("\n").split("\t")
            if len(fl) != 4:
                continue
            out.setdefault(fl[0], {})[fl[1]] = (float(fl[2]), int(fl[3]))
    return out


def ce_lines(ctx, cands, k_pool=None):
    """一窗的 CE 输入行（sim_rerank 每行 ≤10 候选，不足 2 条用「〇」补位——
    CE 逐候选独立，补位候选的结果丢弃）。→ [({"context","cands"}, sid)]
    k_pool 缺省 = K_COLLECT（16，历史口径）；扩池传 K_COLLECT_EXT（64）。"""
    out = []
    pool = list(cands[:(K_COLLECT if k_pool is None else k_pool)])
    for i in range(0, len(pool), 10):
        chunk = pool[i:i + 10]
        if len(chunk) < 2:
            chunk = chunk + ["〇"]
        line = {"context": ctx, "cands": chunk}
        out.append((line, corpus_std.window_key_ctx(ctx, chunk)))
    return out


def load_ce(path=None):
    """ce/results.jsonl → sid → {text: (ce 分量 list, ntok)}。
    path 缺省 = 旧 16 池结果；扩池传 CE_RESULTS_EXT。"""
    path = Path(path or CE_RESULTS)
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                ce, nt = r.get("ce") or {}, r.get("ntok") or {}
                out[r["sid"]] = {t: (ce.get(t), nt.get(t)) for t in ce}
    return out


def load_ce_multi():
    """全链路 CE 合并（sid 幂等，同 sid 同 CE）：旧 16 池 + 扩池 64 两目录。
    扩池后源码目录 `ce/` 与 `ce64/` 都是全链共享，读方一律走这里；
    写方 `run_ce` 仍写单目录（按缺口补算）。"""
    out = load_ce(CE_RESULTS_EXT)
    out.update(load_ce())
    return out


def run_ce(lines, workers=1, out_dir=None):
    """按需补算窗口 3-token CE。lines = ce_lines(...) 的列表 → 追加
    ce/results.jsonl，返回新算行数。workers=1 单进程同步；>1 转交
    ce/run_ce_cohort.py 线程池（按 sid 断点续跑）。
    out_dir 缺省 = 旧 16 池目录；扩池传 CE_DIR_EXT。"""
    out_dir = Path(out_dir or CE_DIR)
    results = out_dir / "results.jsonl"
    have = load_ce(results)
    todo = [(ln, sid) for (ln, sid) in lines if sid not in have]
    if not todo:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SIM_NO_CE_PIPE"] = "1"
    payload = "".join(json.dumps(ln, ensure_ascii=False) + "\n"
                      for ln, _ in todo).encode("utf-8")
    if workers > 1:
        inp = WORK / "ce_on_demand.jsonl"
        with open(inp, "wb") as f:
            f.write(payload)
        script = HERE.parent / "ce" / "run_ce_cohort.py"
        p = subprocess.run([sys.executable, str(script), "--input", str(inp),
                            "--out", str(out_dir), "--workers", str(workers),
                            "--chunk", "500", "--ctx", str(MAX_CTX),
                            "--model", MODEL],
                           capture_output=True, env=env, timeout=7 * 86400)
        if p.returncode != 0:
            raise RuntimeError("run_ce_cohort rc=%d\n"
                               + p.stderr.decode("utf-8", "replace")[-800:])
        return len(todo)
    p = subprocess.run([str(SIM_CPU), MODEL, "--ce", "--ngl", "0",
                        "--ctx", str(MAX_CTX)],
                       input=payload, capture_output=True, env=env,
                       timeout=7 * 86400)
    out_lines = p.stdout.decode("utf-8", "replace").strip().split("\n")
    if p.returncode != 0 or len(out_lines) != len(todo):
        raise RuntimeError(
            f"sim rc={p.returncode} 输出 {len(out_lines)} 行 ≠ 输入 {len(todo)}\n"
            + p.stderr.decode("utf-8", "replace")[-800:])
    with open(results, "a", encoding="utf-8") as f:
        for (ln, sid), ol in zip(todo, out_lines):
            r = json.loads(ol)
            f.write(json.dumps({"sid": sid, "scores": r.get("scores", {}),
                                "ce": r.get("ce", {}),
                                "ntok": r.get("ntok", {})},
                               ensure_ascii=False) + "\n")
    return len(todo)


# ── CE 预算准入（2026-11-26 生产口径）────────────────────────────────
CE_BUDGET = 12


def budget_mask(nt, gshow, pad=None, budget=CE_BUDGET):
    """引擎序扫描的预算准入 → [n,W] bool 窗掩码（与 llm_filter 同构）：

    1-token 免费全收；多 token 花 min(ntok,3)−1，够就收并扣预算，
    不够就跳过继续扫（后面的便宜候选仍可进）。nt=0（研究侧无 CE）按
    2-token 计（cost 1），评分侧会被 has_ce 掩掉。
    """
    import numpy as np
    nt = np.asarray(nt)
    gshow = np.asarray(gshow)
    n, W = nt.shape
    pad = np.zeros_like(gshow) if pad is None else np.asarray(pad)
    win = np.zeros((n, W), dtype=bool)
    rem = np.full(n, int(budget), dtype=int)
    for j in range(W):
        ok = (~gshow[:, j]) & (~pad[:, j])
        cost = np.where(nt[:, j] <= 1, 0, np.minimum(nt[:, j], 3) - 1)
        cost = np.where(nt[:, j] == 0, 1, cost)
        take1 = ok & (nt[:, j] == 1)
        take_m = ok & ((nt[:, j] >= 2) | (nt[:, j] == 0)) & (cost <= rem)
        win[:, j] = take1 | take_m
        rem = rem - np.where(take_m, cost, 0)
    return win
