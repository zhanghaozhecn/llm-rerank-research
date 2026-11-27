#!/usr/bin/env python3
"""
corpus_std.py — 语料分析口径单源（2026-11-05 标准化，详见 口径标准-corpus-standards.md）

活跃脚本（prep_samples_article / calibrate_article / sweep_tau_beta /
calibrate_lenweight / sim_gpu_parallel）一律 import 本模块，禁止本地重写
以下任一函数——window_key 等哈希协议曾有 5 处独立实现，靠"逐字一致"才能
对上缓存，单源后由 import 保证。

【规范口径速查】
  采样    wiki 文章级蓄水池 seed=42、ASCII≤10% 文章级过滤；样本内 shuffle seed=42 截断
  分词    语料重放 = jieba + pdsp 词典全量注入（"词典重放切分"）；
          自由文本评测 = HanLP（"自然文本切分"，见 eval_rerank/eval_prefer，不经本模块）；
          打字记录线不分词（记录本身是词流）
  编码    pdsp_dict.yaml："..." 后、跳 #、\\t 分割；只取多字词且 4 码（=部署 code_pattern .{4}）；
          word→code 取文件序第一个
  窗口    MAX_CAND=5；候选 = 同码词表前 5（正确词按词典位次就座）；正确词 pos≥5 剔除
  eff     formula_d：tick=每中文词提交+1、事前 eff（bump 前取值）、τ=TAU
  上下文  存储 CTX_TAIL 字尾；推理末 MAX_CTX_TOK token（=生产 max_ctx）
  键      window_key = md5(去空白ctx + b"\\0" + 各窗词 + b"\\x01")（不含 eff → τ 改变缓存仍复用）
"""
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# ── 常量（部署口径的唯一出处）──
MAX_CAND = 10         # 标定/融合窗口宽度（2026-11-05 用户定案 5→10：librime userdb
                      # 调频会使成熟用户的可见窗偏离文件词典序 top-5——rank 6~10 的词
                      # 可能被提上来。标定窗扩到 10 覆盖调频后可能进入的词; 标定同时
                      # 报 hit@5（=生产现状 max_candidates 5）与 hit@10 双口径。
                      # 生产 max_candidates 默认仍 5, 是否改 10 由标定增益决定）
CTX_TAIL = 120        # 样本存储的上文末尾字符数（sim 只取末 10 token，足够）
MAX_CTX_TOK = 10      # 推理上文 token 上限（=生产 rime_llm max_ctx）
TAU = 800.0           # formula_d 时间常数（2026-11-02 定案）
DEFAULT_ARM = int(os.environ.get("COHORT_ARM", "30000"))
                      # 用户默认年龄档（2026-11-18 用户定案：10 万 → 3 万次提交；
                      # 同时"不再进行多用户重放"——多年龄档对比退役, 分析默认取 3 万档）
SEED = 42             # 一切随机化的种子
ASCII_LIMIT = 0.20    # 文章级 ASCII 占比过滤阈值（2026-11-05 用户定案 10%→20%：
                      # 10% 是句子级口径误搬到文章级——单峰分布正切峰腹, 计算机
                      # 科学/地质学类核心条目(15-16%)整篇被丢; 20% = 96万/153.7万篇）

# ── 语料路径注册表（env 覆盖集中此处；新脚本不得自带路径常量）──
# 2026-11-18 子目录重组后本模块住 eval/lib/，研究库根 = 上两级
PROJ = Path(__file__).resolve().parents[2]             # 研究库根
EVAL = PROJ / "eval"                                   # eval 根（报告/工作数据统一落这里）
DICT_PATH = Path(os.environ.get(
    "RIME_LLM_PDSP_DICT", str(PROJ / "pdsp_dict.yaml")))   # 词典（布局变体经 env 换入，
    #   如引擎侧 r2 布局重标定：RIME_LLM_PDSP_DICT=...pdsp_dict_r2.yaml）
WIKI_ARTICLES = os.environ.get(
    "RIME_LLM_WIKI_ARTICLES",
    "本机分词注音工程/分读音词频统计/data/wiki_clean.txt")   # 文章级（现行唯一 wiki 口径）
WIKI_SENTENCES = os.environ.get(
    "RIME_LLM_WIKI",
    "本机分词注音工程/分读音词频统计/data/sentences_new.txt")  # 句子池（仅旧工具/回归）
MODEL = os.environ.get("GGUF_MODEL", "d:/gguf_models/Qwen3.5-0.8B-Q4_K_M.gguf")
SIM_EXE = Path(os.environ.get(
    "RIME_LLM_SIM_EXE", str(PROJ / "cpp/build_sim/Release/sim_rerank_gpu.exe")))


# ── 词典 ──
def load_pdsp_dict(dict_path=None):
    """pdsp_dict → (code_to_words, word_code)
    只收多字词且 4 码（部署口径）；word_code 取文件序第一个码；
    code_to_words 保持文件序（窗口次序 = 词典次序，部署一致）。"""
    code_to_words, word_code = {}, {}
    with open(dict_path or DICT_PATH, encoding="utf-8") as f:
        in_body = False
        for line in f:
            line = line.rstrip("\n\r")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line[0] == "#":
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            w, c = parts[0], parts[1]
            if len(w) >= 2 and len(c) == 4:
                code_to_words.setdefault(c, [])
                if w not in code_to_words[c]:
                    code_to_words[c].append(w)
                word_code.setdefault(w, c)
    return code_to_words, word_code


def load_pdsp_dict_single3(dict_path=None):
    """pdsp_dict → (code_to_words_sc, word_code_sc) 单字 3 码口径
    （2026-11-12 一字词重排研究，部署语义 = 用户口授：2 码单字固定顺序
    不重排；3 码 = 声韵+形码（主动辅助码），有重码（1959/5145 码，
    覆盖 4973 词）→ 重排域）。
    收录规则（与旧管线 prep_training 单字口径一致）：CJK 单字、无 1 码、
    全部 3 码前 2 码唯一（声韵一致）、有 3 码（非 \\ 符号码）；
    word_code_sc = 该词文件序第一个 3 码；code_to_words_sc 按全部 3 码行
    收录（文件序去重，与多字词 load_pdsp_dict 同语义——同码窗 = 词典次序）。"""
    code_to_words, word_code = {}, {}
    codes_of = {}
    with open(dict_path or DICT_PATH, encoding="utf-8") as f:
        in_body = False
        for line in f:
            line = line.rstrip("\n\r")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line[0] == "#":
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            w, c = parts[0], parts[1]
            if len(w) != 1 or not ("\u4e00" <= w <= "\u9fff") or len(c) != 3:
                continue
            if c.startswith("\\"):
                continue
            codes_of.setdefault(w, [])
            if c not in codes_of[w]:
                codes_of[w].append(c)
            if w not in code_to_words.get(c, []):
                code_to_words.setdefault(c, []).append(w)
    for w, cs in codes_of.items():
        if any(len(x) == 1 for x in cs):
            continue
        prefixes = {x[:2] for x in cs}
        if len(prefixes) != 1:
            continue
        word_code[w] = cs[0]
    # 只保留仍有正典词的码组（正典 = 文件序第一个 3 码）
    keep = set(word_code.values())
    code_to_words = {c: ws for c, ws in code_to_words.items() if c in keep}
    return code_to_words, word_code


def load_pdsp_dict_single2(dict_path=None):
    """pdsp_dict → (code_to_words_s2, word_code_s2) 单字 2 码口径
    （2026-11-13 扩样研究：2 码窗放开重排的可行性评估——2 码 = 声韵，
    同码组 = 该声韵全部字（同音域，远大于 3 码重码组 1959/5145 码）。
    收录：CJK 单字全部 2 码行（声韵全码，如 部\\tbk）与 3 码行（前 2 位
    归组，如 与\\tvkh → vk）；行内多码逐个取（与\\tv\\tvj 的 vj 也是 2 码）；
    1 码/符号码(\\)/4 码+ 不归组——**不按"无 1 码"排除**（与 single3 不同：
    的/是/与这些 1 码简码字在 2 码窗仍是主角，1 码只是它们的另一条路）。
    word_code_s2 = 该字文件序第一个非 1 码的 2 码前缀（正典 2 码，部署
    词典序第一读音）；多音字不同读音前缀属不同 2 码组（与@vl 与 与@vk
    天然分窗，部署语义一致）；code_to_words_s2[2码] = 该 2 码全部字按
    文件序（窗口次序 = 词典次序，部署 2 码固定顺序语义）。"""
    code_to_words, word_code = {}, {}
    with open(dict_path or DICT_PATH, encoding="utf-8") as f:
        in_body = False
        for line in f:
            line = line.rstrip("\n\r")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line[0] == "#":
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            w = parts[0].strip()
            if len(w) != 1 or not ("\u4e00" <= w <= "\u9fff"):
                continue
            for c in parts[1:]:
                c = c.strip()
                if not c or c.startswith("\\"):
                    continue
                if len(c) == 2:
                    p = c
                elif len(c) == 3:
                    p = c[:2]
                else:
                    continue          # 1 码简码 / 4 码+：不归 2 码组
                if w not in word_code:
                    word_code[w] = p
                if w not in code_to_words.get(p, []):
                    code_to_words.setdefault(p, []).append(w)
    return code_to_words, word_code


def jieba_inject(code_to_words):
    """【冻结·旧口径】jieba 注入词典词（11-02 标定链）。新口径 = HanLP + OOV DP。"""
    import jieba
    for words in code_to_words.values():
        for w in words:
            jieba.add_word(w)
    return jieba


def load_pdsp_words(dict_path=None):
    """pdsp_dict → word→code（**全部词**含单字；一词多码取文件序第一个）。
    OOV 拆词的覆盖词集用（单字也是可提交词，产 bump 不产样本）；
    与 load_pdsp_dict（多字 4 码窗）互补。"""
    word_code = {}
    with open(dict_path or DICT_PATH, encoding="utf-8") as f:
        in_body = False
        for line in f:
            line = line.rstrip("\n\r")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line[0] == "#":
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word_code.setdefault(parts[0], parts[1])
    return word_code


# ── 采样 ──
def reservoir_articles(path, k, seed=SEED, ascii_limit=ASCII_LIMIT,
                       progress_every=200000):
    """【冻结·旧口径】蓄水池抽样 k 篇（11-02 标定链的复现入口，勿用于新分析）。
    新口径 = build_shuffled_corpus + 顺序前缀（2026-11-05 用户定案：文章级整体
    打乱、模拟打字按顺序取、快检取前 10%/1%）。"""
    rng = random.Random(seed)
    picked, n_seen = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            ascii_cnt = sum(1 for c in line if ord(c) < 128)
            if ascii_cnt / len(line) > ascii_limit:
                continue
            n_seen += 1
            if len(picked) < k:
                picked.append(line)
            else:
                j = rng.randrange(n_seen)
                if j < k:
                    picked[j] = line
            if progress_every and n_seen % progress_every == 0:
                print(f"  scanned {n_seen} articles, kept {len(picked)}/{k}",
                      flush=True)
    return picked, n_seen


def build_shuffled_corpus(src, dst, seed=SEED, ascii_limit=ASCII_LIMIT):
    """【新口径·一次性】全部通过 ASCII 过滤的文章整体随机打乱（seed 固定）落盘。
    打字模拟 = 按该文件顺序取前缀（快检取前 10%/1% 数量）。→ 文章数"""
    arts = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            ascii_cnt = sum(1 for c in line if ord(c) < 128)
            if ascii_cnt / len(line) > ascii_limit:
                continue
            arts.append(line)
    random.Random(seed).shuffle(arts)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for a in arts:
            f.write(a + "\n")
    return len(arts)


# ── HanLP 分段缓存（新口径：自然文本切分，方案无关，一次分词长期复用）──
# 解释器优先级: RIME_LLM_VENV_PY（存在才用）→ 当前解释器（本机 main Python312
# 已装 hanlp 2.1.3; 原 bert_seg venv 已不存在, 2026-11-05 实测回退）
_VENV_ENV = os.environ.get("RIME_LLM_VENV_PY", "")
VENV_PY = _VENV_ENV if _VENV_ENV and Path(_VENV_ENV).exists() else sys.executable
HANLP_WORKER = PROJ / "eval" / "prep" / "hanlp_seg_worker.py"


def _build_range(corpus_path, out_path, start, end, batch=64, tag="w0"):
    """[start, end) 篇范围的 HanLP 分段缓存 → out_path（独立 part 文件）。
    常驻 worker 进程一次加载模型, 逐批请求-回写。→ 写入行数"""
    import subprocess
    import tempfile
    errf = os.path.join(tempfile.gettempdir(), f"hanlp_seg_worker_stderr_{tag}.log")
    n_rows = 0
    with open(errf, "w", encoding="utf-8") as ef:
        proc = subprocess.Popen([VENV_PY, str(HANLP_WORKER)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=ef,
                                text=True, encoding="utf-8", bufsize=1)
        with open(corpus_path, encoding="utf-8") as cf, \
                open(out_path, "w", encoding="utf-8") as out:
            for _ in range(start):                 # 跳过段前缀
                next(cf)
            t0 = time.time()
            try:
                while start + n_rows < end:
                    texts = []
                    for _ in range(batch):
                        if start + n_rows + len(texts) >= end:
                            break
                        line = next(cf, None)
                        if line is None:
                            break
                        texts.append(line.rstrip("\n\r"))
                    if not texts:
                        break
                    proc.stdin.write(json.dumps({"texts": texts},
                                                ensure_ascii=False) + "\n")
                    proc.stdin.flush()
                    reply_line = proc.stdout.readline()
                    if not reply_line.strip():     # EOF = 子进程死
                        raise RuntimeError(
                            f"hanlp worker[{tag}] 无响应 (rc={proc.poll()})，"
                            f"stderr 尾 30 行见 {errf}")
                    reply = json.loads(reply_line)
                    for toks in reply["tokens"]:
                        out.write(json.dumps(toks, ensure_ascii=False) + "\n")
                    n_rows += len(texts)
                if n_rows and (start + n_rows) % 5000 < batch:
                    rate = n_rows / max(1e-9, time.time() - t0)
                    print(f"  [{tag}] {start + n_rows}/{end} ({rate:.0f} art/s)",
                          flush=True)
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()
    return n_rows


def ensure_segment_cache(corpus_path, cache_path, n_articles, batch=64,
                         workers=None):
    """增量构建 HanLP 分段缓存（与打乱语料行号一一对齐：缓存第 i 行 =
    语料第 i 篇的 token 列表）。已就绪的前缀直接复用，只补缺的尾部。
    多 worker 段并行：[done, n) 切成 workers 段 → 各自 part 文件 → 按序拼接。
    → 缓存就绪行数。worker 数 env HANLP_BUILD_WORKERS（默认 **1**。2026-11-05 实测：
    GPU 单进程 48 art/s 稳定；GPU 多进程在本机 8GB 卡 2/3 worker 全部冻死——显存
    分配器互踩楔死驱动队列（GPU 100% 但零产出，expandable_segments Windows 不支持
    =no-op）；CPU 单进程 6.4 art/s。多worker留作可选项：GPU+CPU 异构各 1 个,
    按速率分段配额, 未实施）

    并发守卫（2026-11-06 事故后加）: 构建期持 `<cache>.building` 锁文件——
    第二个进程（如 prep 的补缺调用）见锁即返回，**绝不恢复正在被写的 part0**
    （否则恢复路径把活跃 part0 追加进主缓存 + 构建完工再追加一遍 = 重复块,
    当晚实测 30,016 行双份、total 989,477 ≠ 959,461）。锁超 24h 视为陈旧
    （崩溃残留）可强占。"""
    import time as _time
    cache_path = Path(cache_path)
    done = 0
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for _ in f:
                done += 1
    lock = Path(str(cache_path) + ".building")
    if lock.exists() and (_time.time() - lock.stat().st_mtime < 86400):
        print(f"building lock present ({lock.name}) —— 另一构建进程在跑, "
              f"跳过（ready={done}）")
        return done
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _ensure_segment_cache_locked(corpus_path, cache_path,
                                            n_articles, batch, workers, done)
    finally:
        lock.unlink(missing_ok=True)


def _ensure_segment_cache_locked(corpus_path, cache_path, n_articles, batch,
                                 workers, done):
    """ensure_segment_cache 的实体（调用方已持构建锁）。"""
    cache_path = Path(cache_path)
    # 断点自动恢复: 上次中断遗留的 part0（= [done, ...) 连续前缀, 单 worker 顺序写）
    # 并入主缓存; 尾部残缺行（进程被杀的缓冲残留）丢弃
    p0 = Path(str(cache_path) + ".part0")
    if p0.exists():
        n_ok = n_drop = 0
        with open(cache_path, "a", encoding="utf-8") as out, \
                open(p0, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                except Exception:
                    n_drop += 1
                    continue
                out.write(s + "\n")
                n_ok += 1
        p0.unlink()
        done += n_ok
        print(f"recovered part0: +{n_ok} rows (dropped {n_drop} 残尾), done={done}")
    if done >= n_articles:
        return done
    if workers is None:
        workers = int(os.environ.get("HANLP_BUILD_WORKERS", "1"))
    workers = max(1, min(workers, (n_articles - done + 999) // 1000))

    import threading
    todo = n_articles - done
    sizes = [todo // workers + (1 if k < todo % workers else 0)
             for k in range(workers)]
    ranges, pos = [], done
    for s in sizes:
        ranges.append((pos, pos + s))
        pos += s
    parts = [Path(f"{cache_path}.part{k}") for k in range(workers)]
    print(f"segment cache: ready {done}, target {n_articles}, "
          f"{workers} workers × ranges {ranges[0][0]}-{ranges[-1][1]}")

    errs = {}

    def run(k):
        try:
            rows = _build_range(corpus_path, str(parts[k]), ranges[k][0],
                                ranges[k][1], batch, tag=f"w{k}")
            if rows != ranges[k][1] - ranges[k][0]:
                raise RuntimeError(
                    f"worker[w{k}] 产出 {rows} ≠ 计划 {ranges[k][1] - ranges[k][0]}")
        except Exception as e:                     # noqa: BLE001 —— 汇总后统一抛
            errs[k] = e

    ths = [threading.Thread(target=run, args=(k,)) for k in range(workers)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        for k, e in errs.items():
            print(f"  worker w{k} 失败: {e}", flush=True)
        for p in parts:
            p.unlink(missing_ok=True)
        raise RuntimeError(f"分段构建失败（{len(errs)}/{workers}），part 文件已清理；"
                           f"详见 %TEMP%/hanlp_seg_worker_stderr_w*.log")

    with open(cache_path, "a", encoding="utf-8") as out:
        for p in parts:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    out.write(line)
    for p in parts:
        p.unlink()
    return n_articles


class DictSplitter:
    """【新口径·实时】OOV 动态规划拆词（2026-11-05 用户定案）。
    HanLP 切出的词不在当前方案词典时，拆成词典词：目标 = 子词概率**连乘最大**
    （DP 最大化 Σlog p；不按子词数归一 —— 惩罚隐含在 p<1 的连乘里，子词数越少越好）。
    子词必须全部在词典中；无法完整覆盖的 token 整体丢弃（不产 bump 不产样本）。
    实时性：词典（方案）会变 → 本类按调用方当前词典构造，缓存只存 HanLP 原始切分。

    2026-11-05 首实现两处错误（对比抽样抓出：提交 99.3% 碎成单字、可产样本词
    3.3/篇 vs jieba 链 168.9/篇）：① DP 把目标写反成最小化 Σlogp（= 最小乘积）；
    ② unigram 词频只含 HanLP 词表，词典大量词落 floor(-15.1) 而单字 -3~-8，
    拆成单字永远赢。修正 = 最大化 Σlogp + 未见词典词 Laplace 回退 log(1/total)。"""

    def __init__(self, dict_words, logp, unseen_logp):
        """dict_words: set（含单字）；logp: word→log p（语料 unigram，方案无关）；
        unseen_logp: 语料未见词典词的 Laplace 回退 log(1/total)"""
        self.words = set(dict_words)
        self.logp = logp
        self.unseen = unseen_logp

    def _cost(self, w):
        return self.logp.get(w, self.unseen)

    def split(self, token):
        """→ 词典子词列表（连乘最大 = Σlogp 最大）；无法完整覆盖 → None"""
        n = len(token)
        best = [0.0] + [-math.inf] * (n + 1)
        prev = [-1] * (n + 1)          # prev[i] = 最优路径上 token[:i] 的分段点
        for i in range(1, n + 1):
            for j in range(i):
                if best[j] == -math.inf or token[j:i] not in self.words:
                    continue
                c = best[j] + self._cost(token[j:i])
                if c > best[i]:
                    best[i], prev[i] = c, j
        if best[n] == -math.inf:
            return None
        out, i = [], n
        while i > 0:
            out.append(token[prev[i]:i])
            i = prev[i]
        out.reverse()
        return out


# ── eff 重放（formula_d，与部署 llm_processor.freq_bump 同构）──
class FormulaReplay:
    """指数衰减计数。tick=每中文词提交+1；提交 d=1+d·exp((t_w−t)/τ)；
    查询 eff=d·exp((t_w−t)/τ)。打分在提交前 → 先 eff_of 后 bump（事前口径）。"""

    def __init__(self, tau=TAU):
        self.tau = float(tau)
        self.state = {}      # word → [cnt, d, t_w]
        self.tick = 0

    def eff_of(self, w):
        e = self.state.get(w)
        return 0.0 if not e else e[1] * math.exp((e[2] - self.tick) / self.tau)

    def bump(self, w):
        self.tick += 1
        e = self.state.get(w)
        if not e:
            e = [0, 0.0, 0]
            self.state[w] = e
        e[1] = 1.0 + e[1] * math.exp((e[2] - self.tick) / self.tau)
        e[0] += 1
        e[2] = self.tick


# ── 窗口与键 ──
def normalize_ctx(s):
    return re.sub(r"\s+", "", s)


def window_of(s):
    """样本 dict（word/cands/pos）→ 评估窗（正确词按 pos 就座，截 MAX_CAND）"""
    return (s["cands"][:s["pos"]] + [s["word"]] + s["cands"][s["pos"]:])[:MAX_CAND]


def window_key_ctx(ctx, win_words):
    """md5(去空白ctx + \\0 + 各窗词\\x01)。不含 eff —— τ 改变只改 eff_win，
    分数缓存（sim_gpu_parallel/calibrate_article 的 results.jsonl）全部复用。"""
    h = hashlib.md5()
    h.update(normalize_ctx(ctx).encode("utf-8"))
    h.update(b"\x00")
    for w in win_words:
        h.update(w.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def window_key(s):
    """样本 dict → window_key_ctx（calibrate_article 原签名，向后兼容）"""
    return window_key_ctx(s["ctx"], window_of(s))


# ── v2 语料 / debug 记录解析（2026-11-17 记录重构）─────────────────
# 常开语料 rime_llm_corpus.tsv（TSV）：H/W/C/M 行；debug 单独文件，
# 用事件 ID 关联。旧 llm_training.txt 由 load_legacy_training 读（归档）。
RIME_DIR = Path(os.environ.get("APPDATA", "")) / "Rime"
LLM_DIR = RIME_DIR / "llm_rerank"  # 2026-11-21 起 LLM 数据/日志收进子目录


def corpus_files():
    """语料文件清单（2026-11-22 按方案分文件夹）：llm_rerank\\<方案>\\
    rime_llm_corpus.tsv + 旧位置（llm_rerank 根 / Rime 根）的混合历史文件。
    分析脚本可 iterate 全部文件（各方案行互不混合），默认路径取最新的。"""
    paths = []
    for p in (LLM_DIR / "rime_llm_corpus.tsv",
              RIME_DIR / "rime_llm_corpus.tsv"):
        if p.is_file():
            paths.append(p)
    if LLM_DIR.is_dir():
        for d in sorted(LLM_DIR.iterdir()):
            p = d / "rime_llm_corpus.tsv"
            if d.is_dir() and p.is_file():
                paths.append(p)
    return paths


CORPUS_PATHS = corpus_files()
# 注意：RIME_LLM_CORPUS 是既有 wiki 句子语料 env；用户语料用独立名字。
CORPUS_PATH = Path(os.environ["RIME_LLM_USER_CORPUS"]) \
    if os.environ.get("RIME_LLM_USER_CORPUS") else (
        max(CORPUS_PATHS, key=lambda p: p.stat().st_mtime)
        if CORPUS_PATHS else LLM_DIR / "rime_llm_corpus.tsv")


def _unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            out.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(c, c))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _split_cands(s):
    r"""逗号分隔候选；支持 \, 与 \\ 转义。"""
    out, cur, i = [], [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            cur.append(s[i + 1])
            i += 2
        elif s[i] == ",":
            out.append("".join(cur))
            cur = []
            i += 1
        else:
            cur.append(s[i])
            i += 1
    if cur or out:
        out.append("".join(cur))
    return [x for x in out if x != ""]


def _int(s, default=-1):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def iter_events(path=None):
    """逐行产出 v2 语料事件 dict（H/W/C/M）。文件不存在则空。"""
    p = Path(path or CORPUS_PATH)
    if not p.exists():
        return
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fld = line.split("\t")
            t = fld[0]
            if t == "W" and len(fld) >= 14:
                # v9（2026-11-27 联想类标）：可选第 15 字段 comp =
                # 「窗内位次;最终序位次」（1 基、逗号分隔）；旧行 → 空元组
                comp_w, comp_f = (), ()
                if len(fld) >= 15 and fld[14].startswith("comp="):
                    a, _, b = fld[14][5:].partition(";")
                    comp_w = tuple(int(x) for x in a.split(",") if x.strip())
                    comp_f = tuple(int(x) for x in b.split(",") if x.strip())
                yield {"type": "W", "id": fld[1], "ts": fld[2],
                       "src": fld[3], "code": _unesc(fld[4]),
                       "ctx": _unesc(fld[5]), "engine": _split_cands(fld[6]),
                       "window": _split_cands(fld[7]),
                       "final": _split_cands(fld[8]),
                       "tick": _int(fld[9]), "status": fld[10],
                       "skip": _unesc(fld[11]), "prep": fld[12],
                       "flags": fld[13],
                       "comp_window": comp_w, "comp_final": comp_f}
            elif t == "C" and len(fld) >= 11:
                yield {"type": "C", "id": fld[1], "ts": fld[2],
                       "ref": fld[3], "word": _unesc(fld[4]),
                       "code": _unesc(fld[5]), "rank_engine": _int(fld[6]),
                       "rank_window": _int(fld[7]),
                       "rank_final": _int(fld[8]), "tick": _int(fld[9]),
                       "flags": fld[10],
                       # v8（2026-11-26 整句词频归因）：可选第 12 字段 path =
                       # 「词:码/词:码/…」；旧行无该字段 = ""（解析为 []）
                       "path": _unesc(fld[11]) if len(fld) >= 12 else ""}
            elif t == "M" and len(fld) >= 7:
                yield {"type": "M", "id": fld[1], "ts": fld[2],
                       "ref": fld[3], "kind": fld[4], "detail": fld[5],
                       "flags": fld[6]}
            elif t == "H" and len(fld) >= 12:
                yield {"type": "H", "id": fld[1], "ts": fld[2],
                       "edition": fld[3], "schema": fld[4], "model": fld[5],
                       "code_pattern": fld[6], "max_candidates": _int(fld[7]),
                       "max_ctx": _int(fld[8]), "tcs": fld[9],
                       "beta": fld[10], "tau": fld[11]}


RETRACT_KINDS = {"backspace", "delete", "undo-clipboard"}
BOUNDARY_KINDS = {"scene", "focus", "navigation", "enter"}


def parse_attr_path(s):
    """v8 C 行 path 字段 → [(词, 码), …]；空/缺 = []（2026-11-26 归因）。

    格式 = `词:码/词:码/…`（词只含 CJK、码只含小写字母，`:` 与 `/` 不会
    出现在词或码内）；段非法（缺 `:` / 空词）整段跳过。
    """
    out = []
    for seg in (s or "").split("/"):
        if not seg:
            continue
        w, _, c = seg.partition(":")
        if w and c:
            out.append((w, c))
    return out


def is_nofreq(ev):
    """W 行是否"无上文/词频融合"窗（2026-11-24 起 flags 含 `nofreq`）。

    这类窗 status=ok、有 window/final，但**窗内没有 CE**（融合只剩
    β·log1p(eff) + log(uni+1)，词码对修正不参与），debug D 行 ce=-inf。
    CE/融合分/名次算术类分析必须剔除（或单独统计），否则会拿 -inf 当
    数值、把"无 CE"误报成"评分失败"。
    """
    return ev.get("type") == "W" and "nofreq" in (ev.get("flags") or "")


def iter_commits(path=None, include_retracted=True, reset_mode="strict"):
    """C 行 + retracted 标记 + 关联 W 的 engine/window/final。

    reset_mode="strict"：backspace/delete/undo 回撤前一提交；focus/navigation/
    enter 开新场、不回撤（新数据语义）。
    reset_mode="legacy"：除 scene 外所有 M 都回撤前一提交（旧 ← 语义）。
    """
    events = list(iter_events(path))
    wins = {e["id"]: e for e in events if e["type"] == "W"}
    commits, stack, scene = [], [], 0
    for e in events:
        if e["type"] == "M":
            k = e["kind"]
            if reset_mode == "legacy":
                if k == "scene":
                    scene += 1
                    stack = []
                elif stack:
                    stack.pop()["retracted"] = True
            else:
                if k in BOUNDARY_KINDS:
                    scene += 1
                    stack = []
                elif k in RETRACT_KINDS and stack:
                    stack.pop()["retracted"] = True
            continue
        if e["type"] != "C":
            continue
        w = wins.get(e["ref"]) if e.get("ref") else None
        d = dict(e)
        d["scene"] = scene
        # v8（2026-11-26 整句词频归因）：path 原串 → [(词, 码), …]
        d["path"] = parse_attr_path(e.get("path"))
        d["engine"] = w["engine"] if w else []
        d["window"] = w["window"] if w else []
        d["final"] = w["final"] if w else []
        d["cands"] = d["window"]
        d["real_win"] = d["window"] or None
        d["ctx"] = w["ctx"] if w else ""
        d["src"] = w["src"] if w else ""
        d["retracted"] = False
        commits.append(d)
        stack.append(d)
    if include_retracted:
        return commits
    return [c for c in commits if not c["retracted"]]


def iter_scenes(path=None, reset_mode="strict"):
    """场景制提交序列（等价旧 llm_training 行语义的严格/兼容两口径）。"""
    events = list(iter_events(path))
    wins = {e["id"]: e for e in events if e["type"] == "W"}
    scenes, cur = [], []
    for e in events:
        if e["type"] == "M":
            k = e["kind"]
            if reset_mode == "legacy":
                if k == "scene":
                    if cur:
                        scenes.append(cur)
                    cur = []
                elif cur:
                    cur.pop()
            else:
                if k in BOUNDARY_KINDS:
                    if cur:
                        scenes.append(cur)
                    cur = []
                elif k in RETRACT_KINDS and cur:
                    cur.pop()
            continue
        if e["type"] != "C":
            continue
        w = wins.get(e["ref"]) if e.get("ref") else None
        d = dict(e)
        d["engine"] = w["engine"] if w else []
        d["window"] = w["window"] if w else []
        d["final"] = w["final"] if w else []
        d["cands"] = d["window"]
        d["real_win"] = d["window"] or None
        d["ctx"] = w["ctx"] if w else ""
        cur.append(d)
    if cur:
        scenes.append(cur)
    return scenes


def iter_word_stream(path=None, include_retracted=True, expand=False):
    """扁平词流（旧 21 个 sweep/user 脚本口径；默认含被回撤提交）。

    expand=True（2026-11-26 整句词频归因）：C 行带 v8 path 时按归因词
    逐个产出（整句模式的词流从"句子"变"词"）；无 path 的提交仍产出
    word 整体。默认 False = 旧行为（word 原样），既有标定口径不变。
    """
    for c in iter_commits(path, include_retracted=include_retracted):
        if expand and c.get("path"):
            for w, _ in c["path"]:
                yield w
        else:
            yield c["word"]


def load_legacy_training(path):
    """旧 llm_training.txt（场景制 | 词\t码\t快照, ← 退格）→ scenes。

    仅归档旧文件/等价校验用；新语料请用 iter_scenes/iter_commits。
    """
    scenes = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries = []
            for part in line.split("|"):
                if part == "←":
                    entries.append(("←", "", None))
                elif "\t" in part:
                    fl = part.split("\t")
                    cw = fl[2].split(",") if len(fl) > 2 and fl[2] else None
                    entries.append((fl[0], fl[1] if len(fl) > 1 else "", cw))
                else:
                    entries.append((part, "", None))
            scenes.append(entries)
    return scenes
