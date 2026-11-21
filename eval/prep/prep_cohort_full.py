#!/usr/bin/env python3
r"""pairfix 扩样全枚举收窗链（2026-11-16, 用户指令"增加样本量+扣分集中高频"）。

背景: 旧 cohort 链的水库采样把每对 (词,码) 的反事实事件截流到中位 2 个
（40,548 对中 53.5% 零事件）, 人工审核误判多。本脚本对语料**全枚举**
（去水库限流）, 且只把 CE 预算花在"事件可能窗"上。

三阶段（count/emit 同一确定性重放跑两遍, 无 RNG, 窗序列逐位一致）:
  count  全语料单遍重放（多字词4码 + 单字3码双域同流, 各自 bump 口径与
         load_domain 一致）, 每窗算层1（β·log1p(eff)+log1p(uni) 稳定降序
         前 5）; "事件可能窗" = 真值∈层1前5 ∧ 前5存在 uni<FADE（真值∉前5
         的窗不可能产生任何 fix/brk 事件——胜者必在前5, 扣分翻转落点也在
         前5）。统计每对 (词,码) 的 per-arm 供给 pot 与词暴露度 freq
         → eval/pairfix_full_pot<tag>.jsonl / _freq.json / _meta.json
  emit   读 pot → 目标对 = pot_total ≥ --pot-min（按 pot 降序, 预算耗尽时
         高供给对优先）; per-arm 子配额 = ceil(cap × pot_arm/pot_total)
         （保底 min(pot_arm,5), 保证龄档多样性——纯贪心会让浅龄臂吃光
         配额）; 再次重放, 窗选中 = 前5中存在"该臂余量>0"的目标对（选中后
         所有在场目标对消耗配额; 真值=罚词的 brk 侧供给同样计入）; 全局
         --budget 窗数上限（达到即整体停, 含 streams）。产物
         cohort_<dict><tag>/anchors.jsonl（run_ce 兼容 + dom/du/uni/code/
         pos/winsize/t5 附加字段）+ streams.jsonl（用户词流, 供未来换 β
         重新推导）+ meta.json
  smoke  一致性校验: 按 load_domain 同款逻辑重放 streams.jsonl, 对抽样
         选中窗复算 du/uni 逐值比对; 并用 numpy argsort(stable) 复算层1
         前5与 t5 交叉验证。

关键口径（与 derive_pairfix_ratio.load_domain 逐值一致）:
  big 域 bump 键 = (词, 词典首码); sc 域 bump 键 = (词, canonical 3码);
  eff 查询键 = (候选词, 窗码); du = log1p(eff) 事前值; uni = 事前计数;
  eff/uni 均在 bump 前取值。fade 门保留（与生产 pairfix_penalty 一致）,
  样本量靠更多用户 × 全枚举提升, 不截流采集。

用法:
  python -X utf8 prep_cohort_full.py count --tag _full [--limit-articles N]
  python -X utf8 prep_cohort_full.py emit  --tag _full --cap 400 --budget 2500000
  python -X utf8 prep_cohort_full.py smoke --tag _full [--samples 2000]
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
import prep_cohort
from prep_cohort import build_chain, map_article, win_gate
from corpus_std import CTX_TAIL, MAX_CAND, window_key_ctx
from derive_pairfix_ratio import FADE, load_w2codes

sys.stdout.reconfigure(encoding="utf-8")
PROJ = corpus_std.PROJ
BETA = 14.0          # 05b 生产常数（层1 权重, 与 derive --beta 14 同源）
K_TOP = 5            # 层1 前 K（summary_4x4.K, 层2 只在前 K 内判胜负）
CJK_RE = re.compile(r"[一-鿿]")
# 上文有效性门（2026-11-16 用户审核反馈：文章边界标点串上开的窗过滤——
# 无 CJK 字符的上文对生产场景无意义; 裁定前量级 0.2-1.5%/链）
def ctx_ok(s):
    return bool(s) and bool(CJK_RE.search(s))


def s1_top5(du, uni):
    """层1 前5下标（与 derive 的 np.argsort(-s1, kind="stable")[:, :K] 的
    真实候选部分等价, 按分数降序、同分保词典序）。du/uni 长度 = winsize。"""
    n = len(du)
    if n <= K_TOP:
        s1 = [BETA * du[j] + math.log1p(uni[j]) for j in range(n)]
        return sorted(range(n), key=lambda j: -s1[j])
    s1 = [BETA * du[j] + math.log1p(uni[j]) for j in range(n)]
    lo = min(s1)
    if max(s1) == lo:                 # 全同分（浅龄窗常态）→ 恒等序 = 稳定序
        return list(range(K_TOP))
    order = sorted(range(n), key=lambda j: -s1[j])   # python sort = stable
    return order[:K_TOP]


def np_check_t5(du, uni):
    """numpy 复算（smoke 交叉验证用）——与 derive 的矩阵运算完全同构。
    derive 的 win_cols 前 K 可含 pad 列（s1=-1e9 恒排末尾、永不获胜）,
    这里裁到真实候选——与 s1_top5 的语义一致。"""
    import numpy as np
    n = len(du)
    s1 = np.full(MAX_CAND, -1e9)
    # du 已是 log1p(eff)（同 derive 的 s1 = β·du + ln(uni+1), 不可再套 log1p）
    s1[:n] = BETA * np.array(du, dtype=float) + \
        np.log1p(np.array(uni, dtype=float))
    order = np.argsort(-s1, kind="stable")
    return [int(j) for j in order if j < n][:K_TOP]


def parse_arms(s):
    return [tuple(int(x) for x in it.split(":"))[:2] for it in s.split(",")]


# ────────────────────────────── sink 实现 ──────────────────────────────
class CountSink:
    """count 阶段: 每对 (词,码) per-arm 事件可能窗供给 + 词暴露度。"""

    def __init__(self, n_arms):
        self.pot = {"big": {}, "sc": {}}   # (词,码) → [供给 × n_arms]
        self.freq = Counter()              # 词 → DP 流提交次数
        self.tot = Counter()
        self.arms_stat = []
        self.n_arms = n_arms

    def on_commit(self, w, h, wi, ci):
        self.freq[w] += 1
        self.tot["commits"] += 1

    def on_window(self, rec):
        dom, h, wi, ci, w, code, win, widx, winsize, du, un, ctx, t5 = rec
        self.tot[f"win_{dom}"] += 1
        self.tot[f"elig_{dom}"] += 1
        d = self.pot[dom]
        ai = h  # arm 下标在 on_window 时由调用方换好（rec 第二字段 = arm 序号）
        for j in t5:
            k = (win[j], code)
            e = d.get(k)
            if e is None:
                e = d[k] = [0] * self.n_arms
            e[ai] += 1

    def on_user(self, words, h, wi):
        pass

    def on_arm(self, ai, h, stat):
        self.arms_stat.append(dict(stat))

    def want_stop(self):
        return False


class EmitSink:
    """emit 阶段: 目标对定向选窗 + anchors/streams/meta 落盘。"""

    def __init__(self, out_dir, arms, pot_path, pot_min, cap, budget):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.pot_min = pot_min
        self.cap = cap
        self.budget = budget
        self.n_arms = len(arms)
        self.arm_h = [h for h, _n in arms]   # arm 序号 → H 值（anchors.arm 用 H）
        # 目标对: pot_total ≥ pot_min, 按 pot_total 降序（预算耗尽时高供给对优先）
        self.targets = {}
        with open(pot_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pt = sum(r["pot"])
                if pt >= pot_min:
                    self.targets[(r["dom"], r["word"], r["code"])] = r["pot"]
        # per-arm 子配额（保底 min(pot_arm, 5), 上限不超过 pot_arm）
        self.rem = {}
        for p, pots in self.targets.items():
            pt = sum(pots)
            row = []
            for pa in pots:
                if pa <= 0:
                    row.append(0)
                else:
                    sub = max(math.ceil(cap * pa / pt), min(pa, 5))
                    row.append(min(sub, pa))
            self.rem[p] = row
        # per-arm 预算: 10%/臂保底（防贪心顺序让浅龄臂吃光 CE）+ 40% 按需求
        demand = [0] * self.n_arms
        for pots in self.targets.values():
            for ai, pa in enumerate(pots):
                demand[ai] += pa
        dm = sum(demand)
        self.budget_arm = [int(budget * 0.10) + (int(budget * 0.40 * d / dm)
                          if dm else int(budget * 0.40 / self.n_arms))
                          for d in demand]
        self.sel_arm = [0] * self.n_arms
        self.af = open(out_dir / "anchors.jsonl", "w", encoding="utf-8")
        self.stf = open(out_dir / "streams.jsonl", "w", encoding="utf-8")
        self.tot = Counter()
        self.arms_stat = []
        self.arms_sel = Counter()

    def on_commit(self, w, h, wi, ci):
        self.tot["commits"] += 1

    def on_window(self, rec):
        dom, ai, wi, ci, w, code, win, widx, winsize, du, un, ctx, t5 = rec
        self.tot[f"win_{dom}"] += 1
        self.tot[f"elig_{dom}"] += 1
        if self.tot["sel"] >= self.budget or                 self.sel_arm[ai] >= self.budget_arm[ai]:   # 窗级精确预算闸
            return
        present = []
        for j in t5:
            p = (dom, win[j], code)
            if p in self.targets and self.rem[p][ai] > 0:
                present.append(p)
        if not present:
            return
        for p in present:             # 窗内所有在场目标对消耗配额（一窗多用）
            self.rem[p][ai] -= 1
        sid = window_key_ctx(ctx, win)
        self.af.write(json.dumps(
            {"sid": sid, "arm": self.arm_h[ai], "ui": wi, "ai": wi,
             "ci": ci, "w": w, "win": win, "ctx": ctx, "depth": ci,
             "dom": dom, "code": code, "pos": widx, "winsize": winsize,
             "du": [round(x, 6) for x in du], "uni": un, "t5": t5,
             "cands": [x for x in win if x != w][:MAX_CAND - 1]},
            ensure_ascii=False) + "\n")
        self.tot["sel"] += 1
        self.sel_arm[ai] += 1
        self.arms_sel[ai] += 1

    def on_user(self, words, h, wi):
        self.stf.write(json.dumps(words, ensure_ascii=False) + "\n")
        self.tot["users"] += 1

    def on_arm(self, ai, h, stat):
        self.arms_stat.append(dict(stat, sel=self.arms_sel.get(ai, 0)))

    def want_stop(self):
        return self.tot["sel"] >= self.budget

    def close(self):
        self.af.close()
        self.stf.close()


# ────────────────────────────── 重放主循环 ──────────────────────────────
def run_replay(args, sink):
    """确定性全枚举重放（结构照抄 prep_cohort: arm 顺序消费文章, 每用户
    空库起步、整篇消费、满 H commits 即止, 文章不跨用户共享）。"""
    splitter, word_code_mc, code_to_words, _v, _sc_unused, \
        _sc2_unused = build_chain(args.dict, single=False)
    # 双域同流: 单字 3 码词典独立装载（build_chain 仅在 single 模式返回）
    code_to_words_sc, word_code_sc = corpus_std.load_pdsp_dict_single3()
    _w2codes, first_code = load_w2codes()
    arts = Path(args.articles) if args.articles else prep_cohort.SHUFFLED
    cache = Path(args.cache) if args.cache else prep_cohort.CACHE
    arms = parse_arms(args.arms)
    beta = args.beta
    tau = args.tau
    arm_seq = -1
    h, n_users = None, 0
    u_done = 0
    wi = 0
    stopped = False

    rp_big = rp_sc = None
    uni_big = uni_sc = None
    ci = 0
    words = []

    def new_user():
        return (corpus_std.FormulaReplay(tau), corpus_std.FormulaReplay(tau),
                {}, {}, 0, [])

    with open(arts, encoding="utf-8") as cf, \
            open(cache, encoding="utf-8") as tf:
        art_i = -1
        exhausted = False
        for line_art, line_tok in zip(cf, tf):
            art_i += 1
            if args.limit_articles is not None and art_i >= args.limit_articles:
                break
            if h is None:                     # 上一个 arm 恰好在文章边界收尾
                arm_seq += 1
                if arm_seq >= len(arms):
                    break
                h, n_users = arms[arm_seq]
                u_done = 0
                arm_stat = {"h": h, "users": 0, "commits": 0,
                            "art_start": art_i, "art_end": None}
                rp_big, rp_sc, uni_big, uni_sc, ci, words = new_user()
            art = line_art.rstrip("\n\r")
            arm_stat["art_end"] = art_i
            for w, pos in map_article(art, line_tok, splitter):
                # ── 多字词域窗（bump 口径 = 词典首码, 同 load_domain big）──
                g = win_gate(w, False, word_code_mc, code_to_words,
                             None, None)
                if g:
                    code, win, widx, winsize = g
                    du = [math.log1p(e) if (e := rp_big.eff_of((x, code))) else 0.0
                          for x in win]
                    un = [uni_big.get((x, code), 0) for x in win]
                    t5 = s1_top5(du, un)
                    if widx in t5 and any(un[j] < FADE for j in t5)                             and ctx_ok(art[:pos][-CTX_TAIL:]):
                        rec = ("big", arm_seq, wi, ci, w, code, win, widx,
                               winsize, du, un, art[:pos][-CTX_TAIL:], t5)
                        sink.on_window(rec)
                # ── 单字域窗（bump 口径 = canonical 3码, 同 load_domain sc）──
                g2 = win_gate(w, True, word_code_mc, code_to_words,
                              word_code_sc, code_to_words_sc)
                if g2:
                    code, win, widx, winsize = g2
                    du = [math.log1p(e) if (e := rp_sc.eff_of((x, code))) else 0.0
                          for x in win]
                    un = [uni_sc.get((x, code), 0) for x in win]
                    t5 = s1_top5(du, un)
                    if widx in t5 and any(un[j] < FADE for j in t5)                             and ctx_ok(art[:pos][-CTX_TAIL:]):
                        rec = ("sc", arm_seq, wi, ci, w, code, win, widx,
                               winsize, du, un, art[:pos][-CTX_TAIL:], t5)
                        sink.on_window(rec)
                # ── 双域 bump（每个提交都推; 与 load_domain 两域各自全程 bump 一致）──
                kb = (w, first_code.get(w, ""))
                rp_big.bump(kb)
                uni_big[kb] = uni_big.get(kb, 0) + 1
                ks = (w, word_code_sc.get(w, ""))
                rp_sc.bump(ks)
                uni_sc[ks] = uni_sc.get(ks, 0) + 1
                sink.on_commit(w, arm_seq, wi, ci)
                ci += 1
                words.append(w)
                if ci >= h:                   # 用户满 H → 收尾（文章不回退, 同 prep_cohort）
                    sink.on_user(words, arm_seq, wi)
                    arm_stat["users"] += 1
                    arm_stat["commits"] += ci
                    wi += 1
                    u_done += 1
                    if sink.want_stop():
                        stopped = True
                        break
                    if u_done >= n_users:
                        sink.on_arm(arm_seq, h, arm_stat)
                        print(f"arm H={h} done: users {u_done}, "
                              f"commits {arm_stat['commits']}, "
                              f"articles [{arm_stat['art_start']},"
                              f" {arm_stat['art_end']}]", flush=True)
                        h, n_users = None, 0
                    else:
                        rp_big, rp_sc, uni_big, uni_sc, ci, words = new_user()
                    break   # 文章剩余词整体跳过, 下一个用户从下一篇起（同 prep_cohort）
            if stopped:
                break
            if h is None and arm_seq + 1 >= len(arms):
                break
        else:
            exhausted = True
        if not stopped and not exhausted and h is not None:
            pass
        # 语料耗尽: 收尾当前用户与 arm（prep_cohort 同款: 非空词流即落盘）
        if not stopped and h is not None and words:
            sink.on_user(words, arm_seq, wi)
            arm_stat["users"] += 1
            arm_stat["commits"] += ci
            wi += 1
        if not stopped and h is not None:
            sink.on_arm(arm_seq, h, arm_stat)
            print(f"arm H={h} done (corpus end): users {arm_stat['users']}",
                  flush=True)
    return {"stopped": stopped, "users": wi, "articles": art_i + 1}


# ────────────────────────────── 三阶段命令 ──────────────────────────────
def cmd_count(a):
    arms = parse_arms(a.arms)
    sink = CountSink(len(arms))
    st = run_replay(a, sink)
    meta = {"arms": [h for h, _ in arms], "params": vars(a), "replay": st,
            "tot": dict(sink.tot), "arms_stat": sink.arms_stat}
    pot_path = PROJ / "eval" / f"pairfix_full_pot{a.tag}.jsonl"
    rows = []
    for dom, d in sink.pot.items():
        for (w, c), pots in d.items():
            rows.append((sum(pots), dom, w, c, pots))
    rows.sort(reverse=True)
    with open(pot_path, "w", encoding="utf-8") as f:
        for pt, dom, w, c, pots in rows:
            f.write(json.dumps({"dom": dom, "word": w, "code": c,
                                "pot": pots}, ensure_ascii=False) + "\n")
    freq_path = PROJ / "eval" / f"pairfix_full_pot{a.tag}_freq.json"
    freq_path.write_text(json.dumps(sink.freq, ensure_ascii=False),
                         encoding="utf-8")
    meta_path = PROJ / "eval" / f"pairfix_full_pot{a.tag}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    # 摘要
    tot = sink.tot
    print("\n== count 摘要 ==")
    print(f"users {st['users']}, commits {tot['commits']}, "
          f"articles {st['articles']}, stopped {st['stopped']}")
    print(f"windows big {tot['win_big']} (elig {tot['elig_big']}, "
          f"{tot['elig_big'] / max(1, tot['win_big']) * 100:.1f}%)  "
          f"sc {tot['win_sc']} (elig {tot['elig_sc']}, "
          f"{tot['elig_sc'] / max(1, tot['win_sc']) * 100:.1f}%)")
    for dom in ("big", "sc"):
        d = sink.pot[dom]
        pts = sorted((sum(p) for p in d.values()), reverse=True)
        nz = [x for x in pts if x > 0]
        if nz:
            n = len(nz)
            print(f"pot[{dom}]: pairs {n}, pot≥80 {sum(1 for x in nz if x >= 80)}"
                  f", ≥200 {sum(1 for x in nz if x >= 200)}"
                  f", ≥400 {sum(1 for x in nz if x >= 400)}"
                  f", p50 {nz[n // 2]}, p90 {nz[int(n * 0.1)]}")
    print(f"→ {pot_path.name} ({len(rows)} 对), {freq_path.name}, {meta_path.name}")


def cmd_emit(a):
    arms = parse_arms(a.arms)
    pot_path = PROJ / "eval" / f"pairfix_full_pot{a.tag}.jsonl"
    if not pot_path.exists():
        sys.exit(f"缺 {pot_path} —— 先跑 count")
    out_dir = PROJ / f"cohort_{a.dict}{a.tag}"
    sink = EmitSink(out_dir, arms, pot_path, a.pot_min, a.cap, a.budget)
    print(f"targets {len(sink.targets)} 对 (pot≥{a.pot_min}), "
          f"cap {a.cap}, budget {a.budget}", flush=True)
    st = run_replay(a, sink)
    sink.close()
    meta = {"params": {k: v for k, v in vars(a).items()},
            "replay": st, "tot": dict(sink.tot),
            "arms": sink.arms_stat,
            "targets": len(sink.targets),
            "budget_arm": sink.budget_arm,
            "sel_arm": sink.sel_arm}
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== emit 摘要 ==")
    print(f"users {sink.tot['users']}, commits {sink.tot['commits']}, "
          f"selected {sink.tot['sel']} / budget {a.budget}, "
          f"stopped {st['stopped']}")
    for dom in ("big", "sc"):
        print(f"windows {dom} {sink.tot[f'win_{dom}']} "
              f"(elig {sink.tot[f'elig_{dom}']})")
    print(f"selected per arm: {sink.sel_arm} / budget_arm {sink.budget_arm}")
    print(f"→ {out_dir}/anchors.jsonl, streams.jsonl, meta.json")


def cmd_smoke(a):
    out_dir = PROJ / f"cohort_{a.dict}{a.tag}"
    af = out_dir / "anchors.jsonl"
    stf = out_dir / "streams.jsonl"
    for p in (af, stf):
        if not p.exists():
            sys.exit(f"缺 {p} —— 先跑 emit")
    # 水库抽样选中窗（上限 --samples）
    rng = random.Random(corpus_std.SEED)
    picked = []
    n_anch = 0
    with open(af, encoding="utf-8") as f:
        for line in f:
            n_anch += 1
            rec = json.loads(line)
            if len(picked) < a.samples:
                picked.append(rec)
            else:
                j = rng.randrange(n_anch)
                if j < a.samples:
                    picked[j] = rec
    by_ac = {}
    for r in picked:
        by_ac.setdefault((r["ui"], r["ci"]), []).append(r)
    print(f"anchors {n_anch}, sampled {len(picked)}")
    by_ui = {}
    for r in picked:
        by_ui.setdefault(r["ui"], []).append(r)
    # load_domain 同款重放复算
    splitter, word_code_mc, code_to_words, _v, _u1, _u2 = \
        build_chain(a.dict, single=False)
    code_to_words_sc, word_code_sc = corpus_std.load_pdsp_dict_single3()
    _w2codes, first_code = load_w2codes()
    n_checked = n_du_bad = n_t5_bad = 0
    with open(stf, encoding="utf-8") as f:
        for ui, line in enumerate(f):
            rows = by_ui.get(ui)
            if not rows:
                continue          # 用户间账本独立（空库起步）→ 无样本用户整行跳过
            rp_big = corpus_std.FormulaReplay(a.tau)
            rp_sc = corpus_std.FormulaReplay(a.tau)
            uni_big, uni_sc = {}, {}
            for ci, w in enumerate(json.loads(line)):
                for r in by_ac.get((ui, ci), ()):
                    dom, code, win = r["dom"], r["code"], r["win"]
                    rp = rp_big if dom == "big" else rp_sc
                    unp = uni_big if dom == "big" else uni_sc
                    du_raw = [math.log1p(rp.eff_of((x, code))) for x in win]
                    du = [round(x, 6) for x in du_raw]
                    un = [unp.get((x, code), 0) for x in win]
                    if any(abs(x - y) > 6e-7 for x, y in zip(du, r["du"])):
                        n_du_bad += 1
                        if n_du_bad <= 3:
                            print(f"  du 不一致 ui={ui} ci={ci} "
                                  f"{r['w']}@{code}: {du} vs {r['du']}")
                    if un != r["uni"]:
                        n_du_bad += 1
                        if n_du_bad <= 3:
                            print(f"  uni 不一致 ui={ui} ci={ci} "
                                  f"{r['w']}@{code}: {un} vs {r['uni']}")
                    t5 = np_check_t5(du_raw, un)
                    if t5 != r["t5"]:
                        n_t5_bad += 1
                        if n_t5_bad <= 3:
                            print(f"  t5 不一致 ui={ui} ci={ci}: "
                                  f"{t5} vs {r['t5']}")
                    n_checked += 1
                kb = (w, first_code.get(w, ""))
                rp_big.bump(kb)
                uni_big[kb] = uni_big.get(kb, 0) + 1
                ks = (w, word_code_sc.get(w, ""))
                rp_sc.bump(ks)
                uni_sc[ks] = uni_sc.get(ks, 0) + 1
    print(f"\nchecked {n_checked} 窗: du/uni 不一致 {n_du_bad}, "
          f"t5 不一致 {n_t5_bad}")
    if n_checked == 0:
        sys.exit("无窗被校验（抽样为空?）")
    if n_du_bad or n_t5_bad:
        sys.exit("SMOKE FAIL")
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "emit", "smoke"])
    ap.add_argument("--dict", default="big")
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--arms", default="30000:550",
        help="H:n_users（无水库, 每用户全枚举）。2026-11-18 定案：多年龄档/"
             "多用户重放退役，默认只跑 3 万龄单档")
    ap.add_argument("--tau", type=float, default=400.0)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--articles", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pot-min", type=int, default=80)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--budget", type=int, default=2500000)
    ap.add_argument("--limit-articles", type=int, default=None)
    ap.add_argument("--samples", type=int, default=2000)
    a = ap.parse_args()
    {"count": cmd_count, "emit": cmd_emit, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
