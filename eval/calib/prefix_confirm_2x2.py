#!/usr/bin/env python3
"""前缀确认两路的 2×2 对照（2026-11-24）。

问题：前缀自动确认有两条触发路径——
  窗口路径 A：窗口融合序第 1 只覆盖前缀（≥2 字、end < 段末）、与第 2 名
             margin ≥ θ(1.0)；
  句级路径 B：5 条 N-gram 组句按全句 CE 序的首选 lead，其余 4 条 lead 一致，
             或与首个不同 lead 者 margin ≥ θ；在候选全体（引擎池 16 条）里找
             同文本、同 span 的候选执行。
两路是否都必要？在 luna 混合窗 cohort 上按生产逻辑（llm_filter.cc Collect 的
前缀判定段）逐窗复算，分两层：

  ① 逐窗静态（静态窗全集，已打 ≥6 码）：触发率 / 精度 / 召回 / 互补性
     （both、A-only、B-only、neither）、错确认归属、θ 扫描。
     B 分两态：B_pre = 组句 CE 未落地（只看 ngram 序，全 5 条 lead 一致才触发，
     快打时的真实形态）；B_full = CE 已落地（停顿时）。
  ② 整句打字动态仿真（dyn 子集）：{A 开关}×{B 开关} 策略下逐键打完整句，
     统计错确认 / 正确确认 / 手动选择收尾 / 翻页 / 收尾失败。

共同口径：冷账本（层1 = 引擎序）、θ=1.0、单字倍数 1.0、整段 ≥6 码且确认后
剩余 ≥4 码、页宽 5。两路同帧都触发时生产以后执行的 B 为准。
dyn 窗口的 CE / 组句 / 全句 CE 按需补算（luna_mixed_lib），产物幂等追加。

输出：eval/prefix_confirm_2x2.txt
用法：python eval/calib/prefix_confirm_2x2.py [--static] [--dynamic]
        [--dyn-per-domain 100] [--workers 3]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import luna_mixed_lib as L  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

THETA = 1.0
SINGLE_SCALE = 1.0
KCE = (1.0, 1.10, 0.60)      # 生产 kCeW
PAGE = 5                     # menu/page_size
MIN_SEG, MIN_REMAIN = 6, 4
THETAS = (0.5, 1.0, 1.5, 2.0, 3.0)
PATHS = ("A", "B_full", "B_pre")
NAME = {"A": "窗口A", "B_full": "句级B*", "B_pre": "句级B"}


def log(msg):
    print(f"[2x2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Store:
    """窗口状态 = 引擎池 + CE + 组句 + 全句 CE，按生产 Collect 的融合/准入逻辑。"""

    def __init__(self):
        self.cands = json.load(open(L.CANDS, encoding="utf-8"))
        self.wins = {w["wid"]: w for w in L.jsonl_iter(L.WINDOWS)}
        self.grid = json.load(open(L.OUT / "grid.json", encoding="utf-8"))
        self.reload()

    def reload(self):
        self.ce, self.gen, self.full = L.load_ce_multi(), L.load_gen(), L.load_genfull()
        self.tried = L.load_gen_tried()

    def state(self, wid, missing, need_full=False):
        """→ dict；数据不全时把缺口记入 missing 并返回 None。"""
        w = self.wins[wid]
        pool = self.cands.get(w["code"], [])[:L.K_COLLECT]
        texts = [c[0] for c in pool]
        tset = set(texts)
        ce = {}
        for ln, sid in L.ce_lines(w["ctx"], texts):
            r = self.ce.get(sid)
            if r is None:
                missing["ce"].append((ln, sid))
                return None
            for t in ln["cands"]:
                if t in tset and t in r and r[t][0] is not None:
                    ce[t] = r[t]
        g = self.gen.get(wid)
        if g is None:
            # 组句要求码串整段可覆盖；切在音节中间 ⇒ 生成器返回空（已记录在
            # gen_tried），此时句级路径不可用，但这不是数据缺口。
            if len(w["code"]) >= 4 and wid not in self.tried:
                missing["gen"].append((wid, w["ctx"], w["code"]))
                return None
            g = []
        full = self.full.get(wid, {})
        if need_full and any(x["text"] not in full for x in g):
            missing["full"].append((wid, w["ctx"], [x["text"] for x in g]))
            return None
        return self.build(w, pool, g, ce, full)

    @staticmethod
    def build(w, pool, g, ce, full):
        n = len(w["code"])
        texts = [c[0] for c in pool]
        ends = [c[2] for c in pool]
        gset = {x["text"] for x in g}
        nong = [i for i, t in enumerate(texts) if t not in gset]
        window, tail = nong[:L.K_WINDOW], nong[L.K_WINDOW:]
        good = [w["text"].startswith(t) for t in texts]

        # 融合（生产：score + β·log1p(eff) + u1；冷账本后两项为 0。
        # score = -(kCeW·ce 三分量)，不可算分量（哨兵 -1）不参与）
        # 2026-11-25 修：按 ntok 截断，哨兵 -1.0 不再被当数值加进和里
        score = {}
        for i in window:
            c3 = ce.get(texts[i])
            if c3 is None:
                continue
            nt = c3[1] if c3[1] else sum(
                1 for x in c3[0] if x is not None and x > -1e8)
            s = 0.0
            for k in range(min(int(nt), 3)):
                x = c3[0][k] if k < len(c3[0]) else None
                if x is not None and x > -1e8:
                    s += KCE[k] * x
            score[i] = -s
        order0 = sorted(window,
                        key=lambda i: -(score[i] if i in score else -1e10))
        fused, order = {}, []
        if len(order0) > 1:
            min_valid = min(score.values()) if score else 0.0
            for i in order0:
                fused[i] = score.get(i, min_valid)
            order = sorted(order0, key=lambda i: -fused[i])
        st = dict(w=w, n=n, texts=texts, ends=ends, good=good, window=window,
                  tail=tail, fused=fused, order=order, gen=g,
                  gen_scores={t: -v[0] for t, v in full.items() if v[0] < 1e9})
        st["A"] = path_a(st)
        st["B_full"] = path_b(st, use_scores=True)
        st["B_pre"] = path_b(st, use_scores=False)
        return st


def path_a(s):
    """窗口路径：融合序第 1 只覆盖前缀（≥2 字）且 margin ≥ θ → (text, end, margin)。"""
    o, f = s["order"], s["fused"]
    if len(f) < 2 or len(o) < 2:
        return None
    i1, i2 = o[0], o[1]
    t1 = s["texts"][i1]
    if (len(t1) >= 2 and s["ends"][i1] < s["n"] and s["n"] >= MIN_SEG
            and s["n"] - s["ends"][i1] >= MIN_REMAIN
            and f[i1] > -1e307 and f[i2] > -1e307
            and f[i1] - f[i2] >= THETA):
        return (t1, s["ends"][i1], f[i1] - f[i2])
    return None


def path_b(s, use_scores):
    """句级路径 → (text, end, margin)；use_scores=False 为组句 CE 未落地态。"""
    g = s["gen"]
    if not g or s["n"] < MIN_SEG:
        return None
    gs = s["gen_scores"] if use_scores else {}

    def valid(t):
        return t in gs and gs[t] > -1e9
    gord = sorted(range(len(g)), key=lambda k: (
        0 if valid(g[k]["text"]) else 1,
        -(gs[g[k]["text"]]) if valid(g[k]["text"]) else 0.0))
    g1 = g[gord[0]]
    lead, le = g1["lead"], g1["lead_end"]
    if not lead or le < 0 or le >= s["n"]:
        return None
    same = all(g[k]["lead"] == lead for k in range(1, len(g)))
    okc, margin = same, None
    if not same:
        comp = next(k for k in gord[1:] if g[k]["lead"] != lead)
        t1, t2 = g1["text"], g[comp]["text"]
        if valid(t1) and valid(t2):
            theta = THETA * (SINGLE_SCALE if len(lead) < 2 else 1.0)
            margin = gs[t1] - gs[t2]
            okc = margin >= theta
        else:
            okc = False
    if okc and s["n"] - le < MIN_REMAIN:
        okc = False
    if not okc:
        return None
    return (lead, le, margin) if pickable(s, lead, le) else None


def pickable(s, lead, le):
    """生产：在候选全体（引擎池）里找 start=段首、end=le、text==lead 的候选。"""
    gset = {x["text"] for x in s["gen"]}
    return any(t == lead and e == le and t not in gset
               for t, e in zip(s["texts"], s["ends"]))


def display(s, settled):
    """最终序 [组句首选] + 窗口融合序 + 尾窗 → [(text, end)]。"""
    out = []
    if s["gen"]:
        if settled and s["gen_scores"]:
            b = max(s["gen"], key=lambda x: s["gen_scores"].get(x["text"], -1e18))
        else:
            b = s["gen"][0]
        out.append((b["text"], s["n"]))
    out += [(s["texts"][i], s["ends"][i]) for i in s["order"]]
    out += [(s["texts"][i], s["ends"][i]) for i in s["tail"]]
    return out


def a_margin(s):
    """窗口路径的原始判定量：首选是守卫内的 ≥2 字前缀候选 → (margin, 正确否)。"""
    if len(s["order"]) < 2:
        return None
    i1, i2 = s["order"][0], s["order"][1]
    t1 = s["texts"][i1]
    if (len(t1) < 2 or s["ends"][i1] >= s["n"] or s["n"] < MIN_SEG
            or s["n"] - s["ends"][i1] < MIN_REMAIN):
        return None
    if s["fused"][i1] <= -1e307 or s["fused"][i2] <= -1e307:
        return None
    return (s["fused"][i1] - s["fused"][i2], s["good"][i1])


def b_margin(s):
    """句级路径的原始判定量（lead 不一致、两侧都有分、守卫内、池内可执行）
    → (margin, 正确否, lead 字数)。"""
    g, gs = s["gen"], s["gen_scores"]
    if not g or s["n"] < MIN_SEG:
        return None
    scored = [x for x in g if x["text"] in gs]
    if len(scored) < 2 or len({x["lead"] for x in scored}) < 2:
        return None
    top = max(scored, key=lambda x: gs[x["text"]])
    lo = max((x for x in scored if x["lead"] != top["lead"]),
             key=lambda x: gs[x["text"]])
    if top["lead_end"] >= s["n"] or s["n"] - top["lead_end"] < MIN_REMAIN:
        return None
    if not pickable(s, top["lead"], top["lead_end"]):
        return None
    return (gs[top["text"]] - gs[lo["text"]],
            s["w"]["text"].startswith(top["lead"]), len(top["lead"]))


# ── ① 逐窗静态 ────────────────────────────────────────────────────────
def cmd_static(st):
    dom, possible = [], []
    fire = {k: [] for k in PATHS}
    ok = {k: [] for k in PATHS}
    txt = {k: [] for k in PATHS}
    a_raw, b_raw = [], []
    a_raw_idx, a_raw_ok = {}, {}
    n_win = n_elig = 0
    for w in L.jsonl_iter(L.WINDOWS):
        if w["dyn"]:
            continue
        n_win += 1
        n = len(w["code"])
        if n < MIN_SEG:
            continue
        s = st.state(w["wid"], {"ce": [], "gen": [], "full": []})
        if s is None:
            continue
        n_elig += 1
        dom.append(w["domain"])
        gset = {x["text"] for x in s["gen"]}
        # 守卫内、池内可执行的正确前缀（任一路径能确认的上限）
        possible.append(any(s["good"][i] and n - s["ends"][i] >= MIN_REMAIN
                            and s["texts"][i] not in gset
                            for i in range(len(s["texts"]))))
        for k in PATHS:
            r = s[k]
            fire[k].append(r is not None)
            ok[k].append(bool(r) and s["w"]["text"].startswith(r[0]))
            txt[k].append(r[0] if r else None)
        am, bm = a_margin(s), b_margin(s)
        if am:
            a_raw.append(am)
            a_raw_idx[len(dom) - 1] = am[0]
            a_raw_ok[len(dom) - 1] = am[1]
        if bm:
            b_raw.append(bm)

    tot = len(dom)
    rng = range(tot)
    wiki = [i for i in rng if dom[i] == "wiki"]
    user = [i for i in rng if dom[i] == "user"]

    def cnt(x, sel=None):
        sel = rng if sel is None else sel
        return sum(bool(x[i]) for i in sel)

    def pct(x, sel):
        return 100.0 * cnt(x, sel) / max(len(sel), 1)

    def fired(k):
        return [i for i in rng if fire[k][i]]

    out = ["=" * 104,
           f"前缀确认两路 2×2 · ① 逐窗静态 · 静态窗 {n_win}，其中已打 ≥{MIN_SEG} 码 "
           f"{n_elig}（wiki {len(wiki)} / user {len(user)}）",
           f"θ={THETA} 单字倍数={SINGLE_SCALE} 剩余≥{MIN_REMAIN} 码 冷账本",
           "=" * 104,
           f"守卫内、引擎池内有可执行的正确前缀的窗：{cnt(possible)}"
           f"（{pct(possible, rng):.2f}%）——任何确认策略的召回上限"]
    out.append("")
    out.append(f"{'路径':<8}{'触发':>7}{'触发率':>8}{'精度':>8}{'召回':>8}"
               f"{'wiki精度':>10}{'user精度':>10}{'单字':>6}{'多字':>6}")
    for k in PATHS:
        f = fired(k)
        o = [i for i in f if ok[k][i]]
        single = sum(1 for i in f if len(txt[k][i]) == 1)
        out.append(
            f"{NAME[k]:<8}{len(f):>7}{100 * len(f) / max(tot, 1):>7.2f}%"
            f"{100 * len(o) / max(len(f), 1):>7.2f}%"
            f"{100 * len(o) / max(cnt(possible), 1):>7.2f}%"
            f"{pct(ok[k], [i for i in f if dom[i] == 'wiki']):>9.2f}%"
            f"{pct(ok[k], [i for i in f if dom[i] == 'user']):>9.2f}%"
            f"{single:>6}{len(f) - single:>6}")
    out.append("  （B* = 组句 CE 已落地态，停顿打字；B = 未落地态，快打时只看 ngram 序）")

    for k in ("B_full", "B_pre"):
        fA, fB = set(fired("A")), set(fired(k))
        both = sorted(fA & fB)
        aonly, bonly = sorted(fA - fB), sorted(fB - fA)
        same = sum(1 for i in both if txt["A"][i] == txt[k][i])
        oA = [i for i in aonly if ok["A"][i]]
        oB = [i for i in bonly if ok[k][i]]
        out += ["",
                f"互补性（A vs {NAME[k]}）",
                f"  同帧都触发 {len(both)}（同文本 {same}）；"
                f"A 独触发 {len(aonly)}（对 {len(oA)}）；"
                f"{NAME[k]} 独触发 {len(bonly)}（对 {len(oB)}）",
                f"  四象限：both {len(both)} / A-only {len(aonly)} / "
                f"{NAME[k]}-only {len(bonly)} / neither "
                f"{tot - len(both) - len(aonly) - len(bonly)}"]
        out.append(f"  {'2×2 策略':<12}{'触发':>7}{'对':>7}{'错':>5}{'精度':>8}{'召回':>8}")
        for nm, useA, useB in (("none", False, False), ("A", True, False),
                               (NAME[k], False, True), ("A+" + NAME[k], True, True)):
            fs, os_ = [], []
            for i in rng:
                # 生产：两路同帧都触发时以后执行的 B 为准
                hitB = useB and fire[k][i]
                hitA = useA and fire["A"][i] and not hitB
                if hitA or hitB:
                    fs.append(i)
                    os_.append(i if (ok[k][i] if hitB else ok["A"][i]) else -1)
            good = [i for i in os_ if i >= 0]
            out.append(f"  {nm:<12}{len(fs):>7}{len(good):>7}"
                       f"{len(fs) - len(good):>5}"
                       f"{100 * len(good) / max(len(fs), 1):>7.2f}%"
                       f"{100 * len(good) / max(cnt(possible), 1):>7.2f}%")
        wa = {i for i in fired("A") if not ok["A"][i]}
        wb = {i for i in fired(k) if not ok[k][i]}
        out.append(f"  错确认归属：A 错 {len(wa)}，{NAME[k]} 错 {len(wb)}，"
                   f"同帧都错 {len(wa & wb)}；{NAME[k]} 独有的错 {len(wb - wa)}，"
                   f"A 独有的错 {len(wa - wb)}")
        ga = {i for i in fired("A") if ok["A"][i]}
        gb = {i for i in fired(k) if ok[k][i]}
        out.append(f"  正确确认归属：A 对 {len(ga)}，{NAME[k]} 对 {len(gb)}，"
                   f"交 {len(ga & gb)}，并 {len(ga | gb)}")
        agree = {i for i in both if txt["A"][i] == txt[k][i]}
        of = [ok["A"][i] for i in agree]
        out.append(f"  两路一致（同帧同文本）{len(agree)} 帧，精度 "
                   f"{100 * sum(of) / max(len(of), 1):.2f}%，"
                   f"召回 {100 * sum(of) / max(cnt(possible), 1):.2f}%"
                   f"（用作交叉校验门：只认两路一致）")

    out += ["", "θ 扫描（守卫内候选的原始 margin，与路径触发条件同口径）",
            "  「独有」= 该 θ 下 A 触发、但同帧句级 B 不触发的帧（A 的边际价值）"]
    for nm, raw in (("窗口A（首选为 ≥2 字前缀候选的帧）", a_raw),
                    ("句级B*（lead 不一致且可执行的帧）", b_raw)):
        if not raw:
            continue
        parts = []
        for th in THETAS:
            f = [ok_ for (mg, ok_, *_) in raw if mg >= th]
            parts.append(f"θ={th}: 触发 {len(f)} 精度 "
                         f"{100 * sum(f) / max(len(f), 1):.2f}%")
        out.append(f"  {nm} 共 {len(raw)} 帧：" + "；".join(parts))
    if a_raw:
        bfire = {i for i in rng if fire["B_full"][i]}
        for nm_b in ("B_full", "B_pre"):
            bfire = {i for i in rng if fire[nm_b][i]}
            parts = []
            for th in THETAS:
                f = [i for i in rng
                     if a_raw_idx.get(i) is not None and a_raw_idx[i] >= th
                     and i not in bfire]
                g = [i for i in f if a_raw_ok[i]]
                parts.append(f"θ={th}: 独有触发 {len(f)} 精度 "
                             f"{100 * len(g) / max(len(f), 1):.2f}%")
            out.append(f"  A 独有（对照 {NAME[nm_b]} 不触发）" + "；".join(parts))
    if b_raw:
        for lo, hi, nm in ((1, 1, "单字 lead"), (2, 99, "多字 lead")):
            sub = [x for x in b_raw if lo <= x[2] <= hi]
            if sub:
                f = [ok_ for (mg, ok_, _) in sub if mg >= THETA]
                out.append(f"  句级B* {nm}：帧 {len(sub)}，θ={THETA} 触发 {len(f)} "
                           f"精度 {100 * sum(f) / max(len(f), 1):.2f}%")
    txt_out = "\n".join(out)
    (L.EVAL / "prefix_confirm_2x2.txt").write_text(txt_out + "\n", encoding="utf-8")
    print(txt_out)
    cmd_cost(fire, ok, txt, possible, dom)


def cmd_cost(fire, ok, txt, possible, dom):
    """决策表：每次正确确认省 1 次手动选择，每次错确认赔 5 键（撤销重打）。
    追加到报告尾部（前面的静态表保留原样）。"""
    tot = len(dom)
    npos = sum(1 for x in possible if x)
    out = ["", "决策表（正确确认省 1 次手动选择；错确认赔 5 键 = 撤销重打；"
               "分母 = 已打 ≥6 码的静态窗）",
           f"{'策略':<14}{'正确确认':>9}{'错确认':>8}{'精度':>8}{'召回':>8}"
           f"{'净收益(键)':>11}{'每窗净':>9}"]
    combos = [("仅窗口A", ("A",), None),
              ("仅句级B", ("B_pre",), None),
              ("仅句级B*", ("B_full",), None),
              ("A+句级B", ("A", "B_pre"), None),
              ("A+句级B*", ("A", "B_full"), None),
              ("两路一致(B*)", ("A", "B_full"), "agree"),
              ("两路一致(B)", ("A", "B_pre"), "agree")]
    counts = {}
    for nm, paths, mode in combos:
        corr = wrong = 0
        for i in range(tot):
            if mode == "agree":
                if not (fire["A"][i] and fire[paths[-1]][i]
                        and txt["A"][i] == txt[paths[-1]][i]):
                    continue
                good = ok[paths[-1]][i]
            else:
                hit = None
                for p in paths:      # 同帧以 B 为准（生产后执行）
                    if fire[p][i]:
                        hit = p
                if hit is None:
                    continue
                good = ok[hit][i]
            if good:
                corr += 1
            else:
                wrong += 1
        counts[nm] = (corr, wrong)
        net = corr - 5 * wrong
        out.append(f"{nm:<14}{corr:>9}{wrong:>8}"
                   f"{100 * corr / max(corr + wrong, 1):>7.2f}%"
                   f"{100 * corr / max(npos, 1):>7.2f}%{net:>11}"
                   f"{net / max(tot, 1):>9.3f}")
    out.append("  注：无组句资产（ngram 未加载）时句级 B 不可用，届时只能走窗口 A"
               "——此时 A 是回退路径而非并行路径。")
    out += ["", "错确认代价敏感性（净收益(键) = 正确确认 − K×错确认；"
                "负 = 自动确认反而拖慢）",
            f"{'策略':<14}" + "".join(f"{'K=' + str(k):>11}"
                                      for k in (2, 3, 5, 8, 10, 15))]
    for nm, _, _ in combos:
        corr, wrong = counts[nm]
        out.append(f"{nm:<14}" + "".join(f"{corr - k * wrong:>11}"
                                         for k in (2, 3, 5, 8, 10, 15)))
    with open(L.EVAL / "prefix_confirm_2x2.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


# ── ② 动态仿真 ────────────────────────────────────────────────────────
def simulate(st, sent, policy, missing):
    """逐键打完整句。policy ∈ none / A / Bpre / A+Bpre / Bfull /
    A+Bfull / M1 M2 M3（合并路线：组句 lead 一致 ∧ 窗口前 1/2/3 名）。"""
    text, syl = sent["text"], sent["syls"]
    g = st.grid[sent["sid"]]
    settled = "full" in policy
    r = dict(cc=0, wc=0, picks=0, flips=0, fail=0, done=True)
    m, typed = 0, 2
    for _ in range(600):
        if m >= len(text):
            break
        full_len = len("".join(syl[m:]))
        wid = g.get(f"{m}:{typed}")
        if wid is None:
            r["done"] = False
            break
        s = st.state(wid, missing, need_full=settled)
        if s is None:
            return None
        if policy != "none" and typed >= MIN_SEG:
            pick = None
            if policy.startswith("M"):      # 合并路线：组句 lead 一致 ∧ 窗口前 3
                tb = s["B_pre"]
                if tb is not None:
                    need = int(policy[1:])
                    for k, i in enumerate(s["order"][:need]):
                        if s["texts"][i] == tb[0]:
                            pick = tb
                            break
            elif "B" in policy:
                pick = s["B_full" if settled else "B_pre"]
            if pick is None and "A" in policy and not policy.startswith("M"):
                pick = s["A"]
            if pick:
                if text[m:].startswith(pick[0]):
                    m += len(pick[0])
                    typed -= pick[1]
                    r["cc"] += 1
                    continue
                r["wc"] += 1          # 错确认：撤销后照常继续打
        if typed >= full_len:
            disp = display(s, settled)
            best = None
            for p, (t, e) in enumerate(disp):
                if text[m:].startswith(t) and (
                        best is None or len(t) > len(disp[best][0])):
                    best = p
            if best is None:
                r["fail"] += len(text) - m
                r["done"] = False
                break
            t, e = disp[best]
            r["flips"] += best // PAGE
            if len(t) < len(text) - m:
                r["picks"] += 1
            m += len(t)
            typed = 1 if m < len(text) else 0
            continue
        typed += 1
    return r


def require_all(st, sents, rounds=10, workers=3):
    """预算出这批句子全部窗口状态的 CE / 组句 / 全句 CE（幂等补算）。
    仿真本身按需发现缺口会一轮只补一两窗（首缺即返回），故先整体备料。"""
    for rnd in range(1, rounds + 1):
        missing = {"ce": [], "gen": [], "full": []}
        n_state = 0
        for s in sents:
            for key in st.grid[s["sid"]]:
                n_state += 1
                st.state(st.grid[s["sid"]][key], missing, need_full=True)
        n_ce = len({sid for _, sid in missing["ce"]})
        n_gen = len({j[0] for j in missing["gen"]})
        n_full = len({j[0] for j in missing["full"]})
        if not (n_ce or n_gen or n_full):
            log(f"备料完成：{n_state} 个窗口状态（{len(sents)} 句）")
            return
        log(f"备料第 {rnd} 轮：状态 {n_state}，缺 CE {n_ce} 行 / 组句 {n_gen} 窗 / "
            f"全句 CE {n_full} 窗")
        if n_gen:
            L.run_gen(missing["gen"])
        if n_full:
            L.run_full(missing["full"])
        if n_ce:
            L.run_ce(missing["ce"], workers=workers)
        st.reload()
    raise SystemExit("动态仿真备料 10 轮后仍有缺口")


def cmd_dynamic(st, per_domain, workers):
    sents = [s for s in L.jsonl_iter(L.SENTENCES) if s["dyn"]]
    keep, seen = [], Counter()
    for s in sents:
        if seen[s["domain"]] < per_domain:
            seen[s["domain"]] += 1
            keep.append(s)
    sents = keep
    policies = ["none", "A", "Bpre", "A+Bpre", "Bfull", "A+Bfull",
                "M1", "M2", "M3"]
    log(f"动态仿真：{len(sents)} 句（{dict(seen)}）× {len(policies)} 策略")
    require_all(st, sents, workers=workers)
    missing = {"ce": [], "gen": [], "full": []}
    results = {p: [simulate(st, s, p, missing) for s in sents]
               for p in policies}
    if missing["ce"] or missing["gen"] or missing["full"]:
        raise SystemExit(f"仿真缺数据：{ {k: len(v) for k, v in missing.items()} }")
    out = ["=" * 104,
           f"前缀确认两路 2×2 · ② 整句打字动态仿真 · {len(sents)} 句"
           f"（wiki {seen['wiki']} / user {seen['user']}）· 逐键打完整句",
           "每键评分都已落地；pre = 组句 CE 未落地（快打），full = 已落地（停顿）；"
           "收尾 = 显示序里最长的正确前缀（页宽 5，翻页计 1 键/页）",
           "=" * 104,
           f"{'策略':<10}{'错确认句':>9}{'错确认':>7}{'正确确认':>9}{'手动收尾':>9}"
           f"{'翻页':>6}{'收尾失败句':>11}{'失败字':>7}{'总附加键':>9}"]
    for dom in ("wiki", "user", "all"):
        out.append(f"-- {dom} --")
        for p in policies:
            rs = [r for r, s in zip(results[p], sents)
                  if dom == "all" or s["domain"] == dom]
            n = len(rs)
            wc = sum(r["wc"] for r in rs)
            extra = (wc * 5
                     + sum(r["picks"] + r["flips"] + r["fail"] for r in rs))
            out.append(f"{p:<10}{100 * sum(1 for r in rs if r['wc']) / n:>8.1f}%"
                       f"{wc:>7}{sum(r['cc'] for r in rs):>9}"
                       f"{sum(r['picks'] for r in rs):>9}"
                       f"{sum(r['flips'] for r in rs):>6}"
                       f"{100 * sum(1 for r in rs if not r['done']) / n:>10.1f}%"
                       f"{sum(r['fail'] for r in rs):>7}{extra:>9}")
    out.append("  总附加键 = 5×错确认（撤销并重打的代价假设）+ 手动收尾 + 翻页 + 失败字")
    # 动态段单独成文件：静态段由 --static 整篇重写，两段互不覆盖
    (L.EVAL / "prefix_confirm_2x2_dynamic.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--dyn-per-domain", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if not (a.static or a.dynamic):
        a.static = a.dynamic = True
    st = Store()
    if a.static:
        cmd_static(st)
    if a.dynamic:
        cmd_dynamic(st, a.dyn_per_domain, a.workers)


if __name__ == "__main__":
    main()
