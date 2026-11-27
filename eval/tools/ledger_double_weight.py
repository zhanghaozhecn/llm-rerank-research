#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""层1 × 账本源双重计权离线评估（2026-11-27 README「后续方向」P1-1）。

数据口径（无需真机重跑，全部离线）：
  语料 W 行  : window/final/engine_top 候选列表 + tick/status/flags
  语料 C 行  : ref=W-id 的提交（选中词 = 真值）
  账本       : user_ledger.tsv（dee/tick/cum）→ 词级聚合 eff = Σ dee·exp(-Δ/τ)
  debug D 行 : <W-id>|D|<final_rank>|ce=|...|eff=|fb=|u1=|full=|key=|sdelta=
               key = 融合分真值；fb = β·log1p(eff) → log1p(eff) = fb/β
               → 融合分重放：fused'(β') = key + (β'/β - 1)·fb
               （窗口准入组成不重放，属一阶近似——准入随 β 变化极缓）

三组实验：
  A 覆盖与到位率   账本词进窗率、final#1/engine#1 占位率
  B 过强判据       final#1=账本词 的窗里，该词实际被选中率；
                   「挤出」= 账本词占 final#1 而真值词排其后
  C 重放扫描       β 全局扫描（层1 强度）+ 账本词 fb/u1 单独缩放 α
                   （模拟「type==ledger 层1 降权」补救项）

用法：
  python ledger_double_weight.py [--corpus P] [--ledger P] [--debug P ...]
      [--beta 10,12,14,16,18] [--alpha 0,0.5,1.0] [--json OUT]
默认路径走 %APPDATA%/Rime/llm_rerank/<方案>/；方案用 --schema 指定。
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

TAU = 400.0


def unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def split_list(field):
    """候选列表：逗号分隔、词内逗号转义 \\,"""
    if not field:
        return []
    parts, cur, i = [], [], 0
    while i < len(field):
        c = field[i]
        if c == "\\" and i + 1 < len(field):
            cur.append(c)
            cur.append(field[i + 1])
            i += 2
            continue
        if c == ",":
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [unescape(p) for p in parts if p != ""]


def load_ledger(path, now_tick):
    """词级聚合 eff（镜像 llm_filter.cc g_llm_user_source_entries）。"""
    forms = defaultdict(list)  # word -> [(code, cum, dee, tick)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            col = line.split("\t")
            if len(col) < 5:
                continue
            w, c, dee, tick, cum = col[0], col[1], float(col[2]), int(col[3]), int(col[4])
            if c:
                forms[w].append((c, cum, dee, tick))
    eff_of = {}
    for w, v in forms.items():
        eff = 0.0
        for _c, _cum, dee, tick in v:
            if dee > 0:
                eff += dee * math.exp((tick - now_tick) / TAU)
        if eff >= 0.5:
            eff_of[w] = eff
    return eff_of


def load_corpus(path):
    w_rows, c_by_ref, now_tick = [], defaultdict(list), 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue
            col = line.split("\t")
            if col[0] == "W" and len(col) >= 13:
                w = {
                    "id": col[1], "code": col[4], "engine_top": split_list(col[6]),
                    "window": split_list(col[7]), "final": split_list(col[8]),
                    "tick": int(col[9]), "status": col[10], "skip": col[11],
                    "flags": col[13] if len(col) > 13 else "",
                    "ts": col[2],
                }
                w_rows.append(w)
                now_tick = max(now_tick, w["tick"])
            elif col[0] == "C" and len(col) >= 11:
                c_by_ref[col[3]].append({
                    "word": unescape(col[4]), "code": col[5],
                    "rank_engine": int(col[6]), "rank_window": int(col[7]),
                    "rank_final": int(col[8]), "tick": int(col[9]),
                    "path": unescape(col[11]) if len(col) > 11 else "",
                })
            elif col[0] == "H":
                # H id ts edition schema model code_pattern max_candidates max_ctx tcs beta tau
                if len(col) >= 11:
                    try:
                        globals()["LAST_BETA"] = float(col[10])
                    except ValueError:
                        pass
    return w_rows, c_by_ref, now_tick


def load_debug_d(paths):
    """D 行按 W-id 分组：[(final_rank, eff, fb, u1, key, ce, q)]。
    q = A5 可信度惩罚（2026-11-27 后的 D 行才有；旧行 = 0）。"""
    d = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "|D|" not in line:
                    continue
                line = line.rstrip("\n")
                head, _, rest = line.partition("|D|")
                wid = head.split("|")[0]
                kv = {}
                for item in rest.split("|")[1:]:
                    k, _, v = item.partition("=")
                    kv[k] = v
                try:
                    rank = int(rest.split("|")[0])
                    eff = float(kv.get("eff", "0"))
                    fb = float(kv.get("fb", "0"))
                    u1 = float(kv.get("u1", "0"))
                    key = float(kv.get("key", "0"))
                    ce = kv.get("ce", "-inf")
                    q = float(kv.get("q", "0"))
                except ValueError:
                    continue
                d[wid].append((rank, eff, fb, u1, key, ce, q))
    return d


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def fmt_pct(n, d):
    return "%d/%d (%.2f%%)" % (n, d, pct(n, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="luna_pinyin")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--debug", nargs="*", default=None)
    ap.add_argument("--beta", default="10,12,14,16,18")
    ap.add_argument("--alpha", default="0,0.5,1.0")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = os.path.join(os.environ.get("APPDATA", ""), "Rime", "llm_rerank", args.schema)
    corpus = args.corpus or os.path.join(base, "rime_llm_corpus.tsv")
    ledger = args.ledger or os.path.join(base, "user_ledger.tsv")

    w_rows, c_by_ref, now_tick = load_corpus(corpus)
    eff_of = load_ledger(ledger, now_tick)
    beta0 = globals().get("LAST_BETA", 14.0)

    ok = [w for w in w_rows if w["status"] == "ok"]
    ok_by_id = {w["id"]: w for w in ok}
    print("== 数据 ==")
    print("corpus=%s" % corpus)
    print("W=%d ok=%d C(ref)=%d ledger词=%d β=%.2f" %
          (len(w_rows), len(ok), len(c_by_ref), len(eff_of), beta0))

    # ---- A0: 全 ok 窗的账本词覆盖率（不要求有提交）----
    cov = sum(1 for w in ok if any(x in eff_of for x in w["window"]))
    cov_fin1 = sum(1 for w in ok if w["final"] and w["final"][0] in eff_of)
    print("\n== A0 全 ok 窗覆盖 ==")
    print("账本词在窗内 : %s" % fmt_pct(cov, len(ok)))
    print("final#1=账本词: %s" % fmt_pct(cov_fin1, len(ok)))

    # ---- A/B: 覆盖 + placement vs selection ----
    n_led_win = 0            # 窗内含账本词的 ok 窗
    n_eng1_led = 0           # engine#1 是账本词
    n_fin1_led = 0           # final#1 是账本词
    n_fin1_led_commit = 0    # 有对应提交的
    n_fin1_led_sel = 0       # 且 final#1 账本词被选中
    n_fin1_led_sel_any = 0   # 选中了任一账本词
    n_squeeze = 0            # 挤出：final#1 账本词、真值词排其后且非账本#1
    squeeze_ex = []
    n_commit = 0             # 有提交的账本词窗
    sel_examples = []

    for w in ok:
        commits = c_by_ref.get(w["id"])
        if not commits:
            continue
        led_in_win = [x for x in w["window"] if x in eff_of]
        if not led_in_win:
            continue
        n_led_win += 1
        sel_words = {c["word"] for c in commits}
        eng1 = w["engine_top"][0] if w["engine_top"] else ""
        fin1 = w["final"][0] if w["final"] else ""
        n_commit += 1
        if eng1 in eff_of:
            n_eng1_led += 1
        if fin1 in eff_of:
            n_fin1_led += 1
            if fin1 in sel_words:
                n_fin1_led_sel += 1
            if sel_words & set(led_in_win):
                n_fin1_led_sel_any += 1
            if fin1 not in sel_words:
                # 真值词的 final 位次
                pos = next((i + 1 for i, x in enumerate(w["final"]) if x in sel_words), None)
                if pos:
                    n_squeeze += 1
                    if len(squeeze_ex) < 12:
                        sel_eff = {s: round(eff_of[s], 2) if s in eff_of else None
                                   for s in sel_words}
                        squeeze_ex.append((w["code"], fin1, round(eff_of[fin1], 2),
                                           sel_words, pos, sel_eff, w["ts"]))
        if sel_words & set(led_in_win):
            n_fin1_led_sel_any2 = sel_words & set(led_in_win)
            sel_examples.append((w["code"], sorted(n_fin1_led_sel_any2), fin1))

    print("\n== A/B 覆盖与到位率（ok 窗 × 有提交）==")
    print("账本词进窗       : %s" % fmt_pct(n_led_win, n_commit))
    print("engine#1=账本词  : %s" % fmt_pct(n_eng1_led, n_commit))
    print("final#1 =账本词  : %s" % fmt_pct(n_fin1_led, n_commit))
    if n_fin1_led:
        print("  其中该词真被选中   : %s" % fmt_pct(n_fin1_led_sel, n_fin1_led))
        print("  其中选中任一账本词 : %s" % fmt_pct(n_fin1_led_sel_any, n_fin1_led))
        print("  挤出（真值词排其后）: %s" % fmt_pct(n_squeeze, n_fin1_led))
        for ex in squeeze_ex:
            print("    code=%s final#1=%s(eff=%.1f) 真值=%s@%d eff=%s ts=%s" %
                  (ex[0], ex[1], ex[2], sorted(ex[3]), ex[4], ex[5], ex[6]))
    # 反向证据：选中词是账本词、但 final#1 不是它（层1/账本源没把它顶上去）
    n_sel_led = 0
    n_sel_led_not1 = 0
    for w in ok:
        commits = c_by_ref.get(w["id"])
        if not commits:
            continue
        sel_words = {c["word"] for c in commits} & set(w["window"])
        sel_led = {x for x in sel_words if x in eff_of}
        if sel_led:
            n_sel_led += 1
            if not (w["final"] and w["final"][0] in sel_led):
                n_sel_led_not1 += 1
    print("选中=账本词的窗  : %d（final#1 未给账本词 %d）" % (n_sel_led, n_sel_led_not1))

    # ---- C: 重放扫描 ----
    dbg_paths = args.debug
    if dbg_paths is None:
        cand = glob.glob(os.path.join(base, "rime_llm_debug.txt*"))
        dbg_paths = [p for p in cand if not p.endswith(".diag")]
    d_rows = load_debug_d(dbg_paths) if dbg_paths else {}
    betas = [float(x) for x in args.beta.split(",")]
    alphas = [float(x) for x in args.alpha.split(",")]

    print("\n== C 重放（D 行窗；窗口组成不重放）==")
    n_win = 0
    rows = []
    for wid, ds in d_rows.items():
        w = ok_by_id.get(wid)
        if not w or not ds:
            continue
        m = len(w["final"])
        if m == 0:
            continue
        ds = sorted(x for x in ds if x[0] <= m)
        if [r for r, *_ in ds] != list(range(1, m + 1)):
            continue
        commits = c_by_ref.get(wid)
        if not commits:
            continue
        sel = {c["word"] for c in commits}
        words = w["final"][:len(ds)]
        fb = [d[2] for d in ds]
        u1 = [d[3] for d in ds]
        key = [d[4] for d in ds]
        pen = [d[6] for d in ds]
        led = [x in eff_of for x in words]
        n_win += 1
        rows.append((words, fb, u1, key, led, sel, pen))

    print("可用窗=%d（debug D 行 + final 对齐 + 有提交）" % n_win)

    def replay(scale_ledger=1.0, beta=beta0, delta=None):
        """重放融合序。delta = A5 惩罚重放系数（q 行记录在 δ=1 下，
        fused'(δ') = key − q + δ'·q）；None = 维持记录值（δ=1）。"""
        r0 = 0          # 真值词 #1
        led1 = 0        # final#1 账本词
        displ = 0       # 相对基线 final#1 改变
        changed_ex = []
        n_pen = 0
        for words, fb, u1, key, led, sel, pen in rows:
            f = []
            for i in range(len(words)):
                fb_i = fb[i] * (scale_ledger if led[i] else 1.0)
                u1_i = u1[i] * (scale_ledger if led[i] else 1.0)
                fb_new = beta / beta0 * fb_i
                v = key[i] - fb[i] - u1[i] + fb_new + u1_i
                if delta is not None and pen[i] != 0.0:
                    v = v - pen[i] + delta * pen[i]
                    n_pen += 1
                f.append(v)
            order = sorted(range(len(f)), key=lambda i: -f[i])
            top = words[order[0]]
            if top in sel:
                r0 += 1
            if led[order[0]]:
                led1 += 1
            base_top = words[0]
            if top != base_top:
                displ += 1
                if len(changed_ex) < 6:
                    changed_ex.append((base_top, top))
        return r0, led1, displ, changed_ex, n_pen

    print("\nβ 扫描（α=1 即现状账本权重）: 真#1率 / final#1账本率 / 相对基线翻转")
    for b in betas:
        r0, led1, displ, _x, _np = replay(1.0, b)
        print("  β=%-4.1f : %s / %s / %d" %
              (b, fmt_pct(r0, n_win), fmt_pct(led1, n_win), displ))
    print("\n账本词层1 缩放 α 扫描（β=基线）: 真#1率 / final#1账本率 / 翻转")
    for a in alphas:
        r0, led1, displ, ex, _np = replay(a, beta0)
        print("  α=%-4.2f : %s / %s / %d" %
              (a, fmt_pct(r0, n_win), fmt_pct(led1, n_win), displ))
        if a != 1.0 and ex:
            for e in ex[:4]:
                print("      %s -> %s" % e)

    # A5 δ 重放（q 字段，2026-11-27 后 D 行才有）：q 记录于 δ=1
    n_q = sum(1 for _w, _fb, _u1, _k, _led, _sel, pen in rows
              if any(x != 0.0 for x in pen))
    if n_q:
        print("\nA5 δ 重放（含惩罚候选的窗 %d/%d）:" % (n_q, n_win))
        for dv in (0.0, 0.25, 0.5, 1.0, 2.0):
            r0, led1, displ, ex, npn = replay(delta=dv)
            print("  δ=%-4.2f : %s / %s / 翻转%d（惩罚项 %d 个）" %
                  (dv, fmt_pct(r0, n_win), fmt_pct(led1, n_win), displ, npn))
            if ex:
                for e in ex[:3]:
                    print("      %s -> %s" % e)
    else:
        print("\nA5 δ 重放：无 q≠0 候选（数据早于 2026-11-27 或无纠错/补全窗）")

    if args.json:
        out = {
            "n_windows": n_commit, "ledger_in_window": n_led_win,
            "eng1_led": n_eng1_led, "fin1_led": n_fin1_led,
            "fin1_led_selected": n_fin1_led_sel,
            "squeeze": n_squeeze, "replay_windows": n_win,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nJSON -> %s" % args.json)


if __name__ == "__main__":
    sys.exit(main())
