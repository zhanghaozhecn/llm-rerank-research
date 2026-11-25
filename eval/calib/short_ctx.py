#!/usr/bin/env python3
"""短上文（0–7 字）下的前缀确认精度：**截断长上文构造样本**（2026-11-25）。

问题（用户）：
  * 0 上文（无上文且首词未确认）到底要不要做前缀确认？
  * 2 字上文精度偏低（现网只有 44 个样本，不具说服力），倾向取消；
  * 1 字上文已取消，也要测一下。

做法：cohort 里带长上文的句首窗（m=0、before ≥ 20 字）→ 对每句构造
ctx = before[-n:]（n = 0…7），**重新跑组句（ngram 依赖 ctx）与 CE（依赖 ctx）**，
再按生产口径判门并统计精度。

两条判定路径（与生产一致）：
  * n = 0：空上文 → **freq_only**（无 CE，不用模型）：P(lead)≥0.5 ∧ d≥10 ∧
    窗口第 1 = lead ∧ 音节守卫。冷账本下窗口序 = 引擎原序。
  * n ≥ 1：有 CE → 现行 53 特征门（mp ≥ τ ∧ p ≥ 0.5 ∧ 非 ta 族）。
    n = 1 生产已拒绝（只测"如果放行会怎样"）。

跑法（研究库根目录）：
  python eval/calib/short_ctx.py [--n 300] [--ctx 0,1,2,3,4,5,6,7]
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import prefix_confirm_model as M  # noqa: E402
import luna_mixed_lib as L  # noqa: E402
import ss_ce_dist as S  # noqa: E402

TA = {"他", "它", "她", "他们", "它们", "她们"}
TAU = 0.973286164
WORK = L.EVAL / "short_ctx"


def load_model():
    """生产模型（Lua 平坦表）→ python 打分。"""
    import re
    src = open(L.PROJ.parent / "rime-llm-rerank" / "user" /
               "prefix_gate_model.lua", encoding="utf-8").read()
    tau = float(re.search(r"tau=([-0-9.eE+]+)", src).group(1))
    trees = []
    for mm in re.finditer(r"\{f=\{([^}]*)\},t=\{([^}]*)\},l=\{([^}]*)\},"
                          r"r=\{([^}]*)\},v=\{([^}]*)\}\}", src):
        trees.append(([int(x) for x in mm.group(1).split(",")],
                      [float(x) for x in mm.group(2).split(",")],
                      [int(x) for x in mm.group(3).split(",")],
                      [int(x) for x in mm.group(4).split(",")],
                      [float(x) for x in mm.group(5).split(",")]))
    feats = re.search(r'feats=\{([^}]*)\}', src).group(1)
    names = re.findall(r'"([^"]+)"', feats)

    def score(x):
        s = 0.0
        for f, t, l, r, v in trees:
            node = 1
            while f[node - 1] != 0:
                node = l[node - 1] if x[f[node - 1] - 1] <= t[node - 1] \
                    else r[node - 1]
            s += v[node - 1]
        return 1.0 / (1.0 + np.exp(-s))
    return names, tau, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ctx", default="0,1,2,3,4,5,6,7")
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctx.split(",")]
    WORK.mkdir(parents=True, exist_ok=True)
    names, tau, score = load_model()

    # —— 选句：m=0 的句首窗，before ≥ 20 字
    sents = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
        for x in L.jsonl_iter(d / "sentences.jsonl"):
            sents[x["sid"]] = x
    wins = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
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
    print(f"[pick] 句首窗候选 {len(picked)}")
    if len(picked) > a.n:
        picked = picked[::max(1, len(picked) // a.n)][:a.n]
    print(f"[pick] 采样 {len(picked)} 句 × ctx {ctxs}")

    # —— 组句（每个截断 ctx 都要重跑：ngram 用 ctx_tail）
    jobs = []
    for wid, w, s in picked:
        for n in ctxs:
            ctx = (s.get("before") or "")[-n:] if n else ""
            jobs.append((f"{wid}#{n}", ctx, w["code"]))
    out = WORK / "gen.tsv"
    tried = WORK / "gen_tried.txt"
    todo = [(k, c, d) for (k, c, d) in jobs if not k.startswith("#")]
    print(f"[gen] {len(todo)} 窗 →", flush=True)
    L.run_gen(todo, out=out, tried=tried)
    gen = defaultdict(list)
    for r in L.load_gen_raw(out):
        gen[r[0]].append(dict(score=r[2], lead=r[3], lead_end=r[4],
                              text=r[6]))
    for v in gen.values():
        v.sort(key=lambda x: -x["score"])
    print(f"[gen] 有组句 {len(gen)}/{len(jobs)}")

    # —— CE（ctx 变了，全部要重算）
    cands = {}
    for d in (L.OUT, L.EVAL / "prefix_big"):
        cands.update(json.load(open(d / "cands.json", encoding="utf-8")))
    ce_lines = []
    for wid, w, s in picked:
        pool = cands.get(w["code"], [])[:L.K_COLLECT]
        need = [c[0] for c in pool][:10]     # 评分窗只用前 6，池取 10 足够
        for n in ctxs:
            ctx = (s.get("before") or "")[-n:] if n else ""
            ce_lines += L.ce_lines(ctx, need)
    have = L.load_ce()
    todo = [(ln, sid) for (ln, sid) in ce_lines if sid not in have]
    print(f"[ce] 需 {len(ce_lines)} 条，缓存缺 {len(todo)} 条", flush=True)
    STEP = 8000
    for i in range(0, len(todo), STEP):
        L.run_ce(todo[i:i + STEP], workers=8)
        print(f"[ce] {min(i + STEP, len(todo))}/{len(todo)}", flush=True)
    ce = L.load_ce()

    # —— 造行 + 判定
    cfreq = M.char_freq()
    res = defaultdict(lambda: dict(n=0, fire=0, ok=0, bad=[]))
    for wid, w, s in picked:
        pool = cands.get(w["code"], [])[:L.K_COLLECT]
        if len(pool) < 2:
            continue
        for n in ctxs:
            k = f"{wid}#{n}"
            gs = gen.get(k)
            if not gs:
                continue
            ctx = (s.get("before") or "")[-n:] if n else ""
            ced = {}
            for ln, sid in L.ce_lines(ctx, [c[0] for c in pool][:10]):
                rr = ce.get(sid) or {}
                for t in ln["cands"]:
                    if t in rr and rr[t][0] is not None:
                        ced[t] = rr[t]
            lead, p, le, sent, d = M.lead_stats(gs)
            sl, syl = len(lead), len(sent)
            if not (le < len(w["code"]) and syl - sl >= 2):
                continue
            wd = dict(wid=k, sid=w["sid"], domain=w["domain"], m=0,
                      ctx=ctx, code=w["code"], text=s["text"], dyn=False)
            st = M.state_of(wd, pool, gs, ced)
            if st is None:
                continue
            if st["texts"][st["order"][0]] != lead:
                continue                      # 窗口第 1 必须是 lead
            r = M.featurize(wd, st, gs, lead, p, le, sent, d,
                            s["syls"], pool, cfreq, "short")
            e = res[n]
            e["n"] += 1                     # 样本 = 通过结构过滤的窗
            ok = r["ok"]
            if n == 0:
                # 空上文 → freq_only 分支：不用模型，要求 d ≥ 10
                if p < 0.5 or d < 10:
                    continue
            else:
                vec = [r["f"][c] if c in r["f"] else r["g"][c] for c in names]
                mp = score(np.array(vec, float))
                if not (mp >= tau and r["f"]["p"] >= 0.5
                        and lead not in TA):
                    continue
            e["fire"] += 1
            e["ok"] += 1 if ok else 0
            if not ok:
                e["bad"].append((s["text"], lead, r["truth_prefix"], w["code"]))

    print("\n[结果] 按上文长度（全部按生产口径判门）")
    print("ctx   样本   确认   正确率   错")
    for n in ctxs:
        e = res.get(n)
        if not e or not e["n"]:
            print(f" {n:<4}  —（无满足条件的窗）")
            continue
        print(f" {n:<4}{e['n']:>6}{e['fire']:>7}   "
              f"{100*e['ok']/e['fire']:>6.2f}%   {e['fire']-e['ok']:>3}")
    for n in ctxs:
        e = res.get(n)
        if e and e["bad"]:
            print(f"\n[{n} 字上文] 错例")
            for t, ld, tr, cd in e["bad"][:6]:
                print(f"  码={cd:<20} 句「{t}」 确认「{ld}」 vs 真值「{tr}」")


if __name__ == "__main__":
    main()
