"""前缀确认的整句链式仿真：一句内可能确认多次，记录序位与上下文。

口径（与生产 2026-11-24 定案一致）：
- 句源 = luna 混合窗 dyn 子集（前 40 wiki + 40 user；CE / 组句已备料）。
- 组句 = gen20_dyn.tsv（top-20；与生产同一生成器/参数）。
- 窗口 = 生产现行「只排除显示置顶的组句第 1 条」；冷账本 → 层1 = 引擎序。
- 手动收尾 = 理想选词模型：用户按"下一个词"选，一次推进一个词（或当前词
  的剩余部分），**不**使用组句整句候选——否则链一步到底，测不到多次确认。
- 错确认按 (m, lead) 去重：撤销后不再信任同一自动确认；重复触发数单独记。
- 生产门 = P(lead)≥0.95（top-20 分组 softmax）∧ 窗口融合序第 1 ∧ 同文本
  同 span ∧ 剩余≥2 音节；对照规则另加 ngram 句分差 d / 窗口融合 margin wm。

跑法（研究库根目录）：python eval/calib/prefix_confirm_chain.py
产物：`python eval/calib/prefix_confirm_chain.py > eval/prefix_confirm_chain.txt`。
"""
import json
import math
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L
import prefix_confirm_2x2 as P

PGATE = 0.95
PAGE = 5
MIN_REMAIN = 2


class Store2(P.Store):
    @staticmethod
    def build(w, pool, g, ce, full):
        n = len(w["code"])
        texts = [c[0] for c in pool]
        ends = [c[2] for c in pool]
        gshow = g[0]["text"] if g else None
        win = [i for i, t in enumerate(texts) if t != gshow][:L.K_WINDOW]
        tail = [i for i, t in enumerate(texts) if t != gshow][L.K_WINDOW:]
        good = [w["text"].startswith(t) for t in texts]
        score, ce1 = {}, {}
        for i in win:
            c3 = ce.get(texts[i])
            if c3 is None:
                continue
            # 缺失 token 不参与加权（生产 Step2/3 只给真有该 token 的候选加
            # 分；缓存里的 -1.0 是占位哨兵）。2026-11-25 修正——此前把哨兵
            # 当数值加进和里，每缺一项多减 1.1/0.6，与生产不符。
            nt = c3[1] if c3[1] else sum(
                1 for x in c3[0] if x is not None and x > -1e8)
            s = 0.0
            for k in range(min(int(nt), 3)):
                x = c3[0][k] if k < len(c3[0]) else None
                if x is not None and x > -1e8:
                    s += P.KCE[k] * x
            score[i] = -s
            if c3[0] and c3[0][0] is not None and c3[0][0] > -1e8:
                ce1[i] = c3[0][0]
        order0 = sorted(win, key=lambda i: -(score[i] if i in score else -1e10))
        fused, order = {}, []
        if len(order0) > 1:
            min_valid = min(score.values()) if score else 0.0
            for i in order0:
                fused[i] = score.get(i, min_valid)
            order = sorted(order0, key=lambda i: -fused[i])
        st = dict(w=w, n=n, texts=texts, ends=ends, good=good, window=win,
                  tail=tail, fused=fused, order=order, gen=g, ce1=ce1,
                  gen_scores={t: -v[0] for t, v in full.items() if v[0] < 1e9})
        st["A"] = P.path_a(st)
        st["B_full"] = P.path_b(st, use_scores=True)
        st["B_pre"] = P.path_b(st, use_scores=False)
        return st


def load_gen20(path):
    out = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) != 7:
            continue
        try:
            out[f[0]].append(dict(score=float(f[2]), lead=f[3],
                                  lead_end=int(f[4]), text=f[6]))
        except ValueError:
            pass
    for v in out.values():
        v.sort(key=lambda x: -x["score"])
    return out


def lead_stats(gs):
    """top-20 → (lead, p, le, text_of_lead, dscore)。"""
    mx = max(x["score"] for x in gs)
    mass, le_of, txt_of = defaultdict(float), {}, {}
    tot = 0.0
    for x in gs:
        wt = math.exp(x["score"] - mx)
        tot += wt
        mass[x["lead"]] += wt
        le_of.setdefault(x["lead"], x["lead_end"])
        txt_of.setdefault(x["lead"], x["text"])
    lead, best = None, -1.0
    for t, m in mass.items():
        if m > best or (m == best and lead is not None and t < lead):
            lead, best = t, m
    top = gs[0]
    other = next((x for x in gs[1:] if x["lead"] != top["lead"]), None)
    d = (top["score"] - other["score"]) if other else 30.0
    return lead, best / tot if tot else 0.0, le_of[lead], txt_of[lead], d


def gate(s, gs, ctx_len, policy):
    """→ (是否确认, 事件 dict) 或 (False, None)。"""
    if not gs:
        return False, None
    lead, p, le, sent, d = lead_stats(gs)
    sl, syl = len(lead), len(sent)
    n = s["n"]
    rank = -1
    for k, i in enumerate(s["order"]):
        if s["texts"][i] == lead:
            rank = k
            break
    pool_has = any(t == lead and e == le
                   for t, e in zip(s["texts"], s["ends"]))
    wm = 0.0
    if len(s["order"]) >= 2:
        a, b = s["fused"].get(s["order"][0]), s["fused"].get(s["order"][1])
        if a is not None and b is not None:
            wm = a - b
    base = (rank == 0 and pool_has and le < n and syl - sl >= MIN_REMAIN)
    if not base:
        return False, None
    ce1 = s.get("ce1", {}).get(s["order"][0]) if s["order"] else None
    fire = False
    if policy == "cur":
        fire = p >= PGATE
    elif policy == "union":
        fire = p >= PGATE and (wm >= 2 or d >= 10)
    elif policy == "strict":
        fire = p >= PGATE and wm >= 2 and d >= 10
    elif policy == "p99":
        fire = p >= 0.99
    elif policy == "p99plus":
        fire = p >= 0.99 or (p >= PGATE and wm >= 5 and d >= 10)
    elif policy == "ctx":
        fire = p >= PGATE and (ctx_len >= 15 or (wm >= 2 and d >= 5))
    elif policy == "d10":
        fire = p >= PGATE and d >= 10
    elif policy == "ctx2":
        fire = (p >= PGATE and ctx_len >= 2
                and (d >= 10 or wm >= 2))
    elif policy == "ce1_11":
        fire = ce1 is not None and ce1 <= 11
    elif policy == "ce1_10":
        fire = ce1 is not None and ce1 <= 10
    elif policy == "ce1_d10":
        fire = ce1 is not None and ce1 <= 11 and d >= 10
    elif policy == "ce1_union":
        fire = ce1 is not None and ce1 <= 11 and (wm >= 2 or d >= 10)
    elif policy == "lin3":
        sc = (math.log(max(p, 1e-6) / max(1 - p, 1e-6))
              + 0.10 * min(wm, 20) + 0.06 * min(d, 20))
        fire = sc >= 3.6
    if not fire:
        return False, None
    return True, dict(lead=lead, p=p, d=d, wm=wm, syl=syl, sl=sl, le=le,
                      m=s["w"]["m"], ctx_len=ctx_len, rank=rank)


def simulate(st, sent, policy, events, missing):
    text = sent["text"]
    g = st.grid[sent["sid"]]
    r = dict(cc=0, wc=0, picks=0, flips=0, fail=0, done=True, ord=0,
             repeat=0)
    m, typed = 0, 2
    blocked = set()          # 同一 (m, lead) 错确认后不再重复触发（用户撤销后
                             # 不会再信任同一个自动确认；重复触发数单独统计）
    for _ in range(800):
        if m >= len(text):
            break
        full_len = len("".join(sent["syls"][m:]))
        wid = g.get(f"{m}:{typed}")
        if wid is None:
            r["done"] = False
            break
        s = st.state(wid, missing)
        if s is None:
            return None
        gs = st.g20.get(wid, [])
        if typed >= 4:
            fire, ev = gate(s, gs, len(s["w"]["ctx"]), policy)
            if fire:
                key = (m, ev["lead"])
                if key in blocked:
                    r["repeat"] += 1
                    fire = False
            if fire:
                r["ord"] += 1
                ok = text[m:].startswith(ev["lead"])
                events.append(dict(sid=sent["sid"], domain=sent["domain"],
                                   policy=policy, ord=r["ord"], ok=ok,
                                   truth=text[m:], **ev))
                if ok:
                    m += len(ev["lead"])
                    typed = max(2, typed - ev["le"])
                    r["cc"] += 1
                    continue
                r["wc"] += 1
                blocked.add((m, ev["lead"]))
        if typed >= full_len:
            # 手动收尾（理想选词模型）：用户按"下一个词"选，一次推进一个
            # 词（或当前词的剩余部分）；不使用组句整句候选。这样链会走遍
            # 每个词边界，能测到多次确认；各策略的手动基线相同。
            disp = [(s["texts"][i], s["ends"][i]) for i in s["order"]]
            disp += [(s["texts"][i], s["ends"][i]) for i in s["tail"]]
            cum, nw = 0, None
            for wd in sent["segs"]:
                if cum == m:
                    nw = wd
                    break
                if cum < m < cum + len(wd):
                    nw = wd[m - cum:]
                    break
                cum += len(wd)
            if nw is None:
                nw = text[m:]
            rank = None
            for p_, (t, e) in enumerate(disp):
                if t == nw:
                    rank = p_
                    break
            r["picks"] += 1
            if rank is not None:
                r["flips"] += rank // PAGE
            m += len(nw)
            typed = 2 if m < len(text) else 0
            continue
        typed += 1
    return r


def main():
    gen20 = load_gen20(L.OUT / "gen20_dyn.tsv")
    st = Store2()
    st.g20 = gen20
    sents = [s for s in L.jsonl_iter(L.SENTENCES) if s["dyn"]]
    keep, seen = [], Counter()
    for s in sents:
        if seen[s["domain"]] < 40:
            seen[s["domain"]] += 1
            keep.append(s)
    policies = ["none", "cur", "union", "strict", "p99", "p99plus",
                "ctx", "lin3", "d10", "ctx2", "ce1_11", "ce1_10",
                "ce1_d10", "ce1_union"]
    events = []
    res = {}
    for p in policies:
        miss = {"ce": [], "gen": [], "full": []}
        res[p] = [simulate(st, s, p, events, miss) for s in keep]
        bad = sum(1 for x in res[p] if x is None)
        if bad:
            print(f"{p}: {bad} 句缺数据")
    name = {"none": "不确认", "cur": "现行 P≥.95", "union": "P≥.95∧(wm≥2∨d≥10)",
            "strict": "P≥.95∧wm≥2∧d≥10", "p99": "P≥.99",
            "p99plus": "P≥.99∨(P≥.95∧wm≥5∧d≥10)", "ctx": "上下文自适应",
            "lin3": "线性分≥3.6", "d10": "P≥.95∧d≥10",
            "ctx2": "ctx≥2∧(d≥10∨wm≥2)",
            "ce1_11": "ce1≤11", "ce1_10": "ce1≤10",
            "ce1_d10": "ce1≤11∧d≥10",
            "ce1_union": "ce1≤11∧(wm≥2∨d≥10)"}
    print(f"\n链式仿真：{len(keep)} 句（wiki {seen['wiki']} / user {seen['user']}）")
    print(f"{'策略':<26}{'错确认句':>8}{'错确认':>7}{'正确':>6}{'手动收尾':>8}"
          f"{'翻页':>5}{'失败字':>6}{'总附加键':>8}")
    for p in policies:
        rs = res[p]
        n = len(rs)
        wc = sum(x["wc"] for x in rs)
        cc = sum(x["cc"] for x in rs)
        extra = wc * 5 + sum(x["picks"] + x["flips"] + x["fail"] for x in rs)
        print(f"{name[p]:<26}{100*sum(1 for x in rs if x['wc'])/n:>7.1f}%"
              f"{wc:>7}{cc:>6}{sum(x['picks'] for x in rs):>8}"
              f"{sum(x['flips'] for x in rs):>5}{sum(x['fail'] for x in rs):>6}"
              f"{extra:>8}")

    print("\n确认事件按序位（同句内第几次确认）")
    print(f"{'策略':<26}{'序位':>5}{'事件':>6}{'对':>5}{'错':>5}{'精度':>8}")
    for p in policies:
        if p == "none":
            continue
        evs = [e for e in events if e["policy"] == p]
        for o in (1, 2, 3, 4):
            fs = [e for e in evs if (e["ord"] if o < 4 else 4) == (o if o < 4 else e["ord"])]
            if o == 4:
                fs = [e for e in evs if e["ord"] >= 4]
            if not fs:
                continue
            ok = sum(1 for e in fs if e["ok"])
            print(f"{name[p]:<26}{o if o < 4 else '4+':>5}{len(fs):>6}{ok:>5}"
                  f"{len(fs)-ok:>5}{100*ok/len(fs):>7.2f}%")

    print("\n现行规则事件按上下文")
    evs = [e for e in events if e["policy"] == "cur"]
    for nm, fn in (("m=0", lambda e: e["m"] == 0), ("m=1..2", lambda e: 1 <= e["m"] <= 2),
                   ("m>=3", lambda e: e["m"] >= 3),
                   ("ctx<15", lambda e: e["ctx_len"] < 15),
                   ("ctx>=15", lambda e: e["ctx_len"] >= 15),
                   ("wiki", lambda e: e["domain"] == "wiki"),
                   ("user", lambda e: e["domain"] == "user")):
        fs = [e for e in evs if fn(e)]
        if fs:
            ok = sum(1 for e in fs if e["ok"])
            print(f"  {nm:<10}{len(fs):>5} 对 {ok:>4} 错 {len(fs)-ok:>3} "
                  f"精度 {100*ok/len(fs):>6.2f}%")
    print("\n现行规则按句子确认次数分布")
    cnt = Counter()
    for s in keep:
        rs = res["cur"][keep.index(s)]
        cnt[rs["cc"] + rs["wc"]] += 1
    print("  " + "  ".join(f"{k}次:{v}句" for k, v in sorted(cnt.items())))


if __name__ == "__main__":
    main()
