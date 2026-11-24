#!/usr/bin/env python3
"""前缀确认守卫从「码数」改「音节数」的标定（2026-11-24，用户指令）。

背景：旧守卫「整段 ≥6 码、确认后剩余 ≥4 码」的 6/4 是拍的。码数不是通用
单位——全拼 1 音节 2–6 码、简拼 1 码、双拼固定 2 码/音节，同一串混拼后
码数与音节数无固定关系。守卫应改用**音节数**（= 字数；中文一字一音节）。

音节数来源（本标定用）= 组句结果：组句器按码串格逐音节拼词，一条满覆盖
句子的**字符数 = 该句覆盖的音节数**，lead 的字符数 = 已覆盖音节数，
剩余 = 总音节 − lead 音节。（生产实现同源：两版都用组句句子的字数。）

口径：luna 混合窗静态 cohort；gate = top-20 lead 概率 P≥0.95 ∧ 该文本是
窗口融合序第 1 ∧ 引擎池里有同文本同 span 候选（与生产合并门一致）。
"正确" = 窗提交词以该 lead 文本开头。错确认代价 K 键（正确确认省 1 键）。
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "eval/calib")
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L  # noqa: E402
import prefix_confirm_2x2 as P  # noqa: E402

PGATE = 0.95


def gen20_by_wid():
    out = defaultdict(list)
    for line in open(L.OUT / "gen20.tsv", encoding="utf-8"):
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


def nchar(s):
    return len(s or "")


def main():
    g20 = gen20_by_wid()
    st = P.Store()
    miss = {"ce": [], "gen": [], "full": []}
    rows = []
    for w in L.jsonl_iter(L.WINDOWS):
        if w["dyn"]:
            continue
        wid, code = w["wid"], w["code"]
        gs = g20.get(wid)
        if not gs or len(code) < 4:
            continue
        s = st.state(wid, miss)
        if s is None:
            continue
        # ── ngram 侧：top-20 按 lead 分组 softmax ──
        mx = max(x["score"] for x in gs)
        mass, le_of, tot = defaultdict(float), {}, 0.0
        for x in gs:
            wt = 2.718281828459045 ** (x["score"] - mx)
            tot += wt
            mass[x["lead"]] += wt
            le_of.setdefault(x["lead"], x["lead_end"])
        lead, best = None, -1.0
        for t, m in mass.items():          # 同质量取文本序（与生产同）
            if m > best or (m == best and lead is not None and t < lead):
                lead, best = t, m
        p = best / tot if tot > 0 else 0.0
        if p < PGATE:
            continue
        le = le_of[lead]
        n = len(code)
        # 句子/前导词字数 = 音节数（取该 lead 的最高分句）
        sent = next((x for x in gs if x["lead"] == lead), None)
        syll_total = nchar(sent["text"]) if sent else None
        syll_lead = nchar(lead)
        syll_rem = None if syll_total is None else syll_total - syll_lead
        # ── 窗口侧：同文本同 span 候选且是窗口融合序第 1 ──
        pool_has = any(t == lead and e == le
                       for t, e in zip(s["texts"], s["ends"]))
        rank = -1
        for k, i in enumerate(s["order"]):
            if s["texts"][i] == lead:
                rank = k
                break
        trig = pool_has and le < n and rank == 0
        # 召回分母：池内有"正确的前缀候选"（任意位次、严格前缀）
        possible = any(s["good"][i] and s["ends"][i] < n
                       and s["texts"][i] not in {x["text"] for x in s["gen"]}
                       for i in range(len(s["texts"])))
        rows.append(dict(
            wid=wid, trig=trig, ok=(trig and w["text"].startswith(lead)),
            lead=lead, p=p, n_code=n, rem_code=n - le,
            syll=syll_total, srem=syll_rem, slead=syll_lead,
            possible=possible))
    n_trig = sum(1 for r in rows if r["trig"])
    n_ok = sum(1 for r in rows if r["trig"] and r["ok"])
    npos = sum(1 for r in rows if r["possible"])
    print(f"静态窗 {len(rows)}；合并门（无守卫）触发 {n_trig}（{100*n_trig/len(rows):.2f}%），"
          f"精度 {100*n_ok/max(n_trig,1):.2f}%；召回分母 {npos}")
    print(f"旧码守卫（整段≥6 码 ∧ 剩余≥4 码）单独看："
          f"触发 {sum(1 for r in rows if r['trig'] and r['n_code'] >= 6 and r['rem_code'] >= 4)}")

    def table(name, keyf, buckets):
        print(f"\n{name}")
        print(f"{'档':<10}{'触发':>7}{'对':>6}{'错':>6}{'精度':>9}{'K=2':>7}{'K=5':>7}{'K=12':>7}")
        for b in buckets:
            fs = [r for r in rows if r["trig"] and keyf(r) == b]
            if not fs:
                continue
            ok = sum(1 for r in fs if r["ok"])
            bad = len(fs) - ok
            print(f"{b:<10}{len(fs):>7}{ok:>6}{bad:>6}{100*ok/len(fs):>8.2f}%"
                  f"{ok-2*bad:>7}{ok-5*bad:>7}{ok-12*bad:>7}")

    table("① 按总音节数（合并门触发内）", lambda r: r["syll"],
          [1, 2, 3, 4, 5, 6, 7, 8])
    table("② 按剩余音节数（合并门触发内）", lambda r: r["srem"],
          [1, 2, 3, 4, 5, 6])
    table("③ 按总码数（对照：旧单位）", lambda r: r["n_code"],
          [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    table("④ 按剩余码数（对照：旧单位）", lambda r: r["rem_code"],
          [4, 5, 6, 7, 8, 9, 10])

    print("\n⑤ 阈值网格（总音节 ≥ T ∧ 剩余 ≥ R）：触发 / 精度 / 召回 / K=2 / K=5 / K=12")
    print(f"{'T':>3}{'R':>3}{'触发':>7}{'精度':>9}{'召回':>9}{'K=2':>8}{'K=5':>8}{'K=12':>8}")
    for t in (2, 3, 4, 5):
        for r_ in (1, 2, 3):
            fs = [x for x in rows if x["trig"] and x["syll"] is not None
                  and x["syll"] >= t and x["srem"] >= r_]
            if not fs:
                continue
            ok = sum(1 for x in fs if x["ok"])
            bad = len(fs) - ok
            print(f"{t:>3}{r_:>3}{len(fs):>7}{100*ok/len(fs):>8.2f}%"
                  f"{100*ok/max(npos,1):>8.2f}%{ok-2*bad:>8}{ok-5*bad:>8}{ok-12*bad:>8}")

    print("\n⑤b 交叉：lead 音节 × 剩余音节（合并门触发内）——精度 / K=5")
    print(f"{'lead\\余':<8}" + "".join(f"{r_:>14}" for r_ in (1, 2, 3, 4)))
    for ls in (1, 2, 3):
        cells = []
        for r_ in (1, 2, 3, 4):
            fs = [x for x in rows if x["trig"] and x["slead"] == ls
                  and x["srem"] == r_]
            if not fs:
                cells.append(f"{'-':>14}")
                continue
            ok = sum(1 for x in fs if x["ok"])
            bad = len(fs) - ok
            cells.append(f"{len(fs):>5}/{100*ok/len(fs):>5.1f}%{ok-5*bad:>5}"
                         if len(fs) >= 5 else
                         f"{len(fs):>5}/{100*ok/len(fs):>5.1f}%{'--':>5}")
        print(f"{ls:<8}" + "".join(cells))

    print("\n⑤c 交叉：总音节 × 剩余音节（合并门触发内）——触发/精度")
    print(f"{'总\\余':<8}" + "".join(f"{r_:>14}" for r_ in (1, 2, 3, 4)))
    for t in (2, 3, 4, 5, 6):
        cells = []
        for r_ in (1, 2, 3, 4):
            fs = [x for x in rows if x["trig"] and x["syll"] == t
                  and x["srem"] == r_]
            if not fs:
                cells.append(f"{'-':>14}")
                continue
            ok = sum(1 for x in fs if x["ok"])
            cells.append(f"{len(fs):>5}/{100*ok/len(fs):>5.1f}%")
        print(f"{t:<8}" + "".join(cells))

    print("\n⑥ 候选守卫规则对照（同一合并门内）：触发 / 精度 / 召回 / K=2 / K=5 / K=12")
    def guard(r, t, rr):
        return r["syll"] is not None and r["syll"] >= t and r["srem"] >= rr
    rules = (
        ("旧：≥6 码 ∧ 剩余≥4 码",
         lambda r: r["n_code"] >= 6 and r["rem_code"] >= 4),
        ("无实质守卫（总≥2 ∧ 余≥1）", lambda r: guard(r, 2, 1)),
        ("总≥2 ∧ 余≥2", lambda r: guard(r, 2, 2)),
        ("总≥3 ∧ 余≥2", lambda r: guard(r, 3, 2)),
        ("余≥2，单字 lead 例外（余≥1）",
         lambda r: guard(r, 2, 2) or (r["slead"] == 1 and guard(r, 2, 1))),
        ("总≥2 ∧ 余≥3", lambda r: guard(r, 2, 3)),
    )
    print(f"{'规则':<28}{'触发':>7}{'精度':>9}{'召回':>9}{'K=2':>8}{'K=5':>8}{'K=12':>8}")
    for name, sel in rules:
        fs = [x for x in rows if x["trig"] and sel(x)]
        ok = sum(1 for x in fs if x["ok"])
        bad = len(fs) - ok
        print(f"{name:<28}{len(fs):>7}{100*ok/max(len(fs),1):>8.2f}%"
              f"{100*ok/max(npos,1):>8.2f}%{ok-2*bad:>8}{ok-5*bad:>8}{ok-12*bad:>8}")

    # 被音节守卫挤掉/新增的样本归属
    print("\n⑦ 变化来源（合并门触发内）")
    old = [r for r in rows if r["trig"] and r["n_code"] >= 6 and r["rem_code"] >= 4]
    new = [r for r in rows if r["trig"] and guard(r, 3, 1)]
    only_old = [r for r in old if not guard(r, 3, 1)]
    only_new = [r for r in new if not (r["n_code"] >= 6 and r["rem_code"] >= 4)]
    for name, fs in (("仅旧守卫（音节不足）", only_old),
                     ("仅新守卫（码数不足）", only_new)):
        ok = sum(1 for r in fs if r["ok"])
        print(f"{name:<22}{len(fs):>6}  对 {ok} 错 {len(fs)-ok} "
              f"精度 {100*ok/max(len(fs),1):.1f}%")
        for r in fs[:8]:
            print(f"    {r['wid']} lead={r['lead']} 码={r['n_code']}/余{r['rem_code']} "
                  f"音节={r['syll']}/余{r['srem']} {'对' if r['ok'] else '错'}")


if __name__ == "__main__":
    main()
