#!/usr/bin/env python3
"""audit_article_session.py — 真机文章逐词审计（重排功能 + 记录功能）v2。

文章原文 ↔ 语料 W/C/M ↔ rime_llm_debug.txt ↔ user_ledger.tsv 四方对照：
  1. 会话定位   C 行贡献流重建文档文本（带标点回撤提交经下一 W ctx 验证后保词核）
                → 定位文章对应提交区间（边界须落在提交边界上）
  2. 重排触发   逐提交分类: ref 关联 / 带标点回挂前序 W(ref 断链恢复) / 短码路径 /
                纯4码无W(硬缺陷) / W skip 状态分布
  3. 记录完整性 C↔W rank_* ↔ 列表实位、final=window 置换、W.ctx ↔ 重建文档尾部逐词核对、
                tick 连续性、debug E 行逐提交在位、P/D 行在位、X 行按 scope 分流
  4. 融合算术   D 行 ce=-(ce1+1.10·ce2+0.60·ce3)、fb=14·log1p(eff)、
                key=ce+fb+u1+sdelta、final 序 = key 降序稳定排序、finrank/cerank 自洽
  5. 账本对账   user_ledger.tsv 词+码 cum/tick ↔ 语料全史提交（2026-11-20 退款口径：
                回撤提交 bump 已回滚、debug RF 行 refund=1/0 逐条核验）逐键核对
用法:
  python eval/tools/audit_article_session.py --article eval/audit_article_20261120_text.txt \
      --since "2026-11-20 17:29" --until "2026-11-20 17:36"
输出: 控制台摘要 + eval/audit_article_session_<日期>.txt 全量明细
"""
import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std as CS

RE4 = re.compile(r"^[a-z]{4}$")
is_cjk = lambda ch: "\u4e00" <= ch <= "\u9fff"
core_of = lambda w: "".join(ch for ch in w if is_cjk(ch))
rank_in = lambda lst, w: (lst.index(w) + 1) if w in lst else -1
W2, W3, BETA = 1.10, 0.60, 14.0        # 生产常数（ce2/ce3 权重、衰减词频分 β）


def pts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def parse_debug(path):
    """debug 日志 → P{wid:fields}, D{wid:{rank:fields}}, E{cid:fields}, RF{cid:int}, X[list]"""
    P, D, E, X = {}, defaultdict(dict), {}, []
    RF = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        wid, typ = parts[0], parts[1]
        try:
            if typ == "RF":
                f = {}
                for kv in parts[2:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try:
                            f[k] = float(v)
                        except ValueError:
                            f[k] = v
                RF[wid] = int(f.get("refund", 0))
            elif typ == "P":
                P[wid] = {k: float(v) for k, v in
                          (p.split("=", 1) for p in parts[2:] if "=" in p)}
            elif typ == "D":
                r = int(parts[2])
                f = {}
                for kv in parts[3:]:
                    if "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    try:
                        f[k] = float(v)
                    except ValueError:
                        f[k] = v
                D[wid][r] = f
            elif typ == "E":
                E[wid] = {k: float(v) for k, v in
                          (p.split("=", 1) for p in parts[2:] if "=" in p)}
            elif typ == "X":
                X.append(parts)
        except (ValueError, IndexError):
            X.append(["UNPARSED"] + parts)
    return P, D, E, RF, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", required=True)
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="")
    ap.add_argument("--debug-path", default=str(CS.LLM_DIR / "rime_llm_debug.txt"))
    ap.add_argument("--ledger-path", default=str(CS.LLM_DIR / "user_ledger.tsv"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    article = Path(a.article).read_text(encoding="utf-8").strip()
    events = list(CS.iter_events())

    # 提交 + 回撤标记（strict 口径，同 corpus_std.iter_commits）
    commits, stack = [], []
    for e in events:
        if e["type"] == "M":
            k = e["kind"]
            if k in CS.BOUNDARY_KINDS:
                stack = []
            elif k in CS.RETRACT_KINDS and stack:
                stack.pop()["retracted"] = True
        elif e["type"] == "C":
            d = dict(e)
            d["retracted"] = False
            d["legacy"] = "legacy" in d.get("flags", "")   # 2026-11-17 迁移重转写行
            commits.append(d)
            stack.append(d)
    wins = {e["id"]: e for e in events if e["type"] == "W"}
    wseq = [e for e in events if e["type"] == "W"]        # 时间序

    # 贡献流：正常提交=整词；整词回撤=""；带标点回撤→下一 W ctx 含词核则保核
    contrib, punct_keep = {}, []
    wptr = 0
    for c in commits:
        if not c["retracted"]:
            contrib[c["id"]] = c["word"]
            continue
        contrib[c["id"]] = ""                       # 整词回撤默认零贡献
        core = core_of(c["word"])
        if core and not c["code"].isalpha():
            while wptr < len(wseq) and wseq[wptr]["ts"] < c["ts"]:
                wptr += 1
            nxt = wseq[wptr] if wptr < len(wseq) else None
            if nxt and core in nxt["ctx"]:
                contrib[c["id"]] = core
                punct_keep.append((c["id"], c["word"], c["code"], nxt["id"]))

    cw = [c for c in commits
          if (not a.since or c["ts"] >= a.since) and (not a.until or c["ts"] <= a.until)]
    pieces = [(c, contrib[c["id"]]) for c in cw]
    text = "".join(p for _, p in pieces)
    pos = text.find(article)
    if pos < 0:
        print(f"!! 文章未在提交流中找到（窗口内 {len(cw)} 提交 / 文档 {len(text)} 字 / 文章 {len(article)} 字）")
        sys.exit(2)
    start = end = None
    acc = 0
    for i, (_, p) in enumerate(pieces):
        if start is None and acc + len(p) > pos:
            start = i
        acc += len(p)
        if start is not None and acc >= pos + len(article):
            end = i
            break
    assert start is not None and acc == pos + len(article), \
        f"文章边界未对齐提交边界: pos={pos} end_acc={acc}"
    session = [pieces[i][0] for i in range(start, end + 1)]
    sess_ids = {c["id"] for c in session}
    punct_keep = [k for k in punct_keep if k[0] in sess_ids]   # 只报会话内事件

    # 会话生效配置 = 会话首提交前最后一条 H；会话中途出现 H = 配置变更
    h_last = None
    for e in events:
        if e["type"] == "H" and e["ts"] < session[0]["ts"]:
            h_last = e
    for e in events:
        if e["type"] == "H" and session[0]["ts"] <= e["ts"] <= session[-1]["ts"]:
            print(f"!! 会话中途出现 H 行 {e['id']}（配置变更）")

    P, D, E, RF, X = parse_debug(a.debug_path)
    p_exists = bool(P)
    code_to_words, word_code = CS.load_pdsp_dict()

    defects, notes = [], []
    cls, skip_cls = Counter(), Counter()
    n_ok = n_correct = n_moved_ok = n_broken = 0
    total_ms, prep_ms = [], []
    n_ctxchk = 0
    n_nofreq = 0            # 无上文窗（flags=nofreq）：无 CE，跳过 CE 算术校验
    ref_gap = []            # (C id, word, code, linked W id) 验收链断（带标点提交 ref 空）
    retracted_sess = [c for c in session if c["retracted"]]

    # ── tick 连续性 ──（2026-11-20 起纯标点提交不推账本 tick → C 行 tick
    #    允许持平（= 前一词提交的 tick），只查不减）
    ticks = [c["tick"] for c in session]
    for i in range(1, len(ticks)):
        if ticks[i] != ticks[i - 1] + 1 and ticks[i] != ticks[i - 1]:
            defects.append((session[i]["id"], f"tick 不连续 {ticks[i-1]}→{ticks[i]}"))

    # W 按码索引（带标点提交回挂前序 W，5 秒窗）
    w_by_code = defaultdict(list)
    for w in wseq:
        w_by_code[w["code"]].append(w)
    claimed = set()

    # ── 逐提交审计 ──
    rows = []
    for c in session:
        w = wins.get(c["ref"]) if c["ref"] else None
        linked = None
        core = core_of(c["word"])
        rowcls = ""
        if c["ref"] and w is None:
            rowcls = "REF断链"
            defects.append((c["id"], f"C.ref={c['ref']} 指向不存在的 W"))
        elif w is not None:
            claimed.add(w["id"])
            if w["status"] != "ok":
                cls["触发但跳过"] += 1
                skip_cls[f"{w['skip'] or w['status']}({c['word']})"] += 1
                rowcls = f"skip:{w['skip'] or w['status']}"
            else:
                cls["触发且评分"] += 1
                n_ok += 1
                if c["rank_final"] == 1:
                    n_correct += 1
                    if c["rank_engine"] == 1:
                        cls["命中·词典原序"] += 1
                    else:
                        cls["命中·LLM改选"] += 1
                        n_moved_ok += 1
                else:
                    cls["选重·LLM未命中"] += 1
                    if c["rank_engine"] == 1:
                        n_broken += 1
                rowcls = "命中" if c["rank_final"] == 1 else "选重"
            for fld, lst in (("rank_engine", w["engine"]),
                             ("rank_window", w["window"]),
                             ("rank_final", w["final"])):
                want = rank_in(lst, c["word"])
                if c[fld] != want:
                    defects.append((c["id"],
                                    f"C.{fld}={c[fld]} ≠ 实位{want} ({c['word']} @ {fld})"))
            if w["status"] == "ok":                     # skip 窗 final 空为设计口径（0180 实例）
                if sorted(w["window"]) != sorted(w["final"]):
                    defects.append((c["id"], "final 非 window 置换"))
                if not set(w["window"]) <= set(w["engine"]):
                    defects.append((c["id"], "window 含 engine 外候选"))
            kmax = int(h_last["max_candidates"]) if h_last else 6
            if len(w["window"]) > kmax:
                defects.append((c["id"], f"窗宽 {len(w['window'])} > k {kmax}"))
            if len(w["engine"]) > 10:
                defects.append((c["id"], f"engine_top {len(w['engine'])} > 10"))
            if len(w["ctx"]) > CS.CTX_TAIL:
                defects.append((c["id"], f"ctx {len(w['ctx'])} 字 > {CS.CTX_TAIL}"))
            if w["id"] not in D and w["status"] == "ok":
                defects.append((w["id"], "debug 无 D 行（评分窗）"))
            if p_exists and w["id"] not in P and w["status"] == "ok":
                defects.append((w["id"], "debug 无 P 行"))
            if w["id"] in P:
                if "total" in P[w["id"]]:
                    total_ms.append(P[w["id"]]["total"])
                if "prep" in P[w["id"]]:
                    prep_ms.append(P[w["id"]]["prep"])
            dl = D.get(w["id"], {})
            # 无上文窗（2026-11-24 flags=nofreq）：窗内无 CE，D 行 ce=-inf、
            # key 只含词频项 → CE/融合/cerank 算术校验会拿 -inf 当数值误报。
            nofreq = CS.is_nofreq(w)
            if nofreq:
                n_nofreq += 1
            if dl and w["status"] == "ok":
                if len(dl) != len(w["window"]):
                    defects.append((w["id"], f"D 行数 {len(dl)} ≠ 窗宽 {len(w['window'])}"))
                # D 行规格：<W-id>|D|<final_rank>|… —— 行号 = 最终序位置（非窗内位）
                prev_key = float("inf")
                for r, f in sorted(dl.items()):
                    # -inf = 候选 token 数不足该位（缺位不计入加权；生产语义）
                    fin = lambda v: v if isinstance(v, float) and math.isfinite(v) else 0.0
                    if not nofreq:
                        ce_exp = -(fin(f["ce1"]) + W2 * fin(f["ce2"]) + W3 * fin(f.get("ce3", float("nan"))))
                        if abs(ce_exp - f["ce"]) > 0.02:
                            defects.append((w["id"], f"r{r} ce 算术 {ce_exp:.3f}≠{f['ce']:.3f}"))
                    if isinstance(f.get("eff"), float) and math.isfinite(f["eff"]):
                        fb_exp = BETA * math.log1p(max(0.0, f["eff"]))
                        if abs(fb_exp - f["fb"]) > 0.02:
                            defects.append((w["id"], f"r{r} fb 算术 {fb_exp:.3f}≠{f['fb']:.3f}"))
                    if not nofreq:
                        k_exp = f["ce"] + f["fb"] + f["u1"] + f.get("sdelta", 0.0)
                        if abs(k_exp - f["key"]) > 0.02:
                            defects.append((w["id"], f"r{r} key 算术 {k_exp:.3f}≠{f['key']:.3f}"))
                    if f["key"] > prev_key + 1e-6:
                        defects.append((w["id"], f"r{r} key {f['key']:.3f} > 前行 {prev_key:.3f}（final 序非 key 降序）"))
                    prev_key = f["key"]
                    if int(f["finrank"]) != r:
                        defects.append((w["id"], f"r{r} finrank {f['finrank']:g} ≠ 行号"))
                if not nofreq:
                    ces = {r: dl[r]["ce"] for r in dl}
                    corder = sorted(ces, key=lambda r: (-ces[r], r))
                    for r, f in dl.items():
                        if int(f["cerank"]) != corder.index(r) + 1:
                            defects.append((w["id"], f"r{r} cerank {f['cerank']:g} ≠ {corder.index(r)+1}"))
        elif RE4.match(c["code"]):
            rowcls = "无W异常"
            cls["异常·纯4码无W"] += 1
            defects.append((c["id"], f"纯4码({c['code']})提交无 W 记录"))
        elif RE4.match(re.sub(r"[^a-z]", "", c["code"])):
            code_core = re.sub(r"[^a-z]", "", c["code"])
            linked = None
            for w2 in reversed(w_by_code.get(code_core, [])):
                if w2["id"] in claimed or pts(w2["ts"]) > pts(c["ts"]):
                    continue
                if (pts(c["ts"]) - pts(w2["ts"])).total_seconds() <= 5:
                    linked = w2
                    break
                break                                    # 时间序，超 5 秒即止
            if linked is None:
                rowcls = "带标点·无W在先"
                cls["异常·带标点无W在先"] += 1
                defects.append((c["id"], f"带标点提交({c['code']})且 5s 内无同码 W 在先"))
            else:
                claimed.add(linked["id"])
                cls["带标点·W在先(ref断链)"] += 1
                ref_gap.append((c["id"], c["word"], c["code"], linked["id"]))
                if linked["status"] != "ok":
                    rowcls = f"带标点·skip:{linked['skip'] or linked['status']}"
                else:
                    rf = rank_in(linked["final"], core)
                    if rf == 1:
                        n_ok += 1
                        n_correct += 1
                        re_ = rank_in(linked["engine"], core)
                        if re_ == 1:
                            cls["命中·词典原序"] += 1
                        else:
                            cls["命中·LLM改选"] += 1
                            n_moved_ok += 1
                        rowcls = "带标点·命中"
                    elif rf > 1:
                        n_ok += 1
                        cls["选重·LLM未命中"] += 1
                        rowcls = "带标点·选重"
                    else:
                        rowcls = "带标点·词不在窗"
        else:
            rowcls = "短码"
            cls["短码·设计内不触发"] += 1
        rows.append((c, w, linked if w is None else None, rowcls))

    # ── W.ctx ↔ 重建文档尾部核对（时序模型：C 时全文入档；回撤 M 时删对应字符，
    #    带标点回撤经下一 W ctx 验证只删标点尾；场界清空）──
    rm_chars = {c["id"]: len(c["word"]) - len(contrib[c["id"]])
                for c in commits if c["retracted"]}
    doc = ""
    stack2 = []
    t0, t1 = session[0]["ts"], session[-1]["ts"]
    for e in events:
        if e["type"] == "C":
            doc += e["word"]
            stack2.append(e)
        elif e["type"] == "M":
            k = e["kind"]
            if k in CS.RETRACT_KINDS and stack2:
                c = stack2.pop()
                doc = doc[:len(doc) - rm_chars.get(c["id"], len(c["word"]))]
            elif k in CS.BOUNDARY_KINDS:
                doc = ""
                stack2 = []
        elif e["type"] == "W" and e["status"] == "ok" and t0 <= e["ts"] <= t1:
            ctx_n = CS.normalize_ctx(e["ctx"])
            if ctx_n:
                n_ctxchk += 1
                tail = CS.normalize_ctx(doc)[-len(ctx_n):]
                if tail != ctx_n:
                    defects.append((e["id"],
                                    f"W.ctx 与文档不符: ctx尾…{ctx_n[-16:]} / doc尾…{tail[-16:]}"))

    # ── 账本对账（账本 2026-11-17 迁移立账：cum 只承诺覆盖迁移日后提交；
    #    2026-11-20 退款上线 → 回撤提交 bump 已回滚、不计入期望；
    #    每 20 词全量重写 → 尾部有落盘节奏）──
    led = {}
    head_tick = None
    for line in Path(a.ledger_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#tick="):
            head_tick = int(line.split("=", 1)[1])
            continue
        if line.startswith("#"):          # 2026-11-20 起列含义注释行
            continue
        p = line.split("\t")
        if p and p[0] == "s" and len(p) >= 6:      # 旧 s 行（2026-11-20 前）
            led[(p[1], p[2])] = (float(p[3]), int(p[4]), int(p[5]))
        elif len(p) == 5:                          # 新格式（无行类型列）
            led[(p[0], p[1])] = (float(p[2]), int(p[3]), int(p[4]))
    post_cnt, last_tick = Counter(), {}
    flushed_cnt, flushed_last = Counter(), {}      # 末次落盘时刻（tick ≤ #tick）应有状态
    for c in commits:
        key = (c["word"], c["code"])
        if c["legacy"] or c["retracted"]:          # 迁移行不计（账本不承继）；
            continue                               # 回撤提交退款已回滚 → 不计期望
        last_tick[key] = c["tick"]
        post_cnt[key] += 1
        if head_tick and c["tick"] <= head_tick:
            flushed_cnt[key] += 1
            flushed_last[key] = c["tick"]
    sess_keys = {(c["word"], c["code"]) for c in session
                 if core_of(c["word"]) and not c["retracted"]}
    led_missing, led_bad, led_lag, led_seed = [], [], [], []
    for key in sorted(sess_keys):
        lt = last_tick[key]
        ht = head_tick or 0
        if key not in led:
            if lt > ht:
                led_lag.append((key[0], key[1]))          # 末提交晚于末次落盘
            elif flushed_cnt.get(key, 0):
                led_missing.append((key[0], key[1], flushed_cnt[key]))
            # 迁移后从未真实提交且未落账 = 仅旧史键，新账本从未 bump，合法
            continue
        dee, tick, cum = led[key]
        fc = flushed_cnt.get(key, 0)
        if cum < fc:   # 会话/迁移后 bump 丢失（迁移种子只会多不会少）
            led_bad.append((key[0], key[1],
                            f"cum={cum} < 落盘时刻应有 {fc}（迁移日后总数 {post_cnt[key]}）"))
        elif cum > fc:
            led_seed.append((key[0], key[1], cum - fc))   # 差值 = 旧账本迁移种子
        if lt > ht:
            led_lag.append((key[0], key[1]))              # 落盘后又有提交（下次落盘并入）
        if fc and tick != flushed_last.get(key, tick):
            led_bad.append((key[0], key[1], f"tick={tick} ≠ 落盘时刻末 bump {flushed_last[key]}"))
    n_e = sum(1 for c in session if c["id"] in E)

    # ── 汇总 ──
    L = []
    L.append("=" * 100)
    L.append(f"真机文章逐词审计 v2 · {a.since} ~ {a.until} · 提交 {len(session)} 条"
             f"（会话回撤 {len(retracted_sess)} 条）")
    L.append("=" * 100)
    if h_last:
        L.append(f"配置 H: {h_last['model']} pattern={h_last['code_pattern']} "
                 f"k={h_last['max_candidates']} ctx={h_last['max_ctx']} "
                 f"tcs={h_last['tcs']} β={h_last['beta']} τ={h_last['tau']}")
    L.append(f"文章 {len(article)} 字 ↔ 重建文档 {len(text)} 字；对齐 "
             f"[{session[0]['ts']} … {session[-1]['ts']}] tick {ticks[0]}→{ticks[-1]}")
    if punct_keep:
        L.append(f"带标点回撤·保词核 ×{len(punct_keep)}: "
                 + ", ".join(f"{wd}({cd})→{wid[-12:]}" for _, wd, cd, wid in punct_keep))
    L.append("")
    L.append("── 触发分类 ──")
    for k, v in cls.most_common():
        L.append(f"  {k:26s} {v}")
    L.append(f"  合计                       {len(session)}")
    if skip_cls:
        L.append("  W skip 分布: " + ", ".join(f"{k}×{v}" for k, v in skip_cls.most_common()))
    if n_nofreq:
        L.append(f"  无上文窗（flags=nofreq）×{n_nofreq}：无 CE，已跳过 CE/融合算术校验"
                 "（名次与 final 置换校验仍生效）")
    if n_ok:
        L.append("")
        L.append("── 重排质量（含带标点回挂恢复）──")
        L.append(f"  评分 {n_ok} · rank_final=1 正确 {n_correct} ({n_correct/n_ok*100:.1f}%)"
                 f" · 选重 {n_ok-n_correct} ({(n_ok-n_correct)/n_ok*100:.1f}%)")
        L.append(f"  LLM 改选后命中 {n_moved_ok} · 词典首选被打跑且未选回 {n_broken}")
    L.append(f"  W.ctx↔文档尾部核对 {n_ctxchk} 窗")
    if total_ms:
        total_ms.sort()
        prep_ms.sort()
        L.append(f"  P 行延迟 n={len(total_ms)}: total 中位 {total_ms[len(total_ms)//2]:.0f}ms"
                 f" / p95 {total_ms[max(0,int(len(total_ms)*0.95)-1)]:.0f}ms · "
                 f"prep 中位 {prep_ms[len(prep_ms)//2]:.0f}ms")
    L.append("")
    L.append(f"── 缺陷（应为零）: {len(defects)} ──")
    for i, (eid, msg) in enumerate(defects[:80]):
        L.append(f"  [{i+1}] {eid}  {msg}")
    if len(defects) > 80:
        L.append(f"  …（其余 {len(defects)-80} 条）")
    L.append("")
    L.append(f"── 记录观察项 ──")
    L.append(f"  带标点提交 ref 断链 ×{len(ref_gap)}（spec=同码 W 才回挂；验收经前序 W 恢复统计）")
    for cid, wd, cd, wid in ref_gap[:8]:
        L.append(f"    {cid[-14:]} {wd}[{cd}] → 前序 W {wid[-14:]}")
    L.append(f"  会话回撤 {len(retracted_sess)} 条: "
             + ", ".join(f"{c['word']}[{c['code']}]" for c in retracted_sess))
    rf_ok = sum(1 for c in retracted_sess if RF.get(c["id"]) == 1)
    rf_no = sum(1 for c in retracted_sess
                if c["id"] in RF and RF[c["id"]] == 0)
    L.append(f"  退款 RF 行（2026-11-20 口径）: refund=1 ×{rf_ok} / refund=0 ×{rf_no}"
             f"（无账目=标点/栈空不退）→ 回撤提交不入账，账本期望不含回撤")
    L.append("")
    L.append("── 账本对账（迁移 2026-11-17 立账口径；回撤已退款 → 期望不含回撤）──")
    L.append(f"  ledger 词+码条目 {len(led)} · #tick={head_tick}"
             f"（会话末 tick {ticks[-1]}，{'已含会话' if head_tick and head_tick >= ticks[-1] else '尾部滞后=20词落盘节奏'}）")
    L.append(f"  debug E 行逐提交覆盖 {n_e}/{len(session)}")
    if led_lag:
        L.append(f"  落盘节奏内未入账 {len(led_lag)} 键（末提交 tick > #tick）: "
                 + ", ".join(f"{k[0]}[{k[1]}]" for k in led_lag[:10]))
    if led_seed:
        L.append(f"  cum 超出迁移后计数的差值 {len(led_seed)} 键（= 旧账迁移种子 + 已退款回撤，样例）: "
                 + ", ".join(f"{k0}[{k1}]+{v}" for k0, k1, v in led_seed[:6]))
    if led_missing:
        L.append(f"  !! 应落账而缺行 {len(led_missing)}: {led_missing[:10]}")
    if led_bad:
        L.append(f"  !! 会话计数丢失/tick 不符 {len(led_bad)}: {led_bad[:10]}")
    if not led_missing and not led_bad:
        L.append(f"  会话 {len(sess_keys)} 个词+码键无丢失（cum ≥ 落盘时刻应有数，tick 相符；"
                 f"回撤已退款口径，期望不含回撤提交）")
    x_err = [x for x in X if x[1] == "X" and len(x) > 2 and not x[2].startswith("scope=ops")]
    L.append("")
    L.append(f"── debug X 行: ops {len(X)-len(x_err)} / 非ops错误 {len(x_err)} ──")
    for x in x_err[:10]:
        L.append("  " + "|".join(x))

    # ── 逐词明细 ──
    L.append("")
    L.append("── 逐词明细 ──")
    L.append(f"{'词':<9} {'码':<7} {'类':<14} rE rW rF  引擎首选→最终首选  词库位")
    for c, w, linked, rowcls in rows:
        lw = w if w is not None else linked
        eng0 = lw["engine"][0] if lw and lw["engine"] else "-"
        fin0 = lw["final"][0] if lw and lw["final"] else "-"
        dp = ""
        wc = word_code.get(core_of(c["word"]))
        if wc:
            cw_l = code_to_words.get(wc, [])
            if core_of(c["word"]) in cw_l:
                dp = f"dict#{cw_l.index(core_of(c['word']))+1}/{len(cw_l)}"
        L.append(f"{c['word']:<9} {c['code']:<7} {rowcls:<14} "
                 f"{c['rank_engine']:>2} {c['rank_window']:>2} {c['rank_final']:>2}  "
                 f"{eng0}→{fin0}  {dp}")

    rep = "\n".join(L) + "\n"
    out = Path(a.out) if a.out else CS.EVAL / "audit_article_session_20261120.txt"
    out.write_text(rep, encoding="utf-8")
    print("\n".join(L[:55]))
    print(f"\n… 全量 {len(L)} 行 → {out}")
    print(f"缺陷 {len(defects)} · 评分窗 {n_ok} · 正确 {n_correct} ({n_correct/max(1,n_ok)*100:.1f}%)"
          f" · ctx核对 {n_ctxchk} · ref断链恢复 {len(ref_gap)}")


if __name__ == "__main__":
    main()
