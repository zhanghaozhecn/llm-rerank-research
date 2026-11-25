#!/usr/bin/env python3
"""真机语料：整句第 n 个词的自动确认率与准确率（2026-11-25）。

数据 = `%APPDATA%\\Rime\\llm_rerank\\<方案>\\rime_llm_corpus.tsv`（W/C/M）
+ 同目录 `rime_llm_debug.txt`（`pfx: exec select '<词>'` 事件，按「最近一个
W id」锚定到语料行）。

定义：
  * **一句** = 一次组合：从上一个提交（C）或边界 M（focus/navigation/enter…）
    之后到下一个 C 之间的一组 W 行；提交文本 = 这些 C 的 word 拼接。
  * **确认位置** = 确认发生时，锚 W 的 ctx 相对「组合起点 ctx0」多出的字符数
    （即确认词在最终提交文本里的字符偏移）。
  * **正确** = 最终提交文本在该偏移处恰好等于确认文本（用户没删/没改）；
    不一致 = 该次自动确认最终没留在文本里（用户退格重打或选错）。
  * **n** = 该偏移之前有多少个词典词（`pdsp_dict.yaml` 贪心最长匹配）+ 1。

输出：总数 / 准确率、按 n 的分布、按日期的分布（门不同）、失败样本。

跑法（研究库根目录）：
  python eval/calib/confirm_real_corpus.py [--schema luna_pinyin]
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "eval/lib")
sys.stdout.reconfigure(encoding="utf-8")
import corpus_std as C  # noqa: E402

RIME = Path(os.environ.get("APPDATA", "")) / "Rime" / "llm_rerank"
IDPAT = re.compile(r"^([0-9]{8}T[0-9]{9}-[0-9]+-[0-9]+)\|")
BOUND = {"focus", "navigation", "enter", "scene", "reset"}


def load_words(limit=6):
    """pdsp 词典词表（贪心最长匹配用）。"""
    p = C.PROJ / "pdsp_dict.yaml"
    ws = set()
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            w = line.split("\t")[0].strip()
            if w and 2 <= len(w) <= limit:
                ws.add(w)
    return ws


def seg_words(text, lex):
    """贪心最长匹配（最大 6 字）→ 词列表。"""
    out, i = [], 0
    while i < len(text):
        for L in range(6, 1, -1):
            if i + L <= len(text) and text[i:i + L] in lex:
                out.append(text[i:i + L])
                i += L
                break
        else:
            out.append(text[i])
            i += 1
    return out


def parse_confirms(log):
    """debug → [(锚 W id, 确认文本)]。"""
    out, last = [], None
    if not log.exists():
        return out
    for line in log.open(encoding="utf-8", errors="replace"):
        m = IDPAT.match(line)
        if m:
            last = m.group(1)
        k = "pfx: exec select '"
        i = line.find(k)
        if i >= 0 and last:
            x = line[i + len(k):]
            out.append((last, x[:x.find("'")]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="luna_pinyin")
    a = ap.parse_args()
    corpus = RIME / a.schema / "rime_llm_corpus.tsv"
    log = RIME / a.schema / "rime_llm_debug.txt"
    evs = list(C.iter_events(corpus))
    wid = {e["id"]: i for i, e in enumerate(evs) if e["type"] == "W"}
    conf = parse_confirms(log)
    lex = load_words()
    print(f"[data] {a.schema}: 事件 {len(evs)}（W {sum(1 for e in evs if e['type']=='W')}"
          f" / C {sum(1 for e in evs if e['type']=='C')}）；"
          f"确认事件 {len(conf)}；词典词 {len(lex)}")
    print(f"[data] 时间 {evs[0]['ts']} → {evs[-1]['ts']}")

    # 场景切分：一组 W 直到「提交 C」；句文本 = 提交之后第一个 W 的 ctx（含已提交句）
    scenes, cur = [], None
    for i, e in enumerate(evs):
        t = e["type"]
        if t == "M" and e.get("kind") in BOUND:
            if cur and cur["w"]:
                scenes.append(cur)
            cur = None
            continue
        if t == "W":
            if cur is None:
                cur = dict(w=[], cs=[], ctx0=e["ctx"], idx0=i)
            cur["w"].append(i)
        elif t == "C":
            if cur is None:
                cur = dict(w=[], cs=[], ctx0=e.get("ctx", ""), idx0=i)
            cur["cs"].append(i)
            scenes.append(cur)
            cur = None
    if cur and cur["w"]:
        scenes.append(cur)
    # 句文本（ground truth）：C 之后第一个 W 的 ctx 相对 ctx0 的增量
    scenes = [sc for sc in scenes if sc["w"]]
    for k, sc in enumerate(scenes):
        sc["final"] = None
        j = sc["cs"][-1] + 1 if sc["cs"] else sc["w"][-1]
        while j < len(evs):
            if evs[j]["type"] == "W" and evs[j]["ctx"].startswith(sc["ctx0"]):
                sc["final"] = evs[j]["ctx"][len(sc["ctx0"]):]
                break
            if evs[j]["type"] == "M" and evs[j].get("kind") in BOUND:
                break
            j += 1
        if sc["final"] is None:
            # 没有后续 W（句末/会话末）：用场景最后一个 W 的 ctx 增量
            last = evs[sc["w"][-1]]
            if last["ctx"].startswith(sc["ctx0"]):
                sc["final"] = last["ctx"][len(sc["ctx0"]):]
    wid_of = {evs[i]["id"]: i for sc in scenes for i in sc["w"]}
    rows, skipped_bad = [], 0
    rej = Counter()
    for si, sc in enumerate(scenes):
        F = sc["final"]
        if not F:
            continue
        if "�" in F:          # ctx 含非法 UTF-8（写入截断/粘贴）→ 偏移不可信
            skipped_bad += 1
            continue
        for i in sc["w"]:
            e = evs[i]
            for a_id, x in conf:
                if wid_of.get(a_id) != i:
                    continue
                if not e["ctx"].startswith(sc["ctx0"]):
                    rej["ctx 非同一组合起点"] += 1
                    continue
                d = len(e["ctx"]) - len(sc["ctx0"])     # 锚 W 的 ctx 增量
                # 锚行可能已含 X（确认后的重查）或未含（确认前的查询）
                cands = []
                if d >= len(x) and F[d - len(x):d] == x:
                    cands.append((d - len(x), x))
                if F[d:d + len(x)] == x:
                    cands.append((d, x))
                ok = bool(cands)
                off = cands[0][0] if cands else d
                n = len(seg_words(F[:off], lex)) + 1
                rows.append(dict(ci=si, ts=e["ts"], x=x, n=n, ok=ok,
                                 got=F[off:off + len(x)], final=F,
                                 off=off, code=e["code"]))
    print(f"[data] 跳过含非法 UTF-8 的句子 {skipped_bad}；"
          f"未配对确认 {len(conf)-len(rows)}（{dict(rej) or '无后续句文本/边界外'}）")
    nconf = len(rows)
    nok = sum(1 for r in rows if r["ok"])
    print(f"\n[总] 自动确认 {nconf} 次，正确 {nok}，准确率 "
          f"{100*nok/max(nconf,1):.1f}%")
    # 逐句
    ncomp = len({r["ci"] for r in rows})
    lens = [len(seg_words(sc["final"], lex)) for sc in scenes if sc["final"]]
    total_words = sum(lens)
    print(f"[总] 覆盖句子 {ncomp}；全语料句数 {len(lens)}、词数 {total_words}；"
          f"自动确认 / 词 = {100*nconf/max(total_words,1):.1f}%")

    print("\n[按 n 个词]  n   确认   正确   准确率   该位置句子数   确认率")
    byn = defaultdict(lambda: [0, 0])
    for r in rows:
        byn[r["n"]][0] += 1
        byn[r["n"]][1] += 1 if r["ok"] else 0
    for n in sorted(byn):
        c, ok = byn[n]
        denom = sum(1 for L in lens if L >= n)
        print(f"           {n:<4}{c:>4}  {ok:>4}   "
              f"{100*ok/c:>6.1f}%   {denom:>8}      "
              f"{100*c/max(denom,1):>5.1f}%")

    print("\n[按日期]（门不同：11-23 = P≥0.95 单门；11-24 起 ce1 实验）")
    byd = defaultdict(lambda: [0, 0])
    for r in rows:
        byd[r["ts"][:10]][0] += 1
        byd[r["ts"][:10]][1] += 1 if r["ok"] else 0
    for d in sorted(byd):
        c, ok = byd[d]
        print(f"  {d}: {c:>3} 次，正确 {ok:>3}，{100*ok/c:.1f}%")

    print("\n[失败样本] 确认词 → 该位置实际文本")
    bad = [r for r in rows if not r["ok"]]
    for r in bad[:15]:
        print(f"  {r['ts'][11:]} n={r['n']} code={r['code']:<10} "
              f"确认「{r['x']}」实际「{r['got'] or '（空/错位）'}」"
              f" 句「{r['final'][:18]}」")

    print("\n[词频] 确认词 TOP")
    for w, c in Counter(r["x"] for r in rows).most_common(12):
        ok = sum(1 for r in rows if r["x"] == w and r["ok"])
        print(f"  {w:<6} {c:>3} 次，正确 {ok:>3}（{100*ok/c:.0f}%）")


if __name__ == "__main__":
    main()
