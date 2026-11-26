# -*- coding: utf-8 -*-
import json
from collections import defaultdict
EVAL = r"D:\llm-rerank\llm-rerank-research\eval"
DICT = r"C:\Users\Administrator\AppData\Roaming\Rime\wanxiang.dict.yaml"
exact = defaultdict(set); syll=set()
for line in open(DICT, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3 or len(p[0]) != 1: continue
    exact[p[1]].add(p[0]); syll.add(p[1])
want = {"shen","sheng","ji","zhu","yi","yu","shi","de","shi","xian","ta","ne"}
out = defaultdict(list)
for line in open(EVAL + r"\luna_mixed\cohort.jsonl", encoding="utf-8"):
    r = json.loads(line)
    code = r["code"]
    if code not in want: continue
    seen = {}
    for c in r["eng"]:
        text,s,e,typ,q,ce3,nt = c
        if len(text)!=1 or s!=0 or e!=len(code): continue
        if ce3 and ce3[0] is not None and ce3[0]>=0: seen[text]=ce3[0]
    if len(seen)<2: continue
    ranked = sorted(seen.items(), key=lambda x:x[1])
    out[code].append((r["ctx"][-12:], [(t, v, t in exact[code]) for t,v in ranked[:6]]))
for code in ["shen","sheng","ji","zhu","yi","yu","shi","de","xian","ta","ne"]:
    ws = out.get(code, [])
    print(f"=== {code}  windows={len(ws)}  整读字数={len(exact[code])}")
    for ctx, rk in ws[:3]:
        print("   ctx…", ctx, "|", ", ".join(
            f"{'[前]' if not ex else ''}{t}({v:.1f})" for t,v,ex in rk))
