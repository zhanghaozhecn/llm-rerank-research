# -*- coding: utf-8 -*-
"""prefix_big 64 池按需回填：只给「能成为训练行」的窗算 CE。

廉价过滤（不需要 64 池 CE，用 16 池 cands 代理）：
  非 dyn ∧ len(code)≥4 ∧ gen 有 lead ∧ le<码长 ∧ 剩余≥2
  ∧ 池(16)≥2 ∧ lead 在池(16)（start 0, end le）
输出：
  eval/prefix_big/work64/eligible_codes.txt   （探针 64 输入）
  eval/prefix_big/work64/eligible_wids.json   （窗清单，CE/组装用）
"""
import json
import sys
from pathlib import Path

EVAL = Path(r"D:\llm-rerank\llm-rerank-research\eval")
sys.path.insert(0, str(EVAL / "calib"))
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L          # noqa: E402
import prefix_confirm_model as M    # noqa: E402

BIG = EVAL / "prefix_big"
W64 = BIG / "work64"
W64.mkdir(parents=True, exist_ok=True)

gen = M.load_gen_map(BIG / "gen.tsv")
n_win = elig = 0
codes = []
wids = []
miss = {}
for w in L.jsonl_iter(BIG / "windows.jsonl"):
    if w.get("dyn") or len(w["code"]) < 4:
        continue
    n_win += 1
    gs = gen.get(w["wid"])
    if not gs:
        miss["gen"] = miss.get("gen", 0) + 1
        continue
    lead, p, le, sent, d = M.lead_stats(gs)
    if not (le < len(w["code"]) and len(sent) - len(lead) >= 2):
        miss["guard"] = miss.get("guard", 0) + 1
        continue
    codes.append(w["code"])
    wids.append(w["wid"])
    elig += 1

# 去重码串（探针按码缓存，一码多窗共用）
codes_u = sorted(set(codes))
(W64 / "eligible_codes.txt").write_text(
    "\n".join(codes_u) + "\n", encoding="ascii", newline="\n")
(W64 / "eligible_wids.json").write_text(
    json.dumps(wids), encoding="utf-8")
print(f"big 静态窗 {n_win}；廉价过滤通过 {elig} 窗"
      f"（unique codes {len(codes_u)}）；miss={miss}")
print(f"估 64 池 CE 行 ≈ {elig}×5.8 ≈ {int(elig * 5.8)}（含首块复用 ~1/7）")
