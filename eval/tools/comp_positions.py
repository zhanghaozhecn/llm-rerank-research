#!/usr/bin/env python3
"""W 行 v9 comp 类标统计（2026-11-27，todo P0-2「确认门新增触发精度」配套工具）。

读语料 W 行第 15 字段 `comp=窗内位次;最终序位次`（v9 联想类标，1 基、
逗号分隔、空 = 无联想候选），只统计**带 comp 字段的行**（v9 起），
旧行与 skip 行（窗空）不进分母。输出：

  1. 行普查：W 总数 / ok 窗 / v9 行 / 联想窗占比（按方案分列 + 合计）
  2. 位次直方图：comp_window 与 comp_final（1..10, 11+）
  3. 确认门相关：联想占「窗内第 1 位」的窗数与占比 —— 门新增触发候选窗
     上界（该窗此前必返回 nil）；联想占「最终序第 1 位」窗数（显示序
     回归指标，大幅 >0 说明同 span 精确词没压住联想）
  4. C 行 join（按 C.ref → W id）：联想在窗首时用户实际提交的首词
     （path 归因取 path[0]，否则整词）== 窗首联想词？≈ 假如门对这类窗
     自动确认窗首联想词，会确认对的比例；以及联想被选重（提交首词 ∈
     联想集合）的次数、其窗内/最终位次分布

用法：
  python eval/tools/comp_positions.py                  # 全部语料文件
  python eval/tools/comp_positions.py <corpus.tsv> ... # 指定文件
  python eval/tools/comp_positions.py --json out.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "eval" / "lib"))

from corpus_std import _int, _split_cands, _unesc, parse_attr_path  # noqa: E402


def _hist(counter):
    """Counter({1: n, ...}) → [('1', n), ..., ('11+', n)] 按位次排好。"""
    out = []
    for p in sorted(counter):
        out.append((str(p) if p <= 10 else "11+", counter[p]))
    # 合并 11+ 桶
    merged = defaultdict(int)
    for k, v in out:
        merged[k] += v
    return [(k, merged[k]) for k in sorted(merged, key=lambda x: int(x.rstrip('+')))]


def _fmt_hist(counter):
    return "  ".join(f"#{k}:{v}" for k, v in _hist(counter)) or "  (空)"


def scan_file(path):
    """→ (census, w_rows) ；census = dict，w_rows = v9 且 status=ok 的 W dict 列表。"""
    census = {"w_total": 0, "w_ok": 0, "v9": 0, "v9_ok": 0}
    w_rows = []
    p = Path(path)
    if not p.exists():
        return census, w_rows
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fld = line.split("\t")
            t = fld[0]
            if t == "W" and len(fld) >= 14:
                census["w_total"] += 1
                ok = fld[10] == "ok"
                if ok:
                    census["w_ok"] += 1
                is_v9 = len(fld) >= 15 and fld[14].startswith("comp=")
                if not is_v9:
                    continue
                census["v9"] += 1
                if not ok:
                    continue
                census["v9_ok"] += 1
                a, _, b = fld[14][5:].partition(";")
                w_rows.append({
                    "id": fld[1], "ts": fld[2], "schema": p.parent.name,
                    "code": _unesc(fld[4]), "ctx": _unesc(fld[5]),
                    "engine": _split_cands(fld[6]),
                    "window": _split_cands(fld[7]),
                    "final": _split_cands(fld[8]),
                    "tick": _int(fld[9]),
                    "comp_window": tuple(int(x) for x in a.split(",") if x.strip()),
                    "comp_final": tuple(int(x) for x in b.split(",") if x.strip()),
                })
            elif t == "C" and len(fld) >= 11:
                w_rows.append({
                    "type_c": True, "id": fld[1], "ref": fld[3],
                    "word": _unesc(fld[4]), "code": _unesc(fld[5]),
                    "rank_engine": _int(fld[6]), "rank_window": _int(fld[7]),
                    "rank_final": _int(fld[8]),
                    "path": parse_attr_path(_unesc(fld[11])) if len(fld) >= 12 else [],
                })
    return census, w_rows


def main():
    ap = argparse.ArgumentParser(description="W 行 v9 comp 类标统计")
    ap.add_argument("paths", nargs="*", help="语料 tsv；缺省 = corpus_std 全发现")
    ap.add_argument("--json", help="统计结果另存 JSON")
    args = ap.parse_args()

    files = [Path(p) for p in args.paths] or None
    if files is None:
        sys.path.insert(0, str(PROJ / "eval" / "lib"))
        from corpus_std import CORPUS_PATHS
        files = [Path(p) for p in CORPUS_PATHS]

    census_all = defaultdict(int)
    wins, commits = [], []
    for f in files:
        census, rows = scan_file(f)
        for k, v in census.items():
            census_all[k] += v
        if census["w_total"]:
            print(f"[file] {f}")
            print(f"       W 总 {census['w_total']}  ok {census['w_ok']}  "
                  f"v9 {census['v9']}  v9&ok {census['v9_ok']}")
        for r in rows:
            (commits if r.get("type_c") else wins).append(r)

    if not wins:
        print("无 v9 ok 窗（带 comp 字段且 status=ok）——还没有可统计数据。")
        return

    wmap = {w["id"]: w for w in wins}
    base = [w for w in wins if w["window"]]
    n_base = len(base)

    # ② 位次直方图 + ③ 门相关计数
    hist_w, hist_f = Counter(), Counter()
    n_win_first = n_final_first = 0
    for w in base:
        for p in w["comp_window"]:
            hist_w[p] += 1
        for p in w["comp_final"]:
            hist_f[p] += 1
        if 1 in w["comp_window"]:
            n_win_first += 1
        if 1 in w["comp_final"]:
            n_final_first += 1

    print(f"\n== 合计：v9 ok 窗 {census_all['v9_ok']}（非空窗 {n_base}）==")
    if n_base:
        print(f"① 联想窗占比：{sum(1 for w in base if w['comp_window'])}/{n_base} "
              f"({100.0 * sum(1 for w in base if w['comp_window']) / n_base:.1f}%)")
    print(f"② 窗内位次直方图：{_fmt_hist(hist_w)}")
    print(f"   最终序位次：    {_fmt_hist(hist_f)}")
    if n_base:
        print(f"③ 联想占窗内第 1 位：{n_win_first}/{n_base} "
              f"({100.0 * n_win_first / n_base:.2f}%) ← 门新增触发候选窗上界")
        print(f"   联想占最终序第 1 位：{n_final_first}/{n_base} "
              f"({100.0 * n_final_first / n_base:.2f}%) ← 显示序回归指标")

    # ④ C 行 join：联想在窗首时的确认正确率 + 联想被选重
    j = {"n": 0, "lead_eq_commit": 0, "picked": 0, "picked_winpos": Counter(),
         "picked_finpos": Counter(), "no_win": 0}
    for c in commits:
        w = wmap.get(c["ref"])
        if not w:
            continue
        first_word = c["path"][0][0] if c["path"] else c["word"]
        comp_set = {w["window"][p - 1] for p in w["comp_window"]
                    if 0 < p <= len(w["window"])}
        if not comp_set:
            continue
        if 1 in w["comp_window"] and w["window"]:
            j["n"] += 1
            if first_word == w["window"][0]:
                j["lead_eq_commit"] += 1
        if first_word in comp_set:
            j["picked"] += 1
            for p in w["comp_window"]:
                if 0 < p <= len(w["window"]) and w["window"][p - 1] == first_word:
                    j["picked_winpos"][p] += 1
                    break
            for p in w["comp_final"]:
                if 0 < p <= len(w["final"]) and w["final"][p - 1] == first_word:
                    j["picked_finpos"][p] += 1
                    break

    print(f"\n④ C 行 join（关联到 v9 ok 窗的提交）：")
    print(f"   联想占窗首的提交：{j['n']}；其中提交首词 == 窗首联想词 "
          f"{j['lead_eq_commit']}"
          + (f"（{100.0 * j['lead_eq_commit'] / j['n']:.1f}% ≈ 若门自动确认"
             f"窗首联想的正确率）" if j["n"] else ""))
    print(f"   联想被选重：{j['picked']} 次"
          + (f"，窗内位次 {_fmt_hist(j['picked_winpos'])}；"
             f"最终序位次 {_fmt_hist(j['picked_finpos'])}" if j["picked"] else ""))
    total_commits = len([c for c in commits if c.get("ref") in wmap])
    if total_commits:
        print(f"   （同期关联提交总量 {total_commits}；联想窗首占 "
              f"{100.0 * j['n'] / total_commits:.2f}%）")

    if args.json:
        out = {
            "census": dict(census_all),
            "base_windows": n_base,
            "share_comp_window": (sum(1 for w in base if w["comp_window"]),
                                  n_base),
            "hist_window": dict(hist_w), "hist_final": dict(hist_f),
            "n_win_first": n_win_first, "n_final_first": n_final_first,
            "join": {"n": j["n"], "lead_eq_commit": j["lead_eq_commit"],
                     "picked": j["picked"],
                     "picked_winpos": dict(j["picked_winpos"]),
                     "picked_finpos": dict(j["picked_finpos"]),
                     "total_commits": total_commits},
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
