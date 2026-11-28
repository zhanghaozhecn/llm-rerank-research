# -*- coding: utf-8 -*-
"""重放后缓存完整性校验（锚口径失效/重放的收尾断言）。

断言：
  A. 保留行逐字节一致 —— 失效重写 + 重放不得扰动任何非过期条目
     （含 n=0 / n≥2：锚对它们零效应，重放也根本没碰它们）；
  B. 过期 sid 全部已重算回填（新缓存无洞）；
  C. 过期条目新值 vs 旧值：CE 均值应系统性下降（锚效果，ΔCE≈−6 nats），
     下降条目占比应显著偏高（不可全降——个别候选锚后 CE 可升）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # 研究库根
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # eval/calib
sys.stdout.reconfigure(encoding="utf-8")
import luna_mixed_lib as L   # noqa: E402

WORK = L.WORK
BAK = {L.CE_DIR: L.CE_RESULTS.with_suffix(".jsonl.bak_preanchor"),
       L.CE_DIR_EXT: L.CE_RESULTS_EXT.with_suffix(".jsonl.bak_preanchor")}
REFILL = {L.CE_DIR: WORK / "refill_preanchor_16.jsonl",
          L.CE_DIR_EXT: WORK / "refill_preanchor_64.jsonl"}


def load_lines(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
            except ValueError:
                continue
            out[r["sid"]] = l.rstrip("\n")
    return out


def mean_ce(d):
    vs = []
    for _, (ce3, _nt) in d.items():
        if ce3 and ce3[0] is not None:
            vs.append(sum(x for x in ce3 if x is not None))
    return (sum(vs) / len(vs)) if vs else None


def main():
    ok = True
    for d in (L.CE_DIR, L.CE_DIR_EXT):
        bak = load_lines(BAK[d])
        new = load_lines(d / "results.jsonl")
        refill_sids = [json.loads(l) for l in open(REFILL[d], encoding="utf-8")]
        want = {L.corpus_std.window_key_ctx(r["context"], r["cands"])
                for r in refill_sids}
        # A. 保留行一致
        same = diff = 0
        for sid, line in bak.items():
            if sid in want:
                continue
            if new.get(sid) == line:
                same += 1
            else:
                diff += 1
        # B. 过期已回填
        missing = [s for s in want if s not in new]
        # C. 新旧差值
        old_ce, new_ce = L.load_ce(BAK[d]), L.load_ce(d / "results.jsonl")
        deltas, down, up = [], 0, 0
        for sid in want:
            o, n = old_ce.get(sid), new_ce.get(sid)
            if not o or not n:
                continue
            mo, mn = mean_ce(o), mean_ce(n)
            if mo is None or mn is None:
                continue
            deltas.append(mn - mo)
            down += mn < mo
            up += mn > mo
        avg = sum(deltas) / len(deltas) if deltas else 0.0
        print(f"[chk] {d.name}: 保留一致 {same} / 不一致 {diff}；"
              f"过期回填缺 {len(missing)}；过期条目 ΔCE 均值 {avg:+.3f} "
              f"（降 {down} / 升 {up} / 计 {len(deltas)}）")
        if diff or missing or avg >= 0:
            ok = False
    print("[chk] PASS" if ok else "[chk] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
