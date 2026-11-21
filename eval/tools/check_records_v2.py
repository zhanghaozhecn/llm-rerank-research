# -*- coding: utf-8 -*-
"""check_records_v2.py — 语料/ debug 关联与不重复校验（2026-11-17）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std


def main(corpus, debug):
    events = list(corpus_std.iter_events(corpus))
    ids = {"W": set(), "C": set(), "M": set()}
    for e in events:
        if e["type"] in ids:
            ids[e["type"]].add(e["id"])
    want = {"P": "W", "D": "W", "E": "C", "R": "M"}
    bad, counts = [], {}
    dp = Path(debug)
    if dp.exists():
        with open(dp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                fld = line.split("|")
                if len(fld) < 2:
                    bad.append("short: " + line)
                    continue
                eid, typ = fld[0], fld[1]
                counts[typ] = counts.get(typ, 0) + 1
                if typ in want and eid not in ids[want[typ]]:
                    bad.append("missing %s->%s: %s" % (typ, want[typ], line))
                if typ in ("P", "D", "E", "R"):
                    for key in ("word=", "ctx=", "code=", "engine=", "final="):
                        if key in line:
                            bad.append("duplicate field %s: %s" % (key, line))
    print("corpus events", len(events),
          "W", len(ids["W"]), "C", len(ids["C"]), "M", len(ids["M"]))
    print("debug lines", sum(counts.values()), counts)
    if bad:
        print("FAIL", len(bad))
        for b in bad[:10]:
            print(" ", b)
        return 1
    print("OK: every P/D/E/R ID exists in corpus; no duplicated corpus fields")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
