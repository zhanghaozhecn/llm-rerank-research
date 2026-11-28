# -*- coding: utf-8 -*-
"""anchor_*.txt → 无锚/有锚对照表."""
import io, sys, re, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
FORMS = ["bare", "eos+", "eot+", "punct+"]
KEYS = ["世纪", "事迹", "试剂"]

for path in sorted(glob.glob("anchor_*.txt")):
    name = path.replace("anchor_", "").replace(".txt", "")
    print("=" * 24, name, "=" * 24)
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"ctx=(\S+) form=(\S+) ntok=(\d+) \|(.*)", line)
            if not m:
                if line.startswith("model="):
                    print("  ", line.strip())
                continue
            ctx, form, ntok, rest = m.groups()
            vals = dict(re.findall(r" (\S+)=(-?\d+\.\d+|-?inf|nan)", rest))
            rows[(ctx, form)] = (int(ntok), vals)
    for ctx in ("二十", "英雄", "化学"):
        print("  ctx=%s" % ctx)
        for form in FORMS:
            hit = rows.get((ctx, form))
            if not hit:
                continue
            ntok, vals = hit
            seg = "  ".join("%s=%s" % (k, vals.get(k, "?")) for k in KEYS)
            mark = ""
            try:
                # 期望的搭配词
                want = {"二十": "世纪", "英雄": "事迹", "化学": "试剂"}[ctx]
                ok = all(vals[k] < vals[j] for j in KEYS if j != want for k in [want])
                mark = " ✓" if ok else " ✗"
            except Exception:
                pass
            print("    %-6s ntok=%d | %s%s" % (form, ntok, seg, mark))
    print()
