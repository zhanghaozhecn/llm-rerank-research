# -*- coding: utf-8 -*-
"""聚合 docstart_*.csv: 每 (模型, n, 形态) 的均值 CE 与 top1 命中率 + 配对差值."""
import io, sys, glob, csv
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
FORMS = ["bare", "eos+", "eot+", "punct+"]
NS = [1, 2, 4, 8]

for path in sorted(glob.glob("docstart_*.csv")):
    name = path.replace("docstart_", "").replace(".csv", "")
    # per (n, form): list of (ce, hit); 配对 = 按 (句序隐含) 顺序对齐不可靠, 用均值差
    data = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 4:
                continue
            n, form, ce, hit = int(row[0]), row[1], float(row[2]), int(row[3])
            data[(n, form)].append((ce, hit))
    print("=" * 20, name, "(样本=%d)" % (len(data[(1, "bare")]) + 0), "=" * 20)
    print("%-4s | %-22s | %-22s | %-22s | %-22s" % ("n", "bare", "eos+", "eot+", "punct+"))
    for n in NS:
        cells = []
        for form in FORMS:
            v = data.get((n, form))
            if not v:
                cells.append("n/a")
                continue
            mce = sum(x[0] for x in v) / len(v)
            hit = sum(x[1] for x in v) / len(v) * 100
            cells.append("%.3f/%4.1f%%" % (mce, hit))
        print("%-4d | %-22s | %-22s | %-22s | %-22s" % (n, *cells))
    # 全 n 合并 + 相对 bare 的均值 CE 变化
    print("  合并:", end="")
    for form in FORMS:
        allv = [x for n in NS for x in data.get((n, form), [])]
        mce = sum(x[0] for x in allv) / len(allv)
        hit = sum(x[1] for x in allv) / len(allv) * 100
        bare = [x for n in NS for x in data.get((n, "bare"), [])]
        bm = sum(x[0] for x in bare) / len(bare)
        print("  %s: %.3f/%.1f%% (Δ%.3f)" % (form, mce, hit, mce - bm), end="")
    print("\n")
