# -*- coding: utf-8 -*-
"""配对聚合 docstart2_*.csv: 同句同 n 下各形态 vs bare 的逐样本配对差 + 符号检验.
CSV 行序 = 每句 n 外层、form 内层固定 bare,eos+,eot+,punct+ → 下标天然对齐."""
import io, sys, glob, csv, math
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
FORMS = ["bare", "eos+", "eot+", "punct+"]

for path in sorted(glob.glob("docstart2_*.csv")):
    name = path.replace("docstart2_", "").replace(".csv", "")
    data = defaultdict(list)  # (n, form) -> [(ce, hit)]
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 4:
                continue
            n, form, ce, hit = int(row[0]), row[1], float(row[2]), int(row[3])
            data[(n, form)].append((ce, hit))

    print("=" * 18, name, "=" * 18)
    for n in (1, 2):
        cnt = len(data[(n, "bare")])
        print("-- n=%d (样本 %d) --" % (n, cnt))
        # 均值
        for form in FORMS:
            v = data[(n, form)]
            mce = sum(x[0] for x in v) / len(v)
            hit = sum(x[1] for x in v) / len(v) * 100
            print("   %-6s meanCE=%.4f top1=%.2f%%" % (form, mce, hit))
        # 配对 vs bare
        b = data[(n, "bare")]
        for form in FORMS[1:]:
            v = data[(n, form)]
            dce = [v[i][0] - b[i][0] for i in range(min(len(v), len(b)))]
            dh = [v[i][1] - b[i][1] for i in range(min(len(v), len(b)))]
            wins = sum(1 for d in dce if d < -1e-9)
            losses = sum(1 for d in dce if d > 1e-9)
            ties = len(dce) - wins - losses
            mean_d = sum(dce) / len(dce)
            # 符号检验近似 z（无 tie 二项 p=0.5）
            nw = wins + ties * 0.0
            tot = wins + losses
            if tot > 0:
                z = (wins - tot / 2) / math.sqrt(tot * 0.25)
            else:
                z = 0.0
            hw = sum(1 for d in dh if d > 0)
            hl = sum(1 for d in dh if d < 0)
            print("   %-6s vs bare: ΔCE=%+.4f (win/lose/tie=%d/%d/%d, z=%.2f)  Δtop1=%+.2fpp (hit%d/lose%d)" % (
                form, mean_d, wins, losses, ties, z,
                sum(dh) / len(dh) * 100, hw, hl))
    print()
