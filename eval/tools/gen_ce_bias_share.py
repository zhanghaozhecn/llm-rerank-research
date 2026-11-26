#!/usr/bin/env python3
"""生成字-音节 CE 偏置表（2026-11-26 用户定案：T1 = log 读音占比，λ=1）。

两个口径（输出格式相同，**写进各方案目录的 `llm_rerank/<方案>/llm_pairfix.tsv`
——与 (词,码) 偏置同一个表**，2026-11-26 用户定案：一个方案要么整句要么
字词，同格式两种键共用一表，按候选文本长度分流）：
  --mode pinyin  全拼类（luna 等）：码 = 无调拼音，占比直接用
                 `eval/char_readings.tsv` 的 P(拼音|字)。
  --mode pdsp    拼读双拼：码 = 声介+韵调。读音表 `2-1-字.xlsx`「原始信息」
                 （字, 带调读音, 排名, 双拼码）给带调读音→码；把无调占比按
                 「该无调读音下有几个码」**均分**（只用万象占比这一可靠数据，
                 不拿 500/5000 分档反推音调比例）。同码多个读音累加。

输出行：`字<TAB>码<TAB>log(占比)`（**≤0，只打折不加分**；码是单字覆盖的
那段输入码，3 码 = 取前 2 码，1 码简码不输出 = 不折扣）。
表里没有的 (字,码) → 运行期不加修正（默认占比 1）。
同表共存：`--out` 指到 llm_pairfix.tsv 时保留原有**词键行**（多字词开头的
行原样保留），只替换/追加单字行。

用法：
  python eval/tools/gen_ce_bias_share.py --mode pinyin \
      --out "%APPDATA%/Rime/llm_rerank/luna_pinyin/llm_pairfix.tsv"
  python eval/tools/gen_ce_bias_share.py --mode pdsp \
      --out "%APPDATA%/Rime/llm_rerank/pdsp/llm_pairfix.tsv"
  --out 省略 = 只打印统计（不写文件）
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
READINGS = EVAL / "char_readings.tsv"
XLSX = Path(r"D:/OneDrive/typing/拼读双拼/2-1-字.xlsx")


def load_readings(path=READINGS):
    tab = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) == 3:
                try:
                    tab[a[0]][a[1]] = float(a[2])
                except ValueError:
                    pass
    return tab


def strip_tone(reading):
    """de5 / di2 / lüe4 → de / di / lve（与读音表同口径）。"""
    r = reading.strip().lower().replace("ü", "v").replace("u:", "v")
    r = "".join(ch for ch in r if not ch.isdigit())
    return r


def gen_pinyin(tab):
    out = {}
    for ch, rs in tab.items():
        for syl, share in rs.items():
            if share <= 0.0:
                continue
            d = math.log(share)
            if d >= -0.005:       # 占比 ≥99.5%：log ≈ 0，不占表
                continue
            out[(ch, syl)] = d    # 只保留真折扣（≤ -0.005）
    return out


def gen_pdsp(tab, xlsx=XLSX):
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["原始信息"]
    # (字, 无调读音) → [(码, 1/排名权重)]：同一无调读音的几个声调码之间，
    # 用 xlsx 「该读音频次排名」的倒数分配无调占比（排名是分档估计，只作
    # 档位用；无调占比本身来自万象，可靠）。不同无调读音之间直接各用各的
    # 占比，不做任何拆分。
    grp = defaultdict(list)
    rows = 0
    for row in ws.iter_rows(values_only=True):
        if not row or not row[0] or not row[1] or not row[3]:
            continue
        ch, reading, code = str(row[0]), str(row[1]), str(row[3])
        if len(ch) != 1 or len(code) != 2:
            continue
        try:
            rank = max(float(row[2]), 1.0)
        except (TypeError, ValueError):
            rank = 10000.0
        grp[(ch, strip_tone(reading))].append((code, 1.0 / rank))
        rows += 1
    out = {}
    for (ch, syl), items in grp.items():
        share = tab.get(ch, {}).get(syl)
        if share is None:
            # 兜底：`char_readings` 没有该读音的 (字,码)——引擎在 2 码窗里
            # 通常不会给出这种候选（实测仅个位数简码情形），按「1/该读音
            # 码数」保守折扣，宁可少扣不可漏扣（2026-11-26 用户定案）。
            k = len({code for code, _ in items})
            for code, _ in items:
                out.setdefault((ch, code), 1.0 / k)
            continue
        wsum = sum(w for _, w in items)
        for code, w in items:
            out[(ch, code)] = out.get((ch, code), 0.0) + share * w / wsum
    # 门槛：占比 ≥99.5%（|log|≤0.005）视为单读音不折扣；
    # **只打折不加分**：值下钳到 0（同一 (字,码) 汇总两个音节时占比可 >1），
    # 钳成 0 的行本身无信息，丢掉。
    out = {k: min(0.0, math.log(v)) for k, v in out.items()
           if v > 1e-6 and math.log(v) < -0.005}
    return out, rows, len(grp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pinyin", "pdsp"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--readings", default=str(READINGS))
    ap.add_argument("--xlsx", default=str(XLSX))
    a = ap.parse_args()
    tab = load_readings(a.readings)
    if a.mode == "pinyin":
        out = gen_pinyin(tab)
        extra = f"chars={len(tab)}"
    else:
        out, rows, pairs = gen_pdsp(tab, Path(a.xlsx))
        extra = f"xlsx rows={rows} (字,读音)={pairs}"
    print(f"[{a.mode}] {extra} → keys={len(out)}")
    vals = sorted(out.values())
    if vals:
        print(f"  log share: min={vals[0]:.3f} median={vals[len(vals) // 2]:.3f} "
              f"max={vals[-1]:.3f}")
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 保留原有**词键行**（首字段是多字节 UTF-8 汉字 = 多字词）与注释；
        # 单字行（与字-音节键同形）一律由本次生成结果替换。
        kept = []
        if p.exists():
            for line in open(p, encoding="utf-8"):
                body = line.rstrip("\n\r")
                if not body or body.startswith("#"):
                    continue
                head = body.split("\t", 1)[0]
                # 词键 = 多字词（>1 汉字）；单字词键（如旧的「与 vkh -5」）
                # 是旧机制的单字锁，字-音节行已覆盖，不保留
                if len(head) > 1 and len(head.encode("utf-8")) > 3:
                    kept.append(body)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("# CE 偏置（统一表）：词键 `词<TAB>码<TAB>有符号分数` +"
                    " 字-音节键 `字<TAB>码<TAB>log 占比（≤0，只打折）`\n")
            for line in kept:
                f.write(line + "\n")
            for (ch, code), d in sorted(out.items()):
                f.write(f"{ch}\t{code}\t{d:.6f}\n")
        print(f"→ {p} ({p.stat().st_size} B；保留词键 {len(kept)} 行 + "
              f"字-音节 {len(out)} 行)")


if __name__ == "__main__":
    main()
