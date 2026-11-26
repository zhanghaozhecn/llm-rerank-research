#!/usr/bin/env python3
"""生成「音节 → 整读字串」表（2026-11-26 用户定案①：整读分层 + 偏置键解析）。

一个方案一份 `llm_rerank/<方案>/syll_chars.tsv`，两版 filter 共用，两件事：

  ① **单字整读判定**（显示序分层）：单字候选 span 那段码是不是该字的**完整
     读音**？是 → tier 0（整读）；不是 → tier 1（全拼完成匹配 / 双拼简码 /
     拼写纠错的前缀简拼），显示序沉到整读之后。全拼打 `s`/`sh` 这类非音节
     前缀时表里没有键 → 全部 tier 1 → 不重排（缩简习惯不受影响）。
  ② **双拼「音 + 形」偏置键解析**：3/4 码单字 = 2 码音 + 1~2 码形；运行期沿
     span 从长到短找「表里确实含这个字」的前缀，命中即取该前缀作偏置键。
     （旧实现硬编码 `span 长度 == 3 取前 2`，4 码静默失效，且是全拼/双拼不
     分的猜测式规则。）

数据源 = **引擎实际词库**（与引擎码空间严格一致）+ 研究读音表补简繁转换字：
  --mode pinyin  --dict <wanxiang.dict.yaml>  ∪ eval/char_readings.tsv
  --mode pdsp    --dict <pdsp.dict.yaml>      ∪ 2-1-字.xlsx「原始信息」

输出：`音节<TAB>字字字…`（去重、按码排序；不写字频、不写分数）。
表缺失 / 不读 = 运行期退回旧行为（显示序不动、偏置走 `span` / `3 码取前 2`）。

用法：
  python eval/tools/gen_syll_chars.py --mode pinyin \
      --dict "%APPDATA%/Rime/wanxiang.dict.yaml" \
      --out "%APPDATA%/Rime/llm_rerank/luna_pinyin/syll_chars.tsv"
  python eval/tools/gen_syll_chars.py --mode pdsp \
      --dict "%APPDATA%/Rime/pdsp.dict.yaml" \
      --out "%APPDATA%/Rime/llm_rerank/pdsp/syll_chars.tsv"
  --out 省略 = 只打印统计（不写文件）。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
READINGS = EVAL / "char_readings.tsv"
XLSX = Path(r"D:/OneDrive/typing/拼读双拼/2-1-字.xlsx")


def parse_dict(path, mode):
    """引擎词库 → {音节码: set(字)}。只取单字行，码必须是该方案的音节码。"""
    out = defaultdict(set)
    rows = 0
    started = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not started:
                # 跳过 yaml 头（到 `...` 为止），之后才是词条
                if line.strip() == "...":
                    started = True
                continue
            if not line or line[0] == "#":
                continue
            p = line.split("\t")
            if len(p) < 2 or len(p[0]) != 1:
                continue
            code = p[1]
            if mode == "pdsp":
                if len(code) != 2:      # 只取 2 码音（3 码 = 音 + 形）
                    continue
            else:
                if not code.isalpha() or len(code) > 8:
                    continue            # 单字行恒单音节；防御多音节/怪码
            out[code].add(p[0])
            rows += 1
    return out, rows


def add_pinyin_readings(out, path=READINGS):
    """并列补 `char_readings.tsv`（万象 jichu+shici 的读音分布）。"""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 2 and len(a[0]) == 1 and a[1].isalpha():
                out[a[1]].add(a[0])
                n += 1
    return n


def add_pdsp_xlsx(out, xlsx=XLSX):
    """并列补 `2-1-字.xlsx`「原始信息」（字, 带调读音, 排名, 双拼码）。"""
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    n = 0
    for row in wb["原始信息"].iter_rows(values_only=True):
        if not row or not row[0] or not row[3]:
            continue
        ch, code = str(row[0]), str(row[3])
        if len(ch) == 1 and len(code) == 2:
            out[code].add(ch)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pinyin", "pdsp"], required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--readings", default=str(READINGS))
    ap.add_argument("--xlsx", default=str(XLSX))
    a = ap.parse_args()

    out, rows = parse_dict(a.dict, a.mode)
    if a.mode == "pinyin":
        n = add_pinyin_readings(out, Path(a.readings))
        extra = f"dict rows={rows} + readings rows={n}"
    else:
        n = add_pdsp_xlsx(out, Path(a.xlsx))
        extra = f"dict rows={rows} + xlsx rows={n}"

    lines = []
    for code in sorted(out):
        chars = "".join(sorted(out[code]))
        lines.append(f"{code}\t{chars}")
    nchar = sum(len(s) for s in out.values())
    print(f"[{a.mode}] {extra} → 音节 {len(out)} / 字-音节 {nchar}")
    lens = sorted(len(s) for s in out.values())
    if lens:
        print(f"  每音节字数 min={lens[0]} median={lens[len(lens) // 2]} "
              f"max={lens[-1]}；最长行 {max(len(x) for x in lines)} B")
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("# 音节 → 整读字串（2026-11-26 用户定案①）：单字候选 span 命中"
                    "此表且含该字 = 整读（tier 0）；否则 = 前缀简拼（tier 1）。\n")
            f.write("# 同时用于双拼「音+形」偏置键解析（沿 span 从长到短取命中前缀）。"
                    "由 eval/tools/gen_syll_chars.py 生成，勿手改。\n")
            for x in lines:
                f.write(x + "\n")
        print(f"→ {p} ({p.stat().st_size} B)")


if __name__ == "__main__":
    main()
