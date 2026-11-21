#!/usr/bin/env python3
"""用户域文章语料抽取（2026-11-14, 用户指令: OneDrive 个人文件拼接作
模拟用户语料）。

扫描 D:/OneDrive 个人目录的 txt/md/docx:
- txt/md: 编码探测（utf-8 → gbk → utf-16）
- docx:  zipfile 解 word/document.xml, <w:t> 按 <w:p> 段落拼接
- 段落过滤: 非空白字符 CJK 占比 >=0.5 且长度 >=20（代码/数据/目录行剔除）
- 文件保留: CJK 字符数 >=200
- 产出: eval/user_corpus_survey.txt（摸底）+ eval/user_articles.txt
  （每行一篇, prep_cohort 的 article_corpus_shuffled 同格式）+ 明细 jsonl

用法: python build_user_corpus.py
"""
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
EVAL = Path(__file__).resolve().parents[1]      # eval 根（2026-11-18 子目录重组）
OUT_ART = EVAL / "user_articles.txt"
OUT_SURVEY = EVAL / "user_corpus_survey.txt"
OUT_DETAIL = EVAL / "user_corpus_files.jsonl"

ROOTS = ["0-重要资料", "1-本科", "2-美国留学", "3-继续学习", "4-工作",
         "5-其他"]
ONE = Path(r"OneDrive（本机）")
MIN_FILE_CJK = 200
MIN_PARA = 20
CJK_RATIO = 0.5
CJK = re.compile(r"[\u4e00-\u9fff]")
VIS = re.compile(r"[^\s]")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def read_txt(p):
    b = p.read_bytes()
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return b.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def read_docx(p):
    try:
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml")
    except Exception:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    paras = []
    for para in root.iter("{%s}p" % NS["w"]):
        segs = [t.text or "" for t in para.iter("{%s}t" % NS["w"])]
        paras.append("".join(segs))
    return "\n".join(paras)


def para_ok(s):
    vis = VIS.findall(s)
    if len(vis) < MIN_PARA:
        return False
    cjk = sum(1 for ch in vis if CJK.match(ch))
    return cjk / len(vis) >= CJK_RATIO


def main():
    stats = {"dir": {}, "files": []}
    articles = []
    for d in ROOTS:
        base = ONE / d
        if not base.is_dir():
            continue
        n_txt = n_docx = n_keep = 0
        chars = 0
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix in (".txt", ".md", ".log"):
                text = read_txt(p)
                n_txt += 1
            elif suffix == ".docx":
                text = read_docx(p)
                n_docx += 1
            else:
                continue
            if not text:
                continue
            paras = [re.sub(r"\s+", " ", s).strip()
                     for s in re.split(r"[\r\n]+", text) if s.strip()]
            good = [s for s in paras if para_ok(s)]
            n_cjk = sum(len(CJK.findall(s)) for s in good)
            if n_cjk < MIN_FILE_CJK:
                continue
            n_keep += 1
            chars += n_cjk
            art = "\n".join(good)
            articles.append((str(p), art, n_cjk))
            stats["files"].append({"path": str(p), "cjk": n_cjk,
                                   "paras": len(good)})
        stats["dir"][d] = {"txt": n_txt, "docx": n_docx, "kept": n_keep,
                           "cjk": chars}
        print(f"{d}: txt={n_txt} docx={n_docx} kept={n_keep} "
              f"cjk={chars}", flush=True)

    # 长文按 ~1500 CJK 字切分（与 wiki 文章量级接近, HanLP 分块友好）
    out = []
    for path, art, n_cjk in articles:
        if n_cjk <= 1600:
            out.append((path, art, n_cjk))
            continue
        paras = art.split("\n")
        cur, cur_n, parts = [], 0, []
        for s in paras:
            n = len(CJK.findall(s))
            if cur_n + n > 1500 and cur:
                parts.append("\n".join(cur))
                cur, cur_n = [], 0
            cur.append(s)
            cur_n += n
        if cur:
            parts.append("\n".join(cur))
        for i, part in enumerate(parts):
            out.append((path, part, len(CJK.findall(part))))

    with open(OUT_ART, "w", encoding="utf-8") as f:
        for path, art, _ in out:
            f.write(art.replace("\n", " ") + "\n")
    with open(OUT_DETAIL, "w", encoding="utf-8") as f:
        for path, art, n in out:
            f.write(json.dumps({"path": path, "cjk": n},
                               ensure_ascii=False) + "\n")

    total = sum(n for _, _, n in out)
    rep = ["=" * 80, "OneDrive 用户域文章语料抽取", "=" * 80]
    for d, s in stats["dir"].items():
        rep.append(f"{d}: txt/md={s['txt']} docx={s['docx']} "
                   f"保留文件={s['kept']} CJK={s['cjk']}")
    rep.append(f"\n切分后文章数: {len(out)}（长文按 ~1500 CJK 字切分）")
    rep.append(f"总 CJK 字数: {total:,}")
    rep.append(f"产出: {OUT_ART.name} / {OUT_DETAIL.name}")
    OUT_SURVEY.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
