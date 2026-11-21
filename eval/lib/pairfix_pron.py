#!/usr/bin/env python3
r"""pairfix 读音证据 + 语料词频（2026-11-16 用户审核准则落地）。

背景（用户指令"通过审核表发现，自动审核主要看例子确定是否通过，未考虑词本身
是否高频…未考虑多音字的情况，字的前两码可以还原出字的读音进行辅助比较，见拼读
双拼规则，从而可以判断高频字低频读音倾向于扣分。总的来说，只有某词码对系统性
分数偏高时，才应该对其修正"）: 例证驱动的审核把「高频字低频读音」对大量压成
存疑——例证只是少量场景抽样，而这类对天然事件少。判定准则升级为三条:
  ① 系统性原则——只有词码对系统性分数偏高才修正（门控 brk=0/双半复制之外,
    审核层看修对率 = 修对/激活窗 与例证方向一致性）;
  ② 词频先验——语料频次高的词例证代表性低, 少量例证不足以定案;
  ③ 读音证据——拼读双拼每字前两码 = 声介+韵调, 可查字表还原带调读音;
    高频字打低频读音 → 文本 CE 不含读音信息、按常用读法给词加分, 结构性
    偏高 → 倾向扣分（通过）。

数据源:
  拼读双拼 2-1-字.xlsx「原始信息」页 = (字, 带调读音, 该读音频次排名, 双拼码)。
  多音字 = 字读音数 >1; 字频 = 该字最小读音频次; 低读 = 此读频次 > 字频 且
  ≥ LOW_READ。排名是分档估计（主读=真实排名, 次读落 500/1000/1500/3500/5000/
  10000 档），只作档位用。缓存 eval/pdsp_pron_table.tsv（--rebuild 重建）。
  语料词频 = article_corpus_shuffled.txt 子串计数（多进程分块 str.count,
  缓存 eval/pairfix_word_corpus_freq.json 增量累积; 中文无词界, 子串计数是
  频次先验的标准近似）。

还原口径: 2 字词 4 码 = 两字各前两码; 单字 3 码 = 前两码声韵（第三码形码,
与读音无关）; ≥3 字词 4 码是 encoder 缩编（AaBaCaZa 类）拆不回整字 →
pypinyin 词级读音兜底（partial 标记, 多音低读判据同式）。

用法:
  python -X utf8 pairfix_pron.py --words 重申,重庆        # 快查
  python -X utf8 pairfix_pron.py --table <tsv> --out <tsv> [--xlsx]
                                                          # 表格追加证据列
  python -X utf8 pairfix_pron.py --table <tsv> --crosstab # 判定×倾向交叉表
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
EVAL = Path(__file__).resolve().parents[1]     # eval 根（2026-11-18 子目录重组）
PROJ = EVAL.parent
SRC = Path(r"D:/OneDrive/typing/拼读双拼")
XLSX_CHAR = SRC / "2-1-字.xlsx"
PRON_CACHE = EVAL / "pdsp_pron_table.tsv"
FREQ_CACHE = EVAL / "pairfix_word_corpus_freq.json"
CORPUS = PROJ / "article_corpus_shuffled.txt"

LOW_READ = 500    # 此读频次 ≥ 此值且非主读 → 低频读音（字表分档起点）
HF_CHAR = 1000    # 字频 ≤ 此值 → 高频字


# ── 字表（读音还原） ──────────────────────────────────────────

def _build_pron_cache():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX_CHAR, read_only=True)
    ws = wb["原始信息"]
    rows = []
    for r in ws.iter_rows(values_only=True):
        ch, rd, rk, cd = r[0], r[1], r[2], r[3]
        if not ch or not cd or not rd:
            continue
        rows.append((ch, rd, int(rk), cd))
    wb.close()
    with open(PRON_CACHE, "w", encoding="utf-8") as f:
        for ch, rd, rk, cd in rows:
            f.write(f"{ch}\t{rd}\t{rk}\t{cd}\n")
    return rows


def load_pron(rebuild=False):
    """→ (cc, cr): cc[(字,码)] = (读音, 读音频次); cr[字] = [(读音, 次数), …]。"""
    if rebuild or not PRON_CACHE.exists():
        rows = _build_pron_cache()
    else:
        rows = []
        for line in PRON_CACHE.read_text(encoding="utf-8").splitlines():
            c = line.split("\t")
            if len(c) == 4:
                rows.append((c[0], c[1], int(c[2]), c[3]))
    cc, cr = {}, {}
    for ch, rd, rk, cd in rows:            # 同码多读（着 zhao1/zhao2='ee'）取低次
        if (ch, cd) not in cc or rk < cc[(ch, cd)][1]:
            cc[(ch, cd)] = (rd, rk)
        cr.setdefault(ch, []).append((rd, rk))
    return cc, cr


# ── 还原与判据 ────────────────────────────────────────────────

def _py_readings(word):
    """≥3 字词兜底: pypinyin 词级带调读音（TONE3, 轻声=5, 与字表同格式）。"""
    from pypinyin import pinyin, Style
    out = []
    for syl in pinyin(word, style=Style.TONE3, errors=lambda x: [None]):
        out.append(syl[0] if syl and syl[0] else None)
    return out


def pair_evidence(word, code, cc, cr):
    """→ dict(recover, flagged, tendency, partial)。

    recover = [(字, 码段|'', 读音, 此读频次, 该字读音数)]（查不到 → 读音 '?');
    flagged = [(字, 读音, 此读频次, 字频)]（低读字）; tendency ∈
    '高频字低读' / '低读' / ''。"""
    partial = False
    if len(word) == 1 and len(code) >= 2:
        segs = [(word, code[:2])]
    elif len(word) == 2 and len(code) == 4:
        segs = [(word[0], code[0:2]), (word[1], code[2:4])]
    else:
        partial = True
        segs = [(ch, "") for ch in word]
    rec, flagged = [], []
    rd_word = _py_readings(word) if partial else None
    for i, (ch, seg) in enumerate(segs):
        if partial:
            rd = rd_word[i] if i < len(rd_word) and rd_word[i] else None
            hit = None
            if rd:
                for r, k in cr.get(ch, []):
                    if r == rd:
                        hit = (r, k)
                        break
            if hit is None:
                rec.append((ch, "", rd or "?", 0, len(cr.get(ch, []))))
                continue
            rd, rk = hit
        else:
            hit = cc.get((ch, seg))
            if hit is None:
                rec.append((ch, seg, "?", 0, len(cr.get(ch, []))))
                continue
            rd, rk = hit
        n_rd = len(cr.get(ch, []))
        rec.append((ch, seg, rd, rk, n_rd))
        if n_rd > 1:
            cmin = min(k for _, k in cr[ch])
            if rk > cmin and rk >= LOW_READ:
                flagged.append((ch, rd, rk, cmin))
    tendency = ""
    if flagged:
        tendency = "高频字低读" if any(cmin <= HF_CHAR for _, _, _, cmin
                                        in flagged) else "低读"
    return {"recover": rec, "flagged": flagged, "tendency": tendency,
            "partial": partial}


def fmt_recover(ev):
    parts = []
    if ev["partial"]:
        parts.append("词读:" + "".join(r[2] for r in ev["recover"]))
    for ch, seg, rd, rk, n_rd in ev["recover"]:
        poly = f"({n_rd}读)" if n_rd > 1 else ""
        parts.append(f"{ch}={rd}·{rk}{poly}")
    return " ".join(parts)


def fmt_flagged(ev):
    return " ".join(f"{ch}={rd}({rk},主{cmin})"
                    for ch, rd, rk, cmin in ev["flagged"])


# ── 语料词频（多进程分块计数 + 缓存） ─────────────────────────

def _count_shard(words, max_len, chunk_chars):
    cnt = Counter()
    carry = ""
    with open(CORPUS, encoding="utf-8") as f:
        while True:
            block = f.read(chunk_chars)
            if not block:
                break
            text = carry + block
            for w in words:
                cnt[w] += text.count(w)   # 零计数也记账（否则缓存永远缺词重扫）
            carry = block[-(max_len - 1):] if max_len > 1 else ""
    return dict(cnt)


def corpus_freq(words, procs=4, refresh=False):
    """word → 语料子串计数（缓存增量; refresh=True 全量重算）。"""
    words = sorted({w for w in words if w})
    cache = {}
    if FREQ_CACHE.exists() and not refresh:
        cache = json.loads(FREQ_CACHE.read_text(encoding="utf-8"))
    todo = [w for w in words if w not in cache]
    if todo and not CORPUS.exists():
        raise SystemExit(f"语料不存在: {CORPUS}")
    if todo:
        max_len = max(len(w) for w in todo)
        n_ch = len(todo)
        if procs > 1 and n_ch > 8:
            import multiprocessing as mp
            shards = [todo[i::procs] for i in range(procs)]
            with mp.Pool(procs) as pool:
                outs = pool.starmap(_count_shard,
                                    [(s, max_len, 1 << 22) for s in shards])
            for o in outs:
                cache.update(o)
        else:
            cache.update(_count_shard(todo, max_len, 1 << 22))
        FREQ_CACHE.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True),
                              encoding="utf-8")
    return {w: cache.get(w, 0) for w in words}


# ── CLI ───────────────────────────────────────────────────────

def enrich_table(table, out, do_xlsx, procs, rebuild):
    rows = [l.rstrip("\r") for l in
            Path(table).read_text(encoding="utf-8-sig").split("\n") if l.strip()]
    hdr = rows[0].split("\t")
    data = [r.split("\t") for r in rows[1:]]
    cc, cr = load_pron(rebuild)
    freq = corpus_freq([r[0] for r in data], procs=procs)
    extras = ["语料频次", "读音还原", "多音低读", "读音倾向"]
    have = {h: i for i, h in enumerate(hdr)}
    ct = Counter()
    for i, c in enumerate(data):
        ev = pair_evidence(c[0], c[1], cc, cr)
        vals = [str(freq.get(c[0], 0)), fmt_recover(ev), fmt_flagged(ev),
                ev["tendency"]]
        ct[ev["tendency"] or "常规"] += 1
        if "读音还原" in have:          # 已带证据列 → 原地更新
            for name, v in zip(extras, vals):
                c[have[name]] = v
        else:
            c.extend(vals)
    out_rows = ["\t".join(hdr + extras if "读音还原" not in have else hdr)]
    for c in data:
        out_rows.append("\t".join(c))
    Path(out).write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    print(f"已写: {out}（{len(data)} 行 · 追加 {extras}）")
    print(f"读音倾向分布: {dict(ct)}")
    if do_xlsx:
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # lib/
        import audit_xlsx
        audit_xlsx.export_file(Path(out))


def crosstab(table, procs, rebuild):
    rows = [l.rstrip("\r") for l in
            Path(table).read_text(encoding="utf-8-sig").split("\n") if l.strip()]
    hdr = rows[0].split("\t")
    data = [r.split("\t") for r in rows[1:]]
    iv = hdr.index("判定") if "判定" in hdr else None
    cc, cr = load_pron(rebuild)
    freq = corpus_freq([r[0] for r in data], procs=procs)
    grp_ct = Counter()
    fix_med = {}
    i_fix = hdr.index("合计修对") if "合计修对" in hdr else None
    for c in data:
        ev = pair_evidence(c[0], c[1], cc, cr)
        hf = any(cmin <= HF_CHAR for _, _, _, cmin in ev["flagged"])
        grp = ("高频字低读" if hf else ("低读(字低频)" if ev["flagged"]
                                        else "常规"))
        grp_ct[(grp, c[iv] if iv is not None else "?")] += 1
        if i_fix is not None:
            fix_med.setdefault((grp, c[iv] if iv is not None else "?"),
                               []).append(int(c[i_fix]))
    print(f"{'组':<10} {'判定':<4} {'对数':>4} {'修对中位':>6}")
    for grp in ("高频字低读", "低读(字低频)", "常规"):
        for v in ("通过", "存疑", "反对", "?"):
            n = grp_ct.get((grp, v), 0)
            if not n:
                continue
            fs = sorted(fix_med[(grp, v)])
            med = fs[len(fs) // 2] if fs else "-"
            print(f"{grp:<10} {v:<4} {n:>4} {med:>6}")
    if freq:
        zero = [w for w, n in freq.items() if n == 0]
        print(f"语料频次为 0 的词: {len(zero)} / {len(freq)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", help="逗号分隔 快查")
    ap.add_argument("--table", help="审核/复核表 tsv")
    ap.add_argument("--out", help="enrich 输出路径（默认 <table>_pron.tsv）")
    ap.add_argument("--xlsx", action="store_true", help="同时出 xlsx")
    ap.add_argument("--crosstab", action="store_true", help="判定×倾向交叉表")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true", help="重建字表缓存")
    a = ap.parse_args()
    if a.words:
        _, cr = load_pron(a.rebuild)
        for w in a.words.split(","):
            print(w, "→", ", ".join(f"{rd}({k})" for rd, k in cr.get(w, [])))
        return
    if a.table:
        out = a.out or str(Path(a.table).with_name(
            Path(a.table).stem + "_pron.tsv"))
        if a.crosstab:
            crosstab(a.table, a.procs, a.rebuild)
        else:
            enrich_table(a.table, out, a.xlsx, a.procs, a.rebuild)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
