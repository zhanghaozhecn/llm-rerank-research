#!/usr/bin/env python3
r"""审核表 TSV ↔ XLSX 双向转换（手机编辑支持）。

export: OneDrive audited TSV（19 列审核表）→ 同目录同名 .xlsx
  - 样式按 xlsx 技能设计系统（templates/base.py, 缺失时退回内置朴素样式）
  - 冻结 词/码 两列 + 表头；判定列下拉（通过/存疑/反对）+ 三色条件格式
  - 「说明」sheet：判定语义 + 19 列含义 + 回转部署流程
import: 手机上改完判定列的 .xlsx → 覆盖写回 TSV（utf-8, 行序/列序不变；
  数值列还原为字符串——与 TSV 原格式逐格等价）

用法:
  python audit_xlsx.py export [--tag 05b]
  python audit_xlsx.py import --xlsx <file.xlsx> [--tag 05b] [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path

ILLEGAL_RE = re.compile(r"[\000-\010\013\014\016-\037\177]")

sys.stdout.reconfigure(encoding="utf-8")
EVAL = Path(__file__).resolve().parents[1]     # eval 根（2026-11-18 子目录重组）
SOURCE_DIR = Path(r"D:/OneDrive/typing/拼读双拼")

# xlsx 技能设计系统（本机维护脚本; 技能缺失时退回内置样式）
SKILL_TPL = Path(r"C:/Users/Administrator/.zcode/cli/plugins/cache/"
                 r"zcode-plugins-official/spreadsheets/0.2.2/skills/xlsx/templates")
# 数值列按表头名定位（2026-11-16 起 19 列后可追加证据列, 列号不再写死）
NUM_HDR = ("λ", "cf修对偶", "cf改错偶", "cf修对奇", "cf改错奇", "合计修对",
           "词热度", "码位流量", "语料频次")
VERDICTS = ("通过", "存疑", "反对")

COL_MEANING = [
    ("词", "罚词——被判该让位的词"), ("码", "该词的码位（4 码/3 码单字）"),
    ("λ", "预置降分幅度（缺省 5）"), ("判定", "通过 / 存疑 / 反对 ←【手机上编辑这列】"),
    ("cf修对偶", "偶半反事实：罚词让位后修对次数"), ("cf改错偶", "偶半改错次数（应恒 0）"),
    ("cf修对奇", "奇半修对次数"), ("cf改错奇", "奇半改错次数（应恒 0）"),
    ("合计修对", "偶+奇"), ("v2旧态", "相对上一代（0.8B）表：留存/新增"),
    ("被挤目标", "让位给谁（want 词×次数）"), ("证据域", "wiki / 用户域（含单字标记）"),
    ("结构", "词结构（多字词/单字/扩样）"), ("词热度", "该词用户累计频次"),
    ("码位流量", "该码位提交流量"), ("例证1", "反事实例证（want=「…」@码）"),
    ("例证2", "反事实例证"), ("来源", "wiki / 用户域"),
    ("备注", "审查理由（[审查]/[大陆规范]/[沿用] 标注）"),
    # 2026-11-16 审核准则追加列（尾部; 出现在表中才展示）
    ("语料频次", "罚词在 38.5 万篇语料的子串计数（词频先验：频次高→例证代表性低）"),
    ("修对率", "合计修对 / 激活窗——系统性偏高程度（准则①）"),
    ("读音还原", "字前两码还原带调读音·读音频次（(N读)=多音字；长词为 pypinyin 词读兜底）"),
    ("多音低读", "低频读音的字=读音(此读频次,主N)——高频字低读=结构性偏高证据"),
    ("读音倾向", "倾向扣分:高频字低读 / 倾向扣分:低读（准则③）= 结构性偏高、倾向通过"),
]

_GUIDE_TPL = [
    "这张表 = pairfix 预置扣分词码对审核表（{tag}, {n} 对）。",
    "判定分布: 通过 {t0} / 存疑 {t1} / 反对 {t2}。已有判定的是审查推荐, 你可改。",
    "判定语义: 通过 = 收录进部署表（预置降分）; 反对 = 不收录; 存疑 = 暂不收录, 留待复核。",
    "审核准则（2026-11-16）: 只有词码对系统性分数偏高才修正——例证只是少量场景抽样;",
    "语料频次高的词例证代表性低, 少量例证不足以定案; 多音字按字前两码还原读音",
    "（拼读双拼规则）, 高频字打低频读音 → 结构性偏高 → 倾向通过（见「读音倾向」列）。",
    "改完把 xlsx 发回电脑, 运行: python audit_xlsx.py import --xlsx <文件> --tag {tag} → 转回 TSV,",
    "再跑: python pairfix_pipeline.py deploy --tag {tag}（筛判定=通过 行, 终检+备份+部署）。",
    "注意: 只改「判定」列; 其余列是审查证据, 改了会在转回时一并写入。",
]


def guide_lines(tag, n, cnt):
    return [s.format(tag=tag, n=n, t0=cnt["通过"], t1=cnt["存疑"],
                     t2=cnt["反对"]) for s in _GUIDE_TPL]


def _try_skill_base():
    try:
        sys.path.insert(0, str(SKILL_TPL))
        import base  # noqa
        return base
    except Exception:
        return None


def read_tsv(p):
    # TSV 禁 splitlines()（\x0b 幻行, 研究文档教训）——只按 \n 切
    rows = [l.rstrip("\r") for l in p.read_text(encoding="utf-8-sig").split("\n")
            if l.strip()]
    return rows[0].split("\t"), [r.split("\t") for r in rows[1:]]


def _as_num(s):
    s = s.strip()
    return int(s) if s.lstrip("-").isdigit() else s


def export_file(tsv, tag=None):
    import re
    base = _try_skill_base()
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import PatternFill, Font, Alignment

    tag = tag or re.sub(r"^pairfix_review_union_|\.tsv$", "", Path(tsv).stem)
    tsv = Path(tsv)
    xlsx = tsv.with_suffix(".xlsx")
    hdr, data = read_tsv(tsv)
    num_cols = {hdr.index(h) for h in NUM_HDR if h in hdr}
    n = len(data)
    cnt = {v: sum(1 for r in data if r[3].strip() == v) for v in VERDICTS}

    wb = Workbook()
    ws = wb.active
    ws.title = "审核表"
    last_col = 1 + len(hdr)          # B 起
    if base:
        base.setup_sheet(ws, title=f"词码对审核表 {tag}（{n} 对 · "
                                   f"通过{cnt['通过']}/存疑{cnt['存疑']}/反对{cnt['反对']}）",
                         last_col=last_col)
        ws["B3"] = f"来源: {tsv.name} · 手机上只改「判定」列（下拉三值）"
        ws["B3"].font = base.font_caption()
    else:  # 朴素回退
        ws["B2"] = f"词码对审核表 {a.tag}（{n} 对）"

    hrow, drow0 = 4, 5
    for ci, h in enumerate(hdr, start=2):
        ws.cell(row=hrow, column=ci, value=h)
    if base:
        base.style_header_row(ws, row_num=hrow, col_start=2, col_end=last_col)
    for i, row in enumerate(data):
        rn = drow0 + i
        for ci, v in enumerate(row, start=2):
            if (ci - 2) in num_cols:
                cell = _as_num(v)
            else:
                cell = ILLEGAL_RE.sub("", v)  # \x0b 幻行类控制字符 openpyxl 拒写
            ws.cell(row=rn, column=ci, value=cell)
        if base:
            base.style_data_row(ws, row_num=rn, col_start=2, col_end=last_col, row_index=i)
            for j in num_cols:  # 数值右对齐
                c = ws.cell(row=rn, column=2 + j)
                c.alignment = base.align_number()

    # 判定下拉 + 三色
    col_v = 2 + hdr.index("判定")
    letter = ws.cell(row=hrow, column=col_v).column_letter
    rng = f"{letter}{drow0}:{letter}{drow0 + n - 1}"
    dv = DataValidation(type="list", formula1=f'"{",".join(VERDICTS)}"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(rng)
    if base:
        for v, fill, font in (("通过", base.CF_POSITIVE_FILL, base.CF_POSITIVE_FONT),
                              ("存疑", base.CF_WARNING_FILL, base.CF_WARNING_FONT),
                              ("反对", base.CF_NEGATIVE_FILL, base.CF_NEGATIVE_FONT)):
            ws.conditional_formatting.add(
                rng, FormulaRule(formula=[f'${letter}{drow0}="{v}"'],
                                 fill=fill, font=font))

    # 冻结 词/码+表头；筛选；列宽行高
    ws.freeze_panes = "D5"
    ws.auto_filter.ref = f"B{hrow}:{ws.cell(row=hrow, column=last_col).column_letter}{drow0 + n - 1}"
    if base:
        base.auto_fit_columns(ws, min_width=8, max_width=32, header_row=hrow, data_start_row=drow0)
        base.auto_fit_row_heights(ws, header_row=hrow, data_start_row=drow0)

    # 说明 sheet
    info = wb.create_sheet("说明")
    info.sheet_view.showGridLines = False
    info["B2"] = "使用说明"
    info["B2"].font = (base.font_title() if base else
                       Font(size=16, bold=True, color="1B2A4A"))
    r = 4
    for line in guide_lines(tag, n, cnt):
        info.cell(row=r, column=2, value=line).font = (base.font_body() if base else Font(size=11))
        r += 1
    r += 1
    info.cell(row=r, column=2, value="列").font = (base.font_header() if base else Font(bold=True))
    info.cell(row=r, column=3, value="含义").font = (base.font_header() if base else Font(bold=True))
    if base:
        base.style_header_row(info, row_num=r, col_start=2, col_end=3)
    for i, (k, v) in enumerate([x for x in COL_MEANING if x[0] in hdr]):
        rr = r + 1 + i
        info.cell(row=rr, column=2, value=k)
        info.cell(row=rr, column=3, value=v)
        if base:
            base.style_data_row(info, row_num=rr, col_start=2, col_end=3, row_index=i)
    info.column_dimensions["B"].width = 12
    info.column_dimensions["C"].width = 60

    wb.properties.creator = "Z.ai"
    wb.save(xlsx)
    print(f"已写: {xlsx}（{n} 行 · 判定分布 通过{cnt['通过']}/存疑{cnt['存疑']}/反对{cnt['反对']}）")


def cmd_export(a):
    tsv = (Path(a.src) if getattr(a, "src", None)
           else SOURCE_DIR / f"pairfix_review_union_{a.tag}_audited.tsv")
    export_file(tsv, None if getattr(a, "src", None) else a.tag)


def cmd_import(a):
    from openpyxl import load_workbook
    tsv = (Path(a.src) if getattr(a, "src", None)
           else SOURCE_DIR / f"pairfix_review_union_{a.tag}_audited.tsv")
    hdr, data = read_tsv(tsv)
    wb = load_workbook(a.xlsx, data_only=True)
    ws = wb["审核表"]
    hrow, drow0 = 4, 5
    # 列序核对（按表头名对位, 防手机端拖动列）
    xhdr = {}
    for ci in range(2, 2 + len(hdr)):
        v = ws.cell(row=hrow, column=ci).value
        xhdr[v] = ci
    missing = [h for h in hdr if h not in xhdr]
    if missing:
        sys.exit(f"xlsx 表头缺列: {missing}")
    rows = []
    for rn in range(drow0, drow0 + len(data)):
        row = []
        for h in hdr:
            v = ws.cell(row=rn, column=xhdr[h]).value
            if v is None:
                v = ""
            v = str(v).strip()
            if v.startswith("="):  # 防公式注入 TSV
                v = "'" + v
            row.append(v)
        if row[0] == "":
            sys.exit(f"xlsx 第 {rn} 行 词 为空——行数/顺序可能被改动, 拒绝写回")
        rows.append(row)
    if len(rows) != len(data):
        sys.exit(f"行数不符: xlsx {len(rows)} != TSV {len(data)}")

    # 写回前对比: 只报告判定列变化
    diff = [(i + 1, data[i][0], data[i][3], rows[i][3])
            for i in range(len(rows)) if data[i][3] != rows[i][3]]
    other = [(i + 1, data[i][0], h) for i in range(len(rows))
             for j, h in enumerate(hdr) if j != 3 and data[i][j] != rows[i][j]]
    print(f"判定列变化 {len(diff)} 行:")
    for i, w, old, new in diff[:30]:
        print(f"  #{i} {w}: {old or '空'} → {new or '空'}")
    if len(diff) > 30:
        print(f"  …共 {len(diff)} 行")
    if other:
        print(f"⚠ 非判定列也有改动 {len(other)} 格: {other[:10]}")
    if a.dry_run:
        print("dry-run: 不写 TSV")
        return
    out = ["\t".join(hdr)] + ["\t".join(r) for r in rows]
    tsv.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"已写回: {tsv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--tag", default="05b")
    e.add_argument("--src", default=None, help="覆盖源 TSV（默认 OneDrive audited 表）")
    i = sub.add_parser("import")
    i.add_argument("--xlsx", required=True)
    i.add_argument("--tag", default="05b")
    i.add_argument("--src", default=None, help="覆盖目标 TSV（默认 OneDrive audited 表）")
    i.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cmd_export(a) if a.cmd == "export" else cmd_import(a)


if __name__ == "__main__":
    main()
