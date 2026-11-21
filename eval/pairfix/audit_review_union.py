#!/usr/bin/env python3
"""pairfix 审核表全量机械审查（通用版, 按时代 tag 参数化; 取代
audit_review_union_05b.py——默认 --tag 05b 时行为与旧版一致）。

程序侧审查项（语境合理性交子智能体; 大陆规范裁定走人工+S7 变体清单）:
  M1 行完整性: 19 列、词/码非空、λ=5、判定列未填
  M2 词典同码: 词与被挤目标各词在 pdsp_dict 同码（多字词 4 码 / 单字 3 码口径,
     按证据域标签分流）; 用户独有对无例证时被挤目标可空 → 跳过该项
  M3 双半门: wiki 行 = cf修对偶>0 且 cf修对奇>0 且 两半改错 = 0（G3 双半复制）;
     用户行 = 偶(E)修对>0（E∧evA∧evB 三重复制, 奇半仅旁证）——两域门制不同
  M4 合计一致性: 合计修对 = 偶+奇
  M5 来源一致性: 来源=wiki 的行与 pairfix_pool_cfrep_<tag>.tsv 完全一致
  M6 v2旧态核对: wiki 行与 pairfix_pool_ratio_v2.tsv 比对（留存/新增）
  M7 疑似异形词变体: 罚词与被挤目标命中 pairfix_variants.tsv 同一对变体 →
     标记"走大陆规范裁定"（advisory, 不算失败项）
  M8 读音证据一致性（2026-11-16 审核准则; 表含"读音还原"列时启用）:
     重算 pairfix_pron 读音还原/多音低读/读音倾向 与表内逐格比对;
     语料频次须为非负整数（语料计数不重算, 只查格式）

用法: python audit_review_union.py [--tag 05b]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std
from derive_pairfix_ratio import load_w2codes

sys.stdout.reconfigure(encoding="utf-8")
EVAL = corpus_std.EVAL
SOURCE_DIR = Path(r"D:/OneDrive/typing/拼读双拼")  # 审核表工作源目录（既有约定）
V2 = EVAL / "pairfix_pool_ratio_v2.tsv"  # 跨时代对比基准
VARIANTS = EVAL / "pairfix_variants.tsv"

HDRV = 19  # 基础列（追加证据列后的实际列数按表头行推导, 见 main）


def load_pairs(p):
    t = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        c = line.rstrip("\r").split("\t")
        if len(c) >= 2 and c[0] and c[1] and c[0] != "词":
            t.add((c[0], c[1]))
    return t


def load_variants(p=EVAL / "pairfix_variants.tsv"):
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) >= 2 and c[0] and c[1]:
            out.append((c[0], c[1]))
    return out


def variant_hit(w1, targets, variants):
    """罚词 w1 与任一被挤目标命中同一对变体（子串包含, 双向）→ 返回该对。"""
    for a, b in variants:
        for w2 in targets:
            if (a in w1 and b in w2) or (b in w1 and a in w2):
                return (a, b)
    return None


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="05b", help="时代 tag（默认 05b 兼容旧行为）")
    ap.add_argument("--src", default=None, help="覆盖审核表路径（默认按 tag 取 union 表）")
    ap.add_argument("--prev", default=None,
                    help="v2旧态基准 = 上代 audited 表 tag（缺省 = ratio_v2, 兼容旧行为; "
                         "derive_pairfix_full 的 v2旧态按上代 audited 算, 校验须同基准）")
    args = ap.parse_args()
    tag = args.tag

    REVIEW = (Path(args.src) if args.src
              else SOURCE_DIR / f"pairfix_review_union_{tag}.tsv")
    G3 = EVAL / f"pairfix_pool_cfrep_{tag}.tsv"

    rows = []
    for line in REVIEW.read_text(encoding="utf-8-sig").split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        rows.append(line.split("\t"))
    hdr, data = rows[0], rows[1:]
    n_col = len(hdr)
    pron_on = "读音还原" in hdr
    cc = cr = None
    i_cfreq = i_rec = i_flag = i_tend = -1
    if pron_on:
        import pairfix_pron
        cc, cr = pairfix_pron.load_pron()
        i_cfreq, i_rec, i_flag, i_tend = (
            hdr.index(x) for x in ("语料频次", "读音还原", "多音低读", "读音倾向"))
    print(f"行数: {len(data)}（含头共 {len(rows)} 行）  tag={tag}"
          f"  列数={n_col}" + ("（含读音证据列 M8）" if pron_on else ""))

    w2codes, _ = load_w2codes()
    # 多字词 4 码口径 + 单字 3 码口径
    _, wc_big = corpus_std.load_pdsp_dict()
    cw_big, _ = corpus_std.load_pdsp_dict()
    cw_sc, wc_sc = corpus_std.load_pdsp_dict_single3()

    g3 = load_pairs(G3)
    v2 = load_pairs(V2)
    if args.prev:
        prev_ref = load_pairs(SOURCE_DIR / f"pairfix_review_union_{args.prev}_audited.tsv")
        print(f"M6 基准: {args.prev} audited {len(prev_ref)} 对（--prev）")
    else:
        prev_ref = v2
        print(f"M6 基准: pairfix_pool_ratio_v2 {len(v2)} 对（缺省）")
    variants = load_variants()
    print(f"变体清单: {len(variants)} 对（{VARIANTS.name}）")

    issues = defaultdict(list)   # (idx, pair) -> [M1..M7]
    stats = defaultdict(int)
    for i, c in enumerate(data):
        pair = (c[0], c[1])
        tag_ = f"#{i + 1} {pair[0]}@{pair[1]}"
        # M1
        if len(c) != n_col:
            issues[tag_].append(f"M1 列数 {len(c)} != {n_col}")
            continue
        if not pair[0] or not pair[1]:
            issues[tag_].append("M1 词/码空")
        if c[2] != "5":
            issues[tag_].append(f"M1 λ={c[2]} != 5")
        if c[3].strip():
            issues[tag_].append(f"M1 判定列已有值 {c[3]!r}")
        if not c[8]:
            issues[tag_].append("M1 合计修对空")
        # M2 词典同码
        dom = c[11]
        sc_dom = "单字" in dom
        cw = cw_sc if sc_dom else cw_big
        if pair[1] not in cw:
            issues[tag_].append(f"M2 码 {pair[1]} 不在{'单字3码' if sc_dom else '多字4码'}词典")
        elif pair[0] not in cw[pair[1]]:
            issues[tag_].append(f"M2 词 {pair[0]} 不在码 {pair[1]} 候选")
        wants = []
        if c[10]:
            for item in c[10].split("、"):
                w, _, _n = item.partition("×")
                wants.append(w)
        for w in wants:
            if pair[1] not in cw:
                break
            if w not in cw[pair[1]]:
                issues[tag_].append(f"M2 被挤目标 {w} 不在码 {pair[1]} 候选")
        if not wants and c[17] == "wiki":
            issues[tag_].append("M2 wiki 行被挤目标空")
        # M3 门条件（两域门制不同）:
        #   wiki 行 = 双半复制（偶>0 且 奇>0, 改错两半均 0, derive_pairfix_cfrep G3）
        #   用户行 = 偶半反事实 E ∧ evA ∧ evB 三重复制（derive_pairfix_userchain）；
        #     表中"奇"列 = 奇半旁证（O 账, 非门条件）→ 只要求偶(=E)修对>0、改错 0
        try:
            fe, be, fo, bo = (int(c[4]), int(c[5]), int(c[6]), int(c[7]))
        except ValueError:
            issues[tag_].append("M3 cf 计数非整数")
            continue
        is_user = c[17] == "用户域"
        if is_user:
            if not (fe > 0 and fo >= 0):
                issues[tag_].append(f"M3 用户门 偶(E)修对{fe} 应>0")
        elif not (fe > 0 and fo > 0):
            issues[tag_].append(f"M3 双半不满足 偶修对{fe} 奇修对{fo}")
        if be or bo:
            issues[tag_].append(f"M3 改错非零 偶{be} 奇{bo}")
        # M4
        try:
            if int(c[8]) != fe + fo:
                issues[tag_].append(f"M4 合计{c[8]} != 偶{fe}+奇{fo}")
        except ValueError:
            pass
        # M7 疑似异形词变体（advisory, 不入失败项）
        hit = variant_hit(pair[0], wants, variants)
        if hit:
            issues[tag_].append(
                f"M7 疑似异形词变体 {hit[0]}↔{hit[1]} → 走大陆规范裁定"
                "（罚台译/非规范形=通过, 罚大陆规范形=反对）")
        # M8 读音证据一致性（2026-11-16 审核准则; 证据列存在时启用;
        # 语料频次只查格式——重算要扫 1.7GB 语料, 不在逐行审查里做）
        if pron_on:
            if not (c[i_cfreq].isdigit() or c[i_cfreq] == ""):
                issues[tag_].append(f"M8 语料频次非整数 {c[i_cfreq]!r}")
            ev = pairfix_pron.pair_evidence(pair[0], pair[1], cc, cr)
            for name, got, want in (
                    ("读音还原", c[i_rec], pairfix_pron.fmt_recover(ev)),
                    ("多音低读", c[i_flag], pairfix_pron.fmt_flagged(ev)),
                    ("读音倾向", c[i_tend], ev["tendency"])):
                if got != want:
                    issues[tag_].append(
                        f"M8 {name}不符 {got!r} != {want!r}")
        stats["wiki" if c[17] == "wiki" else "user"] += 1

    # M5/M6 集合级
    table_pairs = {(c[0], c[1]) for c in data if len(c) >= 2}
    # wiki 侧含混合域行（来源=wiki+user 的对在 G3 里也算 wiki 行）
    wiki_pairs = {(c[0], c[1]) for c in data
                  if len(c) >= 19 and c[17].startswith("wiki")}
    if wiki_pairs != g3:
        for p in wiki_pairs - g3:
            issues[f"{p[0]}@{p[1]}"].append("M5 wiki 行不在 G3 表")
        for p in g3 - wiki_pairs:
            issues[f"{p[0]}@{p[1]}"].append("M5 G3 表对不在审核表 wiki 行")
    for c in data:
        if len(c) < 19:
            continue
        p = (c[0], c[1])
        if c[17].startswith("wiki"):
            want = "留存" if p in prev_ref else "新增"
            if c[9] != want:
                issues[f"{p[0]}@{p[1]}"].append(
                    f"M6 v2旧态 {c[9]!r} != 实算 {want!r}")

    print(f"来源分布: wiki {stats['wiki']} / 用户域 {stats['user']}")
    print(f"词典核对口径: 多字4码 {len(cw_big)} 码位 / 单字3码 {len(cw_sc)} 码位")
    n_bad = sum(1 for msgs in issues.values() if any(not m.startswith("M7") for m in msgs))
    n_m7 = sum(1 for msgs in issues.values() if any(m.startswith("M7") for m in msgs))
    print(f"\n机械审查结果: {'全部通过' if n_bad == 0 else f'{n_bad} 行有问题'}"
          f"（M7 变体提示 {n_m7} 行, advisory 不算失败）")
    for tg, msgs in sorted(issues.items(), key=lambda kv: [
            int(kv[0].split()[0][1:]) if kv[0].startswith('#') else 10 ** 9]):
        for m in msgs:
            print(f"  {tg}: {m}")

    # 汇总分布（供报告）
    import collections
    fix = collections.Counter()
    for c in data:
        if len(c) == n_col:
            fix[int(c[8])] += 1
    print("\n合计修对分布（对数）:", dict(sorted(fix.items())))
    zero_target = sum(1 for c in data if len(c) == n_col and not c[10])
    print(f"被挤目标空行: {zero_target}")
    if pron_on:
        tend = collections.Counter(c[i_tend] or "常规"
                                   for c in data if len(c) == n_col)
        print(f"读音倾向分布: {dict(tend)}"
              f"（倾向扣分 = 高频字低读/低读, 结构性偏高证据——见审核准则）")


if __name__ == "__main__":
    main()
