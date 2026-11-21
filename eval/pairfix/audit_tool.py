#!/usr/bin/env python3
r"""pairfix 审核工具合并入口（2026-11-18 子目录重组合并，3 脚本 → 1）。

吸收并取代（子命令与原 CLI 一一对应, 参数不变）:
  audit_batches.py → batch   审核表切 N 个批次文件（子智能体语境评审输入）
  audit_merge.py   → merge   批次判定 verdict_*.tsv 回填审核表（只填空判定列）
  audit_deploy.py  → deploy  审核终表 → 部署表 + 备份 + 复制到 %APPDATA%\Rime

共享惯用法收拢: TSV 读取 read_tsv / SOURCE_DIR（OneDrive 拼读双拼）/ EVAL 输出根。
调用方 pairfix_pipeline.py 的 batch/merge/deploy 子命令已改走本工具。

**deploy 有符号格式（2026-11-18 适配落地）**: 审核表 λ 正数 = 降分幅度 →
落地为负分（λ=5 → -5）; 显式 +200（固定首选）/ +100（固定次选）透传;
现行生产表的正分条目全数保留、旧负分由本代审核集取代（同词码对正分冲突
正分胜并报告）——直接跑 deploy 不再抹固顶、不再把降噪写成加分。
"""
import argparse
import hashlib
import os
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import corpus_std

sys.stdout.reconfigure(encoding="utf-8")
EVAL = corpus_std.EVAL
SOURCE_DIR = Path(r"D:/OneDrive/typing/拼读双拼")


def read_tsv(src):
    """审核表 TSV → (hdr, data)（utf-8-sig, 空行跳过, \r 剥除）。"""
    rows = [l.rstrip("\r") for l in src.read_text(encoding="utf-8-sig").split("\n")
            if l.strip()]
    return rows[0].split("\t"), rows[1:]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def audited_path(tag, override):
    return Path(override) if override else SOURCE_DIR / f"pairfix_review_union_{tag}_audited.tsv"


# ─────────────────────────── batch ───────────────────────────

def cmd_batch(a):
    """切批次文件, 列 = 序号+审核表 19 列（取代 audit_batches.py）。"""
    src = audited_path(a.tag, a.src)
    outdir = EVAL / f"audit_{a.tag}_batches"
    outdir.mkdir(exist_ok=True)

    hdr, data = read_tsv(src)
    assert hdr[0] == "词", hdr[0]

    keep = [(i, line) for i, line in enumerate(data)]
    if a.only_pending:
        keep = [(i, line) for i, line in keep
                if not line.split("\t")[3].strip()]
    print(f"源表: {src}  总行 {len(data)}, 参评 {len(keep)}"
          f"（--only-pending={a.only_pending}）")

    N = a.nbatches
    per = (len(keep) + N - 1) // N
    for b in range(N):
        chunk = keep[b * per:(b + 1) * per]
        if not chunk:
            continue
        lines = ["序号\t" + "\t".join(hdr)]
        for i, line in chunk:
            lines.append(f"{i + 1}\t" + line)
        p = outdir / f"batch_{b + 1:02d}.tsv"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(p, len(chunk))
    print("total", len(keep))


# ─────────────────────────── merge ───────────────────────────

def cmd_merge(a):
    """批次判定回填审核表（取代 audit_merge.py）。"""
    tag = a.tag
    src = audited_path(tag, a.src)
    dst = Path(a.dst) if a.dst else src

    batch = EVAL / f"audit_{tag}_batches"
    verdicts = {}
    for vp in sorted(batch.glob("verdict_*.tsv")):
        for line in vp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            c = line.split("\t")
            if len(c) < 3:
                print(f"  {vp.name} 坏行: {line!r}")
                continue
            n, v, r = int(c[0]), c[1].strip(), "\t".join(c[2:]).strip()
            if v not in ("通过", "存疑", "反对"):
                print(f"  {vp.name} 序号{n} 非法判定 {v!r}")
                continue
            verdicts[n] = (v, r)
    print(f"判定覆盖: {len(verdicts)} 行（{batch}\\verdict_*.tsv）")

    hdr, data = read_tsv(src)
    bad = [n for n in verdicts if not (1 <= n <= len(data))]
    if bad:
        sys.exit(f"verdict 序号越界: {bad[:10]}")

    cnt = {"通过": 0, "存疑": 0, "反对": 0}
    filled, skipped, still_empty = 0, 0, []
    out = ["\t".join(hdr)]
    for i, line in enumerate(data):
        c = line.split("\t")
        n = i + 1
        if n in verdicts and len(c) >= 19:
            v, r = verdicts[n]
            if c[3].strip():
                skipped += 1
                print(f"  跳过 #{n} {c[0]}@{c[1]}（判定列已有 {c[3]!r}, 不覆盖）")
            else:
                c[3] = v
                c[18] = (c[18] + " " if c[18] else "") + f"[审查{v}] {r}"
                cnt[v] += 1
                filled += 1
        if len(c) >= 19 and not c[3].strip():
            still_empty.append(n)
        out.append("\t".join(c))

    print(f"合并: 新填 {filled}（通过 {cnt['通过']} / 存疑 {cnt['存疑']} / 反对 {cnt['反对']}）"
          f"，跳过已有判定 {skipped}")
    if still_empty:
        print(f"仍留空 {len(still_empty)} 行: {still_empty[:20]}"
              + (" …" if len(still_empty) > 20 else "")
              + " ← 未评审完, 不得进入部署")
    if a.dry_run:
        print("dry-run: 不写文件")
        return
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"已写: {dst}")


# ─────────────────────────── deploy ───────────────────────────

def cmd_deploy(a):
    """审核终表 → 有符号部署表生成器（取代 audit_deploy.py）。

    2026-11-18 有符号格式适配（用户定案口径 = 迁移件 migrate_pairfix_signed）:
      审核表 λ 列正数 = 降分幅度 → 落地为负分（λ=5 → -5）;
      显式 +200（固定首选）/ +100（固定次选）/ 负数原样透传。
    合并语义（取代"整表覆盖"）: 现行生产表的正分条目（固顶/固次, 用户
    09876 写入或手动编辑）**全数保留**; 旧负分条目由新审核集取代（降噪
    对只来自审查流）; 同词码对正分冲突时正分胜并报告。
    """
    tag = a.tag
    src = audited_path(tag, a.src)
    out_eval = EVAL / f"pairfix_deploy_pass_{tag}.tsv"
    appdata = os.environ.get("APPDATA")
    if not appdata:
        sys.exit("APPDATA 环境变量不存在（非 Windows 环境?）")
    deploy = Path(appdata) / "Rime" / "llm_pairfix.tsv"

    hdr, data = read_tsv(src)
    assert hdr[0] == "词" and len(data[0].split("\t")) >= 19, \
        (hdr[0], "非审核表（基础 19 列起）")

    # 筛通过行（λ 列原样保留, 后面统一有符号化）
    body, skipped = [], []
    for line in data:
        c = line.split("\t")
        if len(c) < 19:
            skipped.append((c[0], "列数<19"))
            continue
        if not c[0] or not c[1]:
            skipped.append((c[0], "词/码空"))
            continue
        if c[3].strip() == "通过":
            body.append((c[0], c[1], (c[2] or "").strip() or "5", c[11]))
        elif c[3].strip():
            pass  # 存疑/反对: 有意不收录
        else:
            skipped.append((c[0], "判定空"))
    print(f"审核终表: {src}")
    print(f"通过 {len(body)} / 跳过 {len(skipped)}"
          f"（判定空 {sum(1 for _, r in skipped if r == '判定空')}）")

    # 词典同码终检（多字 4 码 / 单字 3 码, 按证据域分流——同机械审查 M2 口径）
    cw_big, _ = corpus_std.load_pdsp_dict()
    cw_sc, _ = corpus_std.load_pdsp_dict_single3()
    bad = []
    for w, code, lam, dom in body:
        cw = cw_sc if "单字" in dom else cw_big
        if code not in cw:
            bad.append(f"{w}@{code}: 码不在{'单字3码' if '单字' in dom else '多字4码'}词典")
        elif w not in cw[code]:
            bad.append(f"{w}@{code}: 词不在该码候选")
    if bad:
        print("\n词典同码终检失败（不得部署）:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)
    print("词典同码终检: 全部通过")

    # λ 列 → 有符号分数（口径 = migrate_pairfix_signed 迁移件）:
    # 显式 +/- 原样透传（+200/+100/-5…）; 无符号正数 = 降分幅度 → 取负落地。
    new_entries = []
    for w, code, raw, _dom in body:
        try:
            v = float(raw)
        except ValueError:
            sys.exit(f"λ 列非法值 {raw!r}（{w}@{code}）——检查审核表第 3 列")
        new_entries.append((f"{w}\t{code}",
                            v if raw.startswith(("+", "-")) else -abs(v)))

    # 读现行生产表: 正分（固顶/固次, 用户 09876 写入或手动编辑）全数保留;
    # 旧负分条目由本代审核集取代（降噪对只来自审查流）。
    prod_pos, prod_neg = {}, {}
    if deploy.exists():
        for line in deploy.read_text(encoding="utf-8-sig").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                continue
            try:
                d = float(parts[2]) if len(parts) >= 3 and parts[2].strip() else -5.0
            except ValueError:
                print(f"  生产表跳过坏行: {s!r}")
                continue
            key = parts[0].strip() + "\t" + parts[1].strip()
            (prod_pos if d > 0 else prod_neg)[key] = d

    # 合并: 正分权威（同词码对正分冲突 → 正分胜, 报告）; 新表条目入位。
    merged, conflicts = {}, []
    merged.update(prod_pos)
    for key, d in new_entries:
        if key in prod_pos:
            if prod_pos[key] != d:
                conflicts.append((key, f"新 {d:+g} 让位于现行正分 {prod_pos[key]:+g}"))
            else:
                conflicts.append((key, f"新 {d:+g} 与现行正分相同"))
            continue
        merged[key] = d
    pos = sorted(((k, v) for k, v in merged.items() if v > 0),
                 key=lambda kv: (kv[0].split("\t")[1], kv[0].split("\t")[0]))
    neg = sorted(((k, v) for k, v in merged.items() if v < 0),
                 key=lambda kv: (kv[0].split("\t")[1], kv[0].split("\t")[0]))
    zero = [(k, v) for k, v in merged.items() if v == 0]

    print(f"\n合并计划: 新审核 {len(new_entries)} 对 → 落地 "
          f"{sum(1 for _, d in new_entries if d < 0)} 负 / "
          f"{sum(1 for _, d in new_entries if d > 0)} 正; "
          f"现行正分保留 {len(prod_pos)}（冲突 {len(conflicts)}）; "
          f"旧负分由新集取代 {len(prod_neg)}")
    for key, msg in conflicts:
        print(f"  冲突 {key}: {msg}")
    if zero:
        print(f"  0 分条目 {len(zero)}（无效果, 检查审核表）: "
              + ", ".join(k for k, _ in zero[:10]))

    # 生成部署表内容（有符号头注释, 与迁移件口径一致）
    today = date.today().strftime("%Y-%m-%d")
    header = [
        "# 词码对分数修正（方案特异: 拼读双拼）",
        "# 行格式: 词<TAB>码<TAB>[+|-]分数",
        "#   +200 = 加分（固定首选, 09876 直接上屏并写入）",
        "#   +100 = 加分（固定次选, 手动编辑本表）",
        "#   -5   = 扣分（pairfix 预置降噪; 审核表 λ 正数落地为负分）",
        "#   # 注释; 删本文件 = 机制休眠",
        f"# 来源: {src.name} 判定=通过 {len(new_entries)} 对"
        f"（三层审查: 机械门+子智能体语境评审+大陆规范裁定, {today}）"
        f" + 现行正分 {len(prod_pos)} 对保留（audit_tool deploy 有符号合并）",
        "# 未收录（存疑/反对）行见审核全表（OneDrive typing/拼读双拼/ 工作目录）",
    ]
    lines = list(header)
    for k, d in pos + neg + zero:
        w, c = k.split("\t")
        lines.append(f"{w}\t{c}\t{d:+g}")
    text = "\n".join(lines) + "\n"
    print(f"\n部署表: 正 {len(pos)} / 负 {len(neg)} 对, md5(将写) = "
          f"{hashlib.md5(text.encode('utf-8')).hexdigest()}")

    if a.dry_run:
        print("dry-run: 不写文件、不部署")
        return

    out_eval.write_text(text, encoding="utf-8")
    print(f"已写: {out_eval}  md5={md5(out_eval)}")

    # 备份现有部署表（内容不同才备份; 同名备份已存在则加序号）
    if deploy.exists() and deploy.read_bytes() != out_eval.read_bytes():
        i, bak = 0, deploy.with_suffix(".tsv.bak-" + date.today().strftime("%Y%m%d"))
        while bak.exists():
            i += 1
            bak = deploy.with_suffix(f".tsv.bak-{date.today().strftime('%Y%m%d')}-{i}")
        shutil.copy2(deploy, bak)
        print(f"旧部署表已备份: {bak}")
    elif deploy.exists():
        print("部署表内容相同, 不备份")

    deploy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_eval, deploy)
    print(f"已部署: {deploy}  md5={md5(deploy)}"
          + ("  == eval 副本一致 ✓" if md5(deploy) == md5(out_eval) else "  ✗ 不一致!"))


def main():
    ap = argparse.ArgumentParser(
        description="pairfix 审核工具（batch/merge/deploy, 取代三件套）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("batch", help="审核表切批次文件")
    p.add_argument("--tag", default="05b")
    p.add_argument("--only-pending", action="store_true",
                   help="只切判定列为空的行（SOP S6 新增对评审）")
    p.add_argument("--src", default=None, help="覆盖源表路径")
    p.add_argument("--nbatches", type=int, default=10)
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("merge", help="批次判定回填审核表")
    p.add_argument("--tag", default="05b")
    p.add_argument("--src", default=None)
    p.add_argument("--dst", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("deploy", help="审核终表 → 部署表 + 备份 + 部署")
    p.add_argument("--tag", default="05b")
    p.add_argument("--src", default=None, help="覆盖审核终表路径")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_deploy)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
