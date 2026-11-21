#!/usr/bin/env python3
r"""pairfix 固定重算流程 · 驱动器（SOP 见 freq-fusion-research.md
「pairfix 固定重算流程」节）。

只编排**脚本侧可自动化**的阶段; 重阶段（CE 重算 / 常数标定 / 推导 A/B /
双域验证）不内嵌——CE 要 GPU 铁律人工把关、标定网格每代要设计、推导两脚本
已参数化——本工具负责「校验产物存在」与阶段间的衔接：

  check    校验时代产物齐备（cohort 五链 anchors/samples/streams +
           各链 ce_<tag>/results.jsonl）→ S3 推导的放行门
  manifest 登记时代参数（tag/模型 md5/常数/备注）→ eval/pairfix_epochs/<tag>.json
  diff     时代差分（S5）: 新 union 表 vs 上时代 audited 表 → 留存对自动
           预填旧判定+旧备注、新增对判定列空 → 写本时代 audited 工作表 +
           diff 报告（新增/消失清单）
  batch    切批（S6）: 只切判定列为空的新增对 → audit_tool.py batch --only-pending
  merge    合并子智能体判定（S6）: 只填空判定行 → audit_tool.py merge
  deploy   部署（S9）: 终检+备份+部署 → audit_tool.py deploy

约定（时代 tag 规则）: tag = 模型参数量缩写 + 常数版序（05b = Qwen2.5-0.5B +
kCeW(1,1.10,0.60)/β14; 同模型只换常数 → 05b2; 换模型 → 08b/06b/2b…）。
产物统一 _<tag> 后缀; CE 目录 ce_<tag>; 审核表工作源目录 = OneDrive
typing/拼读双拼/（既有约定, 与旧脚本一致）。

用法:
  python pairfix_pipeline.py check    --tag 05b
  python pairfix_pipeline.py manifest --tag 05b --model d:/gguf_models/X.gguf \
                                      --beta 14 --w2 1.10 --w3 0.60 [--notes "..."]
  python pairfix_pipeline.py diff     --tag 05b2 --prev 05b
  python pairfix_pipeline.py batch    --tag 05b2
  python pairfix_pipeline.py merge    --tag 05b2 [--src F] [--dst F] [--dry-run]
  python pairfix_pipeline.py deploy   --tag 05b2 [--dry-run]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
EVAL = Path(__file__).resolve().parents[1]     # eval 根（2026-11-18 子目录重组）
ROOT = EVAL.parent
SOURCE_DIR = Path(r"D:/OneDrive/typing/拼读双拼")
EPOCHS = EVAL / "pairfix_epochs"

# 推导层依赖的 cohort 链（2026-11-18 研究范围收窄：只留 wiki 多字词大词库）
CHAINS = ["cohort_big"]
CHAIN_FILES = ["anchors.jsonl", "samples.jsonl", "streams.jsonl"]

# v2 扩样全枚举链（2026-11-16, derive_pairfix_full.py 的输入; 不入 CHAINS——
# 它们没有 samples.jsonl, du/uni 内联进 anchors）
FULL_CHAINS = ["cohort_big_full"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_py(script: str, args: list) -> None:
    """跑 pairfix/ 下同级工具脚本（audit_tool.py 等）。"""
    cmd = [sys.executable, str(Path(__file__).resolve().parent / script)] + args
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"{script} 失败（exit {r.returncode}）——按上方输出排查后重跑")


# ---------------------------------------------------------------- check
def cmd_check(a):
    tag = a.tag
    ce_tag = a.ce_tag or tag
    ok = True
    for chain in CHAINS:
        d = ROOT / chain
        for f in CHAIN_FILES:
            p = d / f
            if not p.exists():
                print(f"  缺 {p}")
                ok = False
        ce = d / f"ce_{ce_tag}" / "results.jsonl"
        if not ce.exists():
            print(f"  缺 {ce}")
            ok = False
    for chain in FULL_CHAINS:
        d = ROOT / chain
        p = d / "anchors.jsonl"
        if not p.exists():
            print(f"  缺 {p}")
            ok = False
        ce = d / f"ce_{ce_tag}" / "results.jsonl"
        if not ce.exists():
            print(f"  缺 {ce}")
            ok = False
    print("check:", "产物齐备, S3 推导可放行" if ok else "缺产物——S1 CE 重算未完成或链目录名不符")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- manifest
def cmd_manifest(a):
    EPOCHS.mkdir(exist_ok=True)
    tag = a.tag
    m = {
        "tag": tag,
        "created": date.today().isoformat(),
        "model_path": a.model,
        "model_md5": md5(Path(a.model)) if a.model and Path(a.model).exists() else None,
        "beta": a.beta, "w2": a.w2, "w3": a.w3, "lam": 5.0, "fade_uni": 3,
        "ce_sub": f"ce_{tag}",
        "full_chains": {c: {"anchors_md5": (md5(ROOT / c / "anchors.jsonl")
                            if (ROOT / c / "anchors.jsonl").exists() else None)}
                        for c in FULL_CHAINS},
        "notes": a.notes or "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 时代产物登记（存在才记）
    for key, p in [
        ("union", SOURCE_DIR / f"pairfix_review_union_{tag}.tsv"),
        ("audited", SOURCE_DIR / f"pairfix_review_union_{tag}_audited.tsv"),
        ("deploy_pass", EVAL / f"pairfix_deploy_pass_{tag}.tsv"),
    ]:
        if p.exists():
            m[key] = {"path": str(p), "md5": md5(p), "bytes": p.stat().st_size}
    out = EPOCHS / f"{tag}.json"
    out.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"已写: {out}")
    print(json.dumps(m, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- diff
def cmd_diff(a):
    tag, prev = a.tag, a.prev
    new_p = SOURCE_DIR / f"pairfix_review_union_{tag}.tsv"
    old_p = SOURCE_DIR / f"pairfix_review_union_{prev}_audited.tsv"
    for p in (new_p, old_p):
        if not p.exists():
            sys.exit(f"缺 {p}")

    def load(p):
        rows = [l.rstrip("\r") for l in p.read_text(encoding="utf-8-sig").split("\n")
                if l.strip()]
        return rows[0].split("\t"), rows[1:]

    hdr, new_data = load(new_p)
    _, old_data = load(old_p)
    assert hdr[0] == "词"
    old = {}
    for line in old_data:
        c = line.split("\t")
        if len(c) >= 19 and c[0] and c[1]:
            old[(c[0], c[1])] = (c[3].strip(), c[18])

    kept, added, gone = [], [], []
    out = ["\t".join(hdr)]
    for line in new_data:
        c = line.split("\t")
        if len(c) >= 19 and c[3].strip():
            sys.exit(f"新 union 表 {c[0]}@{c[1]} 判定列已有值 {c[3]!r}——应从推导产物原样进入")
        key = (c[0], c[1]) if len(c) >= 19 else None
        if key in old:
            v, note = old[key]
            if v:  # 上时代有判定 → 自动沿用（含大陆规范裁定备注）
                c[3] = v
                c[18] = (c[18] + " " if c[18] else "") + f"[沿用{prev}] {note}"
            kept.append(key)
        else:
            added.append(key)
        out.append("\t".join(c))
    gone = [k for k in old if k not in set(kept) | set(added)]

    dst = SOURCE_DIR / f"pairfix_review_union_{tag}_audited.tsv"
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    rep = [f"pairfix 时代差分: {prev} → {tag}  {date.today()}",
           f"留存 {len(kept)}（判定自动沿用）/ 新增 {len(added)}（待三层审查）/ 消失 {len(gone)}",
           f"工作表: {dst}", "", "新增对:"] + \
          [f"  {w}@{code}" for w, code in added] + ["", "消失对:"] + \
          [f"  {w}@{code}" for w, code in gone]
    rep_p = EVAL / f"pairfix_diff_{tag}_from_{prev}.txt"
    rep_p.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"留存 {len(kept)}（自动沿用）/ 新增 {len(added)}（待审）/ 消失 {len(gone)}")
    for w, code in added:
        print(f"  新增 {w}@{code}")
    for w, code in gone:
        print(f"  消失 {w}@{code}")
    print(f"工作表: {dst}\n报告: {rep_p}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("check", "diff", "batch", "merge", "deploy", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        if name == "check":
            p.add_argument("--ce-tag", default=None,
                           help="CE 目录后缀（默认 = tag; 05bx 级证据扩样复用"
                                " ce_05b——CE 是原始 nats, 常数/证据规模无关）")
        if name == "diff":
            p.add_argument("--prev", required=True, help="上时代 tag（判定沿用来源）")
        if name == "manifest":
            p.add_argument("--model", default=None, help="模型 GGUF 路径")
            p.add_argument("--beta", type=float, default=None)
            p.add_argument("--w2", type=float, default=None)
            p.add_argument("--w3", type=float, default=None)
            p.add_argument("--notes", default="")
        if name in ("merge", "deploy"):
            p.add_argument("--dry-run", action="store_true")
        if name == "merge":
            p.add_argument("--src", default=None)
            p.add_argument("--dst", default=None)

    a = ap.parse_args()
    if a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "manifest":
        cmd_manifest(a)
    elif a.cmd == "diff":
        cmd_diff(a)
    elif a.cmd == "batch":
        run_py("audit_tool.py", ["batch", "--tag", a.tag, "--only-pending"])
    elif a.cmd == "merge":
        extra = ["merge", "--tag", a.tag]
        if a.src:
            extra += ["--src", a.src]
        if a.dst:
            extra += ["--dst", a.dst]
        if a.dry_run:
            extra += ["--dry-run"]
        run_py("audit_tool.py", extra)
    elif a.cmd == "deploy":
        extra = ["deploy", "--tag", a.tag] + (["--dry-run"] if a.dry_run else [])
        run_py("audit_tool.py", extra)


if __name__ == "__main__":
    main()
