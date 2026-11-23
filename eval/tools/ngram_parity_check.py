#!/usr/bin/env python3
"""ngram_parity_check.py — 组句器离线对拍（C++ 生产实现 vs 研究脚本留档）

做法：
  1) 读研究链留档 `<name>.test.jsonl`（sentence_gen_gram.py 输出，含 code /
     before / truth / cands）；
  2) 生成 TSV（sid \t code \t before）→ 交给插件版 `lua_host.exe` 跑
     `rime-llm-rerank/cpp/ngram_parity.lua`（C++ 组句器，同参数）；
  3) 逐句比较 top-k 文本/分数，并复算 hit@k。

环境变量（不写死本机路径）：
  NGRAM_GRAM / NGRAM_DICT / NGRAM_FILES / NGRAM_ESSAY  同 ngram_parity.lua
用法：
  python ngram_parity_check.py --ref <jsonl> --lua-host <exe> --script <lua> \\
      [--params beam=64,topk=5,min_weight=10]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def load_ref(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_input(rows, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            # before 里的制表符/换行会让列错位（研究链 before 是纯文本尾部）
            before = r.get("before", "").replace("\t", " ").replace("\n", " ")
            f.write("%s\t%s\t%s\n" % (r["sid"], r["code"], before))


def read_out(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sid = int(parts[0])
            out.setdefault(sid, []).append((parts[3], float(parts[2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lua-host", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--params", default="beam=64,topk=5,min_weight=10,ctx_tail=10")
    ap.add_argument("--t2s", action="store_true",
                    help="候选/真值统一转简体再比较（方案词典为繁体时用）")
    ap.add_argument("--keep", action="store_true", help="保留中间文件")
    a = ap.parse_args()

    conv = None
    if a.t2s:
        import opencc  # 本机研究环境已装；缺了就不该用 --t2s
        conv = opencc.OpenCC("t2s")

    def norm(text):
        return conv.convert(text) if conv else text

    rows = load_ref(a.ref)
    tmp = tempfile.mkdtemp(prefix="ngram_parity_")
    in_tsv = os.path.join(tmp, "in.tsv")
    out_tsv = os.path.join(tmp, "out.tsv")
    write_input(rows, in_tsv)

    env = dict(os.environ)
    env.update({
        "NGRAM_IN": in_tsv,
        "NGRAM_OUT": out_tsv,
        "NGRAM_PARAMS": a.params,
    })
    for key in ("NGRAM_GRAM", "NGRAM_DICT", "NGRAM_FILES", "NGRAM_ESSAY"):
        if key not in env:
            print("[warn] %s 未设置" % key, file=sys.stderr)

    proc = subprocess.run([a.lua_host, a.script], cwd=os.path.dirname(
        os.path.abspath(a.lua_host)), env=env)
    if proc.returncode != 0:
        print("[fail] lua_host exit=%d" % proc.returncode, file=sys.stderr)
        return 1

    ours = read_out(out_tsv)
    compared = exact = sameset = 0
    hit_ours = hit_ref = feas = 0
    diffs = []
    for r in rows:
        if not r.get("reachable"):
            continue
        sid = r["sid"]
        ref = [(c["text"], c["score"]) for c in r.get("cands") or []]
        if not ref:
            continue
        get = ours.get(sid) or []
        feas += 1
        compared += 1
        # hit@k 只数前 k 条（topk=10 时不能把 6..10 也算成 hit@5）
        if any(norm(t) == norm(r["truth"]) for t, _ in get[:5]):
            hit_ours += 1
        if any(norm(t) == norm(r["truth"]) for t, _ in ref[:5]):
            hit_ref += 1
        if [norm(t) for t, _ in ref] == [norm(t) for t, _ in get]:
            sameset += 1
            if len(ref) == len(get) and all(
                    abs(x[1] - y[1]) < 1e-6 for x, y in zip(ref, get)):
                exact += 1
            elif len(diffs) < 8:
                diffs.append((r["truth"], "score", ref, get))
        elif len(diffs) < 8:
            diffs.append((r["truth"], "text", ref, get))

    print("reachable(compared)=%d  exact=%d(%.1f%%)  same-order-set=%d(%.1f%%)"
          % (compared, exact, 100.0 * exact / max(compared, 1), sameset,
             100.0 * sameset / max(compared, 1)))
    print("hit@5: cpp=%.1f%%  ref=%.1f%%  (reachable n=%d)"
          % (100.0 * hit_ours / max(feas, 1), 100.0 * hit_ref / max(feas, 1),
             feas))
    for truth, kind, ref, get in diffs:
        print("\n[mismatch:%s] truth=%s" % (kind, truth))
        for i in range(max(len(ref), len(get))):
            l = "%s %.3f" % ref[i] if i < len(ref) else "-"
            r = "%s %.3f" % get[i] if i < len(get) else "-"
            print("  %d  ref=%-28s cpp=%s" % (i + 1, l, r))
    if not a.keep:
        for name in ("in.tsv", "out.tsv"):
            try:
                os.remove(os.path.join(tmp, name))
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass
    return 0 if exact == compared else 2


if __name__ == "__main__":
    sys.exit(main())
