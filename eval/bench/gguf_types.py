#!/usr/bin/env python3
"""GGUF 张量类型清单（排查 base vs instruct 文件大小/速度差异用）。"""
import collections
import struct
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

TSIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TN = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
      8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
      14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ3_XXS", 18: "IQ4_NL"}


def gguf_tensors(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        struct.unpack("<I", f.read(4))          # version
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rd_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rd_val(t):
            if t == 8:                            # STRING
                rd_str()
            elif t == 9:                          # ARRAY
                et, = struct.unpack("<I", f.read(4))
                cnt, = struct.unpack("<Q", f.read(8))
                if et == 8:
                    for _ in range(cnt):
                        rd_str()
                else:
                    f.read(TSIZE[et] * cnt)
            else:
                f.read(TSIZE[t])

        for _ in range(n_kv):
            rd_str()
            t, = struct.unpack("<I", f.read(4))
            rd_val(t)

        out = []
        for _ in range(n_tensors):
            name = rd_str()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            ttype, = struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            out.append((name, ttype, dims))
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="GGUF 张量类型清单")
    ap.add_argument("paths", nargs="*",
                    help="GGUF 路径；缺省 = 0.5B base/instruct 对照")
    ap.add_argument("--keys", default="token_embd,output,blk.0.attn_v,blk.23.attn_output",
                    help="关键张量前缀，逗号分隔")
    args = ap.parse_args()
    if args.paths:
        pairs = [(Path(p).stem, p) for p in args.paths]
    else:
        pairs = [("base(QuantFactory)",
                  r"本机 GGUF 模型目录\Qwen2.5-0.5B-Q4_K_M.gguf"),
                 ("instruct(官方)",
                  r"本机 GGUF 模型目录\qwen2.5-0.5b-instruct-q4_k_m.gguf")]
    for name, p in pairs:
        ts = gguf_tensors(p)
        agg = collections.defaultdict(lambda: [0, 0, ""])
        for nm, t, dims in ts:
            n = 1
            for d in dims:
                n *= d
            agg[t][0] += 1
            agg[t][1] += n
            if not agg[t][2] and "attn" in nm or not agg[t][2]:
                agg[t][2] = nm
        print(f"== {name}: {len(ts)} tensors")
        for t, (cnt, tot, ex) in sorted(agg.items()):
            print(f"  {TN.get(t, str(t)):>8}: {cnt:>3} 张量 "
                  f"{tot / 1e6:>8.1f}M 参数  例: {ex}")
    # 关键张量逐个对照
    print("\n== 关键张量对照")
    for key in [k.strip() for k in args.keys.split(",") if k.strip()]:
        for name, p in pairs:
            for nm, t, dims in gguf_tensors(p):
                if nm.startswith(key):
                    n = 1
                    for d in dims:
                        n *= d
                    print(f"  {name:>24} {nm:<28} {TN.get(t, t):>5} "
                          f"{n / 1e6:8.1f}M")
                    break


if __name__ == "__main__":
    main()
