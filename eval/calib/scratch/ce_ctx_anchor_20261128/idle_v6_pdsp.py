# -*- coding: utf-8 -*-
"""idle_v3.py — 模拟用户真实流程: 提交探测词清场 → akjk → 静置观测.
打字期 60ms 采样抓候选窗; 静置后 WM_GETTEXT 取文档精确文本; TSF/服务端日志对齐."""
import os, sys, time, ctypes, hashlib
from ctypes.wintypes import RECT

sys.path.insert(0, r"D:\llm-rerank\rime-llm-ime\scripts")
os.environ["WEASEL_TSF_DBG"] = "1"

import test_ctx_auto as H          # noqa: E402
import verify_rel10 as V           # noqa: E402
from PIL import ImageGrab          # noqa: E402

TEMP = os.environ.get("TEMP", ".")
SRV = os.path.join(TEMP, "weasel_srv_dbg.log")
TSF = os.path.join(TEMP, "weasel_tsf_dbg.log")
SHOTS = r"D:\llm-rerank\probe-scratch\ce_invest\idle_pdsp_fix"
os.makedirs(SHOTS, exist_ok=True)
CORPUS = r"C:\Users\Administrator\AppData\Roaming\Rime\llm_rerank\pdsp\rime_llm_corpus.tsv"
CORPUS_LUNA = r"C:\Users\Administrator\AppData\Roaming\Rime\llm_rerank\luna_pinyin\rime_llm_corpus.tsv"
u32 = ctypes.windll.user32

def log_off(p):
    try: return os.path.getsize(p)
    except OSError: return 0

def log_delta(p, off):
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            f.seek(off); return f.read().splitlines()
    except OSError: return []

def corpus_rows_since(off):
    """只读 off 之后的新行、只认 pdsp 语料（2026-11-28 教训: seek(0)+双语料
    让历史行也能满足探针 = 等于没确认 IME，v6 曾因此把微软拼音当 pdsp 测）."""
    out = []
    try:
        with open(CORPUS, "r", encoding="utf-8", errors="replace") as f:
            f.seek(off)
            for line in f:
                if line.startswith("W\t") or line.startswith("C\t"):
                    out.append(line.rstrip("\n")[:130])
    except OSError:
        pass
    return out

def doc_text(hwnd):
    """WM_GETTEXT 跨进程取 TextBox 全文（AGENTS #10 口径）."""
    WM_GETTEXTLENGTH, WM_GETTEXT = 0x000E, 0x000D
    # 找 TextBox 子窗口
    hbox = None
    CB = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
    found = []
    def enum(child, lp):
        cls = ctypes.create_unicode_buffer(64)
        u32.GetClassNameW(child, cls, 64)
        if "Edit" in cls.value or "WindowsForms" in cls.value:
            found.append(child)
        return 1
    proc = CB(enum)
    u32.EnumChildWindows(hwnd, proc, 0)
    ctypes.WINFUNCTYPE.restype = None
    if found:
        hbox = found[0]
    target = hbox or hwnd
    ln = u32.SendMessageW(target, WM_GETTEXTLENGTH, 0, 0)
    buf = ctypes.create_unicode_buffer(ln + 4)
    u32.SendMessageW(target, WM_GETTEXT, ln + 2, buf)
    return buf.value

def grab(hwnd, tag):
    r_ = RECT(); u32.GetWindowRect(hwnd, ctypes.byref(r_))
    img = ImageGrab.grab(bbox=(max(0, r_.left-10), max(0, r_.top-130),
                               r_.right+380, r_.bottom+260))
    fn = os.path.join(SHOTS, tag + ".png")
    img.save(fn)
    return fn

def sig(img):
    return hashlib.md5(img.convert("L").resize((48, 16)).tobytes()).hexdigest()[:8]

def main():
    corpus_off = log_off(CORPUS)
    host, hwnd, stop_file = V.host_start()
    ime_ok = False
    for r in range(6):
        if not V.force_foreground(hwnd): break
        V.switch_ime(hwnd)
        H.type_code("wkjl", gap=0.10)
        time.sleep(0.6)
        if any("\twkjl\t" in row for row in corpus_rows_since(corpus_off)):
            ime_ok = True; break
    if not ime_ok:
        open(stop_file, "w").close(); print("ABORT: IME"); return
    # 空格提交探测词 → 清场（无 Esc）
    H.key_tap(0x20); time.sleep(0.5)
    doc = doc_text(hwnd)
    print("doc after probe commit: |%s|" % doc)
    # IME 签名校验: 「我看见了」= 微软拼音简拼 w-k-j-l 的签名输出，
    # pdsp 不可能出 → 说明激活的是微软拼音而非小狼毫 pdsp
    if "我看见了" in doc:
        open(stop_file, "w").close()
        print("ABORT: 微软拼音签名 (我看见了) — IME 未切到 pdsp")
        return
    # 化学上屏（生产流程: 上文词先提交）
    H.type_code("hxxu", gap=0.10)
    H.key_tap(0x20); time.sleep(0.6)
    doc2 = doc_text(hwnd)
    print("doc after hxxu commit: |%s|" % doc2)
    if "化学" not in doc2:
        print("WARN: hxxu 未提交出「化学」— IME 可疑，结果存疑")

    srv0, tsf0 = log_off(SRV), log_off(TSF)
    r_ = RECT(); u32.GetWindowRect(hwnd, ctypes.byref(r_))
    box = (max(0, r_.left-10), max(0, r_.top-130), r_.right+380, r_.bottom+260)
    print("doc before typing: |%s|" % doc_text(hwnd)[:60])
    for ch in "akjk":
        H.key_tap(ord(ch.upper()))
        time.sleep(0.10)
    t_last = time.time()
    idle = []
    last = None
    while time.time() - t_last < 4.0:
        img = ImageGrab.grab(bbox=box)
        s = sig(img)
        if s != last:
            idle.append((round(time.time()-t_last, 3), s, img))
            last = s
        time.sleep(0.06)
    print("idle distinct states: %d" % len(idle))
    for i, (t, s, img) in enumerate(idle):
        if i < 14:
            img.save(os.path.join(SHOTS, "idle_%02d.png" % i))
        print("  +%ss %s" % (t, s))
    print("doc after idle: |%s|" % doc_text(hwnd)[:80])
    rows = corpus_rows_since(corpus_off)
    print("=== corpus rows (last 8) ===")
    for row in rows[-8:]:
        print(row)
    print("=== srv delta (%d) ===" % len(log_delta(SRV, srv0)))
    for line in log_delta(SRV, srv0)[-10:]:
        print(line)
    print("=== tsf delta (comp/ctx lines) ===")
    for line in log_delta(TSF, tsf0):
        if any(k in line for k in ("Ctx", "EndEdit", "Drop")):
            print(line)
    open(stop_file, "w").close()
    try: host.wait(timeout=8)
    except Exception: host.kill()
    print("host closed")

if __name__ == "__main__":
    main()
