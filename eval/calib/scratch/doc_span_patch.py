# -*- coding: utf-8 -*-
"""把「同字数档首选」口径的文档表述改成「同 span 首选」（2026-11-26 定案③）。"""
import re

FILES = ["AGENTS.md", "rime-llm-ime/README.md", "rime-llm-rerank/README.md",
         "llm-rerank-research/README.md", "llm-rerank-research/freq-fusion-research.md"]

SPAN = "同 span（`[seg_start, seg_start+lead_end)`）"
NOTES = {
    "AGENTS.md": [("的融合序首选**（不再要求全局第 1）",
                   "的融合序首选**（2026-11-26 定案③：取代同字数档，"
                   "A/B `RIME_LLM_GATE_SPAN=0` 回旧口径；不再要求全局第 1）")],
    "rime-llm-research": [],
}

SUBS = [
    # A) 其同字数档（1 字档 / 多字档）的融合序首选 → 其同 span（…）的融合序首选
    (re.compile(r"其同字数档（[^）]*）的融合序首选"), "其" + SPAN + "的融合序首选"),
    # B) 与同字数档首选比较 → 与同 span 首选比较
    (re.compile(r"与同字数档首选比较"), "与同 span 首选比较"),
    # C) 1 字档另跳过音节不完整的单字 → 音节不完整的单字不参与
    (re.compile(r"\*\*1 字档另跳过音节不完整的单字\*\*"), "**音节不完整的单字不参与**"),
    # D) 研究库：结构证据记述加一句现状
    (re.compile(r"结构证据改\*\*同字数档首选\*\*（62 可判窗"),
     "结构证据改**同字数档首选**（后于同日改为**同 span 首选**，见「整读分层」节；"
     "62 可判窗"),
    (re.compile(r"结构证据改\*\*同字数档首选\*\*（特征仍以 lead 为锚"),
     "结构证据改**同字数档首选**（**2026-11-26 晚改为同 span 首选**，"
     "见文末「整读分层」节；特征仍以 lead 为锚"),
    (re.compile(r"「同字数档首选」的结构证据"), "「同 span 首选」在单音节窗上的结构证据"),
    (re.compile(r"`pick`（同字数档首个）"),
     "`pick`（**同 span 首个**，2026-11-26 定案③取代同字数档）"),
    # E) 申请表格里的简写
    (re.compile(r"且该文本必须是其同字数档（1 字/多字）的融合序首选且 span 对齐"),
     "且该文本必须是其" + SPAN + "的融合序首选"),
]

for p in FILES:
    s = open(p, encoding="utf-8").read()
    n = 0
    for rx, rep in SUBS:
        s, k = rx.subn(rep, s)
        n += k
    open(p, "w", encoding="utf-8", newline="").write(s)
    left = s.count("同字数档")
    print(f"{p}: 替换 {n} 处，剩余「同字数档」{left} 处")
