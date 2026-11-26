# -*- coding: utf-8 -*-
"""补第二遍：加「定案③ + A/B 开关」标注、修研究库落地段。"""
SUBS = [
    ("AGENTS.md",
     "的融合序首选**（不再要求全局第 1）∧ span/文本一致（**音节不完整的单字不参与**",
     "的融合序首选**（2026-11-26 定案③：取代「同字数档首选」，"
     "A/B `RIME_LLM_GATE_SPAN=0` 回旧口径；不再要求全局第 1）∧ 文本一致"
     "（**音节不完整的单字不参与**"),
    ("rime-llm-ime/README.md",
     "的融合序首选** ∧ span/文本一致（**音节不完整的单字不参与**",
     "的融合序首选**（2026-11-26 定案③：取代同字数档，"
     "A/B `RIME_LLM_GATE_SPAN=0` 回旧口径）∧ 文本一致"
     "（**音节不完整的单字不参与**"),
    ("rime-llm-rerank/README.md",
     "的融合序首选** ∧ span/文本一致（**音节不完整的单字不参与**",
     "的融合序首选**（2026-11-26 定案③：取代同字数档，"
     "A/B `RIME_LLM_GATE_SPAN=0` 回旧口径）∧ 文本一致"
     "（**音节不完整的单字不参与**"),
    ("llm-rerank-research/freq-fusion-research.md",
     "证据改**同字数档首选**（特征仍以 lead 为锚、吃完整窗口）",
     "证据改**同字数档首选**（**2026-11-26 晚改为同 span 首选**，"
     "见文末「整读分层」节；特征仍以 lead 为锚、吃完整窗口）"),
    ("llm-rerank-research/README.md",
     "（后于同日改为**同 span 首选**，见「整读分层」节",
     "（后于同日改为**同 span 首选**，见 `freq-fusion-research.md`「整读分层」节"),
]
for p, old, new in SUBS:
    s = open(p, encoding="utf-8").read()
    n = s.count(old)
    if n == 0:
        print(f"!! {p}: 未命中")
        continue
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
    print(f"{p}: {n} 处")
