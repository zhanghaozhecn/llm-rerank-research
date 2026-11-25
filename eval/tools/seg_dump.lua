-- seg_dump.lua — 批量导出候选的 token 段显（2026-11-25；用于词模式预算实验）
package.cpath = [[build/Release/?.dll]]
local m = require("rime_llm")
m.model_path = "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"
m.max_ctx = 20; m.n_threads = 8; m.n_seq_max = 96
m.score("预热", { "x" })
local t0 = os.clock()
while not m.is_ready() do if os.clock() - t0 > 120 then error("load timeout") end end
local IN = "D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/wm_texts.txt"
local OUT = "D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/wm_segs.tsv"
local fin = assert(io.open(IN, "r"))
local fout = assert(io.open(OUT, "w"))
local batch, nb = {}, 0
local function flush()
  if #batch < 2 then return end
  m.score("今天", batch)
  local seg = m.get_seg() or {}
  for _, t in ipairs(batch) do
    fout:write(t, "\t", seg[t] or "", "\n")
  end
  batch = {}
end
for line in fin:lines() do
  if line ~= "" then batch[#batch + 1] = line; if #batch >= 64 then flush() end end
end
flush()
fin:close(); fout:close()
print("[seg] done")
