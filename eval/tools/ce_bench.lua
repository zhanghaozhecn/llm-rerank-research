-- ce_bench.lua — 对整池候选打分并导出 ce1/ce2/ce3（2026-11-25）
package.cpath = [[build/Release/?.dll]]
local m = require("rime_llm")
m.model_path = "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"
m.max_ctx = 20
m.n_threads = 8
m.n_seq_max = 96
m.score("预热", { "x" })
local t0 = os.clock()
while not m.is_ready() do if os.clock() - t0 > 120 then error("load timeout") end end
local fin = assert(io.open("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/ce_bench_input.tsv", "r"))
local fout = assert(io.open("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/ce_bench_out.tsv", "w"))
local n = 0
for line in fin:lines() do
  local a, b = line:find("\t")
  if a then
    local wid = line:sub(1, a - 1)
    local rest = line:sub(b + 1)
    local c2, d = rest:find("\t")
    local ctx = c2 and rest:sub(1, c2 - 1) or rest
    local joined = c2 and rest:sub(d + 1) or ""
    local cands = {}
    for t in (joined .. "\x1f"):gmatch("(.-)\x1f") do if t ~= "" then cands[#cands + 1] = t end end
    if #cands >= 2 then
      local t0w = os.clock()
      m.score(ctx, cands)
      local ms = (os.clock() - t0w) * 1000
      local ce = m.get_ce3() or {}
      fout:write("W\t", wid, "\t", string.format("%.1f", ms), "\n")
      for _, c in ipairs(cands) do
        local v = ce[c] or {}
        fout:write("C\t", wid, "\t", c, "\t",
          v[1] and string.format("%.4f", v[1]) or "", "\t",
          v[2] and string.format("%.4f", v[2]) or "", "\t",
          v[3] and string.format("%.4f", v[3]) or "", "\n")
      end
      n = n + 1
    end
  end
end
fin:close(); fout:close()
print("[ce_bench] windows=" .. n)
