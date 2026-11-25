-- score_bench2.lua — 排查候选数耗时来源（2026-11-25）
package.cpath = [[build/Release/?.dll]]
local m = require("rime_llm")
m.model_path = "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"
m.max_ctx = 20
m.n_threads = 8
m.n_seq_max = 96
m.score("预热", { "x" })
local t0 = os.clock()
while not m.is_ready() do
  if os.clock() - t0 > 90 then error("load timeout") end
end
local CTX = "今天天气很好我们一起去公园散步吧"

local function load_json(p)
  local f = assert(io.open(p, "r")); local s = f:read("*a"); f:close()
  local t = {}; for x in s:gmatch('"([^"]*)"') do t[#t + 1] = x end; return t
end
local WORD = load_json("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/bench_word.json")
local SING = load_json("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/bench_single.json")

-- 1) 单字 / 词的 token 数（score_full 返回 ntok）
for _, c in ipairs({ "是", "的", "一", "嗯", SING[1], SING[2], WORD[1], WORD[2] }) do
  local r = m.score_full(CTX, { c })
  print(string.format("[tok] %-6s ntok=%s", c, r and r[1] and tostring(r[1].ntok) or "?"))
end

local function bench(name, cands)
  local reps = 4
  m.score(CTX, cands)
  local ts = {}
  for i = 1, reps do
    local a = os.clock(); m.score(CTX, cands); ts[i] = (os.clock() - a) * 1000
  end
  local s = {}
  for i = 1, reps do s[i] = string.format("%.0f", ts[i]) end
  print(string.format("[bench] %-24s n=%-3d  %s ms", name, #cands, table.concat(s, " / ")))
end

local function rep(t, n)
  local o = {}
  for i = 1, n do o[i] = t[((i - 1) % #t) + 1] end
  return o
end

bench("6 words", rep(WORD, 6))
bench("64 words", rep(WORD, 64))
bench("64 x 同一词", rep({ WORD[1] }, 64))
bench("64 x 同一单字", rep({ "是" }, 64))
bench("64 singles", rep(SING, 64))
bench("64 单字(常用)", rep({ "是", "的", "一", "在", "有", "和", "了", "不", "人", "我" }, 64))
