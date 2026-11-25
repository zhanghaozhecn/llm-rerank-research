-- score_bench3.lua — 行数与耗时（2026-11-25）
package.cpath = [[build/Release/?.dll]]
local m = require("rime_llm")
m.model_path = "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"
m.max_ctx = 20
m.n_threads = 8
m.n_seq_max = 96
m.score("预热", { "x" })
local t0 = os.clock()
while not m.is_ready() do if os.clock() - t0 > 90 then error("load timeout") end end
local CTX = "今天天气很好我们一起去公园散步吧"
local function load_json(p)
  local f = assert(io.open(p, "r")); local s = f:read("*a"); f:close()
  local t = {}; for x in s:gmatch('"([^"]*)"') do t[#t + 1] = x end; return t
end
local SING = load_json("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/bench_single.json")
-- bench_single 里各候选的 token 数分布
local c1, c2, c3 = 0, 0, 0
for _, c in ipairs(SING) do
  local r = m.score_full(CTX, { c })
  local n = r and r[1] and r[1].ntok or 0
  if n == 1 then c1 = c1 + 1 elseif n == 2 then c2 = c2 + 1 else c3 = c3 + 1 end
end
print(string.format("[tok] bench_single ntok 分布: 1tok=%d 2tok=%d 3+tok=%d", c1, c2, c3))

local function bench(name, cands)
  m.score(CTX, cands)
  local a = os.clock(); m.score(CTX, cands); local ms = (os.clock() - a) * 1000
  print(string.format("[bench] %-26s n=%-3d %7.1f ms", name, #cands, ms))
end
local function rep(t, n)
  local o = {}; for i = 1, n do o[i] = t[((i - 1) % #t) + 1] end; return o
end
local WORDS = { "今天", "天气", "我们", "一起", "公园", "散步", "你好", "世界",
                "中国", "北京", "上海", "电脑", "软件", "输入", "候选", "模型" }
for _, n in ipairs({ 2, 8, 16, 32, 64 }) do bench(n .. " x 同一2token词", rep({ "今天" }, n)) end
for _, n in ipairs({ 2, 8, 16, 32, 64 }) do bench(n .. " 个不同2token词", rep(WORDS, n)) end
