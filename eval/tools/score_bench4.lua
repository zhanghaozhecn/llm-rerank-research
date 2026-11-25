-- score_bench.lua — 候选数 × 前缀共享 对 3-token CE 打分耗时的影响（2026-11-25）
-- 用真机同款 rime_llm.dll / Qwen3-0.6B；候选列表取自 luna 混合窗探针的真实窗口。
-- 跑法（cwd = 插件 cpp\）：lua_host.exe <abs path>/score_bench.lua
package.cpath = [[build/Release/?.dll]]
local m = require("rime_llm")
m.model_path = "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"
m.max_ctx = 20
m.n_threads = 8
m.n_seq_max = 96
m.log_dir = "C:/Users/Administrator/AppData/Roaming/Rime/llm_rerank/luna_pinyin"          -- 默认 12 只够 ~11 条 worker seq（>11 个不同首 token 会越界）

local function load_json(p)
  local f = assert(io.open(p, "r"))
  local s = f:read("*a"); f:close()
  -- 极简字符串数组解析（候选文本不含引号/逗号）
  local t = {}
  for x in s:gmatch('"([^"]*)"') do t[#t + 1] = x end
  return t
end

local WORD = load_json("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/bench_word.json")
local SING = load_json("D:/llm-rerank/llm-rerank-research/eval/luna_mixed/wx/bench_single.json")
print(string.format("[bench] word list %d (1char %d) / single list %d (1char %d)",
                    #WORD, (function() local n=0 for _,c in ipairs(WORD) do if utf8.len(c)==1 then n=n+1 end end return n end)(),
                    #SING, (function() local n=0 for _,c in ipairs(SING) do if utf8.len(c)==1 then n=n+1 end end return n end)()))

m.score("预热", { "x" })
local t0 = os.clock()
while not m.is_ready() do
  if os.clock() - t0 > 90 then error("model load timeout") end
end
print(string.format("[bench] model ready %.0fs", os.clock() - t0))

local CTX = "今天天气很好我们一起去公园散步吧"

local function take(list, n)
  local out = {}
  for i = 1, math.min(n, #list) do out[i] = list[i] end
  return out
end

local function bench(name, cands, reps)
  reps = reps or 3
  -- 预热一次（prep 缓存/分配器）
  m.score(CTX, cands)
  os.clock()
  local t = os.clock()
  for _ = 1, reps do m.score(CTX, cands) end
  local ms = (os.clock() - t) * 1000 / reps
  print(string.format("[bench] %-22s n=%-3d  %7.1f ms", name, #cands, ms))
  return ms
end

bench("word 6", take(WORD, 6))
bench("word 12", take(WORD, 12))
bench("word 24", take(WORD, 24))
bench("word 48", take(WORD, 48))
bench("word 64", take(WORD, 64))
bench("single-heavy 64", take(SING, 64))
-- 首 token 共享对照：取同一首字的多个候选（从 word 列表里筛）
local by1 = {}
for _, c in ipairs(WORD) do
  local h = utf8.len(c) == 1 and c or c:sub(1, 3)
  by1[h] = by1[h] or {}
  table.insert(by1[h], c)
end
local share = {}
for _, v in pairs(by1) do
  for _, c in ipairs(v) do
    if #share < 24 then table.insert(share, c) end
  end
  if #share >= 24 then break end
end
bench("24 same-first", share)
