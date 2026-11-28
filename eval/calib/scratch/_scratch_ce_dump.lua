-- _scratch_ce_dump.lua — CE 转储（CE 行级并行化前后逐位对比；不入库）
-- 用法: CE_DLL=<dll路径> lua_host.exe _scratch_ce_dump.lua > out.txt
package.cpath = os.getenv("CE_DLL") or [[build/Release/?.dll]]

function yield(c) end
ShadowCandidate = function(c, t, txt, cm, inherit) return c end
_G.llm_context_get = function() return "", "" end

local llm = require("rime_llm")
llm.live_write = function() end

llm.model_path = "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf"
llm.score("预热", { "x" })
local t0 = os.clock()
while not llm.is_ready() do
  if os.clock() - t0 > 90 then error("model load timeout") end
end

local ctx = "我们明天早上八点在公司门口集合然后一起去图书馆借几本关于计算机系统的书顺便买点水果回来"
local cands = { "集合", "会议", "计划", "回家", "图书馆", "计算机" }

local function dump(tag)
  llm.score(ctx, cands)
  local ce3 = llm.get_ce3()
  for _, w in ipairs(cands) do
    local c = ce3[w]
    io.write(tag, "\t", w, "\t",
      string.format("%.17g", c and c[1] or -1), "\t",
      string.format("%.17g", c and c[2] or -1), "\t",
      string.format("%.17g", c and c[3] or -1), "\n")
  end
end

dump("fresh")  -- 全流程（S1 全量 + CE1/2/3）
dump("hit")    -- prep 命中路径重评
print("CE-DUMP-OK")
