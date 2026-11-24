-- luna_mixed_gen.lua -- luna 混合窗 cohort 的插件 DLL 批跑驱动（2026-11-24）
-- 由插件 cpp\lua_host.exe 运行（cwd = cpp\，package.cpath 取 build/Release）。
-- 两种模式（环境变量 LM_MODE），输出均为追加写，已完成的 wid 由调用方去重：
--   gen  LM_JOBS: wid \t ctx \t code
--        N-gram 组句（生产参数，与源码版 llm_ngram.cc 同构）
--        LM_OUT: wid \t rank \t ngram分 \t lead \t lead_end \t end:word;… \t text
--   full LM_JOBS: wid \t ctx \t 文本\x1f文本…（≤5）
--        全句 CE（生产 score_full：完整候选 token 的条件 CE 和，上文截 max_ctx）
--        LM_OUT: wid \t text \t sum_ce \t ntok
-- 其余环境变量：LM_MODEL / LM_THREADS / LM_MAXCTX，gen 模式另有
-- LM_GRAM / LM_DICT / LM_FILES / LM_CACHE / LM_PARAMS（见 luna_mixed_lib.py）。
package.cpath = [[build/Release/?.dll]]

local function env(name)
  local v = os.getenv(name)
  if v == nil or v == "" then return nil end
  return v
end

local mode = assert(env("LM_MODE"), "LM_MODE not set")
local jobs_path = assert(env("LM_JOBS"), "LM_JOBS not set")
local out_path = assert(env("LM_OUT"), "LM_OUT not set")

local llm = require("rime_llm")
local model = env("LM_MODEL")
local thr = env("LM_THREADS")
local maxctx = env("LM_MAXCTX")
if model then llm.model_path = model end
if thr then llm.n_threads = tonumber(thr) end
if maxctx then llm.max_ctx = tonumber(maxctx) end

local fi = assert(io.open(jobs_path, "r"))
local fo = assert(io.open(out_path, "a"))
local n = 0

if mode == "gen" then
  local gram = assert(env("LM_GRAM"), "LM_GRAM not set")
  local dict_dir = assert(env("LM_DICT"), "LM_DICT not set")
  local files = {}
  for name in string.gmatch(env("LM_FILES") or "", "[^,]+") do
    files[#files + 1] = name
  end
  local params = {}
  for k, v in string.gmatch(env("LM_PARAMS") or "",
                            "([%w_]+)=([%w%.%-]+)") do
    params[k] = tonumber(v) or v
  end
  local t0 = os.clock()
  local ok, err = llm.ngram_load(gram, dict_dir, files, "", env("LM_CACHE"))
  print(string.format("[lm] ngram_load ok=%s %s (%.1fs)", tostring(ok),
                      tostring(err or ""), os.clock() - t0))
  if not ok then error("ngram_load failed: " .. tostring(err)) end
  print("[lm] " .. tostring(llm.ngram_status()))
  for line in fi:lines() do
    local wid, ctx, code = line:match("^([^\t]*)\t([^\t]*)\t([^\t\r]*)")
    if wid then
      local cands = llm.ngram_sentences(code, ctx, params)
      if cands then
        for i, c in ipairs(cands) do
          local ws = {}
          for _, w in ipairs(c.words or {}) do
            ws[#ws + 1] = tostring(w.e) .. ":" .. tostring(w.w)
          end
          fo:write(string.format("%s\t%d\t%.6f\t%s\t%d\t%s\t%s\n", wid, i,
                                 c.score, tostring(c.lead or ""),
                                 tonumber(c.lead_end) or -1,
                                 table.concat(ws, ";"), c.text))
        end
      end
      n = n + 1
      if n % 500 == 0 then
        fo:flush()
        print(string.format("[lm] gen %d (%.1fs)", n, os.clock() - t0))
      end
    end
  end
elseif mode == "full" then
  llm.score("预热", { "x" })
  local t0 = os.clock()
  while not llm.is_ready() do
    if os.clock() - t0 > 180 then error("model load timeout") end
  end
  llm.score_full("今天天气", { "很好", "很坏" })  -- 预热：创建 g_full_ctx
  print("[lm] model ready")
  t0 = os.clock()
  for line in fi:lines() do
    local wid, ctx, tail = line:match("^([^\t]*)\t([^\t]*)\t([^\r]*)")
    if wid then
      local cands, seen = {}, {}
      for t in (tail .. "\x1f"):gmatch("([^\x1f]*)\x1f") do
        if t ~= "" and not seen[t] and #cands < 5 then
          seen[t] = true
          cands[#cands + 1] = t
        end
      end
      if #cands >= 1 then
        local okf, r = pcall(llm.score_full, ctx, cands)
        if okf and type(r) == "table" then
          for i, c in ipairs(r) do
            fo:write(string.format("%s\t%s\t%.6f\t%d\n", wid, cands[i],
                                   c.sum_ce, c.ntok))
          end
        else
          print(string.format("[lm] score_full failed wid=%s %s", wid,
                              tostring(r)))
        end
      end
      n = n + 1
      if n % 200 == 0 then
        fo:flush()
        print(string.format("[lm] full %d (%.1fs)", n, os.clock() - t0))
      end
    end
  end
else
  error("unknown LM_MODE: " .. mode)
end
fi:close()
fo:close()
print(string.format("[lm] %s done jobs=%d", mode, n))
