-- sgen_llm_greedy.lua — LLM 首 token CE 贪心组句（可行性验证）
-- 由 lua_host.exe 运行（cwd = 插件 cpp\，package.cpath 取 build/Release）：
--   lua_host.exe <本文件>
-- 环境变量：
--   GRD_JOBS   输入 TSV：sid \t before \t pos|end|word\x1e...（每位置 ≤N 条边）
--   GRD_OUT    输出 TSV：sid \t text
--   GRD_MODEL  模型路径（缺省 = DLL 内置默认）
--   GRD_THREADS 推理线程数
-- 贪心：pos=0，每步取 by_pos[pos] 的候选，llm.score(before+prefix, cands)
-- 选最高分（负加权 CE，越大越好），前进到词尾。
package.cpath = [[build/Release/?.dll]]

local jobs_path = assert(os.getenv("GRD_JOBS"), "GRD_JOBS not set")
local out_path = assert(os.getenv("GRD_OUT"), "GRD_OUT not set")
local model = os.getenv("GRD_MODEL")

local llm = require("rime_llm")
if model and model ~= "" then llm.model_path = model end
local thr = os.getenv("GRD_THREADS")
if thr and thr ~= "" then llm.n_threads = tonumber(thr) end

llm.score("预热", { "x" })
local t0 = os.clock()
while not llm.is_ready() do
    if os.clock() - t0 > 120 then error("model load timeout") end
end
print("[greedy] model ready")

local fi = assert(io.open(jobs_path, "r"))
local fo = assert(io.open(out_path, "wb"))
local n_done = 0
for line in fi:lines() do
    line = line:gsub("\r$", "")
    local sid, before, edges_s = line:match("^([^\t]*)\t([^\t]*)\t(.*)$")
    if sid then
        local by_pos = {}
        local code_len = 0
        for e in (edges_s .. "\x1e"):gmatch("([^\x1e]*)\x1e") do
            local pos, epos, w, wt = e:match("^(%d+)|(%d+)|(.+)|(-?[%d%.]+)$")
            if pos then
                pos = tonumber(pos)
                epos = tonumber(epos)
                by_pos[pos] = by_pos[pos] or {}
                table.insert(by_pos[pos], { endp = epos, word = w, wt = tonumber(wt) or 0 })
                if epos > code_len then code_len = epos end
            end
        end
        local prefix = ""
        local pos = 0
        local steps = 0
        while pos < code_len and steps < 64 do
            steps = steps + 1
            local outs = by_pos[pos]
            if not outs or #outs == 0 then break end
            local cands = {}
            for _, o in ipairs(outs) do cands[#cands + 1] = o.word end
            local best = outs[1]
            if #cands >= 2 then
                -- 分批：llm.score 一次最多 n_seq_max(12) 个候选
                local BATCH = 11
                local bi, bs = 1, -1e18
                local alpha = tonumber(os.getenv("GRD_ALPHA") or "0")
                local use_mean = (os.getenv("GRD_MEAN") == "1")
                local kws = { 1.0, 1.1, 0.6 }
                for b = 1, #outs, BATCH do
                    local hi = math.min(b + BATCH - 1, #outs)
                    local batch = {}
                    for i = b, hi do batch[#batch + 1] = outs[i].word end
                    if #batch >= 2 then
                        local ok, ret = pcall(function()
                            return llm.score(before .. prefix, batch)
                        end)
                        if ok and type(ret) == "table" then
                            local ok2, s = pcall(function()
                                return llm.get_scores()
                            end)
                            local ok3, ce3 = pcall(function()
                                return llm.get_ce3()
                            end)
                            if ok2 and type(s) == "table" then
                                for i = b, hi do
                                    local w = outs[i].word
                                    local v = s[w]
                                    if use_mean and type(ce3) == "table"
                                       and type(ce3[w]) == "table" then
                                        local ws, wsum = 0, 0
                                        for k = 1, 3 do
                                            local c = ce3[w][k]
                                            if type(c) == "number" then
                                                ws = ws + kws[k] * c
                                                wsum = wsum + kws[k]
                                            end
                                        end
                                        if wsum > 0 then v = -ws / wsum end
                                    end
                                    if type(v) == "number" then
                                        v = v + alpha * (outs[i].wt or 0)
                                        if v > bs then bs, bi = v, i end
                                    end
                                end
                            end
                        end
                    end
                end
                if bs > -1e17 then best = outs[bi] end
            end
            prefix = prefix .. best.word
            pos = best.endp
        end
        fo:write(sid .. "\t" .. prefix .. "\n")
        n_done = n_done + 1
        if n_done % 20 == 0 then
            io.stderr:write("[greedy] " .. n_done .. "\n")
        end
    end
end
fo:close()
fi:close()
print("[greedy] done " .. n_done)
