-- precommit_conf_score.lua — 批量给「头部词候选」打分（预提交效果研究）
-- 由 lua_host.exe 运行（cwd 自动切到 cpp\，package.cpath 取 build/Release）：
--   lua_host.exe <本文件>   （环境变量传参）
-- 环境变量：
--   PM_JOBS   输入 TSV：sid \t i \t ctx \t 候选1\x1f候选2...
--   PM_OUT    输出 TSV：sid \t i \t 候选 \t score（每次评分一行）
--   PM_MODEL  可选模型路径（缺省 = DLL 内置 Qwen3-0.6B 默认路径）
-- 说明：llm.score 只接受 ≤ n_seq_max(12) 个候选；ctx 为空/候选<2 返回 nil，
-- 本脚本跳过（不写行）。评分失败（哨兵分）仍写出，报告侧按 -1e9 过滤。
package.cpath = [[build/Release/?.dll]]

local jobs_path = assert(os.getenv("PM_JOBS"), "PM_JOBS not set")
local out_path = assert(os.getenv("PM_OUT"), "PM_OUT not set")
local model = os.getenv("PM_MODEL")

local llm = require("rime_llm")
if model and model ~= "" then llm.model_path = model end

-- 预热 + 等模型就绪（lua_host 下 is_ready 轮询；超时 120s）
llm.score("预热", { "x" })
local t0 = os.clock()
while not llm.is_ready() do
    if os.clock() - t0 > 120 then error("model load timeout") end
end
print(string.format("[score] model ready %.0fs", os.clock() - t0))

local fi = assert(io.open(jobs_path, "r"))
local fo = assert(io.open(out_path, "wb"))

local n_jobs, n_ok, n_bad, n_scored = 0, 0, 0, 0
for line in fi:lines() do
    line = line:gsub("\r$", "")   -- 防 CRLF 残留（末候选会带 \r）
    local sid, i, ctx, tail = line:match("^([^\t]*)\t([^\t]*)\t([^\t]*)\t(.*)$")
    if sid then
        n_jobs = n_jobs + 1
        local cands = {}
        for t in (tail .. "\x1f"):gmatch("([^\x1f]*)\x1f") do
            if t ~= "" then cands[#cands + 1] = t end
        end
        if #cands >= 2 then
            local ok, ret = pcall(function() return llm.score(ctx, cands) end)
            local sc = nil
            if ok and type(ret) == "table" then
                local ok2, s = pcall(function() return llm.get_scores() end)
                if ok2 and type(s) == "table" then sc = s end
            end
            -- 防陈旧：本次评分必须覆盖全部候选（否则是上一次快照残留）
            local complete = sc ~= nil
            if complete then
                for _, t in ipairs(cands) do
                    if type(sc[t]) ~= "number" then complete = false; break end
                end
            end
            if complete then
                n_ok = n_ok + 1
                for _, t in ipairs(cands) do
                    fo:write(string.format("%s\t%s\t%s\t%.6f\n", sid, i, t, sc[t]))
                    n_scored = n_scored + 1
                end
            else
                n_bad = n_bad + 1
            end
        end
    end
end
fi:close()
fo:close()
print(string.format("[score] jobs=%d ok=%d skipped=%d rows=%d",
                    n_jobs, n_ok, n_bad, n_scored))
