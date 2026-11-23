-- sgen_latency.lua — 5 个组句候选：生产 3-token 与全句 CE 的延迟基准
-- 由 lua_host.exe 运行（cwd = 插件 cpp\，package.cpath 取 build/Release）：
--   lua_host.exe <本文件>
-- 环境变量：
--   LAT_JOBS   输入 TSV：sid \t ctx \t 候选1\x1f候选2...（≤5 个）
--   LAT_OUT    输出 TSV：sid \t mode(3tok|full) \t ms \t ntok
--   LAT_MODEL  模型路径（缺省 = DLL 内置默认）
-- 说明：每句先跑 3-token（生产 score_batch 路径），再跑 score_full（整句 CE）。
-- 全句上下文 g_full_ctx 在预热时创建（不计入测量）；每句 ctx 预填充计入全句耗时。
package.cpath = [[build/Release/?.dll]]

local jobs_path = assert(os.getenv("LAT_JOBS"), "LAT_JOBS not set")
local out_path = assert(os.getenv("LAT_OUT"), "LAT_OUT not set")
local model = os.getenv("LAT_MODEL")

local llm = require("rime_llm")
if model and model ~= "" then llm.model_path = model end
local thr = os.getenv("LAT_THREADS")
if thr and thr ~= "" then llm.n_threads = tonumber(thr) end

-- 触发加载 + 等就绪
llm.score("预热", { "x" })
local t0 = os.clock()
while not llm.is_ready() do
    if os.clock() - t0 > 120 then error("model load timeout") end
end
-- 预热：创建 g_full_ctx（首次调用含上下文创建，不计入）
llm.score_full("今天天气", { "很好", "很坏" })
print("[lat] model ready")

local fi = assert(io.open(jobs_path, "r"))
local fo = assert(io.open(out_path, "wb"))
local n_jobs, n_ok = 0, 0
for line in fi:lines() do
    line = line:gsub("\r$", "")
    local sid, ctx, tail = line:match("^([^\t]*)\t([^\t]*)\t(.*)$")
    if sid then
        n_jobs = n_jobs + 1
        local cands = {}
        for t in (tail .. "\x1f"):gmatch("([^\x1f]*)\x1f") do
            if t ~= "" then cands[#cands + 1] = t end
        end
        if #cands >= 2 then
            -- 3-token（生产 score_batch）：先热一次 prep（同 ctx），再计时
            -- —— 生产路径命中 prepare 缓存，计时只含 Step2/3（不含 ctx 预填充）
            llm.score(ctx, cands)
            local a = llm.now_ms()
            local r3 = llm.score(ctx, cands)
            local b = llm.now_ms()
            fo:write(string.format("%s\t3tok\t%.3f\t%d\n", sid, b - a,
                                   r3 and #r3 or 0))
            -- 全句 CE
            local c = llm.now_ms()
            local rf = llm.score_full(ctx, cands)
            local d = llm.now_ms()
            local maxn = 0
            if rf then
                for i = 1, #rf do
                    if rf[i].ntok > maxn then maxn = rf[i].ntok end
                end
            end
            fo:write(string.format("%s\tfull\t%.3f\t%d\n", sid, d - c,
                                   maxn))
            n_ok = n_ok + 1
        end
    end
end
fi:close()
fo:close()
print(string.format("[lat] jobs=%d ok=%d", n_jobs, n_ok))
