-- 模型加载耗时（dofile 一次）
local path = assert(os.getenv("LGB_MODEL"))
local t0 = os.clock()
local M = dofile(path)
local dt = os.clock() - t0
local nodes = 0
for i = 1, #M.trees do nodes = nodes + #M.trees[i].f end
print(string.format("[load] trees=%d nodes=%d clock=%.1fms", #M.trees, nodes, dt * 1000))
