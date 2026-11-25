-- lgb model inference benchmark (2026-11-25)
-- env: LGB_MODEL = path to the dumped model table; LGB_ROWS = iterations
local path = assert(os.getenv("LGB_MODEL"), "LGB_MODEL not set")
local rows = tonumber(os.getenv("LGB_ROWS") or "20000")
local M = dofile(path)

local nf = #(M.feats or {}) > 0 and #M.feats or 41
-- 伪随机特征向量（固定种子，模拟真实分支路径规模）
math.randomseed(20261125)
local X = {}
for i = 1, rows do
  local x = {}
  for k = 1, nf do x[k] = math.random() * 20.0 - 2.0 end
  X[i] = x
end

local function pred(x)
  local s = 0
  local trees = M.trees
  for i = 1, #M.trees do
    local tr = trees[i]
    local f, t, l, r, v = tr.f, tr.t, tr.l, tr.r, tr.v
    local node = 1
    while f[node] ~= 0 do
      if x[f[node]] <= t[node] then node = l[node] else node = r[node] end
    end
    s = s + v[node]
  end
  return 1.0 / (1.0 + math.exp(-s))
end

-- 预热
for i = 1, math.min(2000, rows) do pred(X[i]) end

local t0 = os.clock()
local acc = 0.0
for i = 1, rows do acc = acc + pred(X[i]) end
local dt = os.clock() - t0
print(string.format(
  "[lgb] trees=%d rows=%d clock=%.3fs per=%.1fus acc=%.3f",
  M.n, rows, dt, dt / rows * 1e6, acc / rows))
