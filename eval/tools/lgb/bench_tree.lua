-- bench_tree.lua — 平坦树表推理基准（2026-11-25，LGB / XGB 通用）
-- env:
--   TREE_MODEL = dofile 的模型表（trees={f,t,l,r,v} 或 {f,t,l,r}）
--   TREE_ROWS  = 迭代次数（默认 20000）
--   TREE_CMP   = "le"（LightGBM，x <= t）| "lt"（XGBoost，x < t）
--   TREE_BASE  = 加性偏置（XGBoost 的 logit(base_score)），默认 0
-- 注意：Lua 5.4 无 float32；XGBoost 表在 double 下只作**成本**参考
-- （分支语义 float32 的原版见 eval/calib/xgb_vs_lgb.py 的对拍）。
local path = assert(os.getenv("TREE_MODEL"), "TREE_MODEL not set")
local rows = tonumber(os.getenv("TREE_ROWS") or "20000")
local cmp = os.getenv("TREE_CMP") or "le"
local base = tonumber(os.getenv("TREE_BASE") or "0")
local t0 = os.clock()
local M = dofile(path)
local load_ms = (os.clock() - t0) * 1000

local nf = #(M.feats or {})
if nf == 0 then nf = 41 end
math.randomseed(20261125)
local X = {}
for i = 1, rows do
  local x = {}
  for k = 1, nf do x[k] = math.random() * 20.0 - 2.0 end
  X[i] = x
end

local nodes = 0
for i = 1, #M.trees do nodes = nodes + #M.trees[i].f end

local function pred(x)
  local s = base
  local trees = M.trees
  for i = 1, #trees do
    local tr = trees[i]
    local f, t, l, r, v = tr.f, tr.t, tr.l, tr.r, tr.v
    local node = 1
    if cmp == "lt" then
      while f[node] ~= 0 do
        if x[f[node]] < t[node] then node = l[node] else node = r[node] end
      end
    else
      while f[node] ~= 0 do
        if x[f[node]] <= t[node] then node = l[node] else node = r[node] end
      end
    end
    s = s + v[node]
  end
  return s
end

for i = 1, math.min(2000, rows) do pred(X[i]) end
local t1 = os.clock()
local acc = 0.0
for i = 1, rows do acc = acc + pred(X[i]) end
local dt = os.clock() - t1
print(string.format(
  "[tree] trees=%d nodes=%d rows=%d cmp=%s load=%.1fms per=%.1fus",
  M.n or #M.trees, nodes, rows, cmp, load_ms, dt / rows * 1e6))
