-- 模型导出数值对拍：逐行读 bench_rows.tsv → 预测 → 与 LightGBM 参考值比
local model = assert(os.getenv("LGB_MODEL"))
local instr = assert(os.getenv("LGB_ROWS"))
local M = dofile(model)

local function predict(x)
  local s = 0
  for i = 1, #M.trees do
    local tr = M.trees[i]
    local f, t, l, r, v = tr.f, tr.t, tr.l, tr.r, tr.v
    local node = 1
    while f[node] ~= 0 do
      if x[f[node]] <= t[node] then node = l[node] else node = r[node] end
    end
    s = s + v[node]
  end
  return 1.0 / (1.0 + math.exp(-s))
end

local nf = #(M.feats or {}) > 0 and #M.feats or 41
local maxdiff = 0.0
local n = 0
for line in io.lines(instr) do
  local x = {}
  local k = 0
  for tok in line:gmatch("([^\t]+)") do
    k = k + 1
    x[k] = tonumber(tok)
  end
  if k == nf then
    local p = predict(x)
    n = n + 1
    if n <= 5 then print(string.format("row %d p=%.12f", n, p)) end
    io.write(string.format("%.12f\n", p))
  end
end
io.stderr:write(string.format("[parity] rows=%d\n", n))
