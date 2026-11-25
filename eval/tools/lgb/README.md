# eval/tools/lgb — 树模型导出的 Lua 侧基准/对拍（2026-11-25）

配套 `eval/calib/prefix_confirm_model.py`（`export` 子命令把 LightGBM 导成
平坦节点表）与 `eval/calib/xgb_vs_lgb.py`（XGBoost 同格式导出，含 `base`
偏置）：

```
python eval/calib/prefix_confirm_model.py export --trees 150 --leaves 15 \
    --out eval/lgb/prefix_gate.lua
python eval/calib/xgb_vs_lgb.py --export eval/lgb/xgb_gate.lua
```

四个脚本都用插件版 `rime-llm-rerank/cpp/lua_host.exe` 跑（Lua 5.4），
cwd 无所谓、路径全用环境变量传：

| 脚本 | 环境变量 | 作用 |
|---|---|---|
| `bench_lgb.lua` | `LGB_MODEL`、`LGB_ROWS` | 单次判门耗时（µs/次，随机特征向量） |
| `bench_tree.lua` | `TREE_MODEL`、`TREE_ROWS`、`TREE_CMP`(`le`/`lt`)、`TREE_BASE` | 同上，但通用两语义 + XGB 的 margin 偏置；报加载/节点数/µs |
| `load_lgb.lua` | `LGB_MODEL` | 模型 dofile 加载耗时（一次） |
| `parity_lgb.lua` | `LGB_MODEL`、`LGB_ROWS`=特征 TSV | 逐行预测 → stdout，与 LightGBM 对拍 |

实测（本机、已降频）：150 树/15 叶 = 4,350 节点 / 130 KB / 加载 5 ms /
判门 **34 µs**；对拍 |Δp| 5e-13。300 树/15 叶（已落地档，53 特征 =
41+ss12）= 8,700 节点 / 树表 225 KB / 加载 19 ms / 判门 **76 µs**；
XGBoost 300×深 4 同档也是 8,700 节点 / 208 KB / **76 µs**。单键 CE ≈ 100 ms、
组句 14–16 ms，所以判门成本可忽略。

⚠ `bench_tree.lua` 的 `TREE_CMP=lt` 在 Lua 里仍是 **double** 比较；XGBoost
原版是 `float32(x) < float32(t)`（见 `calib/xgb_vs_lgb.py` 对拍），成本
口径不受影响，但要复现分数必须走 float32。

