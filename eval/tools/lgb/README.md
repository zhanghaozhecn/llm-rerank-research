# eval/tools/lgb — 树模型导出的 Lua 侧基准/对拍（2026-11-25）

配套 `eval/calib/prefix_confirm_model.py`（`export` 子命令把 LightGBM 导成
平坦节点表）：

```
python eval/calib/prefix_confirm_model.py export --trees 150 --leaves 15 \
    --out eval/lgb/prefix_gate.lua
```

三个脚本都用插件版 `rime-llm-rerank/cpp/lua_host.exe` 跑（Lua 5.4），
cwd 无所谓、路径全用环境变量传：

| 脚本 | 环境变量 | 作用 |
|---|---|---|
| `bench_lgb.lua` | `LGB_MODEL`、`LGB_ROWS` | 单次判门耗时（µs/次，随机特征向量） |
| `load_lgb.lua` | `LGB_MODEL` | 模型 dofile 加载耗时（一次） |
| `parity_lgb.lua` | `LGB_MODEL`、`LGB_ROWS`=特征 TSV | 逐行预测 → stdout，与 LightGBM 对拍 |

实测（本机、已降频）：150 树/15 叶 = 4,350 节点 / 130 KB / 加载 5 ms /
判门 **34 µs**；对拍 |Δp| 5e-13。单键 CE ≈ 100 ms、组句 14–16 ms，
所以判门成本可忽略。
