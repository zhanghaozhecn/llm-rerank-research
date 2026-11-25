# eval/ — 评测工具链（2026-11-18 子目录重组）

42 个核心脚本按用途分 7 个子目录（重组前一轮大清洗 140→40、二轮合并 40→30；
2026-11-18 新增延迟/扫参/前沿/拟合工具 7 个：`lib/gguf_bpe`、`calib/fusion_sweep_all`、
`bench/latency_synth`、`bench/latency_grid`、`tools/pareto_front`、`tools/plot_latency`、
`tools/fit_prefill`；2026-11-20 新增 `calib/matrix_2x2`（LLM×词频四格，复用
fusion_sweep_all 装载器）与 `tools/audit_article_session`（真机文章逐词审计：
文章↔语料↔debug↔账本四方对照，见 freq-fusion-research.md 审计节）；
旧脚本见 `legacy-20261117/removed-scripts-20261117/`）。报告与工作数据 TSV
统一落 **eval 根**（`corpus_std.EVAL`），与本目录既有历史产物同目录。

## 目录与脚本

- **lib/**（10）共享模块，均可 import 亦可直接运行：
  `corpus_std`（口径单源: PROJ/EVAL、词典/语料解析、FormulaReplay、
  window_key、CE 缓存规范、**`is_nofreq`**——2026-11-24：W 行 `flags`
  含 `nofreq` = 无上文窗，窗内无 CE，CE/融合分/名次算术分析须剔除或
  单独统计）、`summary_4x4`（4×4 终表 + cohort_matrices/
  user_matrices/build_ce_matrix）、`derive_pairfix_ratio`（load_w2codes/
  load_domain/FADE 库；**load_domain 的累计项按词+码计数 = 生产口径**）、
  `calibrate_model`（kCeW×β 标定，`--protocol split|4half`）、
  `cohort_layer1_feat`（layer_eval）、`prep_cohort`（队列模拟引擎）、
  `pairfix_pron`（读音证据库）、`audit_xlsx`（TSV↔XLSX）、
  **`gguf_bpe`（2026-11-18：GGUF 词表复刻的字节级 BPE，选 3-token 候选/量 ctx 覆盖）**、
  **`wxgram`（2026-11-23：万象 LTS `.gram` 纯 Python 读取——Rime::Grammar/1.0
  与 DartsGrammar/3 双格式、Darts 双数组 lookup、`Octagram::Query` 复刻）**
- **prep/**（7）语料/队列/用户域准备：`build_segment_cache`、
  `cohort_unigram`、`hanlp_seg_worker`（HanLP 常驻子进程, 由
  corpus_std.ensure_segment_cache 拉起）、`prep_cohort_full`（全枚举链）、
  `prep_user_sc`、`build_user_corpus`、`user_corpus_eval`
- **ce/**（1）`run_ce_cohort` = CE 跑批统一入口：anchors/samples/input
  三源 × 分片/线程池并行 × 断点续跑 × `--seed-from` 旧链播种
- **calib/**（20）标定与首选率：`analyze_sc`（单字/词码对 5 合 1）、
  `analyze_sel_by_code`、`grid_ctx_cand`（ctx×候选数网格，生产参数）、
  `compare_models`（多模型同窗对照）、`fusion_joint`（单格联合扫参）、
  **`fusion_sweep_all`（2026-11-18：逐 (模型,ctx,k) 全网格扫参，
  输出 err / honest(split-half) / err_prod）**、
  **`matrix_2x2`（2026-11-20：LLM×词频四格（引擎序/层1/纯CE/两层融合），
  `--ce` 传完整 CE 目录名，固定生产点参数，双链可复算）**、
  **`length_formula_sweep`（2026-11-24：CE 权重 × 长度项联合标定——F0 现行 /
  F1 加权均值 / F2 n^α / F3 λ·n / F4 组合；复用 fusion_sweep_all 装载器）、
  `lenformula_compare`（定点复核 / `--isolate` 固定权重隔离 α/λ）**、
  **`build_luna_mixed`（2026-11-24：luna 混合窗 cohort——句子→逐字音节→
  窗口网格→无头引擎探针→CE/组句/全句 CE 批补，子命令
  sentences/windows/ce_input/gen/genfull/assemble）、`luna_mixed_lib`
  （探针、插件 DLL 批跑、按需补算的公共件 + `luna_mixed_gen.lua` 驱动）、
  `luna_mixed_eval`（混合窗上的排序公式族评测）、`prefix_confirm_2x2`
  （前缀确认「窗口路径 × 句级路径」2×2：逐窗静态 + 整句打字动态仿真）、
  `guard_syllables`（2026-11-24：前缀确认守卫标定——码数 vs 音节数，
  按「lead 音节 × 剩余音节」出精度/K 敏感表；结论 = 剩余 ≥2 音节、单字 lead 无例外
  ——2026-11-24 真机定案）、
  **`prefix_confirm_precision`（2026-11-25：前缀确认精度的上下文/置信度
  分析——真实上文长/短、空上文、仅前缀 m 字三态 + 规则对照（首 token CE /
  ngram 句分差 d / 窗口融合 margin wm）+ 逻辑回归 PR + 句级 bootstrap；
  结论 = 加 `ce1≤11` 把精度 95.27→97.15%（确认量 95.3%，K=2/5/12 全部
  上升），`ce1≤11∧d≥10` → 98.94% @ 60.8%；空/1 字上文与第 2 次确认是
  弱档，见研究库 2026-11-25 节）**、
  **`prefix_confirm_chain`（2026-11-25：整句链式仿真——80 句逐键打完整句、
  理想选词手动收尾、同句多次确认的序位/上下文精度；产物
  `eval/prefix_confirm_chain.txt`）**、
  **`token_bucket_sweep`（2026-11-25：按候选总 token 数分桶的 CE 系数标定
  ——1-token 固定 1，其余 5 个系数 (a21,a22,a31,a32,a33) 随机采样+坐标下降
  联合搜索；句子分半诚实；产物 `eval/token_bucket_sweep.txt/.tsv`。
  结果 = wrong −0.15pp / nonideal −1.54pp，收益主要在 2-token 窗、参数跨半
  不稳，未改生产常数）**、
  **`build_prefix_big`（2026-11-25：前缀确认大 cohort——wiki 12,001 句（8×）
  与 luna_mixed 完全同口径；**只对判定区窗算 CE**（P≥0.5 ∧ lead 在引擎池同
  span ∧ 剩余 ≥2 音节，约 21% 窗），CE 落 luna_mixed 共用缓存；子命令
  sentences/windows/gen/ce/assemble）**、
  **`prefix_confirm_model`（2026-11-25：46 特征 + LightGBM/逻辑回归 + 结构性
  错判挖掘——句子奇偶分半诚实协议；易错项判据 = 错误率 CI 下界 > 1/(1+K) 才
  禁用、点估计超才加强；产物 `eval/prefix_confirm_model.txt`。
  结果（15,746 行 / 13,482 触发 / 错 820）：树模型 **+4.45pp**（93.92→
  98.37%，同触发量）、3 特征线性 `logitP/ce1/wm` **+3.10pp**；错确认
  **99.1% 是同音异字**；**ta 族（他/它/她/他们/它们/她们）40 触发 13 错
  （user 域 8/10）→ 禁用 +0.07~0.08pp（user 域 +0.55pp）**，数据挖掘式
  名单不做（两半不重合）；树模型部署成本 150 树/15 叶 = 130KB / 34µs
  （`export` 子命令导出平坦 Lua 表，与 LightGBM 对拍 <1e-12））**、
  **`export_prefix_gate`（2026-11-25：训练 + 双导出落地物——41 个可部署
  特征 / 300 树 / τ 由 OOF 90% 量分位；同时写 Lua 平坦表与 C++ 头文件
  （含 8,421 字字频表），落地到插件 `user/prefix_gate_model.lua` 与源码
  `librime/src/rime/gear/prefix_gate_model.h`）**
- **pairfix/**（4）现行链：`pairfix_pipeline`（SOP 驱动）、
  `derive_pairfix_full`（唯一现行推导）、`audit_tool`（batch/merge/deploy
  三合一）、`audit_review_union`（M1–M8 机械审查）
- **bench/**（6）`speed_bench`（推理速度）、`ctx_latency`（ctx×设备均值）、
  **`latency_synth`（2026-11-18：合成最坏情况候选取自 anchors，
  `--metric total|prep|step23` 分别给整窗/预填充/候选段，逐窗 SIM_* 计时）**、
  **`latency_grid`（ctx×k×设备的分位延迟，均值口径）**、
  `ab_upgrade`（llama.cpp 升级 A/B）、`gguf_types`（GGUF 张量体检）
- **tools/**（19）`check_records_v2`、`run_tests`、`eval_rerank`、
  **`lgb/`（2026-11-25：树模型导出的 Lua 侧基准/加载/对拍三脚本，
  配套 `calib/prefix_confirm_model.py export`；150 树/15 叶实测 130 KB /
  34 µs/次、对拍 5e-13）**、
  **`pareto_front`（2026-11-18：(中位延迟, 选重率) 帕累托前沿，可 `--metric
  err|honest|prod`、可选 ctx 池化）**、**`plot_latency`（②③ 的 2×2 拼图，部署档 Q4_K_M）**、
  **`fit_prefill`（T_prefill 线性拟合 + ctx* 选择：<150ms 且 ≤20；`--extra` 可拟合量化变体）**、
  **`audit_article_session`（2026-11-20：真机文章逐词审计，文章↔语料↔debug↔账本四方
  对照 + W.ctx 文档核对 + 融合算术 + 账本对账；`--since/--until` 定会话）**、
  **`wiki_fp_pos`（2026-11-23：wiki 全拼候选位置分布，Collect 上限 16 的依据）**、
  **`precommit_conf_collect` / `precommit_conf_score.lua` /
  `precommit_conf_report`（2026-11-23：「确定前缀预提交」模式效果研究——
  真实引擎取头候选 → LLM 词级 CE → margin→准确率/覆盖率/按键成本；
  结论见设计文档 §十）**、
  **`sentence_gen_gram` / `sgen_llm_rerank` / `sgen_llm3_rerank`（2026-11-23：
  组句候选生成 + LLM 句级重排——任意 Rime 格式 `.gram`（万象 LTS / 官方
  八股文 bgw / 千问内置）+ 万象词库词级 beam 生成 top-5：万象两区命中率
  71%/62%（词表内 83%/85%）、官方 67%/53%、千问 54%/46%；LLM 整句 CE 重排
  top-1 万象 64%/57%、官方 62%/49%、千问 50%/43%；`sgen_llm3_rerank` 用生产
  DLL 的 3-token 加权 CE 做安全基线（万象 56%/41%，明显弱于全句）；配套
  `cpp/sent_score.cpp`（整句 CE 打分，链接同一 llama 静态库）；结论见设计
  文档 §十一）**、
  **`sentence_gen_gram --lm`（2026-11-23 晚：商业输入法 bigram 适配器接入——
  `lib/wxbigram_lm.py`（微信 `lmcnmac.bin`，纯哈希表：CityHash(词对 UTF-8)
  → 16-bit 量化条件分）、`lib/doubao_bigram_lm.py`（豆包 `sys_bigram` S6 +
  `sys.dat` 词表，id 索引 + Katz 退避）；同口径 A/B：微信 56%/41%、
  豆包 56%/32%（词表内 68.4/69.7）；格式推导与阻塞点见
  `商业输入法ngram逆向-微信豆包-2026-11-23.md`）**、
  **`sgen_latency.lua` / `sgen_latency.py`（2026-11-23：5 个组句候选的评分
  延迟基准——生产 3-token vs 插件新增 `llm.score_full` 整句 CE；199 句语料，
  8 线程 p50 65.5 / 252.2 ms；同日两路合并到同一 context 后 56.2 / 200.1 ms，
  回归 `51.0 + 3.99×Σtoken`；结论见设计文档 §11.7，跨键复用测量见 §11.8）**、
  **`ngram_parity_check.py`（2026-11-23 晚：生产组句器对拍——读留档
  `sgen_gram_*.test.jsonl` → 跑插件版 `cpp/ngram_parity.lua`（C++ 组句器）
  → 逐句比对 top-k 文本/分数并复算 hit@k（只数前 k 条）；全参数下
  100/100 一致，生产裁剪参数 14–16ms/句、hit@5 73%/61%；`--t2s` 供繁体
  方案词典口径；结论见设计文档 §11.10）**、
  **`llm_gen_a_index.py` / `llm_gen_a_eval.py`（2026-11-24 夜：**A 路线**——
  拼音约束的 LLM 直接组句，不要 gram、不要 1.7M 词库。索引 = 词表 token 的
  「拼音字母串 → id」（读音取万象单字条、剔叹词野读音、只留简体）；评估 =
  整句/首词/字级 + **约束合规校验**（生成文本拼音逐字母等于输入码）。
  配套 `cpp/llm_gen_a.cpp`（约束 beam search，GPU；另有 `dump-vocab` /
  `selftest` / `next-top` / `mass` / `path-logp` / `beam-ref` 探查模式）与
  `cpp/bench_llm_gen.cpp`（纯延迟基准）。
  **实测（A/B 各 100 句，DIV=4 落点分组 beam）：hit@1 69/63%、hit@5 84/74%、
  首词 89/86%、字级 94.7/91.4%——全面超过 N-gram（50/40、71/62、83/85、85/86）；
  `lead_p≥0.95` 首词精度 95.7/91.9%（同精度覆盖率 70/89%，gram 门仅 42%）。
  代价 = 0.62–0.85s/句（gram 14–16ms CPU）。**注意**：初版曾因两个搜索 bug
  （完成态被丢 + 全局 top-B 压掉长 token 假设）误判为负结论，修正与证据见
  研究库 `freq-fusion-research.md`「2026-11-24 夜（三）」。
  **逐键增量**（`type` 模式，跨键复用前缀 KV）：K=1/rowcap8 每键 40ms（p50 18 /
  p95 80）但 hit@1 掉到 47%；K=2/rowcap24 每键 107ms、54%——质量与延迟强冲突，
  最优形态是 gram 管逐键、A 路线管异步句末。
  **2026-11-24 用户定案：结论记录在案、不采用**（理由 = 逐键延迟进不了按键路径；
  现行架构不变）**

## 约定

- 跨目录 import 一律 `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))`；
  lib 内互导无需 hack（脚本目录自动在 sys.path）。
- `corpus_std.PROJ` = 研究库根、`corpus_std.EVAL` = eval 根——**报告/工作
  数据写 EVAL，链数据写 PROJ**（cohort_* 链、CE 缓存等）。
- 合并映射与归档原件见 `legacy-20261117/removed-scripts-20261117/merged-20261118/README.md`。
