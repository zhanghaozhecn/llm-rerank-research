# 语料口径标准（2026-11-05 定案）

> 目的：所有语料分析共用一套采样/分词/编码/重放/键口径，杜绝各脚本口径漂移。
> **单源 = `eval/lib/corpus_std.py`**（2026-11-18 子目录重组后住 lib/）——新脚本禁止重写本文任何函数/常量，一律 import。
> 长度分组实验与标准化动机见 `freq-fusion-research.md` §二十三/§二十四。

## 零、总则

1. **单源铁律**：常量（MAX_CAND/CTX_TAIL/TAU/SEED/ASCII_LIMIT）、词典解析、采样、
   formula_d 重放、window_key 全部只在 corpus_std 定义。window_key 曾有 5 处独立实现，
   全靠"逐字一致"对上缓存——单源后由 import 保证。
2. **两条授权分词口径**（用途不同，并存但语义命名固定）：
   - **自然文本切分**（HanLP + 混排拆边界）——语料线（新）与自由文本评测线共用；
   - **词典重放切分**（jieba + pdsp 注入）——**冻结**，仅 11-02 标定链复现用。
3. **冻结链不追改**：下表所列旧脚本保持原样（历史分析可复现即达目的），新工作不引用。

## 一、语料源

| 语料 | 粒度 | 状态 |
|------|------|------|
| `wiki_clean.txt`（分词注音工程） | 文章（一行一篇） | **唯一 wiki 源**；新口径经 `build_shuffled_corpus` 产 `article_corpus_shuffled.txt` |
| `article_corpus_shuffled.txt` | 文章（打乱后） | **现行规范语料**：整体随机打乱（seed=42），打字模拟按顺序取前缀 |
| `sentences_new.txt` | 句子 | 旧工具/回归专用 |
| `rime_llm_corpus.tsv`（%APPDATA%\Rime） | 真实打字事件流 | **2026-11-17 现行**：W 推理窗/C 提交/M reset/H 配置；v2 单源解析 `corpus_std.iter_*` |
| `llm_training.txt`（旧，已归档） | 真实打字词流 | 历史语料，仅在 `legacy-20261117/` 作等价校验；旧准备链 prep_training 已切 v2 |

### 用户语料 v2（2026-11-17 记录重构）

- 文件 `%APPDATA%\Rime\rime_llm_corpus.tsv`：UTF-8/TAB；`H` 配置起点、`W` 满码推理窗
  （编码/上文/引擎原序前10/评分窗/最终序/状态或 skip）、`C` 提交（词/完整码/名次/ref W）、
  `M` reset（focus/backspace/delete/navigation/enter/undo-clipboard/scene）。
- `%APPDATA%\Rime\rime_llm_debug.txt` 只写语料没有的 P/D/E/R/X，按事件 ID 关联；
  不重复词/码/上下文/候选序。
- 解析单源 `eval/lib/corpus_std.py`：
  - `iter_scenes(strict)`：backspace/delete/undo 回撤前一提交；focus/navigation/enter 分场、
    不回撤（新数据语义）。
  - `iter_scenes(legacy)`：除 scene 外所有 M 都回撤（旧 `←` 语义；迁移旧数据/复现历史）。
  - `iter_commits(include_retracted=...)`：需要原始序列 vs 幸存序列的脚本用。
  - `iter_word_stream()`：旧扁平脚本口径，默认含被回撤提交。
- 旧 `llm_training.txt` 迁移：`migrate_corpus_v2.py`（旧提交→C/M、旧 filter_log 事件→W，
  `legacy=1`）；等价校验 `check_migration_v2.py`（旧训练词流/场景 0 差异）。
  两脚本 2026-11-17 已归档（`legacy-20261117/removed-scripts-20261117/bulk-20261117/`）。
- `window_key`/ctx 重建口径不变：`iter_scenes` 仍用提交词流重建 ctx（避免 CE 缓存失效）；
  W 的真实上文/引擎序/最终序作为可选字段供新分析。

## 二、采样（2026-11-05 用户定案）

- **新口径**：文章级**整体随机打乱**（`build_shuffled_corpus`，seed=42，**ASCII≤20%** 文章级过滤），
  模拟打字**按顺序取前缀**；快速检验取前 **10% / 1%** 数量的文章（`--prefix`）。
  动机：文章内部不分散（连续上文），语料库级随机由"打乱一次 + 顺序前缀"承载，
  实验间可完全对齐（同一打乱语料的 k 篇前缀 = 1% ⊂ 10% ⊂ 100% 嵌套）。
- **ASCII 10%→20%（2026-11-05 用户定案）**：10% 沿自旧管线句子级口径，搬到文章级
  正切在单峰分布（峰值桶 10-15%）的腹部——〖计算机科学〗〖地质学〗类核心条目
  （15-16%，括号拉丁转写/年份所致）整篇被丢，系统性偏掉科技/统计内容。
  20% = 959,461/1,537,490 篇（62%）；30-50% 桶（表格/链接/IPA 堆）仍被挡。
  原始顺序测试（相邻同词条率 0.0%、≤τ 与打乱仅差 1pp）证明 wiki_clean 原序无话题
  结构，打乱不损失任何东西。
- **旧口径（冻结）**：蓄水池抽样 k 篇（seed=42）——11-02 标定链复现入口，勿用于新分析。
- 样本截断：重放产出样本后 shuffle（seed=42）再截 `--max-samples`（避免"取前 M"偏向冷启动样本）。

## 三、分词（2026-11-05 用户定案）

**语料线 = HanLP 自然切分 + OOV 动态规划拆词**，两级结构：

1. **HanLP 层（缓存，方案无关）**：`hanlp_seg_worker.py`（常驻子进程）
   COARSE_ELECTRA_SMALL_ZH；混排 token 按 CJK/非CJK 边界拆（"450毫米"→"450"+"毫米"）。
   `ensure_segment_cache` 增量构建 `segments_hanlp_cache.jsonl`（行号 = 打乱语料行号，
   一次分词长期复用）。**吞吐基准（2026-11-05）**：GPU 单进程 48 art/s（16k 字符预算/
   批 64——更大批反慢，256k 预算 OOM）、CPU 6.4 art/s。多 worker 基建保留
   （`HANLP_BUILD_WORKERS`，part 文件按序拼接）但**默认 1**——§七单 CUDA 进程铁律
   （≥2 worker = ≥2 CUDA 进程，三轮实测全楔死）；协议坑两处：hanlp/torch 加载
   进度条污染 stdout（worker 用 os.dup(1) 复制真 stdout，sys.stdout 改道 stderr）、
   64 篇长文整批 electra 注意力矩阵 CUDA OOM（按字符预算组批 + expandable_segments）。
2. **拆词层（实时，随方案词典变）**：HanLP 词不在当前方案词典 → `DictSplitter.split`
   动态规划拆成词典词：
   - 目标 = **子词概率连乘最大**（log 域求和最小；**不按子词数归一**——惩罚隐含在
     p<1 的连乘里，子词数越少越好）；
   - 子词概率 = 语料 unigram 频率（从分段缓存前缀统计，**方案无关**）；
     语料未见词典词回退 `log(0.01/total)`（强偏向少拆）；
   - 子词必须全部在词典（含单字——单字也是可提交词，产 bump 不产样本）；
   - **无法完整覆盖的 token 整体丢弃**（不产 bump 不产样本，计数上报）。
- **缓存与实时的边界**：缓存只存 HanLP 原始切分（换方案零成本）；拆词在每次重放时
  按当前词典实时做——方案会变，拆词结果不缓存。
- **自由文本评测线**（eval_rerank/eval_prefer/compare_dicts）：同样 HanLP 自然切分，
  不做 DP 拆词（评测的是各方案词典自身的覆盖）。
- **冻结**：jieba + pdsp 注入（`jieba_inject`）仅 11-02 标定链复现用。

## 四、编码（pdsp_dict.yaml 解析）

- 规则：`...` 后开始、跳 `#`、`\t` 分割；`load_pdsp_dict()` 只取**多字词且 4 码**
  （= 部署 `code_pattern .{4}` 口径），word→code 取**文件序第一个**；
  `load_pdsp_words()` 取**全部词含单字**（拆词覆盖用，单字产 bump 不产样本）。
- 窗口：**评测口径 = 5 候选窗（生产镜像, 2026-11-05 用户两步定案）**——生产
  max_candidates 维持 5，测评窗同样只取前 5（目标词不在前 5 的锚点不评 =
  生产不可见）；**预计算仍覆盖 10 个候选**（sim n_seq_max=11；调频把 rank 6~10
  的词提进可见 5 窗时仍有 CE 可比——这是预计算 10 的唯一意义，生产 CE 本就
  实时算）。hit@10 表仅作预计算覆盖参考，**标定/选点一律 hit@5**。
  **终表口径（2026-11-06 用户定案）**：有 CE 参与的模型（LLM±eff）= 5 窗
  hit@5（生产速度约束，10 候选无意义）；其余模型（词典/纯eff/trigram±eff/
  LSTM±eff）= 10 窗 hit@10（打分近零成本，10 候选可负担）——各技术按可部署
  形态对齐，终表逐行标注口径。11+ 位为
  空集（4 码组最大 11、仅 2 组达 11，1380 万窗 pos≥10 命中=0），10 即完备
  预计算深度；prep 输出 `窗11+位命中` 计数防词典扩容。
  sim n_seq_max=11（seq0+10 worker）；生产 rime_llm n_seq_max=12 已够。
  候选 = 同码词表前 10（正确词按词典位次就座）；正确词 pos≥10 剔除
  （prep 输出 `窗11+位命中` 计数）。**11+ 位实测为空集**（2026-11-05）：
  pdsp 多字 4 码组最大规模 11（186,123 组中仅 2 组达 11），1380 万窗中
  pos≥10 命中 = 0；10 窗已覆盖全部可能被调频提到可见 5 窗的词——生产侧
  CE 本就实时计算，10 窗纯属**评测覆盖口径**（调频把 6~10 位词提进可见窗后
  仍有预计算 CE 可评），不是生产能力边界。词典未来扩容若出现更深的组,
  看 `窗11+位命中` 计数再议（>1% 再考虑 15 窗敏感性检查）。

## 五、eff 重放（formula_d）

- 语义：tick = **每词典子词提交 +1**（新口径；冻结链 = 每任意 CJK 切词提交）；
  提交 `d = 1 + d·exp((t_w−t)/τ)`；查询（事前，打分在提交前）
  `eff = d·exp((t_w−t)/τ)`；τ = `corpus_std.TAU` = **800**（2026-11-02 定案）。
- window_key **不含 eff** → τ/β 等参数改变不废分数缓存，只有窗口内容改变才重算。

## 六、上下文与键

- 存储：上文末尾 `CTX_TAIL=120` 字；推理：末 `MAX_CTX_TOK=10` token（=生产 max_ctx）。
- 键：`window_key = md5(去空白ctx + b"\0" + 各窗词 + b"\x01")`——全流程共用。

## 七、推理与分数

- **canonical 后端 = `sim_rerank_gpu`**（管线化版，workers=3 标准姿势；86b94708 镜像纪律；
  GGML_CUDA_GRAPHS=OFF）。**本机单 CUDA 进程铁律（2026-11-05 三轮实测定案）**：8GB
  GeForce + Windows WDDM 下，任何两个 CUDA 进程并发（HanLP worker ×N、HanLP+sim
  混跑）都会显存分配器互踩楔死驱动队列——GPU 100% 但两边零产出。GPU 作业一律串行
  排队；GPU 上跑批期间禁跑其它 CUDA 作业（CPU 分析不受限）。
  落地修正：calibrate_article `RIME_LLM_EVAL_WORKERS` 默认 3→1（同日实测
  默认 3 起跑即楔死现行）。
- **标定链流式契约（2026-11-05 OOM 死机后定案, 详见研究库 §三十）**：prep 两遍式
  （pass1 词流逐行落盘 streams_std.jsonl + 选样; pass2 仅对选中窗重放采 eff）,
  锚点 sel_anchors_std.jsonl 是 sweep 的事实源——**streams/anchors/samples 必须
  同一 prep 进程产出**（不同语料 n 的 logp → DictSplitter 分割不同, 实测错位
  21%），sweep 不再有"按缓存行数兜底重算"。内存峰值 ~300MB（旧版物化 ~15GB
  是一次 OOM 死机成因）。多 τ 重放 = 单遍多 FormulaReplay 实例（8 τ 同时推进）。CPU 引擎仅回归对照用；**CPU/GPU 缓存不合并**
  （量化路径差 0.05~0.4 nats；`RIME_LLM_REUSE_CPU_CACHE=1` 显式 opt-in）。
- **散热限频前置检查（2026-11-05 用户铁律：本机散热弱，满载过热死机，参考 calc-schema
  项目经验）**：GPU 作业前确认 SM 锁频 **2100MHz**（开机任务 `GPU_limit_2100` 持久生效，
  `schtasks /Run /TN GPU_limit_2100` 免提权重锁；撤销 = 管理员 `nvidia-smi -rgc` + 删任务；
  代价约 −20% 吞吐，纯算力受限、结果不变）；长期 CPU 作业前确认**关睿频 + 70% 上限**
  （`PERFBOOSTMODE=0` + `PROCTHROTTLEMAX=70`，电源方案级持久）。两项一条命令核验
  （calc-schema 项目 `稳频.py --查`，去掉 `--查` 则幂等修正；数值规范源在该脚本，
  改目标须同步其常量）。全量 8.6h 量级的作业**必检**，短作业可跳过。
- 冻结数值纪元：eval_logz/build-mt 产物（9 月 dump 族）不与新分析混合。
- 分数符号：sim 输出 = **−加权CE（越大越好）**；eval_logz dump = +cost（越小越好）。
- CE 位置权重 (1.0, 1.13, 0.61) 不做配置项（§20/§23 两次复核），按长度分组已证伪（§23）。

## 八、冻结链与产物对照

**2026-11-06 实验文件清理（用户定案：无效实验删文件、只留结果记录）**：
已删 = 自训 n-gram（trigram_baseline.py / ngram_baseline.py / 万象驱动与
418MB gram 模型 / cpp/ngram_wanxiang.* / darts.h / 分数缓存）/ LSTM
（lstm_lm.py + 两份 162MB 权重）/ 开源神经重排器（test_msime.py + 30MB
safetensors）/ 旧 §三十 n-gram 实验（exp_user_ngram_freq.py）/ 全部运行
日志。**结果记录全保留**（eval/*_report.txt：trigram/lstm/user_lstm/
msime_reranker/user_corpus/user_ce/user_adaptive_ngram/user_ngram_gamma/
user_unigram_delta/gamma_wiki/llm_opt_gamma）+ 本文档与研究文档归档。
user_corpus_eval.py 已摘除 ngram_baseline 依赖（window_of 改用 corpus_std）。
保留 = 核心脚本 40 个（2026-11-17 大清洗；2026-11-18 子目录重组二轮合并后
**30 个 × 7 子目录**，明细见 §十四·补三与 eval/README.md）：lib 共享模块
（corpus_std / summary_4x4 / derive_pairfix_ratio / calibrate_model /
cohort_layer1_feat / prep_cohort / pairfix_pron / audit_xlsx）、prep 准备
（build_segment_cache / cohort_unigram / hanlp_seg_worker / prep_cohort_full /
prep_user_sc / build_user_corpus / user_corpus_eval）、ce/run_ce_cohort
（三合一）、calib（analyze_sc 五合一 / analyze_sel_by_code / grid_ctx_cand /
compare_models）、pairfix（pipeline / derive_pairfix_full / audit_tool 三合一 /
audit_review_union）、bench（speed_bench / ab_upgrade / gguf_types）、tools
（check_records_v2 / run_tests / eval_rerank）；数据链
（segments 缓存、CE、锚点/样本）保留。旧脚本归档
`legacy-20261117/removed-scripts-20261117/`。
冻结记录：早期（τ=200 jieba 时代）脚本与产物仍按本节下表冻结留存。

| 产物/脚本 | 口径 | 状态 |
|-----------|------|------|
| `article_samples_tau200.jsonl` + `article_corpus_4000.txt` + `tmp_sweep_streams.pkl` | 蓄水池 + jieba + τ=200 | **已删（2026-11-18 语料单源定案）**；仅报告留档 |
| prep_samples.py / calibrate_freq_beta.py / sweep_w2w3 / sample_size / sweep_fusion / compare_rerank / step1_singlechar / compare_elw_abs / compare_dicts / list_long_cands / run_tests / eval_rerank / eval_prefer / prep_training | 各自历史口径（头部注明） | 冻结（脚本除 `run_tests`/`eval_rerank` 外已于 2026-11-17 归档；`eval_prefer` 有 seed=42，可复现——2026-11-05 审计更正） |
| `article_corpus_shuffled.txt` + `segments_hanlp_cache.jsonl` | **新口径**（本文档） | 现行（`article_samples_std.jsonl` 已于 2026-11-08 清理，口径以 §十三 cohort 为准） |
| corpus_std.py + prep_cohort.py / prep_cohort_full.py + run_ce_cohort.py + summary_4x4.py + calibrate_model_05b(_i_final).py | corpus_std 单源 + cohort 链 | 现行工具链（2026-11-17 核心集；旧全文重放链已归档） |

## 九、迁移验证记录（2026-11-05）

- sweep_tau_beta τ=200 multiset 强校验：58,955 行 **0 不一致**（迁移前后数字逐位一致）
- calibrate_lenweight 基线：纯CE 0.9317 / +freq β=8 0.9510 精确复现；分量重构对拍
  7,505 条 max|Δ|=0.002（GPU 重跑噪声级）

## 十、标定语料口径（2026-11-06 用户定案；同日 v3 修订为两步法）

- **融合公式常数标定两步法（2026-11-06 v3 定案，现行）**：
  ① **用户真实语料跑全参数联合网格**（6 维 w2×w3×β×τ×γ×λ，68,640 点，
  `eval/calibrate_joint6_user.py`）——用户域定向，argmax 代表用户想要的取向；
  ② **在 wiki 平台内（距 wiki argmax ≤0.05pp）取距用户结果最近的点**为部署值
  （归一距离，τ 按 log2）——wiki 只提供稳健性约束，不参与定向。
  现行部署常数即此法产出：**w2=1.05 / w3=0.45 / β=12 / τ=400 / γ=1 / λ=0.5**
  （该点用户域选重率 1.90%，距用户 argmax 仅 +0.12pp）。点云缓存
  `wiki_joint6_results.npz` / `user_joint6_results.npz`。
- **联合网格取代分维标定**：公式 log 域可加 → 全部参数落在预计算缓存上
  （CE/eff/bigram 矩阵），6 维一次网格分钟级（`eval/calibrate_joint6.py`）；
  历史上"分维扫、逐次定"的协议不再使用。
- **历史口径（2026-11-06 上午，仍适用于非公式常数）**：常数只在 wiki 上标定、
  用户语料只测试——测试集上选参 = 过拟合（实锤：γ 用户域 argmax=4 在 wiki 上
  收益掉到 +0.24pp、γ=8 转负）。两步法 = 该原则的精化：wiki 仍不提供定向，
  只提供平台边界。
- 自适应 n-gram 的标定形态 = **时序前缀计数**（生产精确语义的 wiki 镜像）：
  跨篇滚动历史, 打分点捕获此前计数, 需键过滤控内存（
  eval/calibrate_gamma_wiki.py 为样板）。
- 基线对比模型（LSTM/trigram 等）维持自训自评口径（2026-11-06 用户裁定，
  不受本条约束——本条只管生产融合公式的常数）。

## 十一、描述口径（2026-11-06 用户定案；2026-11-07 CE 行口径根本修正）

- **对外描述与总表一律用选重率 = 1 - 正确率（hit@k）, 越低越好**。
- **CE 行 = 端到端口径（2026-11-07 用户指出旧口径缺陷后修正）**: k=5 生产
  配置下, 目标在前 5 的窗按融合序判定, **目标在前 5 之外的窗按现生产行为
  （尾窗保持引擎序, 目标不可能第一）计失的, 全部窗口参评**——旧的"目标不在
  前 k 不评"条件口径与全窗参评的非 CE 行不可比, 已作废（条件值降级为
  "融合器质量"诊断指标）。非 CE 行（词典/词频/bigram/外部模型）本就全窗
  参评, 不变。
- **不给无 LLM 分的候选加 eff+bigram 融合**（混合融合尾窗实测大幅更差,
  且模拟文件序为静态词典序、真实生产已含引擎调频, 双重计频伪影;
  eval/sweep_kmixed.py）——现行 -huge 尾窗实现正确。
- 文档中历史段落保留当时的数字不改（以段落自身口径为准）。

## 十二、小词库测评基准（2026-11-07 新增）

**位置**：`词库-测评6万/`（pdsp_dict_small6w.yaml 基准词典 + 词音/编码表中间产物）。

**来源与生成**：`05设计数据/词库-测评6万.xlsx`（59,133 多字词，按词频降序）→
`03处理脚本/词音-测评6万.txt`（词\t拼音）→ `拼音转编码.py + 生成编码表.py
--scheme 拼读双拼`（4 码词编码，每码≤9 候选）→ `pdsp_dict_small6w.yaml`
（Rime 词典格式，频序保留）。**全部 4 码、无单字、无简码**。

**质量校验**：与生产 pdsp_dict.yaml 交集 54,945 词编码一致 **99.50%**——差异
两类：① 多音字读音源不同（怎么着/标识/重新 的读音选择）；② 大词典含 2 码
简码（如"物质"=ue），本表恒 4 码。转换引擎与生产编码规则一致。

**口径**：
- 加载：`RIME_LLM_PDSP_DICT=<词库-测评6万/pdsp_dict_small6w.yaml>` 跑 eval
  管线即为小词库口径（48,647 个 4 码 → 59,132 词）。
- **OOV 规则（用户定案）**：小词库无单字、无词频——分词评测时 OOV 一律
  落到**单字最小粒度**（单字概率兜底）。

**补充（2026-11-07 晚定案; 2026-11-08 更新）**：
- **DP 概率口径（用户确认）**：小词库词的概率 = 语料 unigram（非 xlsx 词频，
  频率列只用于基准词典词序）；单字兜底 = log(1/total)。
- ~~前缀自洽重放~~（2026-11-08 已被 §十三 cohort 口径取代，数据文件已删）。
- **CE 跑批（2026-11-08 终版结论）**：**GPU + CUDA graphs = 唯一正确档位
  ~40.2 窗/s**（0.8B Q4; 12 万窗实测 1h）。此前"CPU 3×6 贴核 18–25 窗/s
  胜 GPU 2 倍"的结论已被 llama.cpp keying 补丁推翻——graph 激活失败才是
  GPU 慢的根因（prompt/decode 图共用 nodes[0] 缓存条目 → warmup 永不完成;
  补丁 = 按全节点形状签名哈希分桶, 归档 patches/ggml_cuda_graph_shape_keying.patch,
  镜像本地提交 0d29a383）。注意: **必须用 sim_rerank_gpu.exe**——
  sim_rerank.exe 是 CPU 静态链接, 其 --ngl 99 实际走 CPU（静态 CPU 后端
  先注册 → 动态 CUDA DLL 永不加载, 2026-11-08 实证）。机制细节（发射
  CPU 侧开销/WDDM/多 graph map/KV pad-256）见研究库 §三十七 补记。
- **清理（2026-11-08, 用户指令）**：过期口径实验数据已删除（全文重放链
  streams_std/streams_std_r2/article_samples_std/sel_anchors_std/std_ce_work、
  小词库前缀链 4 件 + std_ce_work_small6w、jieba/τ200/ascii10/r2 布局各代、
  joint6/split_features/decay_ctx npz 缓存、CPU 宇宙 CE 备份）——口径以
  §十三 cohort 为准, 旧脚本保留作方法记录（数据可用语料源+缓存重生成）。

## 十三、cohort 队列模拟标定口径（2026-11-08 新增，取代全文重放标定）

**动机**：旧标定链在 3.18 亿提交（≈真实用户 #tick 2.8 万的 1 万倍）的特征
分布上标定/评估——累计词频项实际是"全局 LM"，量级失真（§研究库 §三十七）。

- **设计**：语料切互不重叠连续文章段 = 虚拟用户打字史；每用户**空库起步**
  （eff/n_seen/uni/h2 逐用户重置，h2 跨文章不跨用户 = streams 行边界）；
  段内水库采样 K 窗/用户；每窗记录 depth（历史深度）。
- **年龄网格**：H ∈ {3千, 1万, 3万, 10万, 30万, 100万} 提交/用户；
  **基准龄 = 10 万**（2026-11-08 用户定案），其余档参考。
- **τ 口径**：eff 重放 τ=400（生产 v3 值；corpus_std.TAU=800 是 2026-11-02
  旧值勿用——prep_cohort --tau/--redo-eff 支持）。
- **p1 分母**：与生产逐字一致 `n_seen+20000`（BIGRAM_VCAP）；旧评测
  d=n_seen+V+1（V=词典帽）已废——与 δ 数学冗余、深历史等价但年轻档失真。
- **DP 先验**：全文语料 unigram（cohort_unigram_<chain>.json），各档共用
  固定先验（≈生产词典先验地位，消除"先验随重放长度变"的附带失真）。
- **工具链（eval/，2026-11-18 子目录重组后路径）**：`prep/cohort_unigram.py` →
  `lib/prep_cohort.py`（--pilot 测产量 / --redo-eff 只重算 eff）→
  `prep/prep_cohort_full.py`（v2 全枚举）→ `ce/run_ce_cohort.py`
  （sid 去重断点续跑, 3 分片贴核; 已吸收 sim_gpu_parallel/seed_ce_full）→
  `lib/summary_4x4.py`（4×4 首选率）→ `lib/calibrate_model.py`
  （--protocol split|4half; 吸收原 calibrate_model_05b / 05bi_final）。
  旧龄档消融/门控脚本 2026-11-17 已归档（报告留档）。产物 cohort_big/（生产词典）与
  cohort_small6w/。
- **CE 复用合法性**：窗与 sid 只依赖 (ctx, win)，τ/eff 变更不变 → CE 结果
  按 sid 跨口径复用。
- **注意**：两条重放链（streams_std 全文 / cohort 分段）的锚点总体不同，
  绝对值不可比；生产量级的数字以 cohort 为准。

**补充（2026-11-08 晚）**：4×4 汇总工具 `eval/summary_4x4.py`（四列
大/小词库 × wiki10万龄/用户语料 × 4 行, 统一 E+U1 口径一次产出;
报告 summary_4x4_report.txt）。CE 跑批标准命令：
`python eval/ce/run_ce_cohort.py --dict <chain> --ngl 99`（GPU graphs
~40 窗/s, 单进程; --ngl 0 = CPU 多分片贴核, 仅当 GPU 不可用时）。
显著性核验（`cohort_sigtest.py`，脚本已归档、报告留档；held-out 分窗 + 用户级
簇/块 bootstrap 95% CI, 补同窗标定与窗间相关两漏洞; 报告
cohort_sigtest_report.txt; 矩阵装载单源 = summary_4x4.cohort_matrices /
user_matrices, 改动窗构造或特征口径须两脚本同源）。
**2026-11-09~11 误差攻关工具组**（脚本 2026-11-17 已归档，报告 `.txt` 保留）:
`cohort_gate_probe.py`（改对/改错分解 + 按编码门控 CV + 在位者 h 扫描）→
`cohort_gate_oracle.py`（门控族 oracle 上界）; `cohort_layer1_feat.py`
（未及分解 + k 扫描 + 层1 特征网格, 矩阵含 wins/ci 字段）;
`cohort_skip_feedback.py`（跳选负反馈仿真）; `cohort_err_taxonomy.py`
（错误病理分型）; `ft_extract_errors.py → ft_train.py → ft_export.py →
ft_user_ce.py → ft_eval_report.py`（微调管线, margin-MSE, 批处理+gather
内核 48ms/窗——只对 Σmin(ntok,3) 位置过词表 softmax, 全位 softmax 是
GB 级中间量）+ `ft_batch_check.py`（批处理数值一致性, argmin 须逐窗一致）;
`ft_ce_compare.py`（转换噪声底测: HF→Q4_K_M 自转 vs 生产 Unsloth,
CE 均值差 0.44 nat 但选重率指标仅 +0.06~0.10pp——误差率指标对量化噪声
稳健, 同链路对比即可）; `ft_distill.py`（教师→小模型 margin-MSE 蒸馏,
V1 hashed-linear/V2 char-transformer——负结果, 见研究文档 §四十三）;
`ctx_eval_report.py`（ctx×候选数矩阵, 配 sim `--ctx` 参数）。训练链
`cohort_big_train/`（25 万窗 CE 全套）= 教师标签库; 训练 venv 用
`python -m venv --system-site-packages ft_env` + `pip -U transformers`
（需 ≥5.22 才支持 qwen3_5 多模态壳; transformers 保存丢 mtp.* 张量,
转换前须从原始 safetensors 拼回, 见研究文档 §四十二）。

## 十四、2026-11-17 研究库清理（用户指令）

- 判定线 = 现行口径/被现行脚本引用/文档冻结留档保留；重复副本、可再生中间产物、
  被新链取代的实验数据、失效日志删除。清理后 16 GB → 12 GB（释放约 4.5 GB）。
- 删除：`segments_hanlp_cache.jsonl.dup.bak`（3.2 GB，去重前旧缓存）、
  `smoke_86b94708/` 的 8 个 DLL（804 MB，与 `cpp/base_86b94708/` 逐字节相同）、
  `cohort_r2/`、`cohort_big_sc2/`（2 码域结论已记录）、`cohort_big_smoke/`、
  `eval/{std_cal_work_ascii10,std_cal20_work,article_cal_work,lenweight_work,freq_beta_work}/`
  （旧标定中间结果，报告全保留）、`eval/pairfix_full_pot_smoke*`、
  `eval/cohort_unigram_r2.json`、`cpp/ngram_wanxiang.*` + `deps/`（11-06 已宣布
  删除的 n-gram 残件）、`cpp/*.obj`、`eval/__pycache__`、空 `models/`、
  `ab_upgrade_*.stderr.log` / `smoke_28b812d6/*.err` / 残跑与 0 B 输出 / `oracle.err` /
  孤立 `streams_std*.count`。
- 保留：语料源（`article_corpus_shuffled.txt` + `segments_hanlp_cache.jsonl`）、
  现行 `cohort_big_full/` + `cohort_big_ufull/`、旧 5 域证据池、小词典链与两条训练链、
  用户链、CE/特征缓存、现行 `.py` 与全部 `*_report.txt`、`cpp/` 评测器与 86b 基线运行时、
  `legacy-20261117/` 冻结记录。明细见 `freq-fusion-research.md`
  "研究库清理（2026-11-17）"节。

### 十四·补：年龄口径 + 旧 pairfix 脚本归档（2026-11-17 用户定案）

- **默认用户年龄 = 3 万次提交（2026-11-18 用户定案，取代 11-17 的 10 万）**；
  **跨年龄比较改为特定年龄**（默认 3 万，按需指定），不再出
  3 千/1 万/3 万/10 万/30 万/100 万龄档对比表。
- **2026-11-18 起不再进行多用户重放**（multi-user / multi-age cohort replay 退役）：
  `prep_cohort` / `prep_cohort_full` 的 `--arms` 默认只剩 3 万龄单档；相关 cohort 语料
  按"合并或删除"处置（见 `freq-fusion-research.md` "多用户重放退役"节）。
- 旧 pairfix 推导/探索链 12 个脚本从 `eval/` 移入
  `legacy-20261117/removed-scripts-20261117/`（清单与原因见该目录 `README.md`）：
  `derive_pairfix_{allwords,preset,cfrep,expand,rerate,userchain}.py`、
  `validate_pairfix_user.py`、`validate_pairfix_v2_user.py`、`pairfix_gap_analysis.py`、
  `pairfix_gate_{explore,debug,debug2}.py`。
- 现行推导唯一路径 = `eval/derive_pairfix_full.py`（05bi 全枚举链 + 旧五域池化，
  含部署态全量自测）；共享模块 `eval/derive_pairfix_ratio.py` 保留
  （`load_w2codes` 被审核/分析/标定脚本导入）。

### 十四·补二：eval 脚本大清洗（2026-11-17 用户定案）

- 原则：**已定结论只保留核心脚本；重复功能删除或合并；语料/词典/模型/ctx/
  候选数/参数一律走参数**，不再保留功能近似的变体。
- 结果：`eval/*.py` 140 → **40 个**；**103 个**旧脚本移入
  `legacy-20261117/removed-scripts-20261117/bulk-20261117/`（逐脚本清单、原因、
  替代入口见该目录 `README.md`）。`*_report.txt` 与数据文件未动。
- 三类重复功能合并：
  - `speed_bench.py` ← `latency_probe_model.py` + `latency_probe_upgrade.py`
    （`--models` / `--sims` / `--rounds`）；
  - `grid_ctx_cand.py` ← `ctx_eval_report.py` + `sweep_maxcand/kdepth/kmixed/
    2layer/grid2layer/pack2x2.py`（`--ks` / `--ctx-ce` / `--domains`；
    生产两层语义 `cohort_layer1_feat.layer_eval`）；
  - `compare_models.py` ← `compare_model_2b/sizes/sizes_wiki.py`
    （`--domain user|wiki|all` / `--models name=ce_sub` / `--limit`）。
- `run_ce_cohort.py` 扩展 `--anchors/--samples/--out`，吸收
  `run_ce_small6w.py` / `run_ce_user_sc.py` / `ft_user_ce.py`；`--dict r2`
  死选项删除。
- 注意：~~`audit_deploy.py` 无符号 λ=5 表与现行有符号生产表漂移~~
  **已于 2026-11-18 适配落地**（见 §十四·补四）。

### 十四·补三：eval 子目录重组 + 二轮合并（2026-11-18 用户指令）

- **按用途分 7 个子文件夹**（40 → 30 个脚本；目录级明细见 `eval/README.md`）：
  `lib/`（8 共享模块）/ `prep/`（7）/ `ce/`（1）/ `calib/`（4）/
  `pairfix/`（4）/ `bench/`（3）/ `tools/`（3）。
- **跨目录 import 统一** `sys.path.insert(0, parents[1] / "lib")`；
  `corpus_std.PROJ` 改 `parents[2]`（lib/ 下沉一层），新增
  `corpus_std.EVAL = PROJ/"eval"`——报告/工作 TSV 统一写 eval 根
  （原 12 处 `Path(__file__).with_name` 全部改指，历史产物不孤立）。
- **二轮合并 5 项**（数字口径与报告文件名不变；吸收件归档
  `legacy-20261117/removed-scripts-20261117/merged-20261118/`）：
  - `calib/analyze_sc.py` ← test_sc / analyze_sc_final / resample_percode_clean /
    inspect_word_percode / inspect_codes_behavior（子命令
    `test|final|clean|wordflip|behavior`；共享引擎 = 词典全码表/双键重放/
    两层终选/资格剔除集收拢为一；mince/s1fallback 两打分模式保留）；
  - `ce/run_ce_cohort.py` ← + sim_gpu_parallel（`--input/--workers/
    --scores-only`）+ seed_ce_full（`--seed-from`）；GPU 铁律扩为
    `--nshards 1 --workers 1`；
  - `pairfix/audit_tool.py` ← audit_batches/merge/deploy（子命令
    `batch|merge|deploy`；`pairfix_pipeline` 三处 subprocess 调用已同步改）；
  - `lib/calibrate_model.py` ← calibrate_model_05b + 05bi_final
    （`--protocol split|4half`；`load_raw`/`ce_matrix` 保持模块级可导入,
    `analyze_sel_by_code` import 已改）;
  - `bench/ab_upgrade.py` ← ab_llama_upgrade + rerate_impact_upgrade
    （`--phase run|impact|both`；wsum 统一）。
- **不合并并固化理由**：prep_cohort/prep_cohort_full（samples 与全枚举两种
  数据契约 + import 耦合）、derive_pairfix_ratio/full（方法不同 + ratio 为
  7 处 import 的库 + full 运行时改其全局）、speed_bench/grid_ctx_cand/
  compare_models（上轮合并产物）、hanlp_seg_worker（独立 venv 子进程）、
  eval_rerank/run_tests/check_records_v2/gguf_types（运行时各异入口）。
- 验证：ast 30/30、lib import 中性 cwd 全过、`--help` ×12、
  grid_ctx_cand 复现 4.46/4.26%、compare_models/analyze_sc test/
  run_ce_cohort 单窗/audit_tool 三子命令冒烟/check_records_v2 真语料/
  speed_bench 真模型全过。

### 十四·补四：S9 部署有符号格式适配落地（2026-11-18）

补二记录的 `audit_deploy` 无符号 λ=5 漂移在本日解决（`pairfix/audit_tool.py
deploy`，原 `migrate_pairfix_signed.py` 适配器退役归档）：

- **λ 列有符号化**：无符号正数 = 降分幅度 → 取负落地（λ=5 → `-5`）;
  显式 `+200`（固定首选）/ `+100`（固定次选）/ 负数原样透传——审核表
  第 3 列从此可直写有符号值。
- **合并取代覆盖**：现行生产表的正分条目（固顶/固次, 09876 写入或手动
  编辑）**全数保留**; 旧负分条目由本代审核集取代（降噪对只来自审查流）;
  同词码对正分冲突 → 正分胜并打印冲突清单。deploy 直接跑不再抹 71 条
  固顶、不再把降噪写成 `+5` 加分。
- **词典同码终检只查新审核行**（生产 +200 的 1-2 码顶屏对不在 3/4 码
  终检词典、也不参检）; 非法 λ 值（非数字）整批拒绝部署。
- 输出 = 有符号头注释（与迁移件口径一致）+ 正分在前（码/词序）负分在后;
  `eval/pairfix_deploy_pass_<tag>.tsv` 与 `%APPDATA%\Rime\llm_pairfix.tsv`
  同内容, 备份链 `.bak-<日期>` 不变。
- **落地验证（05bi 表真机部署）**：dry-run 合并计划 = 21 新负 / 71 正保留 /
  0 冲突 / 21 旧负取代——与生产表实况完全吻合; 部署后条目级校验
  **92 = 92 词码→分数零差异**（仅头注释规范化, 备份
  `llm_pairfix.tsv.bak-20261118`）; 合成分支合成表验证：+200 冲突让位、
  `+100` 透传、λ=5 → -5 三分支全对。

### 十四·补五：2026-11-18 保守档清理（用户指令"先做保守档"）

- **11.95 GB → 10.87 GB（唯一字节数，回收 1.08 GB）**。删：`dll_bak_gfx/`（35 MB，
  与 `base_86b94708/` 逐字节相同的旧引擎备份）、`cohort_big_full/ce_05b`（176.6 MB，
  427,128 sid 全含于 ce_05bi 的残跑）、`cohort_big_ufull/ce_05b`（4.4 MB）、
  `cohort_small6w_train/ce`（24.4 MB，96k/150k 残跑）、`eval/tmp_sweep_streams.pkl`
  （33.4 MB，0 引用临时物化）。
- **硬链接去重**：`cpp/base_86b94708/` 的三件 CUDA 运行时（cublasLt 692 MB / cublas
  114 MB / cudart 0.6 MB）指向 `build_sim/Release/` 同名文件；SHA256 与包内
  `SHA256SUMS.txt` 逐字节一致，两处路径仍可独立运行。
- 保留：两份 pairfix 全量账本（11-17 清理已明列保留；05b 审核判定以其为推导依据）。
  明细见 `freq-fusion-research.md` "2026-11-18 保守档清理"节。

### 十四·补六：多用户重放语料处置（2026-11-18 用户指令）

- 6 条仍保留的 cohort 链**原地裁剪为 3 万龄单档**（`legacy-20261117/oneoff-20261118/
  subset_arm.py --arm 30000`：anchors/samples 按 arm 过滤 + ui/ai 重映射、streams 只留
  相应用户、CE 按 sid 去重、meta 记 `subset` 溯源）。裁剪后窗数：big_full 238,582 /
  big 20,004 / big_x2 20,004 / ufull 28,112 / big_user 8,000 / small6w 25,000。
- `cohort_big_sc` / `_sc_x2` **整套保留**（无 3 万臂：3千/10万/100万）；
  `cohort_big_train` / `_small6w_train` **删除**（只有 10 万臂、0 现行引用，666 MB）。
- 校验：裁剪前后 `summary_4x4.cohort_matrices` + `layer_eval` 选重率逐位一致
  （big 4.4441 / big_x2 4.2691 / big_user 1.8875 / small6w 3.4560 %），
  `pairfix_pipeline check --tag 05bi` 产物齐备。lp1 分母 n_seen 变小但为窗内常数，
  排序不变；旧链绝对 lp1 数值不再逐位可比。
- **总量 11.95 GB → 7.06 GB**（两批共回收 4.89 GB；语料源 4.86 GB 占 69%）。

### 十四·补七：语料单源收敛（2026-11-18 用户定案）

- **语料只保留两份**：wiki 域 `article_corpus_shuffled.txt`（959,461 篇）与用户域
  `eval/user_articles.txt`（1,253 篇）；两者的逐篇 HanLP 分词缓存
  （`segments_hanlp_cache.jsonl` / `eval/user_segments_cache.jsonl`）是派生缓存、
  行号对齐，小任务按文件顺序**取前缀**（`build_segment_cache` 已支持 1%/10%）。
- **单字/多字不需要两份语料**：语料本体同时含单字与多字；`cohort_big_sc` 与
  `cohort_big` 只是同一份语料按"窗口资格"（3 码单字重码 / 4 码多字）筛出的两套窗。
- 已删：`eval/article_corpus_2000.txt` / `_4000.txt`（随机抽样快照，行级哈希核验
  100% 含于大语料但非前缀）、`singlechar_corpus.tsv`（3,206 窗样本，0 引用）、
  `评测文本/理性存钱理财.txt`（1.2 KB，0 引用）。
- 保留的 `eval_samples.tsv` / `train_samples.tsv` / `fusion_dump_*.tsv` 是
  **采样窗口与 CE 中间量**（非语料），暂作历史留档。

### 十四·补八：研究范围收窄（2026-11-18 用户定案）

- **只做 wiki 大词库多字词**：小词库（`cohort_small6w/`、`词库-测评6万/`）、
  单字 3 码（`cohort_big_sc/` `_sc_x2`）、用户域（`cohort_big_user/` `_ufull`
  与 `user_*_std` / `user_ce_work` / `user_sc_ce`）数据已删除；`cohort_big_x2/`
  （第二半区）亦删。用户语料文件（`eval/user_articles.txt` + 缓存）保留；
  2026-11-20 用户语料确认实验时按补十口径重建了 `cohort_big_user/`。
- **只留生产模型 CE**：`cohort_big` 只留 `ce_05bi`；非 instruct CE（0.8B/0.5B-base/
  0.6B/2B）已删——② 的对照数字与实验参数见 `eval/core_conclusions_params.md` +
  `compare_models_report.txt`。
- **① 的上下文网格**（历史；链与 CE 已于 2026-11-19 删除，见补九）：ctx = 1–20 逐个值、
  每档 10,000 窗、只多字词、只 wiki、模型 = 0.5B instruct。10k 窗链 `cohort_big_ctx10k/`
  （seed=42）；CE 目录 `ce_c1…ce_c20`；精度表 `grid_ctx_cand.py`，延迟 `bench/ctx_latency.py`。
- 工具同步：pairfix 池化链只剩 `cohort_big`（全枚举 `cohort_big_full`）；
  4×4 终表退役；`compare_models` 域只剩 wiki。

### 十四·补九：模型归一与清理（2026-11-19 用户定案，取代补八的模型/CE 保留规则）

- **保留模型**：3 个 base 部署候选自建 Q4_K_M（0.5B / 0.6B / 0.8B）
  + 4B base 能力上限（2026-11-19 用户指令，同链 6 万窗实测后保留）
  + 4 份原始 base 权重（`hf/`）。chat/对话版权重、全部量化变体、BF16 中间件与 2B 文件已删除。
- **保留 CE**：只有 6 万窗独立链 `cohort_big_60k/` 的 ctx20 目录
  （`ce_05b_base_q4km_c60k_c20` / `ce_06bb_base_q4km_c60k_c20` /
  `ce_08bb_base_q4km_c60k_c20` / `ce_4bb_base_q4km_c60k_c20`）。
- `ce_05bi`、`ce_c1…c20`、chat 与量化变体 CE 全部删除；**1 万窗链
  `cohort_big_ctx10k` 及其 CE/扫参文件亦于 2026-11-19 按用户指令删除**
  （实验记录只保留在 `freq-fusion-research.md`）。
  `cohort_big` / `cohort_big_full` 窗口链保留（纯词频 ① 与历史母链）。
- 量化结论只剩两条（README 附录 A）：**base 优于 chat**、**Q4_K_M 性价比最佳**
  （含 0.5B 因 hidden 896 无法标准 K 量化、回退 Q5_0/Q8_0 的证据）。
- 2B 只在附录 A.3 保留一句：最低档位（k=2 候选段 121.7ms / 预填充 260ms）已超即时性要求。
- 4B（Qwen3-4B-Base Q4_K_M）只作能力上限：6 万窗 k4/5/6 = 3.06/2.80/2.63%，
  但预填充 462–472ms、候选段 k6 438ms、整窗 k6 916ms，同样只在附录 A.3 记录。

### 十四·补十：用户语料确认链重建（2026-11-20 用户指令）

- **用途**：研究叙事收尾步「用户语料确认」（只确认生产选择、不参与选型、不调参）。
- **链**：`cohort_big_user/` = `prep_cohort.py --dict big
  --articles eval/user_articles.txt --cache eval/user_segments_cache.jsonl
  --tag _user --arms 30000:34:1000`（单龄档，多年龄档重放已退役不恢复）；
  实得 **27 用户 × 1000 = 27,000 窗**——语料 81 万提交在 34 用户用尽前枯竭
  （1,253 篇全部消费，V_cap 419,625）；同 seed/分词/窗门，τ=400。
- **CE + 扫参**：四模型 base Q4_K_M ctx20（`ce_*_base_q4km_cuser_c20` ×4，
  GPU 串行 34/31.8/30.4/18.6 窗/s）；`fusion_sweep_all --chain big_user
  --ks 4,5,6`（扫参 `eval/fusion_sweep__base_models_c20_user.tsv`，
  `--model` 传 CE 目录名前缀、ctx 由工具追加，`--lp1-ce` 传完整名）。
- **结论**（README 附录 C）：基线 5.43%；k6 最优 err 0.6B 1.51% /
  0.5B 1.52% / 0.8B 1.53% / 4B 1.41%（诚实 1.56/1.56/1.60/1.45）——
  小模型互差噪声级（wiki 上 0.6B 对 0.5B 的 0.28pp 优势消失）、4B 领先
  从 0.65pp 收窄到 0.10pp；生产点 0.6B k6 = 1.60%（+0.09pp）。
  用户域 argmax 参数漂移（b_cum 顶格 2.0–2.5、w2/w3 偏高）不采纳——
  维持 wiki 平台标定的生产参数。
