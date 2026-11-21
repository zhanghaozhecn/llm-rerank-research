# 核心结论实验参数（2026-11-18 记录，2026-11-19 补 base 0.6B/0.8B）

> 目的：三条核心结论（①模型×ctx×候选数、②模型规模、③融合 2×2）的参数留档。
> 删除非生产模型 CE 之后，凭本文件 + 语料/缓存/窗口集即可重跑对应 CE。

## 口径常数（生产）

| 项 | 值 |
|---|---|
| 融合公式 | `fused = CE + β·log1p(eff) + log(uni+1)`（CE = 负的 kCeW 加权交叉熵） |
| β | 14.0 |
| CE 位置权重 kCeW | (1.0, **1.10**, **0.60**)（第 2/3 token） |
| 两层 k | **6**（0.5B / 0.6B）；**4**（0.8B）；2B 不用（最低档位延迟已超即时性要求，见 README 附录 A.3） |
| 生产推荐模型 | **Qwen3-0.6B base Q4_K_M**（**2026-11-19 已切生产**：安装器 URL/默认名/默认值、lua 与 llm_filter.cc 默认、本机 pdsp.schema.yaml 均已改；0.5B base 为回退档） |
| 上文 token | **20**（2026-11-18 定案；原 10。拟合见 README ②：三个入选模型都撞 20 上限） |
| 衰变 τ | 400 次提交（formula_d） |
| V_CAP | 20,000 |
| 默认用户龄 | 3 万次提交（`corpus_std.DEFAULT_ARM=30000`） |
| 指标 | 选重率 = 错误率 = 1 − 首选正确率（越低越好） |

## 模型（GGUF, Q4_K_M；2026-11-19 清理后只留这 4 个）

| 角色 | 文件 | sha256 | 字节 |
|---|---|---|---|
| **0.6B base（生产，自建）** | `d:/gguf_models/quant_probe/base06b_q4km.gguf` | `5dd353df7326c7a6e0574de74eeb8fb1786a00b2fa9404182368b9b0cd30a036` | 396,704,640 |
| **0.5B base（回退档，自建）** | `d:/gguf_models/quant_probe/05b_base_q4km.gguf` | `8f9342b201053121be7647ac50d8ff048c51e8c8325df20f5ba3cd57c81410bf` | 397,807,392 |
| **0.8B base（规模对照，自建）** | `d:/gguf_models/quant_probe/base08b_q4km.gguf` | `450169cdd1505b96300b12d7cce6aae56ebefbf563c009a80cf2419cec3a455b` | 547,308,352 |
| **4B base（能力上限，不作部署）** | `d:/gguf_models/quant_probe/base4b_q4km.gguf` | `0a0b0ed0cd7b93cbafa69a3c31bcf4ee5b8be8052717c81e9947c059bd371520` | 2,497,280,416 |

**实际张量组成（`eval/bench/gguf_types.py`，量化调研结论的关键证据）**：

| 文件 | 张量组成 | 等效位宽 |
|---|---|---|
| `05b_base_q4km` | Q5_0 132 + Q8_0 13（含嵌入）+ Q4_K 12 + Q6_K 12 | ≈6.4 bpw——**0.5B 无法标准 Q4_K_M**（hidden 896 不整除 256，自动回退） |
| `base06b_q4km` | Q4_K 168 + Q6_K 29（含 tied 嵌入） | ≈5.3 bpw，零回退 |
| `base08b_q4km` | Q4_K 132 + Q6_K 27（含 tied 嵌入） | ≈5.35 bpw，零回退 |
| `base4b_q4km` | Q4_K 216 + Q6_K 37（含 tied 嵌入） | ≈4.95 bpw，标准 Q4_K_M，零回退 |

**原始权重（保留，供重转）**：`d:/gguf_models/hf/{Qwen2.5-0.5B, Qwen3-0.6B-Base, Qwen3.5-0.8B-Base, Qwen3-4B-Base}`
（model.safetensors + tokenizer/config）。

> **2026-11-19 清理**：chat/对话版权重与全部量化变体（Q4_0 / Q5_0 / Q6_K / Q3_K_M / Q2_K /
> e4k / out40 / BF16 中间件）、对应 CE 目录、0.5B instruct 与 2B 文件均已删除；
> 相关结论作为实测记录保留在 README 附录 A，不保留可复跑文件。
> 分发件：`本机 GGUF 模型目录\Qwen3-0.6B-Base-Q4_K_M.gguf`（硬链）与 `upload_06b_base\`
> 上传包（0.5B 回退档上传包已于 2026-11-19 按用户指令删除）。

CE 目录（3 个 base 部署候选 + 4B 能力上限，6 万窗主口径）：

- `cohort_big_60k/ce_05b_base_q4km_c60k_c20`、
  `ce_06bb_base_q4km_c60k_c20`、`ce_08bb_base_q4km_c60k_c20`、
  `ce_4bb_base_q4km_c60k_c20`
- 用户语料确认（2026-11-20）：`cohort_big_user/ce_05b_base_q4km_cuser_c20`、
  `ce_06bb_base_q4km_cuser_c20`、`ce_08bb_base_q4km_cuser_c20`、
  `ce_4bb_base_q4km_cuser_c20`

扫参：`eval/fusion_sweep__base_models_c20_60k.tsv`（4B 能力上限：`fusion_sweep__qwen3_4b_c20_60k.tsv`；
用户语料：`fusion_sweep__base_models_c20_user.tsv`）。
全部口径与结论见 README**附录 A/B/C**。

## 数据（窗口集 + CE，2026-11-19 清理后）

| 域 | 链 / 臂 | 窗数 | CE 目录 |
|---|---|---|---|
| wiki 多字词（主线 ②③④） | `cohort_big_60k` @30000（20 用户 × 3000，文章 168631–170675） | 60,000 | `ce_05b_base_q4km_c60k_c20` / `ce_06bb_base_q4km_c60k_c20` / `ce_08bb_base_q4km_c60k_c20` |
| wiki 多字词（4B 能力上限） | `cohort_big_60k` @30000（同上） | 60,000 | `ce_4bb_base_q4km_c60k_c20` |
| wiki 多字词（6 万窗巩固） | `cohort_big_60k` @30000（20 用户 × 3000，文章 168631–170675） | 60,000 | `ce_05b_base_q4km_c60k_c20` / `ce_06bb_base_q4km_c60k_c20` / `ce_08bb_base_q4km_c60k_c20` |
| 用户域确认（2026-11-20） | `cohort_big_user` @30000（27 用户 × 1000，OneDrive 1253 篇单遍消费） | 27,000 | `ce_05b/06bb/08bb/4bb_base_q4km_cuser_c20` |
| wiki 多字词（纯词频 ①） | `cohort_big_full` @30000（big 域） | 205,966 | 无需 CE（纯词频基线） |

> 单字 3 码 / 用户域 / 小词库链及其 CE 已于 2026-11-18 删除；
> chat/对话版与量化变体 CE 已于 2026-11-19 删除；**1 万窗链 `cohort_big_ctx10k`
> 及其 CE/扫参文件同日删除**（实验记录保留在 `freq-fusion-research.md`）。
> `cohort_big`（20,004 窗）保留为历史母链。

sid = `window_key_ctx(ctx, 候选窗)`（与 sim 的 `--ctx` 截断无关，同一窗在不同 ctx 的 CE 可并存于不同目录）。

## 环境

- sim = `cpp/build_sim/Release/sim_rerank{_gpu}.exe`，llama.cpp `9f26fe11c`（v1.0.0 时代）
- GPU：RTX 4060 Laptop，SM 锁 2100MHz；CPU：睿频关 + 70% 上限（`稳频.py --查` 已核验）
- GPU 单 CUDA 进程铁律：`--nshards 1 --workers 1`

## 延迟口径（2026-11-18，两次修正后定案）

- **计入 = 整窗全程**：ctx 预填充 + step2/step3 两次 decode + CE 计算（sim `SIM_TIMES` 逐窗）。
  **旧口径（只计两次 decode、预解码不计）作废**——它把 ctx 的作用完全抹掉，而 ctx 成本
  全在预填充里（实测 ≈5–6 ms/token；0.5B ctx1→20 在 k=5 上 +110ms）。
- **取最坏情况**：所有候选按 **3 token** 构造（k 个候选 = k 条 3-token 行），且**首 token 两两不同**
  （前缀去重失效 → step2/step3 各 k 行）。
- 测量：**一个进程跑完该模型全部 (ctx, k)**（模型只加载一次、机器状态一致），
  n=12 窗 × 2 轮（部署档/量化复测；legacy 初测为 n=4），逐窗计时取**中位数**
  （p90/p50 中位 1.05，噪声 5%）。
- 环境：**机器空闲**（无其他重负载）、睿频关 + 70% 上限（长期 CPU 作业标准状态，持久生效）。
- 工具：`eval/bench/latency_synth.py --metric total`；合成候选由 `eval/lib/gguf_bpe.py`
  （复刻 llama.cpp 分词，CE `ntok` 复核 302/302）挑选。
- 历史对照：旧链路每候选各 prefill 一遍 → "10→17 token 延迟翻倍"；现行共享 prefill 后
  ctx 边际 ≈5ms/token（研究库 §四十一 的预测得到实测确认）。

## 产出与复现

- 主口径 ④（6 万窗）：三模型 CE + 联合扫参
  ```bash
  for m in 05b_base_q4km:05b_base_q4km base06b_q4km:06bb_base_q4km base08b_q4km:08bb_base_q4km; do
    python eval/ce/run_ce_cohort.py --dict big --tag _60k --ce-tag "_${m%%:*}_c60k_c20" \
      --ctx 20 --model d:/gguf_models/quant_probe/${m%%:*}.gguf --ngl 99 --nshards 1 --workers 1
  done
  python eval/calib/fusion_sweep_all.py --chain big_60k \
    --model 0.5B-base=ce_05b_base_q4km_c60k_c --model 0.6B-base=ce_06bb_base_q4km_c60k_c \
    --model 0.8B-base=ce_08bb_base_q4km_c60k_c --ctxs 20 --ks 4,5,6 \
    --tag _base_models_c20_60k --lp1-ce ce_05b_base_q4km_c60k_c20
  ```
- 4B 能力上限（2026-11-19，同 6 万窗）：
  ```bash
  python eval/ce/run_ce_cohort.py --dict big --tag _60k --ce-tag "_4bb_base_q4km_c60k_c20" \
    --ctx 20 --model d:/gguf_models/quant_probe/base4b_q4km.gguf --ngl 99 --nshards 1 --workers 1
  python eval/calib/fusion_sweep_all.py --chain big_60k \
    --model 4B-base=ce_4bb_base_q4km_c60k_c --ctxs 20 --ks 4,5,6 \
    --tag _qwen3_4b_c20_60k --lp1-ce ce_05b_base_q4km_c60k_c20
  ```
  延迟：`latency_synth.py --model 4B-base=<gguf> --ctxs 20 --ks 2,4,6 --metric prep|step23|total`。
- 用户语料确认（2026-11-20，README 附录 C）：
  ```bash
  python eval/lib/prep_cohort.py --dict big --articles eval/user_articles.txt \
    --cache eval/user_segments_cache.jsonl --tag _user --arms 30000:34:1000
  # CE ×4（GPU 串行 --nshards 1 --workers 1；06bb/08bb/4bb 同式换 --model/--ce-tag）
  python eval/ce/run_ce_cohort.py --dict big --tag _user --ce-tag "_05b_base_q4km_cuser_c20" \
    --ctx 20 --model d:/gguf_models/quant_probe/05b_base_q4km.gguf --ngl 99 --nshards 1 --workers 1
  python eval/calib/fusion_sweep_all.py --chain big_user \
    --model 0.5B-base=ce_05b_base_q4km_cuser_c --model 0.6B-base=ce_06bb_base_q4km_cuser_c \
    --model 0.8B-base=ce_08bb_base_q4km_cuser_c --model 4B-base=ce_4bb_base_q4km_cuser_c \
    --ctxs 20 --ks 4,5,6 --tag _base_models_c20_user --lp1-ce ce_05b_base_q4km_cuser_c20
  ```
  （注意 `--model` 传 CE 目录名**前缀**，ctx 值由工具追加；`--lp1-ce` 传完整目录名。
  链实测 27 用户（语料 81 万提交在 34 用户用尽前枯竭）= 27,000 窗。）
- ②③ 延迟：`python eval/bench/latency_synth.py --model 0.6B-base=<gguf> \
  --ctxs 1-20 --ks 2 --metric prep --tag 06bb_q4km_prep`；候选段 `--ctxs 20 --ks 2-9 \
  --metric step23 --tag 06bb_q4km_s23`。
- ①：README「核心结论 ①」（`cohort_big_full` @3万，205,966 窗，纯词频口径；无需 CE）。
  `python eval/calib/fusion_sweep_all.py --model 0.5Bi=ce_c --ctxs 1-20 --ks 2-9 --lp1-ce ce_c10`
- 帕累托前沿：`eval/pareto_report.txt`
  `python eval/tools/pareto_front.py --acc eval/fusion_sweep_05bi.tsv \
      --acc eval/fusion_sweep_08b.tsv --lat eval/latency_synth_0.5Bi_total.tsv \
      --lat eval/latency_synth_0.8B_total.tsv --device cpu --metric honest`

## 帕累托前沿（整窗中位延迟 × 选重率）——**已归档**

> **2026-11-18 定稿后作废**：现行前沿 = README「帕累托前沿（(ctx\*, k\*) 版）」
> （预解码全命中口径 + 量化变体）。以下整窗表只作历史对照，证明 ctx 与 k 在同一预算内竞争。

整窗延迟（CPU 中位 ms，0.5B-Instruct；完整表见 `eval/latency_synth_0.5Bi_total.txt`）：

| ctx \ k | 2 | 3 | 4 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|
| 1 | 61.2 | 87.4 | 96.8 | 118.9 | 155.3 | 193.7 |
| 5 | 86.1 | 114.5 | 118.6 | 138.6 | 186.3 | 223.9 |
| 10 | 114.3 | 144.7 | 154.3 | 179.9 | 219.9 | 245.4 |
| 15 | 153.1 | 178.9 | 177.4 | 204.5 | 248.3 | 273.6 |
| 20 | 168.1 | 183.0 | 217.6 | 233.6 | 261.9 | 302.8 |

（ctx 成本 ≈5.6 ms/token，与 k 的成本 ~20ms/候选 同一量级。）

**ctx 与 k 在同一预算内竞争**：ctx1→20 在 k=5 上加 +115ms；k 2→9 加 +135ms。
前沿（诚实口径，摘录；共 33 点）：

| 点 | 中位 ms | 选重率（诚实） |
|---|---|---|
| 0.5B ctx1 k2 | 61.2 | 11.84% |
| 0.5B ctx5 k2 | 86.1 | 7.11% |
| 0.5B ctx7 k3 | 115.5 | 5.92% |
| 0.5B ctx9 k4 | 142.3 | 5.35% |
| 0.5B ctx12 k5 | 169.8 | 4.82% |
| 0.5B ctx16 k5 | 211.7 | 4.42% |
| 0.5B ctx19 k7 | 266.1 | 4.06% |
| 0.5B ctx19 k9 | 317.4 | 3.94% |
| 0.8B ctx19 k7 | 334.2 | 3.92% |
| 0.8B ctx19 k9 | 406.1 | 3.84% |

0.8B 只在 ≥334ms 末端出现（−0.02~−0.10pp，+17~89ms），性价比不成立；2B 起点 179ms
（ctx1 k2）且全程 ≥500ms，整体出局。
用生产固定参数代替逐格最优只差 0.02–0.05pp，前沿结论不变。

## 状态（2026-11-19）

- ② **完成并定案**：T_prefill(ctx) 3 个 base 模型 × ctx1–20；线性拟合 → ctx\*=20
  （0.5B/0.6B/0.8B 都先撞 20 上限）。2B 不参与主线（附录 A.3）。
- ③ **完成并定案**：T_cand(k) k2–9；k\*=6（0.5B/0.6B）、4（0.8B）。
- ④ **完成**（每模型一次，格 = ctx20 × k\*；**主口径 6 万窗**）：
  **0.6B-base 3.28%** / 0.5B-base 3.56% / 0.8B-base 3.55%（标定后最优）
  → **0.6B base 成为帕累托最优，已切生产**。1 万窗小样本结果与文件已移除
  （实验记录见 `freq-fusion-research.md`）。
- **量化收尾**：结论只保留两条（base > chat、Q4_K_M 性价比最佳），全部 chat/量化变体
  与中间件已于 2026-11-19 清理；0.5B 无法标准 Q4_K_M（hidden 896）的张量证据见
  README 附录 A.2。
- **6 万窗巩固（2026-11-19）**：独立链 `cohort_big_60k`（20 用户 × 3000 窗，
  文章 168,631–170,675；纯词频基线 10.27%）三模型 CE + 扫参：
  0.6B k6 **3.28%** / 0.8B k5 3.34% / 0.8B k6 3.20% / 0.5B k6 3.56%；
  生产选择不变（0.6B k6 在 110ms 内最优），详见 README 附录 B。
- **4B 能力上限（2026-11-19 用户指令）**：Qwen3-4B-Base base 自建 Q4_K_M
  （2.50GB，4.95bpw，Q4_K 216 + Q6_K 37，零回退），同 6 万窗 k4/5/6 =
  **3.06% / 2.80% / 2.63%**（比 0.6B 同 k 好 0.59–0.65pp）；预填充 ctx20
  462–472ms、候选段 218/273/438ms、整窗 k6 916ms → **远超即时性，只记能力上限，
  不进入选型**（README 附录 A.3）。
- **用户语料确认（2026-11-20）**：`cohort_big_user` 重建（27 用户 × 3 万龄
  = 27,000 窗，纯词频基线 5.43%）+ 四模型 CE（34/31.8/30.4/18.6 窗/s）
  + 扫参：k6 最优 err 0.6B **1.51%** / 0.5B 1.52% / 0.8B 1.53% / 4B 1.41%
  （诚实 1.56/1.56/1.60/1.45）；小模型互差噪声级、4B 领先比 wiki（0.65pp）
  收窄至 0.10pp；生产点 0.6B k6 = 1.60%（+0.09pp vs argmax）→
  **生产选择在用户域确认**（README 附录 C）；用户域 argmax 参数漂移不采纳。
- **分发待办**：0.6B base Q4_K_M 自建文件，目标仓
  `zhanghaozhecn/Qwen3-0.6B-Base-GGUF`（**2026-11-19 已上传**：GGUF 396,704,640 B +
  模型卡 + LICENSE；resolve URL 实测 200）；
  安装器 URL/默认名已改。
- 待办：量化文件真机验证；用户语料确认、pairfix 后置。
