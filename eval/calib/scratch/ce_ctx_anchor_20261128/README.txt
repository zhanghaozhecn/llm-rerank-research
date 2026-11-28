CE 上文锚实验归档（2026-11-28）— 见研究库 README「CE 上文锚」节

排查链（CE 回归排除）:
  ce_probe.cpp   精确复刻生产 CE1（tokenize add_special=true → ctx decode → normalizer skip20 → ce_target）
  ce_probe2.cpp  单 token 上文健康探针（top-5 + 指定词 CE, --out 直写文件）
  ce_probe3.cpp  pos-0 隔离（@pos0/@pos5/两token批row0/。row1 → 排除位置与批形状, 4B 对照正常）
  ce_probe4.cpp  锚 A/B（bare/eos+/eot+/punct+ × 二十/英雄/化学 → 三例全对验证）

批量实验（wiki 分条首句 800 样本）:
  prep_samples.py  抽样（剥标题, 16-60 字, 首句）
  ce_docstart.cpp  n∈{1,2,4,8} × 4 形态 × 真值续 token CE + top1（CSV 逐行）
  agg_docstart.py  聚合表
  docstart_*.csv   三模型原始数据（每模型 ~12,700 decode）
  scope_*.txt / anchor_*.txt  小样本可读输出
  samples.txt      800 句样本

结论: 仅 Qwen3-0.6B 裸 n=1 上下文盲（top1 0.0%）；最佳锚 = 「。」（唯一 n≤4 无负作用）；
endoftext/im_end 锚 n=2 起转负；0.8B 不需要锚。生产规则 = ctx ≤2 token 垫「。」。

关联修复（2026-11-28 下午，真机键盘测试发现）:
  用户报告「打码后静置候选窗闪烁自动消失」→ 二分定罪 = TSF 无按键刷新 tick
  (120ms) 的响应处理错杀活组合（tick 回包偶发 composing=0/幽灵 commit →
  _EndComposition；与 CE 锚无关，锚关/28-3 载荷同现，luna/pdsp 双复现）。
  修复 = tick 编辑会话「只读刷新 UI」（m_tick_refresh 守卫，EditSession.cpp:
  无 commit 不裁决组合生死）。验证: idle_v6_pdsp.py（严格 IME 确认: 只认本轮
  新 pdsp W 行 + 微软拼音签名拦截——wkjl→「我看见了」= 微软拼音简拼 w-k-j-l，
  曾把微软拼音当 pdsp 测过一轮，教训已写进脚本注释）。

n=0（真空上文）专项（2026-11-28 深夜，定案 = 维持 n=0 不算 CE）:
  n0_rerank.py            cohort m=0 句首窗两臂切片（生产口径：条件真值∈池 +
                          完整词覆盖真值）；--apply 断点续跑
  n0_rerank.txt           切片结果（锚净效应随码长符号翻转）
  n0_rerank_hard12k.txt   首版硬子集三臂存档（≥6 字母过滤假象的对照）
  n0_firstword.py         wiki 95.9 万条正文首句首词 Top10k 三臂
                          （引擎原序 / 整句显示序 span 分组 / 锚·span 分组）
  n0_firstword.txt        结果（锚加权 +25.52pp 净害）；反面教材 = 无 span
                          分组纯 CE 序 +71.98pp（1-token 候选只付一笔 token 钱）
  n0_firstword_top.json   首词频次表（Top10k 缓存）
  n0_firstword_probe/     首词引擎池探针缓存（run_probe work 目录）
  n0_bare_punct_ce/       裸「。」CE 临时缓存（SIM_CTX_ANCHOR=0 裸跑；主缓存
                          ctx=「。」的 sid 行是「。。」锚口径——不可写回共享缓存）
  结论与全部数字见研究库 README 锚节清单⑧；AGENTS 本地状态 ⑰。

