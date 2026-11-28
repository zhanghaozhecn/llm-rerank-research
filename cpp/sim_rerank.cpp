/*
 * sim_rerank.cpp — 用生产 DLL 同款算法仿真 LLM 重排
 * 读取 JSON 输入（context + candidates list），输出 LLM 重排结果
 *
 * 输入格式 (stdin, 一行一个 JSON):
 *   {"context": "上文文本", "cands": ["候选1", "候选2", ...]}
 *
 * 输出格式 (stdout, 对每行输入):
 *   ["首选", "次选", ...]
 *
 * 编译: 与 rime_llm.cpp 相同方式（llama.lib + GGUF 模型）
 *
 * CE 管线化 (2026-11-04): decode 主线程逐窗串行不变, CE(14.9ms/窗全词表双精度循环)
 * 移后台线程与下一窗 decode 重叠, 输出由后台线程按序写 → 分数位级不变 (验证:
 * CPU 引擎 SIM_NO_CE_PIPE A/B 字节级 diff + GPU 2000 窗对拍门禁)。
 * 详见 GPU批推理提速调研-2026-11-04.md。
 */
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "llama.h"

#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

// Config
static const char* DEFAULT_MODEL_PATH = "d:/gguf_models/Qwen3.5-0.8B-Q4_K_M.gguf";
static const int   N_CTX     = 64;
static const int   N_THREADS = 6;
static const int   MAX_CTX_TOK_DEF = 10;
static int         g_max_ctx_tok = MAX_CTX_TOK_DEF;  // --ctx 覆盖（§四十一）

static llama_model*       g_model = nullptr;
static llama_context*     g_ctx   = nullptr;
static const llama_vocab* g_vocab = nullptr;

// 分段计时 (2026-11-14): prep = Step1 ctx prefill (生产由预解码摊销),
// cand = Step2/3 候选并行 decode (生产感知延迟的可比部分)
static double g_ms_step1 = 0, g_ms_step23 = 0;
static double g_last_step1 = 0;    // 最近一窗的 ctx 预填充段（2026-11-18）
static double g_last_step23 = 0;   // 最近一窗的 step2+step3 段（候选段延迟, 2026-11-18）
static int    g_n_windows = 0;

// Tokenize
static std::vector<llama_token> tokenize(const char* text) {
    std::vector<llama_token> toks(128);
    int n = llama_tokenize(g_vocab, text, (int)strlen(text), toks.data(), (int)toks.size(), true, true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(g_vocab, text, (int)strlen(text), toks.data(), (int)toks.size(), true, true); }
    toks.resize(std::max(0, n));
    return toks;
}

// CE 上文锚（2026-11-28，与两版生产同构）：仅 n=1 裸单词上文垫「。」——
// Qwen3-0.6B 裸单词上文预测退化为先验序（wiki 首句配对大样本：bare n=1
// top1 0.0%→锚后 6.71%、ΔCE −6.10、z=47.9；n=2 专项 3000 句验证句号锚
// ΔCE −0.0096 / z=−0.15 = 零效应，故只收 n=1；endoftext/im_end 锚 n≥2
// 显著为负）。重放生产语料必须带此规则才能与生产对拍。A/B：
// SIM_CTX_ANCHOR=0。
static const llama_vocab* g_anchor_vocab = nullptr;
static int g_anchor_tok = -1;
static std::vector<llama_token> ctx_ids_for_ce(std::vector<llama_token> ids) {
    static const bool kOn = !(getenv("SIM_CTX_ANCHOR") && getenv("SIM_CTX_ANCHOR")[0] == '0');
    if (!kOn || ids.size() > 1) return ids;
    if (g_anchor_vocab != g_vocab) {
        auto t = tokenize("。");
        g_anchor_tok = t.empty() ? -1 : (int)t[0];
        g_anchor_vocab = g_vocab;
    }
    if (g_anchor_tok >= 0) ids.insert(ids.begin(), g_anchor_tok);
    return ids;
}

// Cross-entropy for a single token
// 2026-11-06 与生产同步 (rime_llm.cpp / llm_filter.cc 同款): 尾部跳过
// (与 max 相差 >20 nats 的项不 exp, 单项贡献 < e^-20 ≈ 2.1e-9, 全词表
// lse 偏差 < 0.0006 nats) + expf。循环形状与生产逐字同款, 维持 gold
// 镜像位级同源; 生产侧归一化量随 prep 缓存不影响数值 (同一条求和路径)。
static const float kLseSkipNat = 20.0f;
static double cross_entropy(float* logits, int vs, int tid) {
    float m = -1e30f;
    for (int k = 0; k < vs; k++) if (logits[k] > m) m = logits[k];
    const float cut = m - kLseSkipNat;
    double se = 0;
    for (int k = 0; k < vs; k++) if (logits[k] >= cut) se += expf(logits[k] - m);
    return -((double)(logits[tid] - m) - log(se));
}

// Gold scoring: 生产同款三步批量 (rime_llm.cpp score_batch 镜像, 2026-11-02):
//   Step1 ctx decode 一次 (seq0 共享) → 全候选首 token CE 用 ctx logits
//   Step2 seq_cp 0→1..M, M 个 len≥2 候选首 token 并行 decode → 次 token CE
//   Step3 K 个 len≥3 候选次 token 并行 decode → 第三 token CE
// 加权前三 token CE (kW = 1.0, 1.13, 0.61), 无外推 —— 与旧单候选实现同数学,
// 只省掉每候选重复 prefill (旧版 5 次 memory_clear + 5 次 ctx decode;
// 生产实测同款批量 43ms/窗 vs 旧版 ~0.45s/窗)。4+ token 候选尾部不 decode
// (因果注意力下不影响前三项 CE, 旧版仅"保留 decode 为最小改动")。
//
// ★ 跨窗批处理判负 (2026-11-04, 已回滚): 曾实现 --wbatch N (B 窗合并 decode),
// 经 2000 窗×4 种批形状对拍: 任何偏离"逐窗生产形状"的配置都有 ~3% 首选翻转
// (68% 窗口分数偏 >0.01) —— 多序列×多 token 混批、逐轮 1-token 推进 (AR kernel
// 与 chunked 有精度差)、worker 状态从非 seq0 拷贝 / prefill 时其他 seq 持活状态,
// 全部引入确定性偏差 (该镜像 Qwen3.5 GDN 层多序列路径的固有特性)。
// 提速走多进程并行 (eval/sim_gpu_parallel.py), 精度零风险。
// ── CE 管线化 (2026-11-04): score_candidates 拆 decode_phase (llama 调用, 主线程,
//    逐窗串行序列一字不动) + ce_phase (纯 CPU 计算)。原实现里 CE(全词表双精度循环,
//    实测 14.9ms/窗) 卡在关键路径上让 GPU 空等 59%; 管线后主线程 decode 下一窗,
//    后台线程对"位级拷贝的 logits 行"算 CE 并按序输出。ce_phase 的运算表达式与
//    累加顺序逐字复刻原实现 → 分数位级不变 (CPU 引擎 A/B 字节级验证 + GPU 2000 窗
//    对拍门禁, 见 GPU批推理提速调研-2026-11-04.md §3)。
//    开关: 环境变量 SIM_NO_CE_PIPE=1 = 关闭管线走同步路径 (对照/排查用)。
struct CeTask {
    bool passthrough = false;          // 空 cands: 直接输出 raw_line
    std::string raw_line;              // passthrough 时的输出行
    int n = 0, vs = 0;
    std::vector<std::string> texts;    // 候选文本 (输出用)
    bool emit_scores = false;
    bool emit_ce = false;              // --ce: 追加输出逐 token CE 分量 + ntok
    std::vector<int> ntok;             // 候选 token 数 (decode_phase 填, 输出用)
    std::vector<std::array<double, 3>> ce;   // 逐 token CE 分量 (ce_phase 填; -1=不可算)
    // decode 结果 (decode_phase 填, ce_phase 读)
    bool ok1 = false, has_ctx = false;
    std::vector<float> ctx_row;        // ctx 末位 logits 位级拷贝
    std::vector<char> has_tok0;        // 候选非空 (有首 token)
    std::vector<llama_token> tok0;
    std::vector<int> idx2, idx3;
    std::vector<llama_token> tok1, tok2;             // 与 idx2/idx3 对位
    bool ok2 = false, ok3 = false;
    std::vector<char> row2_ok, row3_ok;              // 逐 s 行 logits 是否有效
    std::vector<std::vector<float>> rows2, rows3;    // 有效行的位级拷贝

    // 每窗状态复位 (任务池跨窗复用; 2026-11-04 实测教训: idx2/idx3 残留 push_back
    // 会让 M 滚大 → seq_cp 越界断言 + 残留索引污染累加翻转排序)。
    // clear() 保留容量, rows2/rows3 内层脏数据由 row2_ok/row3_ok 守卫不读。
    void reset() {
        passthrough = false; raw_line.clear();
        ok1 = has_ctx = ok2 = ok3 = false;
        n = 0; vs = 0; emit_scores = false; emit_ce = false;
        texts.clear(); ctx_row.clear();
        ntok.clear(); ce.clear();
        has_tok0.clear(); tok0.clear();
        idx2.clear(); idx3.clear();
        tok1.clear(); tok2.clear();
        row2_ok.clear(); row3_ok.clear();
    }
};

// decode 主线程段: llama 调用序列与原实现逐字一致, 仅把原处内联 CE 的位置改为拷行
static void decode_phase(const std::vector<llama_token>& ctx_ids,
                         const std::vector<std::vector<llama_token>>& cand_ids,
                         CeTask& t)
{
    t.reset();
    int n = (int)cand_ids.size(), ctx_len = (int)ctx_ids.size();
    t.n = n; t.vs = llama_vocab_n_tokens(g_vocab);
    t.has_tok0.assign(n, 0); t.tok0.assign(n, 0);
    t.ntok.resize(n);
    for (int i = 0; i < n; i++) t.ntok[i] = (int)cand_ids[i].size();

    llama_memory_t mem = llama_get_memory(g_ctx);
    llama_memory_clear(mem, false);   // 新窗全量重置元数据, 不零化 (同生产)
    auto tp0 = std::chrono::high_resolution_clock::now();

    // ---- Step 1: ctx decode (seq 0) ----
    llama_batch b1 = llama_batch_init(ctx_len, 0, 1);
    for (int j = 0; j < ctx_len; j++) {
        b1.token[j] = ctx_ids[j]; b1.pos[j] = j;
        b1.n_seq_id[j] = 1; b1.seq_id[j][0] = 0;
    }
    b1.logits[ctx_len - 1] = 1;
    b1.n_tokens = ctx_len;
    t.ok1 = llama_decode(g_ctx, b1) == 0;
    llama_batch_free(b1);
    if (!t.ok1) return;

    float* cl = llama_get_logits_ith(g_ctx, ctx_len - 1);
    if (!cl) return;                  // has_ctx=false → ce_phase 全 -1e10 (同原)
    t.has_ctx = true;
    t.ctx_row.assign(cl, cl + t.vs);
    for (int i = 0; i < n; i++)
        if (!cand_ids[i].empty()) { t.has_tok0[i] = 1; t.tok0[i] = cand_ids[i][0]; }
    auto tp1 = std::chrono::high_resolution_clock::now();
    double s1ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
    g_ms_step1 += s1ms;
    g_last_step1 = s1ms;

    // ---- Step 2: M 个 len≥2 候选首 token 并行 (KV copy ctx → worker seqs) ----
    for (int i = 0; i < n; i++) {
        if (cand_ids[i].size() >= 2) t.idx2.push_back(i);
        if (cand_ids[i].size() >= 3) t.idx3.push_back(i);
    }
    int M = (int)t.idx2.size();
    if (M > 0) {
        for (int s = 0; s < M; s++)
            llama_memory_seq_cp(mem, 0, s + 1, 0, -1);
        llama_batch b2 = llama_batch_init(M, 0, M);
        for (int s = 0; s < M; s++) {
            b2.token[s] = cand_ids[t.idx2[s]][0];
            b2.pos[s] = ctx_len;
            b2.n_seq_id[s] = 1; b2.seq_id[s][0] = s + 1; b2.logits[s] = 1;
        }
        b2.n_tokens = M;
        t.ok2 = llama_decode(g_ctx, b2) == 0;
        if (t.ok2) {
            t.row2_ok.assign(M, 0);
            if (t.rows2.size() < (size_t)M) t.rows2.resize(M);  // 只增不减: 内层缓冲随任务池复用
            t.tok1.resize(M);
            for (int s = 0; s < M; s++) {
                float* l = llama_get_logits_ith(g_ctx, s);
                if (l) { t.row2_ok[s] = 1; t.rows2[s].assign(l, l + t.vs); }
                t.tok1[s] = cand_ids[t.idx2[s]][1];
            }
        }   // !ok2 → ce_phase 对 idx2 全置 -1e10 (同原)
        llama_batch_free(b2);
    }

    // ---- Step 3: K 个 len≥3 候选次 token 并行 (与 Step2 同 worker seq) ----
    int K = (int)t.idx3.size();
    if (K > 0 && M > 0) {
        llama_batch b3 = llama_batch_init(K, 0, K);
        for (int s = 0; s < K; s++) {
            int ci = t.idx3[s];
            int seq = (int)(std::find(t.idx2.begin(), t.idx2.end(), ci)
                            - t.idx2.begin()) + 1;
            b3.token[s] = cand_ids[ci][1];
            b3.pos[s] = ctx_len + 1;
            b3.n_seq_id[s] = 1; b3.seq_id[s][0] = seq; b3.logits[s] = 1;
        }
        b3.n_tokens = K;
        t.ok3 = llama_decode(g_ctx, b3) == 0;
        if (t.ok3) {
            t.row3_ok.assign(K, 0);
            if (t.rows3.size() < (size_t)K) t.rows3.resize(K);  // 同上
            t.tok2.resize(K);
            for (int s = 0; s < K; s++) {
                float* l = llama_get_logits_ith(g_ctx, s);
                if (l) { t.row3_ok[s] = 1; t.rows3[s].assign(l, l + t.vs); }
                t.tok2[s] = cand_ids[t.idx3[s]][2];
            }
        }
        llama_batch_free(b3);
    }

    // worker seq KV 释放 (生产同款尾部清理; 下一窗 clear(mem,false) 亦会重置)
    for (int s = 1; s <= M; s++) llama_memory_seq_rm(mem, s, -1, -1);
    auto tp2 = std::chrono::high_resolution_clock::now();
    double s23 = std::chrono::duration<double, std::milli>(tp2 - tp1).count();
    g_ms_step23 += s23;
    g_last_step23 = s23;
    g_n_windows++;
}

// CE 纯计算段: 运算表达式与累加顺序逐字复刻原实现 (位级一致的前提)。
// comps 非空时 (--ce) 顺带记录逐 token CE 分量 (先算后加权, 表达式顺序不变;
// -1 = 该位不可算: 候选空 / 行 logits 无效 / 整步 decode 失败 / Step1 失败)
static std::vector<double> ce_phase(const CeTask& t,
                                    std::vector<std::array<double, 3>>* comps = nullptr)
{
    static const double kW[3] = {1.0, 1.13, 0.61};
    int n = t.n, vs = t.vs;
    std::vector<double> scores(n, -1e10);
    const std::array<double, 3> kNone = {-1.0, -1.0, -1.0};
    if (comps) comps->assign(n, kNone);
    if (!t.ok1 || !t.has_ctx) return scores;

    for (int i = 0; i < n; i++)
        if (t.has_tok0[i]) {
            double c = cross_entropy(const_cast<float*>(t.ctx_row.data()),
                                     vs, t.tok0[i]);
            if (comps) (*comps)[i][0] = c;
            scores[i] = -kW[0] * c;
        }

    int M = (int)t.idx2.size();
    if (M > 0) {
        if (t.ok2) {
            for (int s = 0; s < M; s++) {
                if (!t.row2_ok[s]) { scores[t.idx2[s]] -= kW[1] * 1e10; continue; }
                double c = cross_entropy(const_cast<float*>(t.rows2[s].data()),
                                         vs, t.tok1[s]);
                if (comps) (*comps)[t.idx2[s]][1] = c;
                scores[t.idx2[s]] -= kW[1] * c;
            }
        } else {
            for (int i : t.idx2) scores[i] = -1e10;
        }
    }

    int K = (int)t.idx3.size();
    if (K > 0 && M > 0) {
        if (t.ok3) {
            for (int s = 0; s < K; s++) {
                if (!t.row3_ok[s]) { scores[t.idx3[s]] -= kW[2] * 1e10; continue; }
                double c = cross_entropy(const_cast<float*>(t.rows3[s].data()),
                                         vs, t.tok2[s]);
                if (comps) (*comps)[t.idx3[s]][2] = c;
                scores[t.idx3[s]] -= kW[2] * c;
            }
        } else {
            for (int i : t.idx3) scores[i] = -1e10;
        }
    }
    return scores;
}

// 同步路径 (SIM_NO_CE_PIPE=1 / 兼容入口): 两段原样拼接 = 原实现
static std::vector<double> score_candidates(
    const std::vector<llama_token>& ctx_ids,
    const std::vector<std::vector<llama_token>>& cand_ids,
    std::vector<std::array<double, 3>>* comps = nullptr)
{
    CeTask t;
    decode_phase(ctx_ids, cand_ids, t);
    return ce_phase(t, comps);
}

// 转义并加引号的字符串条目 ("词" 形式, 与原 printf("\"%s\"") 逐字节一致)
static void append_quoted(std::string& out, const std::string& text)
{
    out += '"';
    for (char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    out += '"';
}

// scores 段条目: 追加 转义词 + ":%.6f (调用方负责前置的 ")
static void append_scores_entry(std::string& out, const std::string& text, double score)
{
    for (char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "\":%.6f", score);
    out += buf;
}

// 输出格式化: 两种模式共用一份 (管线/同步两路字节一致)
// ce/ntok 非空 (--ce) 时在 scores 后追加逐 token CE 分量与候选 token 数:
//   ...,"ce":{"词":[c1,c2,c3],...},"ntok":{"词":n,...}  (c=-1 = 该位不可算)
static std::string format_output(const std::vector<std::string>& texts,
                                 const std::vector<double>& scores, bool emit_scores,
                                 const std::vector<std::array<double, 3>>* ce = nullptr,
                                 const std::vector<int>* ntok = nullptr)
{
    std::vector<int> order(texts.size());
    for (int i = 0; i < (int)order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    std::string out;
    if (emit_scores) {
        out += "{\"ranked\":[";
        for (int i = 0; i < (int)order.size(); i++) {
            if (i > 0) out += ",";
            append_quoted(out, texts[order[i]]);
        }
        out += "],\"scores\":{";
        bool first = true;
        for (int i = 0; i < (int)texts.size(); i++) {
            if (scores[i] <= -1e9) continue;  // 失败哨兵不输出
            if (!first) out += ",";
            first = false;
            out += "\"";
            append_scores_entry(out, texts[i], scores[i]);
        }
        if (ce && ntok) {
            out += "},\"ce\":{";
            for (int i = 0; i < (int)texts.size(); i++) {
                if (i > 0) out += ",";
                out += "\"";
                for (char c : texts[i]) {
                    if (c == '"' || c == '\\') out += '\\';
                    out += c;
                }
                char buf[80];
                snprintf(buf, sizeof(buf), "\":[%.6f,%.6f,%.6f]",
                         (*ce)[i][0], (*ce)[i][1], (*ce)[i][2]);
                out += buf;
            }
            out += "},\"ntok\":{";
            for (int i = 0; i < (int)texts.size(); i++) {
                if (i > 0) out += ",";
                out += "\"";
                for (char c : texts[i]) {
                    if (c == '"' || c == '\\') out += '\\';
                    out += c;
                }
                char buf[32];
                snprintf(buf, sizeof(buf), "\":%d", (*ntok)[i]);
                out += buf;
            }
        }
        out += "}}\n";
        return out;
    }
    out += "[";
    for (int i = 0; i < (int)order.size(); i++) {
        if (i > 0) out += ",";
        append_quoted(out, texts[order[i]]);
    }
    out += "]\n";
    return out;
}

// Simple JSON parser: extract string value for key
static std::string json_get_string(const std::string& json, const char* key) {
    std::string search = std::string("\"") + key + "\"";
    size_t p = json.find(search);
    if (p == std::string::npos) return "";
    p = json.find("\"", p + search.size());
    if (p == std::string::npos) return "";
    size_t start = p + 1;
    // Handle escaped quotes
    std::string result;
    for (size_t i = start; i < json.size(); i++) {
        if (json[i] == '\\' && i + 1 < json.size()) { result += json[++i]; continue; }
        if (json[i] == '"') break;
        result += json[i];
    }
    return result;
}

// Extract int value for key（缺失返回 fallback；2026-11-18 逐行 ctx 覆盖用）
static int json_get_int(const std::string& json, const char* key, int fallback) {
    std::string search = std::string("\"") + key + "\"";
    size_t p = json.find(search);
    if (p == std::string::npos) return fallback;
    p = json.find(':', p + search.size());
    if (p == std::string::npos) return fallback;
    p++;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    size_t q = p;
    while (q < json.size() && (json[q] == '-' || (json[q] >= '0' && json[q] <= '9'))) q++;
    if (q == p) return fallback;
    return std::atoi(json.substr(p, q - p).c_str());
}

// Extract string array for key
static std::vector<std::string> json_get_array(const std::string& json, const char* key) {
    std::vector<std::string> result;
    std::string search = std::string("\"") + key + "\"";
    size_t p = json.find(search);
    if (p == std::string::npos) return result;
    p = json.find("[", p);
    if (p == std::string::npos) return result;
    size_t start = p + 1;
    std::string item;
    bool in_string = false;
    for (size_t i = start; i < json.size(); i++) {
        char c = json[i];
        if (c == '"' && (i == start || json[i-1] != '\\')) { in_string = !in_string; continue; }
        if (!in_string) {
            if (c == ',' || c == ']') {
                if (!item.empty()) { result.push_back(item); item.clear(); }
                if (c == ']') break;
                continue;
            }
            if (c == ' ' || c == '\n' || c == '\t') continue;
        }
        if (in_string) item += c;
    }
    return result;
}

// ── CE 管线: 主线程 decode → 入队; 后台线程算 CE + 按序写 stdout ──
// FIFO 单消费者 → 输出严格按输入序; 队列上限 8 (CE 慢于 decode 时自然背压);
// 任务对象走空闲池循环用 (每窗 ~5MB logits 行拷贝, 避免每窗 malloc 峰值)。
class CePipeline {
public:
    void start() { th = std::thread([this] { worker(); }); }
    CeTask* acquire() {
        std::lock_guard<std::mutex> lk(mu);
        if (freelist.empty()) return new CeTask();
        CeTask* t = freelist.back(); freelist.pop_back();
        return t;
    }
    void submit(CeTask* t) {
        std::unique_lock<std::mutex> lk(mu);
        cvSpace.wait(lk, [this] { return (int)q.size() < kMaxQ; });
        q.push_back(t);
        cvItem.notify_one();
    }
    void finish() {                     // 排干队列后收线程 (输出全部落盘)
        {
            std::lock_guard<std::mutex> lk(mu);
            stopping = true;
            cvItem.notify_all();
        }
        if (th.joinable()) th.join();
    }
private:
    static const int kMaxQ = 8;
    void worker() {
        for (;;) {
            CeTask* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cvItem.wait(lk, [this] { return stopping || !q.empty(); });
                if (q.empty()) return;  // stopping 且已排干
                t = q.front(); q.pop_front();
                cvSpace.notify_one();
            }
            if (t->passthrough) {
                fwrite(t->raw_line.data(), 1, t->raw_line.size(), stdout);
            } else {
                std::vector<double> scores = ce_phase(*t, t->emit_ce ? &t->ce : nullptr);
                std::string out = format_output(t->texts, scores, t->emit_scores,
                                                t->emit_ce ? &t->ce : nullptr,
                                                t->emit_ce ? &t->ntok : nullptr);
                fwrite(out.data(), 1, out.size(), stdout);
            }
            fflush(stdout);
            {
                std::lock_guard<std::mutex> lk(mu);
                freelist.push_back(t);   // rows 向量随任务复用 (assign 会复用容量)
            }
        }
    }
    std::thread th;
    std::mutex mu;
    std::condition_variable cvItem, cvSpace;
    std::deque<CeTask*> q, freelist;
    bool stopping = false;
};

int main(int argc, char* argv[]) {
    const char* model_path = (argc > 1 && argv[1][0] != '-') ? argv[1] : DEFAULT_MODEL_PATH;
    // --scores: 输出 {"ranked":[...],"scores":{词:原始分,...}} (权重分析用;
    // 默认仍输出纯 ranked 数组, eval_rerank.py 依赖不变)
    // --ce:    在 --scores 基础上追加逐 token CE 分量 "ce":{词:[c1,c2,c3]} 与
    //          候选 token 数 "ntok" (按长度分组标定用, 2026-11-05; -1 = 该位不可算)
    // --ngl N: GPU 卸载层数 (CUDA 构建; 默认 99=全卸载, 失败自动回退 CPU)
    // --ctx N: 上文截断 token 数 (默认 10 = 生产 max_ctx; §四十一 ctx×候选数
    //          等延迟再平衡实验用, 2026-11-10)
    bool emit_scores = false, emit_ce = false;
    int ngl = 99;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scores") == 0) emit_scores = true;
        else if (strcmp(argv[i], "--ce") == 0) { emit_ce = true; emit_scores = true; }
        else if (strcmp(argv[i], "--ngl") == 0 && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) g_max_ctx_tok = atoi(argv[++i]);
    }
    fprintf(stderr, "Model: %s%s%s ngl=%d ctx=%d\n", model_path,
            emit_scores ? " (--scores)" : "", emit_ce ? " (+ce)" : "", ngl,
            g_max_ctx_tok);

    // Init model
    auto t0 = std::chrono::high_resolution_clock::now();
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();  // use_mmap 上游已删（默认即 mmap）
    mparams.n_gpu_layers = ngl;
    g_model = llama_model_load_from_file(model_path, mparams);
    if (!g_model && ngl > 0) {
        fprintf(stderr, "WARN: GPU offload failed, retry CPU (ngl=0)\n");
        mparams.n_gpu_layers = 0;
        g_model = llama_model_load_from_file(model_path, mparams);
    }
    if (!g_model) { fprintf(stderr, "FATAL: model load failed\n"); return 3; }
    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_CTX;
    cparams.n_seq_max = 11;  // seq0 + 10 worker seq（标定窗 MAX_CAND=10, 2026-11-05;
                             // 默认 1 会拒收 seq>=1; 生产 rime_llm 为 12）
    cparams.n_threads = N_THREADS;
    cparams.n_threads_batch = N_THREADS;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) { fprintf(stderr, "FATAL: context create failed\n"); return 3; }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "Model loaded in %lld ms (n_ctx=%d threads=%d)\n", load_ms, N_CTX, N_THREADS);

    // CE 管线开关: SIM_NO_CE_PIPE=1 关闭 (同步对照/排查; CPU 模式多进程跑批建议关,
    // 后台 CE 线程会与 ggml 计算线程抢核, 破坏"进程数×6线程≤物理核"的贴核铁律)
    bool pipe_on = true;
    const char* no_pipe = getenv("SIM_NO_CE_PIPE");
    if (no_pipe && no_pipe[0] && strcmp(no_pipe, "0") != 0) pipe_on = false;
    fprintf(stderr, "CE pipe: %s\n", pipe_on ? "on" : "off (SIM_NO_CE_PIPE)");
    // 逐窗耗时输出（2026-11-18）：SIM_TIMES=path → 每窗一行 "decode_ms"，
    // 供延迟分位数（中位数/p90）统计。默认不写，行为与旧版逐字节一致。
    FILE* times_fp = nullptr;
    FILE* step23_fp = nullptr;
    FILE* prep_fp = nullptr;
    if (const char* tp = getenv("SIM_TIMES"); tp && tp[0]) {
        times_fp = fopen(tp, "w");
        if (!times_fp) fprintf(stderr, "warn: 打不开 SIM_TIMES=%s\n", tp);
    }
    if (const char* sp = getenv("SIM_STEP23"); sp && sp[0]) {
        step23_fp = fopen(sp, "w");
        if (!step23_fp) fprintf(stderr, "warn: 打不开 SIM_STEP23=%s\n", sp);
    }
    if (const char* pp = getenv("SIM_PREP"); pp && pp[0]) {
        prep_fp = fopen(pp, "w");
        if (!prep_fp) fprintf(stderr, "warn: 打不开 SIM_PREP=%s\n", pp);
    }
    CePipeline pipe;
    if (pipe_on) pipe.start();
    auto w0 = std::chrono::high_resolution_clock::now();

    // Process stdin line by line
    std::string line;
    int count = 0;
    double total_ms = 0;   // 管线化后只计 decode 段 (CE 已移出关键路径)
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::string context = json_get_string(line, "context");
        if (context.empty()) context = " ";
        auto cand_texts = json_get_array(line, "cands");
        if (cand_texts.empty()) {
            if (pipe_on) {                       // 经队列保序 (主线程不得直接写 stdout)
                CeTask* t = pipe.acquire();
                t->passthrough = true; t->raw_line = "[]\n";
                pipe.submit(t);
            } else {
                printf("[]\n");
            }
            continue;
        }

        // Tokenize context (trim to g_max_ctx_tok, --ctx 覆盖; 逐行 "ctx" 再覆盖)
        auto ctx_ids = tokenize(context.c_str());
        int ctx_limit = json_get_int(line, "ctx", g_max_ctx_tok);
        if (ctx_limit <= 0) ctx_limit = g_max_ctx_tok;
        if ((int)ctx_ids.size() > ctx_limit)
            ctx_ids.erase(ctx_ids.begin(), ctx_ids.end() - ctx_limit);
        ctx_ids = ctx_ids_for_ce(std::move(ctx_ids));  // 仅 n=1 垫「。」锚（与生产同口径）

        // Tokenize candidates
        std::vector<std::vector<llama_token>> cand_ids;
        for (auto& c : cand_texts)
            cand_ids.push_back(tokenize(c.c_str()));

        auto t_s0 = std::chrono::high_resolution_clock::now();
        if (pipe_on) {
            CeTask* t = pipe.acquire();
            decode_phase(ctx_ids, cand_ids, *t);
            t->texts = std::move(cand_texts);
            t->emit_scores = emit_scores;
            t->emit_ce = emit_ce;
            pipe.submit(t);
        } else {
            std::vector<std::array<double, 3>> comps;
            auto scores = score_candidates(ctx_ids, cand_ids, emit_ce ? &comps : nullptr);
            std::vector<int> ntokv;
            if (emit_ce) {
                ntokv.reserve(cand_ids.size());
                for (auto& c : cand_ids) ntokv.push_back((int)c.size());
            }
            std::string out = format_output(cand_texts, scores, emit_scores,
                                            emit_ce ? &comps : nullptr,
                                            emit_ce ? &ntokv : nullptr);
            fwrite(out.data(), 1, out.size(), stdout);
            fflush(stdout);
        }
        auto t_s1 = std::chrono::high_resolution_clock::now();
        double one_ms = std::chrono::duration<double, std::micro>(t_s1 - t_s0).count() / 1000.0;
        total_ms += one_ms;
        if (times_fp) fprintf(times_fp, "%.3f\n", one_ms);
        if (step23_fp) fprintf(step23_fp, "%.3f\n", g_last_step23);
        if (prep_fp) fprintf(prep_fp, "%.3f\n", g_last_step1);

        count++;
        if (count % 10 == 0)
            fprintf(stderr, "  processed %d entries\n", count);
    }
    if (pipe_on) pipe.finish();   // 排干队列, 全部输出落盘后才进 summary
    if (times_fp) fclose(times_fp);
    if (step23_fp) fclose(step23_fp);
    if (prep_fp) fclose(prep_fp);

    double avg_ms = count > 0 ? total_ms / count : 0;
    double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - w0).count();
    fprintf(stderr, "Done. %d entries, decode %.1f ms avg, wall %.0f ms (pipe %s)"
            ", prep %.1f ms avg, cand %.1f ms avg (%d scored)\n",
            count, avg_ms, wall_ms, pipe_on ? "on" : "off",
            g_n_windows ? g_ms_step1 / g_n_windows : 0.0,
            g_n_windows ? g_ms_step23 / g_n_windows : 0.0,
            g_n_windows);

    llama_free(g_ctx);
    llama_model_free(g_model);
    llama_backend_free();
    return 0;
}
