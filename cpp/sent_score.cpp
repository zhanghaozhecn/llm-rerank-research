// sent_score.cpp — 整句 LLM 似然打分（研究工具，2026-11-23）
//
// 用途：给「N-gram 组句候选」逐句算 Qwen3-0.6B 的对数似然，供 LLM 句级重排
// 实验（先 N-gram 生成 top-5，再用 LLM 句子得分选优）。生产 DLL 的
// score_batch 只算候选前 3 个 token（词级重排用），不适用于整句；本工具独立
// 链接同一份 llama.cpp 静态库与同一个 GGUF，算完整句子的 CE。
//
// 输入（stdin，TAB 分隔，每行一个候选）：
//   sid \t idx \t context \t candidate
// 输出（stdout，TAB 分隔）：
//   sid \t idx \t sum_ce \t mean_ce \t n_tok \t ctx_tok
//   sum_ce  = Σ -log P(t_i | context, t_<i)（candidate 的全部 token）
//   mean_ce = sum_ce / n_tok
//
// 用法：
//   sent_score.exe [model.gguf] < jobs.tsv > scores.tsv
//   model 缺省 d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf
//
// 构建：build_sent_score.bat（与 bench_threads 同一套 MT 静态库）。
#define NOMINMAX
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_model *g_model = nullptr;
static const llama_vocab *g_vocab = nullptr;

static std::vector<llama_token> tokenize(const std::string &text) {
  std::vector<llama_token> toks(512);
  int n = llama_tokenize(g_vocab, text.c_str(), (int)text.size(), toks.data(),
                         (int)toks.size(), false, false);
  if (n < 0) {  // buffer too small
    toks.resize(-n);
    n = llama_tokenize(g_vocab, text.c_str(), (int)text.size(), toks.data(),
                       (int)toks.size(), false, false);
  }
  if (n <= 0)
    return {};
  toks.resize(n);
  return toks;
}

static double cross_entropy(const float *logits, int vs, int target) {
  float m = -1e30f;
  for (int k = 0; k < vs; k++)
    if (logits[k] > m)
      m = logits[k];
  double se = 0;
  for (int k = 0; k < vs; k++)
    se += std::exp((double)(logits[k] - m));
  return -((double)(logits[target] - m) - std::log(se));
}

int main(int argc, char **argv) {
  const char *model_path =
      argc > 1 ? argv[1] : "d:/gguf_models/Qwen3-0.6B-Base-Q4_K_M.gguf";
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
  mp.n_gpu_layers = 0;
  g_model = llama_model_load_from_file(model_path, mp);
  if (!g_model) {
    fprintf(stderr, "model load failed: %s\n", model_path);
    return 1;
  }
  g_vocab = llama_model_get_vocab(g_model);
  const int vs = llama_vocab_n_tokens(g_vocab);
  const int kCtx = 1024;

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = kCtx;
  cp.n_threads = 8;
  cp.n_threads_batch = 8;
  cp.n_seq_max = 1;
  llama_context *ctx = llama_init_from_model(g_model, cp);
  if (!ctx) {
    fprintf(stderr, "ctx create failed\n");
    llama_model_free(g_model);
    llama_backend_free();
    return 1;
  }

  char line[16384];
  while (fgets(line, sizeof(line), stdin)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    if (!len)
      continue;
    // sid \t idx \t ctx \t cand
    char *t1 = strchr(line, '\t');
    if (!t1)
      continue;
    *t1++ = 0;
    char *t2 = strchr(t1, '\t');
    if (!t2)
      continue;
    *t2++ = 0;
    char *t3 = strchr(t2, '\t');
    if (!t3)
      continue;
    *t3++ = 0;
    const std::string sid = line, idx = t1, text_ctx = t2, text_cand = t3;

    std::vector<llama_token> cids = tokenize(text_ctx);
    std::vector<llama_token> kids = tokenize(text_cand);
    if (kids.empty()) {
      printf("%s\t%s\t0\t0\t0\t%d\n", sid.c_str(), idx.c_str(),
             (int)cids.size());
      continue;
    }
    if ((int)cids.size() > kCtx - 64)
      cids.erase(cids.begin(), cids.end() - (kCtx - 64));
    const int nc = (int)cids.size();
    const int nk = (int)kids.size();
    const int total = nc + nk;
    if (total > kCtx) {  // clamp candidate
      kids.resize(kCtx - nc);
    }
    const int ntot = nc + (int)kids.size();

    llama_memory_clear(llama_get_memory(ctx), false);
    llama_batch b = llama_batch_init(ntot, 0, 1);
    for (int i = 0; i < nc; i++) {
      b.token[i] = cids[i];
      b.pos[i] = i;
      b.n_seq_id[i] = 1;
      b.seq_id[i][0] = 0;
      b.logits[i] = 0;
    }
    for (int j = 0; j < (int)kids.size(); j++) {
      int i = nc + j;
      b.token[i] = kids[j];
      b.pos[i] = i;
      b.n_seq_id[i] = 1;
      b.seq_id[i][0] = 0;
      // 预测第 j 个候选 token 的 logits 来自位置 nc-1+j（batch 行号同）
      b.logits[i] = (i < ntot - 1) ? 1 : 0;
    }
    if (nc > 0)
      b.logits[nc - 1] = 1;
    b.n_tokens = ntot;

    double sum = 0.0;
    int counted = 0;
    if (llama_decode(ctx, b) == 0) {
      for (int j = 0; j < (int)kids.size(); j++) {
        int row = nc - 1 + j;
        if (row < 0 || row >= ntot - 1)
          continue;  // 无上文时首个 token 无法计算
        const float *lg = llama_get_logits_ith(ctx, row);
        if (!lg)
          continue;
        sum += cross_entropy(lg, vs, kids[j]);
        counted++;
      }
    }
    llama_batch_free(b);
    double mean = counted ? sum / counted : 0.0;
    printf("%s\t%s\t%.6f\t%.6f\t%d\t%d\n", sid.c_str(), idx.c_str(), sum, mean,
           counted, nc);
    fflush(stdout);
  }

  llama_free(ctx);
  llama_model_free(g_model);
  llama_backend_free();
  return 0;
}
