// ce_probe2.cpp — 单 token(pos-0) 解码健康度探针
//   每个 ctx: ① top-5 argmax 词元与概率 ② 指定搭配词的 CE1
//   用以判定 bare 单 token 上下文的预测分布是否被先验主导/结构异常
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static const float kLseSkipNat = 20.0f;

static void logits_normalizer(const float *logits, int vs, float &m, double &lse) {
  m = -1e30f;
  for (int k = 0; k < vs; k++)
    if (logits[k] > m) m = logits[k];
  const float cut = m - kLseSkipNat;
  double se = 0;
  for (int k = 0; k < vs; k++)
    if (logits[k] >= cut) se += expf(logits[k] - m);
  lse = log(se);
}
static double ce_target(const float *logits, int target_id, float m, double lse) {
  return -((double)(logits[target_id] - m) - lse);
}
static std::vector<llama_token> tokenize(const llama_vocab *v, const char *text,
                                         bool add_special) {
  std::vector<llama_token> toks(128);
  int n = llama_tokenize(v, text, (int)strlen(text), toks.data(),
                         (int)toks.size(), add_special, true);
  if (n < 0) {
    toks.resize(-n);
    n = llama_tokenize(v, text, (int)strlen(text), toks.data(),
                       (int)toks.size(), add_special, true);
  }
  toks.resize(n > 0 ? n : 0);
  return toks;
}

struct Case { const char *ctx; std::vector<const char *> cands; };

int main(int argc, char **argv) {
  const char *model = "C:/Users/Administrator/AppData/Roaming/Rime/Qwen3-0.6B-Base-Q4_K_M.gguf";
  int threads = 14;
  FILE *fout = NULL;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-' && argv[i][0] != '\0' && strcmp(argv[i], "model") != 0) {}
    if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) fout = fopen(argv[++i], "wb");
    else if (argv[i][0] != '-') model = argv[i];
  }
#define P(...) do { printf(__VA_ARGS__); if (fout) fprintf(fout, __VA_ARGS__); } while (0)

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
#ifdef LLAMA_LOAD_MODE_MMAP_MLOCK
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
#endif
  llama_model *m = llama_model_load_from_file(model, mp);
  if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab *v = llama_model_get_vocab(m);
  const int vs = llama_vocab_n_tokens(v);

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.n_seq_max = 20;
  cp.kv_unified = true;
  llama_context *ctx = llama_init_from_model(m, cp);
  if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }

  Case cases[] = {
    {"中国", {"人民", "世纪", "事迹", "试剂"}},
    {"改革", {"开放", "世纪", "事迹", "试剂"}},
    {"今天", {"天气", "世纪", "事迹", "试剂"}},
    {"英雄", {"事迹", "试剂", "世纪"}},
    {"化学", {"试剂", "事迹", "世纪"}},
    {"二十", {"世纪", "事迹", "试剂"}},
  };
  for (auto &c : cases) {
    auto ids = tokenize(v, c.ctx, true);
    P("ctx=\"%s\" ntok=%d | ", c.ctx, (int)ids.size());
    if (ids.empty()) continue;
    llama_memory_clear(llama_get_memory(ctx), false);
    llama_batch b = llama_batch_init((int)ids.size(), 0, 1);
    for (int j = 0; j < (int)ids.size(); j++) {
      b.token[j] = ids[j];
      b.pos[j] = j;
      b.n_seq_id[j] = 1;
      b.seq_id[j][0] = 0;
      b.logits[j] = (j + 1 == (int)ids.size()) ? 1 : 0;
    }
    b.n_tokens = (int)ids.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    llama_batch_free(b);
    const float *cl = llama_get_logits_ith(ctx, (int)ids.size() - 1);
    if (!cl) { fprintf(stderr, "no logits\n"); return 1; }
    float m0; double lse0;
    logits_normalizer(cl, vs, m0, lse0);
    // top-5 argmax（排除 control/EOG）
    int top[5]; double tv[5];
    for (int k = 0; k < 5; k++) { top[k] = -1; tv[k] = -1e30; }
    for (int k = 0; k < vs; k++) {
      if (llama_vocab_is_control(v, k) || llama_vocab_is_eog(v, k)) continue;
      if (cl[k] > tv[4]) {
        int p = 4;
        while (p > 0 && cl[k] > tv[p-1]) { tv[p] = tv[p-1]; top[p] = top[p-1]; p--; }
        tv[p] = cl[k]; top[p] = k;
      }
    }
    printf("top5:");
    for (int k = 0; k < 5 && top[k] >= 0; k++)
      P(" %s(%.2f)", llama_vocab_get_text(v, top[k]), tv[k] - m0 - lse0);
    P(" | ce:");
    for (const char *cand : c.cands) {
      auto t = tokenize(v, cand, false);
      double ce = ce_target(cl, t[0], m0, lse0);
      P(" %s=%.3f", cand, ce);
    }
    printf("\n");
  }
  llama_free(ctx);
  llama_model_free(m);
  return 0;
}
