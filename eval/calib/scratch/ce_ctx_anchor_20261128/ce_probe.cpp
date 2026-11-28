// ce_probe.cpp — 精确复刻 rime-llm-ime llm_filter.cc 的 CE1 计算路径
//   tokenize(add_special, parse_special) -> ensure_ctx_kv 式 ctx decode
//   -> get_logits_ith(ctx_len-1) -> logits_normalizer(skip 20nats) -> ce_target
// A/B 开关：
//   --nospecial  tokenize add_special=false (默认 true = 生产)
//   --split      kv_unified=false      (默认 true = 生产)
//   --mmap       load_mode=MMAP        (默认 MMAP_MLOCK = 生产)
//   --threads N  默认 14 (生产 cpu_cores)
// 输出每个 (ctx, cand) 的 CE1(nats)。
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

int main(int argc, char **argv) {
  bool add_special = true, unified = true, mmap_only = false;
  int threads = 14;
  const char *model = "C:/Users/Administrator/AppData/Roaming/Rime/Qwen3-0.6B-Base-Q4_K_M.gguf";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--nospecial")) add_special = false;
    else if (!strcmp(argv[i], "--split")) unified = false;
    else if (!strcmp(argv[i], "--mmap")) mmap_only = true;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (argv[i][0] != '-') model = argv[i];
  }

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
#ifdef LLAMA_LOAD_MODE_MMAP_MLOCK
  mp.load_mode = mmap_only ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_MMAP_MLOCK;
#else
  (void)mmap_only;  // 旧 API：默认 mmap，无 MLOCK 档
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
  cp.kv_unified = unified;
  llama_context *ctx = llama_init_from_model(m, cp);
  if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }

  const char *ctx_texts[] = { "二十", "英雄", "化学", "，", "二十，", "。英雄" };
  const char *cand_texts[] = { "世纪", "事迹", "试剂", "事绩", "时机", "时节", "实际" };

  for (const char *ctx_text : ctx_texts) {
    auto ids = tokenize(v, ctx_text, add_special);
    printf("ctx=\"%s\" tokens=%d [", ctx_text, (int)ids.size());
    for (size_t j = 0; j < ids.size(); j++)
      printf("%s%s", j ? "," : "", llama_vocab_get_text(v, ids[j]));
    printf("]\n");
    if (ids.empty()) continue;

    // ensure_ctx_kv 复刻：全量重建（memory_clear + decode，末行 logits）
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

    for (const char *cand : cand_texts) {
      auto t = tokenize(v, cand, false);  // 候选词按生产候选 tokenize（无 special）
      double ce = 0;
      std::string toks_desc;
      // 单 token 候选走 CE1；多 token 词这里也算出首 token 分量供对照
      double c1 = ce_target(cl, t[0], m0, lse0);
      ce = c1;
      for (size_t j = 0; j < t.size(); j++)
        toks_desc += (j ? "," : "") + std::string(llama_vocab_get_text(v, t[j]));
      printf("  %-6s ntok=%d [%s] ce1=%.4f\n", cand, (int)t.size(),
             toks_desc.c_str(), c1);
    }
  }
  llama_free(ctx);
  llama_model_free(m);
  return 0;
}
