// ce_probe3.cpp — pos-0 异常的三向隔离实验
//   A: [改革]@pos0        —— 基线（已知坏）
//   B: [改革]@pos5        —— 同 token 放中间位置（前面无内容）
//   C: [改革,改革] 读 row0 —— 两 token 批中第 0 行的分布
//   D: [。改革] 读 row1    —— 对照（已知好）
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
static std::vector<llama_token> tokenize(const llama_vocab *v, const char *text) {
  std::vector<llama_token> toks(128);
  int n = llama_tokenize(v, text, (int)strlen(text), toks.data(),
                         (int)toks.size(), false, true);
  if (n < 0) {
    toks.resize(-n);
    n = llama_tokenize(v, text, (int)strlen(text), toks.data(),
                       (int)toks.size(), false, true);
  }
  toks.resize(n > 0 ? n : 0);
  return toks;
}
static void run_case(llama_context *ctx, const llama_vocab *v, int vs, const char *name,
                     const std::vector<llama_token> &ids, int pos_base,
                     int read_row, int cand_id) {
  llama_memory_clear(llama_get_memory(ctx), false);
  llama_batch b = llama_batch_init((int)ids.size(), 0, 1);
  for (int j = 0; j < (int)ids.size(); j++) {
    b.token[j] = ids[j];
    b.pos[j] = pos_base + j;
    b.n_seq_id[j] = 1;
    b.seq_id[j][0] = 0;
    b.logits[j] = (j == read_row) ? 1 : 0;
  }
  b.n_tokens = (int)ids.size();
  if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return; }
  llama_batch_free(b);
  const float *cl = llama_get_logits_ith(ctx, read_row);
  if (!cl) { fprintf(stderr, "no logits\n"); return; }
  float m0; double lse0;
  logits_normalizer(cl, vs, m0, lse0);
  // top-3
  int top[3]; double tv[3];
  for (int k = 0; k < 3; k++) { top[k] = -1; tv[k] = -1e30; }
  for (int k = 0; k < vs; k++) {
    if (llama_vocab_is_control(v, k) || llama_vocab_is_eog(v, k)) continue;
    if (cl[k] > tv[2]) {
      int p = 2;
      while (p > 0 && cl[k] > tv[p-1]) { tv[p] = tv[p-1]; top[p] = top[p-1]; p--; }
      tv[p] = cl[k]; top[p] = k;
    }
  }
  printf("%-22s top3:", name);
  for (int k = 0; k < 3 && top[k] >= 0; k++)
    printf(" %s(%.2f)", llama_vocab_get_text(v, top[k]), tv[k] - m0 - lse0);
  printf(" | 开放=%.3f\n", ce_target(cl, cand_id, m0, lse0));
}

int main(int argc, char **argv) {
  const char *model = "C:/Users/Administrator/AppData/Roaming/Rime/Qwen3-0.6B-Base-Q4_K_M.gguf";
  for (int i = 1; i < argc; i++)
    if (argv[i][0] != '-') model = argv[i];

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
  cp.n_threads = 14;
  cp.n_threads_batch = 14;
  cp.n_seq_max = 20;
  cp.kv_unified = true;
  llama_context *ctx = llama_init_from_model(m, cp);
  if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }

  auto gaifang = tokenize(v, "开放");
  int kai_id = gaifang.empty() ? -1 : gaifang[0];
  auto t_ref = tokenize(v, "改革");
  if (t_ref.size() != 1 || kai_id < 0) {
    printf("tokenize unexpected: 改革=%d 开放=%d\n", (int)t_ref.size(), kai_id);
    return 1;
  }
  // token 文本确认（UTF-8 原样输出，控制台可能乱码但字节可核对）
  printf("token(改革)=%d token(开放)=%d\n", t_ref[0], kai_id);

  run_case(ctx, v, vs, "A [改革]@pos0", {t_ref[0]}, 0, 0, kai_id);
  run_case(ctx, v, vs, "B [改革]@pos5", {t_ref[0]}, 5, 0, kai_id);
  run_case(ctx, v, vs, "C [改革,改革]row0", {t_ref[0], t_ref[0]}, 0, 0, kai_id);
  run_case(ctx, v, vs, "D [。改革]row1", {tokenize(v, "。")[0], t_ref[0]}, 0, 1, kai_id);

  llama_free(ctx);
  llama_model_free(m);
  return 0;
}
