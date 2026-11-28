// ce_docstart.cpp — wiki 分条首句批量实验: 真实文档起始场景下的锚 A/B
//   每样本句: 取 n∈{1,2,4,8} token 前缀为上文, 真值续 token = T[n]
//   四形态: bare / [eos]+ / [eot]+ / [。]+ ; 度量 = 真值 token 的 CE + top1 命中
//   输出 CSV: n,form,ce,hit (逐样本行, 由 python 聚合)
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
  std::vector<llama_token> toks(256);
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

int main(int argc, char **argv) {
  const char *model = "C:/Users/Administrator/AppData/Roaming/Rime/Qwen3-0.6B-Base-Q4_K_M.gguf";
  const char *samples = "samples.txt";
  const char *outp = "docstart_out.csv";
  int threads = 14;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = argv[++i];
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (argv[i][0] != '-') model = argv[i];
  }
  FILE *fo = fopen(outp, "w");

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
#ifdef LLAMA_LOAD_MODE_MMAP_MLOCK
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
#endif
  llama_model *m = llama_model_load_from_file(model, mp);
  if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab *v = llama_model_get_vocab(m);
  const int vs = llama_vocab_n_tokens(v);
  llama_token eos = llama_vocab_eos(v);
  llama_token eot = llama_vocab_eot(v);
  auto dun = tokenize(v, "。");
  llama_token punct = dun.empty() ? -1 : dun[0];
  fprintf(stderr, "vs=%d eos=%d eot=%d punct=%d\n", vs, (int)eos, (int)eot, (int)punct);

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.n_seq_max = 20;
  cp.kv_unified = true;
  llama_context *ctx = llama_init_from_model(m, cp);
  if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }

  const int NS[] = { 1, 2, 4, 8 };
  const char *FN[] = { "bare", "eos+", "eot+", "punct+" };

  FILE *fs = fopen(samples, "rb");
  if (!fs) { fprintf(stderr, "samples open failed\n"); return 1; }
  char line[512];
  int sent_idx = 0, cases = 0;
  while (fgets(line, sizeof(line), fs)) {
    size_t L = strlen(line);
    while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
    if (L < 4) continue;
    auto T = tokenize(v, line);
    if ((int)T.size() < 10) continue;  // 需要 8 前缀 + 续 token
    for (int ni = 0; ni < 4; ni++) {
      int n = NS[ni];
      if ((int)T.size() <= n) continue;
      int target = T[n];
      if (llama_vocab_is_control(v, target) || llama_vocab_is_eog(v, target)) continue;
      for (int fi = 0; fi < 4; fi++) {
        std::vector<llama_token> ids;
        if (fi == 1 && eos >= 0) ids.push_back(eos);
        else if (fi == 2 && eot >= 0) ids.push_back(eot);
        else if (fi == 3 && punct >= 0) ids.push_back(punct);
        else if (fi >= 1) continue;
        for (int j = 0; j < n; j++) ids.push_back(T[j]);
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
        if (!cl) continue;
        float m0; double lse0;
        logits_normalizer(cl, vs, m0, lse0);
        double ce = ce_target(cl, target, m0, lse0);
        // top1 命中(排除 control/EOG)
        int am = -1; float bv = -1e30f;
        for (int k = 0; k < vs; k++) {
          if (llama_vocab_is_control(v, k) || llama_vocab_is_eog(v, k)) continue;
          if (cl[k] > bv) { bv = cl[k]; am = k; }
        }
        fprintf(fo, "%d,%s,%.4f,%d\n", n, FN[fi], ce, am == target ? 1 : 0);
        cases++;
      }
    }
    sent_idx++;
  }
  fclose(fs);
  fclose(fo);
  fprintf(stderr, "sentences=%d cases=%d\n", sent_idx, cases);
  llama_free(ctx);
  llama_model_free(m);
  return 0;
}
