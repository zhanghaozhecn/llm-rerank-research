// ce_probe4.cpp — EOS anchor 修复验证 + 多模型对照
//   每个 ctx 词测三种形态: bare / EOS+word / 。+word, 读末行 CE1
//   输出写到文件 (raw vocab 字节串 = GPT-2 byte-unicode 形式, 由 python 解码)
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

int main(int argc, char **argv) {
  const char *model = "C:/Users/Administrator/AppData/Roaming/Rime/Qwen3-0.6B-Base-Q4_K_M.gguf";
  const char *outp = "probe4_out.txt";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (argv[i][0] != '-') model = argv[i];
  }
  FILE *fo = fopen(outp, "wb");

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
#ifdef LLAMA_LOAD_MODE_MMAP_MLOCK
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
#endif
  llama_model *m = llama_model_load_from_file(model, mp);
  if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
  const llama_vocab *v = llama_model_get_vocab(m);
  const int vs = llama_vocab_n_tokens(v);
  llama_token eot = llama_vocab_eot(v);   // EOT 类锚（im_end）
  llama_token eos = llama_vocab_eos(v);   // EOS 类锚（endoftext，训练文档分隔符）
  fprintf(fo, "model=%s vs=%d eos=%d eot=%d\n", model, vs, (int)eos, (int)eot);

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_threads = 14;
  cp.n_threads_batch = 14;
  cp.n_seq_max = 20;
  cp.kv_unified = true;
  llama_context *ctx = llama_init_from_model(m, cp);
  if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }

  const char *ctx_words[] = { "二十", "英雄", "化学" };
  const char *cand_words[] = { "世纪", "事迹", "试剂", "实际", "时机" };

  for (const char *w : ctx_words) {
    auto base = tokenize(v, w);
    if (base.empty()) continue;
    // 四种形态: bare / [eos]+base / [eot]+base / [。]+base
    auto dun = tokenize(v, "。");
    std::vector<std::vector<llama_token>> forms = { base, {}, {}, {} };
    for (int fi = 1; fi <= 2; fi++) {
      llama_token anch = (fi == 1) ? eos : eot;
      if (anch >= 0) {
        forms[fi].push_back(anch);
        forms[fi].insert(forms[fi].end(), base.begin(), base.end());
      }
    }
    forms[3] = dun;
    forms[3].insert(forms[3].end(), base.begin(), base.end());

    const char *names[] = { "bare", "eos+", "eot+", "punct+" };
    for (int fi = 0; fi < 4; fi++) {
      auto &ids = forms[fi];
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
      if (!cl) continue;
      float m0; double lse0;
      logits_normalizer(cl, vs, m0, lse0);
      fprintf(fo, "ctx=%s form=%s ntok=%d |", w, names[fi], (int)ids.size());
      for (const char *cand : cand_words) {
        auto t = tokenize(v, cand);
        if (t.empty()) continue;
        // 候选词按其真实 token 序列算 CE: 首 token 用 ctx 行, 与生产 CE1 同口径
        double ce = ce_target(cl, t[0], m0, lse0);
        fprintf(fo, " %s=%.3f", cand, ce);
      }
      fprintf(fo, "\n");
    }
  }
  fclose(fo);
  llama_free(ctx);
  llama_model_free(m);
  return 0;
}
