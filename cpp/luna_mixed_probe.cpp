// luna_mixed_probe.cpp -- headless engine probe for the luna mixed-window
// cohort (research toolchain, 2026-11-24).
//
// For every code read from stdin, enumerates the schema's candidates in
// engine order and prints text + comment. The research schema luna_mixed
// ends its filter chain with lua_filter@dump_span, which packs each
// candidate's start/end/quality/type into the comment (the C API candidate
// struct carries no span). Engine candidate generation in luna does not
// depend on the context, so a bare code per line is enough.
//
// usage: luna_mixed_probe.exe <rime.dll> <shared_data_dir> <user_data_dir>
//                          <schema_id> [max_cands=32]
// stdin : one code per line (lowercase letters); blank and '#' lines skipped
// stdout: Q\t<code>\t<count>\t<ms>\t<text\x1e comment\x1f text\x1e comment...>
//         (UTF-8 bytes verbatim; TAB/newline in text replaced by a space)
//         A code that produces no candidate is printed with count 0.
// The probe only links kernel32 (rime.dll is loaded at run time).
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <windows.h>
#include "rime_api.h"

static RimeApi* g_api = nullptr;

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(
             steady_clock::now().time_since_epoch()).count();
}

static void append_clean(std::string& out, const char* s) {
  for (const char* p = s; *p; ++p) {
    if (*p == '\t' || *p == '\n' || *p == '\r' || *p == '\x1e' ||
        *p == '\x1f')
      out.push_back(' ');
    else
      out.push_back(*p);
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: luna_mixed_probe <rime.dll> <shared_data_dir> "
            "<user_data_dir> <schema_id> [max_cands]\n");
    return 2;
  }
  const char* dll = argv[1];
  const char* shared = argv[2];
  const char* user = argv[3];
  const char* schema = argv[4];
  const int max_cands = (argc > 5) ? atoi(argv[5]) : 32;

  // rime.dll depends on opencc.dll sitting next to it; the altered search
  // path makes that directory the first lookup location (cwd-independent).
  HMODULE h = LoadLibraryExA(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!h) {
    fprintf(stderr, "LoadLibrary failed err=%lu (dll=%s)\n", GetLastError(),
            dll);
    return 1;
  }
  RimeApi* (*get_api)() = (RimeApi * (*)())GetProcAddress(h, "rime_get_api");
  if (!get_api) {
    fprintf(stderr, "GetProcAddress(rime_get_api) failed\n");
    return 1;
  }
  g_api = get_api();

  RimeTraits traits = {};
  RIME_STRUCT_INIT(RimeTraits, traits);
  traits.shared_data_dir = shared;
  traits.user_data_dir = user;
  traits.distribution_name = "Rime";
  traits.distribution_code_name = "weasel";
  traits.distribution_version = "0.17.4";
  traits.app_name = "rime.luna_mixed_probe";
  traits.min_log_level = 2;  // ERROR: deployment/engine noise off stderr
  // LUNA_PROBE_LOG=<dir> turns on verbose glog for deployment debugging
  if (const char* ld = getenv("LUNA_PROBE_LOG")) {
    traits.log_dir = ld;
    traits.min_log_level = 0;
  }
  g_api->initialize(&traits);
  if (g_api->start_maintenance) {
    g_api->start_maintenance(True);
    g_api->join_maintenance_thread();
  }
  fprintf(stderr, "[probe] staging=%s prebuilt=%s\n",
          g_api->get_staging_dir ? g_api->get_staging_dir() : "?",
          g_api->get_prebuilt_data_dir ? g_api->get_prebuilt_data_dir() : "?");
  // LUNA_PROBE_SCHEMA=<schema file> deploys that schema synchronously and
  // reports the result; LUNA_PROBE_DEPLOY=1 additionally runs the
  // synchronous workspace_update task. Both exist because the threaded
  // maintenance has been seen to build default.yaml but skip the schema
  // compile in a fresh user directory.
  if (const char* sf = getenv("LUNA_PROBE_SCHEMA")) {
    const Bool ok = g_api->deploy_schema ? g_api->deploy_schema(sf) : False;
    fprintf(stderr, "[probe] deploy_schema=%d\n", (int)ok);
  }
  if (getenv("LUNA_PROBE_DEPLOY") && g_api->run_task) {
    const Bool ok = g_api->run_task("workspace_update");
    fprintf(stderr, "[probe] run_task(workspace_update)=%d\n", (int)ok);
  }

  RimeSessionId sid = g_api->create_session();
  if (!sid) {
    fprintf(stderr, "create_session failed\n");
    return 1;
  }
  Bool sel = g_api->select_schema(sid, schema);
  fprintf(stderr, "[probe] select_schema(%s)=%d\n", schema, (int)sel);
  {
    char cur[256] = {0};
    if (g_api->get_current_schema)
      g_api->get_current_schema(sid, cur, sizeof(cur));
    fprintf(stderr, "[probe] schema=%s shared=%s user=%s\n", cur,
            g_api->get_shared_data_dir ? g_api->get_shared_data_dir() : "?",
            g_api->get_user_data_dir ? g_api->get_user_data_dir() : "?");
  }

  char line[512];
  while (fgets(line, sizeof(line), stdin)) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    g_api->process_key(sid, 0xFF1B /*Escape*/, 0);
    if (n == 0 || line[0] == '#')
      continue;
    const double t0 = now_ms();
    if (g_api->simulate_key_sequence)
      g_api->simulate_key_sequence(sid, line);
    else
      for (size_t i = 0; i < n; i++)
        g_api->process_key(sid, (int)(unsigned char)line[i], 0);
    const double t1 = now_ms();
    if (getenv("LUNA_PROBE_LOG")) {  // context preedit for debugging
      RimeContext rc;
      RIME_STRUCT_INIT(RimeContext, rc);
      if (g_api->get_context(sid, &rc)) {
        fprintf(stderr, "[probe] preedit='%s' len=%d\n",
                rc.composition.preedit ? rc.composition.preedit : "",
                (int)rc.composition.length);
        if (g_api->free_context)
          g_api->free_context(&rc);
      }
    }
    std::string out;
    int count = 0;
    RimeCandidateListIterator it;
    memset(&it, 0, sizeof(it));
    if (g_api->candidate_list_begin(sid, &it)) {
      while (count < max_cands && g_api->candidate_list_next(&it)) {
        if (!it.candidate.text)
          continue;
        if (count)
          out.push_back('\x1f');
        append_clean(out, it.candidate.text);
        out.push_back('\x1e');
        append_clean(out, it.candidate.comment ? it.candidate.comment : "");
        count++;
      }
      g_api->candidate_list_end(&it);
    }
    printf("Q\t%s\t%d\t%.2f\t%s\n", line, count, t1 - t0, out.c_str());
    fflush(stdout);
  }

  g_api->destroy_session(sid);
  g_api->finalize();
  return 0;
}
