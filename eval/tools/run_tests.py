#!/usr/bin/env python3
"""
LLM 重排自动化测试

用法:
  python eval/run_tests.py               运行全部测试（CPU 后端）
  python eval/run_tests.py --baseline    生成/更新基线
  python eval/run_tests.py --samples 500 指定样本数（默认 200）
  python eval/run_tests.py --quick       快速模式（仅 CE 正确性，跳过回归）

测试内容:
  1. CE 正确性 — 分层 vs gold，固定 6 个用例
  2. 准确率回归 — wiki 语料采样（RIME_LLM_WIKI 可指定语料），对比基线
  3. 延迟回归 — p50 对比基线
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # 研究库根
CPP_DIR = PROJECT_ROOT / "cpp"                          # 评测器源码（2026-09-02 自插件版迁入）
BUILD_DIR = CPP_DIR / "build_sim"
TEST_EXE = BUILD_DIR / "Release/test_core.exe"
BASELINE_FILE = BUILD_DIR / "test_baseline.json"


def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"


def build():
    """编译 test_core.exe（本库 cpp\\build_sim，MSVC Release）"""
    print(f"\n{yellow('[build]')} Compiling test_core.exe...")

    r = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "test_core"],
        cwd=str(CPP_DIR), capture_output=True, text=True, errors='replace'
    )
    if r.returncode != 0:
        print(red(f"  Build failed:\n{r.stderr[-500:]}"))
        return False
    if not TEST_EXE.exists():
        print(red(f"  Build dir not configured: {BUILD_DIR}"))
        print(red(f"  First configure: cmake -B cpp/build_sim -S cpp -G \"Visual Studio 17 2022\" -A x64"))
        return False

    print(green("  Build OK"))
    return True


def run_test(args: list) -> tuple[int, str]:
    """运行 test_core.exe，返回 (exit_code, output)"""
    r = subprocess.run(
        [str(TEST_EXE)] + args,
        cwd=str(BUILD_DIR),
        capture_output=True, text=True, errors='replace',
        timeout=600  # 10 分钟超时
    )
    return r.returncode, r.stdout + r.stderr


def main():
    quick = "--quick" in sys.argv
    baseline = "--baseline" in sys.argv
    samples = 200
    for i, a in enumerate(sys.argv):
        if a == "--samples" and i + 1 < len(sys.argv):
            samples = int(sys.argv[i + 1])

    print(f"{'='*50}")
    print(f"  LLM Rerank Automated Tests")
    print(f"  Samples: {samples}" + (" (quick mode)" if quick else ""))
    print(f"{'='*50}")

    # Step 1: Build
    if not build():
        sys.exit(3)

    # Step 2: Generate baseline (if requested)
    if baseline:
        print(f"\n{yellow('[baseline]')} Generating baseline with {samples} samples...")
        code, out = run_test(["--baseline", "--samples", str(samples)])
        print(out)
        if code == 0:
            print(green("\n  Baseline created: " + str(BASELINE_FILE)))
        else:
            print(red("\n  Baseline generation failed"))
        sys.exit(code)

    # Step 3: Run tests
    if quick:
        # CE correctness only — skip accuracy/latency regression
        # Run with 0 samples to skip wiki tests
        code, out = run_test(["--samples", "0"])
        print(out)
        # Filter to show only CE test results
        for line in out.splitlines():
            if "PASS:" in line or "FAIL:" in line:
                print(f"  {line.strip()}")
        sys.exit(code)
    else:
        code, out = run_test(["--samples", str(samples)])
        print(out)

    # Step 4: Report
    print(f"\n{'='*50}")
    if code == 0:
        print(green("  ALL TESTS PASSED"))
    else:
        print(red(f"  TESTS FAILED (exit code {code})"))
    print(f"{'='*50}")
    sys.exit(code)


if __name__ == "__main__":
    main()
