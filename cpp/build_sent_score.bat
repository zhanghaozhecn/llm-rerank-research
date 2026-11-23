@echo off
REM build_sent_score.bat - build sent_score.exe (sentence-level LLM likelihood)
REM usage: build_sent_score.bat  (output: sent_score.exe in this directory)
REM prereq: llama.cpp MT static build (LLAMA_ROOT, default D:/llama.cpp-mirror)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
if "%LLAMA_ROOT%"=="" set LLAMA_ROOT=D:/llama.cpp-mirror
cl /O2 /std:c++17 /EHsc /DNDEBUG /MT /utf-8 sent_score.cpp /I %LLAMA_ROOT%\include /I %LLAMA_ROOT%\ggml\include %LLAMA_ROOT%\build-mt\src\Release\llama.lib %LLAMA_ROOT%\build-mt\ggml\src\Release\ggml.lib %LLAMA_ROOT%\build-mt\ggml\src\Release\ggml-base.lib %LLAMA_ROOT%\build-mt\ggml\src\Release\ggml-cpu.lib /Fe:sent_score.exe /link /LTCG advapi32.lib
exit /b %errorlevel%
