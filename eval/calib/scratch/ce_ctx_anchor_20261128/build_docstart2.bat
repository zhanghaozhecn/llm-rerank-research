@echo off
REM build_patched.bat - ce_probe against CURRENT patched mirror (build-mt)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
set INC1=D:/llama.cpp-mirror/include
set INC2=D:/llama.cpp-mirror/ggml/include
set LIB1=D:/llama.cpp-mirror/build-mt/src/Release/llama.lib
set LIB2=D:/llama.cpp-mirror/build-mt/ggml/src/Release/ggml.lib
set LIB3=D:/llama.cpp-mirror/build-mt/ggml/src/Release/ggml-base.lib
set LIB4=D:/llama.cpp-mirror/build-mt/ggml/src/Release/ggml-cpu.lib
cl /O2 /std:c++17 /EHsc /DNDEBUG /MT /utf-8 ce_docstart2.cpp /I %INC1% /I %INC2% %LIB1% %LIB2% %LIB3% %LIB4% /Fe:ce_docstart2.exe /link advapi32.lib
exit /b %errorlevel%
