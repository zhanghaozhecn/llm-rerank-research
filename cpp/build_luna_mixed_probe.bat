@echo off
rem build_luna_mixed_probe.bat - build the luna mixed-window cohort probe
rem (headless engine candidate dump with spans; links nothing but kernel32)
rem usage: build_luna_mixed_probe.bat  (output: luna_mixed_probe.exe here)
rem prereq: sibling checkout ..\..\rime-llm-ime (for rime_api.h)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
cl /nologo /O2 /EHsc /utf-8 /I "%~dp0..\..\rime-llm-ime\librime\src" luna_mixed_probe.cpp /Fe:luna_mixed_probe.exe
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
del /q luna_mixed_probe.obj >nul 2>&1
echo BUILD OK
exit /b 0
