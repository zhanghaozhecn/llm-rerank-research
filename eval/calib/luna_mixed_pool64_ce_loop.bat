@echo off
rem luna mixed-window CE pool 64 backfill loop (research, single process).
rem Repeats `luna_mixed_pool64.py ce --batch N` until the gap step reports 0.
rem usage: luna_mixed_pool64_ce_loop.bat [batch] [workers]
setlocal
set /a BATCH=%1
if "%BATCH%"=="%1" set BATCH=2000
set /a WORKERS=%2
if "%WORKERS%"=="%2" set WORKERS=1
cd /d "%~dp0..\.."
:loop
python -I -X utf8 eval\calib\luna_mixed_pool64.py ce --batch %BATCH% --workers %WORKERS%
if errorlevel 1 goto fail
python -I -X utf8 eval\calib\luna_mixed_pool64.py gap
if errorlevel 2 goto fail
if not errorlevel 1 goto done
goto loop
:done
echo POOL64_CE_ALL_DONE
exit /b 0
:fail
echo POOL64_CE_FAILED
exit /b 1
