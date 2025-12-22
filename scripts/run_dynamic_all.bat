@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run all scenarios under scenarios\dynamic except drl_eval and drl_train
rem Each scenario is executed with -b 100 (rngSeed 1..100)

set "BASE=scenarios\dynamic"
if not exist "%BASE%" (
  echo [ERROR] Folder not found: %BASE%
  exit /b 1
)

set "FAIL_LOG=scripts\run_dynamic_all_failed.txt"
del /q "%FAIL_LOG%" 2>nul

set /a COUNT=0
for /R "%BASE%" %%F in (*.txt) do (
  set "FILE=%%F"
  set "SKIP=0"
  rem Exclude any DRL scenarios/directories
  echo !FILE! | findstr /I /C:"\drl_eval\" >nul && set "SKIP=1"
  echo !FILE! | findstr /I /C:"\drl_train\" >nul && set "SKIP=1"
  echo !FILE! | findstr /I /C:"\drl\" >nul && set "SKIP=1"
  if "!SKIP!"=="0" (
    set /a COUNT+=1
    echo [!COUNT!] Running: "%%F"
    call one.bat -b 100 "%%F"
    if errorlevel 1 (
      echo FAILED: %%F >> "%FAIL_LOG%"
      echo   -> Recorded failure to %FAIL_LOG%
    )
  ) else (
    echo Skipping: "%%F"
  )
)

echo.
echo Completed. Total executed: %COUNT%
if exist "%FAIL_LOG%" (
  echo Check failures listed in %FAIL_LOG%
)

endlocal
