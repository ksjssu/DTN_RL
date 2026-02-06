@echo off
setlocal enabledelayedexpansion

set "SCEN_DIR=C:\dtn\DTN_RL\scenarios\dynamic\low_high_combi_delta010"
set "ONE_BAT=%~dp0one.bat"

if not exist "%SCEN_DIR%" (
  echo Scenario directory not found: %SCEN_DIR%
  exit /b 1
)

if not exist "%ONE_BAT%" (
  echo one.bat not found at: %ONE_BAT%
  exit /b 1
)

set "JAVA_EXE="
if defined JAVA_HOME if exist "%JAVA_HOME%\\bin\\java.exe" set "JAVA_EXE=%JAVA_HOME%\\bin\\java.exe"
if not defined JAVA_EXE for %%P in (
  "C:\\Program Files\\Java\\*\\bin\\java.exe"
  "C:\\Program Files\\Eclipse Adoptium\\*\\bin\\java.exe"
  "C:\\Program Files\\AdoptOpenJDK\\*\\bin\\java.exe"
  "C:\\Program Files\\Zulu\\*\\bin\\java.exe"
  "C:\\Program Files\\Amazon Corretto\\*\\bin\\java.exe"
  "C:\\Program Files\\BellSoft\\*\\bin\\java.exe"
  "C:\\Java\\*\\bin\\java.exe"
) do (
  if exist "%%~fP" (
    set "JAVA_EXE=%%~fP"
    goto :java_found
  )
)

:java_found
if not defined JAVA_EXE (
  echo Java not found. Install a JDK or set JAVA_HOME.
  exit /b 1
)
for %%D in ("%JAVA_EXE%") do set "JAVA_BIN=%%~dpD"
set "PATH=%JAVA_BIN%;%PATH%"

pushd "%SCEN_DIR%"
for %%F in (*.txt) do (
  echo Running %%F...
  call "%ONE_BAT%" -b 1 "%%F"
  if errorlevel 1 (
    echo Failed on %%F
    popd
    exit /b 1
  )
)
popd

echo All scenarios completed.
