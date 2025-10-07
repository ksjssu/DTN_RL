@echo off
setlocal enableextensions
rem Configure DRL reward weights
set W_RELAY=1
rem PPO stability: large batch, few epochs (early update gating)
set PPO_BATCH_SIZE=2048
set PPO_EPOCHS=2
rem Start DRL PPO server in a new window
start "DRL Server" cmd /c python toolkit\drl_server.py
rem Give the server a moment to start
timeout /t 2 >nul
rem Run The ONE simulation (batch mode 1 run)
call one.bat -b 1 dynamic_traffic_hml_1h_settings.txt
endlocal
