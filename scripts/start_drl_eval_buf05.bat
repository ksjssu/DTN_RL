@echo off
setlocal enableextensions
rem DRL evaluation server for buf05 using the latest trained checkpoint
set EVAL_ONLY=1
set DRL_POLICY=ppo
set DRL_PORT=5005
set MODEL_PATH=models\buf05\dtn_sim_t97200_ppo.pt

rem Optional: align reward weights with training
set W_DELIVER=20.0
set W_RELAY=10.0
set W_DROP=2.0
set W_ABORT=0.5
set W_DELAY=0.0

echo Starting DRL eval server on port %DRL_PORT% with model %MODEL_PATH%
python toolkit\drl_server.py
endlocal

