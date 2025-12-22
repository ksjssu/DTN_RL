@echo off
REM Batch-run DRL training scenarios (buf60–buf100) with 50 runs each.
REM Assumes DRL server is already running with matching ports.

call one.bat -b 1:50 scenarios/dynamic/drl_eval/drl_eval_buf60_100k.txt
call one.bat -b 1:50 scenarios/dynamic/drl_eval/drl_eval_buf70_100k.txt
call one.bat -b 1:50 scenarios/dynamic/drl_eval/drl_eval_buf80_100k.txt
call one.bat -b 1:50 scenarios/dynamic/drl_eval/drl_eval_buf90_100k.txt

