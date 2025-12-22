@echo off
REM Run heuristic_d020 scenarios for buffer sizes 60–100 MB (100 runs each)
call one.bat -b 1:100 scenarios/dynamic/heuristic_d020/heuristic_d020_buf60_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d020/heuristic_d020_buf70_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d020/heuristic_d020_buf80_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d020/heuristic_d020_buf90_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d020/heuristic_d020_buf100_100k.txt
