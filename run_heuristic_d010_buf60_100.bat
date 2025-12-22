@echo off
REM Run heuristic_d010 scenarios for buffer sizes 60–100 MB (100 runs each)
call one.bat -b 1:100 scenarios/dynamic/heuristic_d010/heuristic_d010_buf60_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d010/heuristic_d010_buf70_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d010/heuristic_d010_buf80_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d010/heuristic_d010_buf90_100k.txt
call one.bat -b 1:100 scenarios/dynamic/heuristic_d010/heuristic_d010_buf100_100k.txt
