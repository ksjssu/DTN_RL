@echo off
setlocal EnableExtensions

rem Run all non-DRL scenarios in scenarios\dynamic with -b 100
rem Explicit commands as requested (no loops)

echo === Prophet ===
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf05_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf10_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf15_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf20_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf25_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf30_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf35_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf40_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf45_100k.txt
call one.bat -b 100 scenarios\dynamic\prophet\prophet_buf50_100k.txt

echo === Heuristic d010 ===
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf05_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf10_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf15_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf20_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf25_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf30_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf35_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf40_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf45_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d010\heuristic_d010_buf50_100k.txt

echo === Heuristic d020 ===
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf05_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf10_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf15_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf20_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf25_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf30_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf35_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf40_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf45_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d020\heuristic_d020_buf50_100k.txt

echo === Heuristic d030 ===
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf05_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf10_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf15_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf20_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf25_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf30_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf35_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf40_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf45_100k.txt
call one.bat -b 100 scenarios\dynamic\heuristic_d030\heuristic_d030_buf50_100k.txt

echo Done.
endlocal

