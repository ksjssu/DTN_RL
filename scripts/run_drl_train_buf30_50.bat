@echo off
setlocal

echo === DRL Train: buf40 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf40_100k.txt

echo === DRL Train: buf30 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf30_100k.txt

echo === DRL Train: buf35 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf35_100k.txt

echo === DRL Train: buf45 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf45_100k.txt

echo === DRL Train: buf50 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf50_100k.txt

endlocal
