@echo off
setlocal

echo === DRL Train: buf10 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf10_100k.txt

echo === DRL Train: buf15 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf15_100k.txt

echo === DRL Train: buf20 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf20_100k.txt

echo === DRL Train: buf25 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf25_100k.txt

echo === DRL Train: buf05 ===
call .\one.bat -b 50 scenarios/dynamic/drl_train/drl_train_buf05_100k.txt

endlocal
