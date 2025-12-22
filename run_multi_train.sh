#!/bin/bash
set -euo pipefail


./one.sh -b 20 scenarios/dynamic/drl_train/drl_train_buf10_100k_1.txt
./one.sh -b 20 scenarios/dynamic/drl_train/drl_train_buf10_100k_2.txt
./one.sh -b 20 scenarios/dynamic/drl_train/drl_train_buf10_100k_3.txt
./one.sh -b 20 scenarios/dynamic/drl_train/drl_train_buf10_100k_4.txt