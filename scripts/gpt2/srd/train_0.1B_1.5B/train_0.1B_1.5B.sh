#!/bin/bash

BASE_PATH=${1-"/hy-tmp/srd"}

bash ${BASE_PATH}/scripts/gpt2/srd/train_0.1B_1.5B/t0.sh
bash ${BASE_PATH}/scripts/gpt2/srd/train_0.1B_1.5B/t1.sh
bash ${BASE_PATH}/scripts/gpt2/srd/train_0.1B_1.5B/t2.sh
