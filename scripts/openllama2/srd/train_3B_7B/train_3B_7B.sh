#!/bin/bash

BASE_PATH=${1-"/home/srd"}

bash ${BASE_PATH}/scripts/openllama2/srd/train_3B_7B/t0_rkl.sh
bash ${BASE_PATH}/scripts/openllama2/srd/train_3B_7B/t1_rkl.sh
bash ${BASE_PATH}/scripts/openllama2/srd/train_3B_7B/t2_rkl.sh
