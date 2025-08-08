#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/srd"}
CKPT_NAME="open_llama_3b_v2_4"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
TEACHER_CKPT_NAME="open_llama_7b_v2"
TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
TEACHER_PEFT_CKPT_NAME="sft_7B/e20-bs4-lr0.0005-G2-N4-NN1-lora-16-64-0.1/2856"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/openllama2/train/sft/${TEACHER_PEFT_CKPT_NAME}/"
MP_SIZE=4

# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/openllama2/train0/"

# hp
BATCH_SIZE=2
LR=0.0001
GRAD_ACC=4
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/openllama2/train/pocl_rkl/t0"
# seed
SEED=10



OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 4"
OPTS+=" --kd-ratio 0.7"
OPTS+=" --stage 0"
OPTS+=" --num-stages 3"


# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval 89"
OPTS+=" --eval-interval 89"
OPTS+=" --log-interval 32"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"
# type
OPTS+=" --type tfkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# distillm
#OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
OPTS+=" --skew-alpha 0.1"
OPTS+=" --mixed-alpha 0.8"
OPTS+=" --temperature-kld 1"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
