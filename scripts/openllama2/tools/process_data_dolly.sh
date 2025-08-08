BASE_PATH=${1-"/home/srd"}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/gsm8k/main \
    --processed-data-dir ${BASE_PATH}/processed_data/gsm8k/prompt \
    --model-path ${BASE_PATH}/checkpoints/ckptsQwen2.5-Math-1.5B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type qwen2.5

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${BASE_PATH}/checkpoints/open_llama_3b_v2_4 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type openllama2
