#!/bin/bash
export N_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray stop --force && ray start --head --include-dashboard=False
export BASE_MODEL="/path/to/basemodel"

export TRAIN_DATA="data/train.parquet"
export VAL_DATA="data/dev.parquet"
export TRAIN_BATCH_SIZE=4
export VAL_BATCH_SIZE=24
export MAX_PROMPT_LENGTH=28000
export MAX_RESPONSE_LENGTH=512
export ACTOR_LR=1e-5
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE=8
export LOG_PROB_MICRO_BATCH_SIZE=8
export KL_LOSS_COEF=0.001
export KL_CTRL_COEF=0.001
export ROLLOUT_TP_SIZE=8
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.8
export ROLLOUT_N_AGENT=8
export ROLLOUT_TEMPERATURE=1
export PROJECT_NAME="train"
export EXPERIMENT_NAME="train"
export SAVE_FREQ=10
export TEST_FREQ=10
export TOTAL_EPOCHS=8
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE='offline'
export LOG_FILE=train.log

bash ./examples/grpo_trainer/train.sh
