set -x

# hdfs_path=hdfs://user/verl/experiments/gsm8k/deepseek-coder-6.7b-instruct/ # replace to your own hdfs/local path

nproc_per_node=$2
export CUDA_VISIBLIE_DEVICES=2,3
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/gemini-1/space/data01/r1-tool/gsm8k/train-00000-of-00001.parquet \
    data.val_files=/gemini-1/space/data01/r1-tool/gsm8k/test-00000-of-00001.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.micro_batch_size=8 \
    model.partial_pretrain=/gemini-1/space/data01/r1-tool/qwen_3b \
    trainer.default_hdfs_dir=None\
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen3b \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']