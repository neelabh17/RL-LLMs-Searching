#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=3
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --output="/home/mila/b/buvanesa/code/nano-vppo/logs/out_%j.out"
#SBATCH --error="/home/mila/b/buvanesa/code/nano-vppo/logs/err_%j.err"
#SBATCH --partition=long
export WANDB_PROJECT="sft"
export WANDB_ENTITY="anirudhb11"

module load anaconda/3
module load cuda/12.4.0/cudnn/8.9

conda activate vppo


DATASET_NAME="deg-5-path-5-nodes-300-qwen-14bi-final-all"
mkdir -p /network/scratch/b/buvanesa/mini-rl-project/sft-qwen-1.5bi-${DATASET_NAME}

python train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name anirudhb11/${DATASET_NAME} \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --eval_strategy no \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir /network/scratch/b/buvanesa/mini-rl-project/sft-qwen-1.5bi-${DATASET_NAME} \
    --run_name sft-qwen-1.5bi-${DATASET_NAME} \
    --push_to_hub

DATASET_NAME="deg-5-path-5-nodes-300-qwen-14bi-final-corr"
mkdir -p /network/scratch/b/buvanesa/mini-rl-project/sft-qwen-1.5bi-${DATASET_NAME}

python train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name anirudhb11/${DATASET_NAME} \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --eval_strategy no \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir /network/scratch/b/buvanesa/mini-rl-project/sft-qwen-1.5bi-${DATASET_NAME} \
    --run_name sft-qwen-1.5bi-${DATASET_NAME} \
    --push_to_hub
