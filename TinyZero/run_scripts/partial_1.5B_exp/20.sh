#!/bin/bash
#SBATCH --job-name=tinyzero3b
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_dev,a100_short,a100_long

conda activate
conda activate verl

pwd

export WANDB_MODE=online

DATA_DIR=/gpfs/data/ranganathlab/Jatin/verl_neel/data/neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_20
ROLLOUT_TP_SIZE=1
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
# BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
# BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
# BASE_MODEL=Qwen/Qwen2.5-14B-Instruct

# BASE_MODEL=anirudhb11/sft-qwen-1.5bi-deg-5-path-5-nodes-300-qwen-14bi-final2-all
# BASE_MODEL=anirudhb11/sft-qwen-1.5bi-deg-5-path-5-nodes-300-qwen-14bi-final2-corr

EXPERIMENT_NAME=1.5B_partial_corr_20
N_GPUS=1 # use 4xA100

CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main_ppo \
reward_model.reward_manager=partial \
\
data.truncation=right \
\
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=64 \
data.val_batch_size=128 \
data.max_prompt_length=2000 \
data.max_response_length=1200 \
\
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
\
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3300 \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.kl_loss_coef=1e-3 \
\
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
\
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
\
critic.optim.lr=1e-5 \
\
algorithm.kl_ctrl.kl_coef=0.01 \
algorithm.adv_estimator=grpo \
\
trainer.logger=['console','wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=50 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee verl_demo.log

# actor_rollout_ref.actor.ppo_micro_batch_size=2 \ # this will be deprecated: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html
# actor_rollout_ref.actor.use_dynamic_bsz=True \ # we don't need to set batch sizes if we have this enabled: actor_rollout_ref.actor.ppo_max_token_len_per_gpu needs to be set: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#tuning-for-dynamic-batch-size