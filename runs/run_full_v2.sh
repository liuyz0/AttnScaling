#!/bin/bash
#SBATCH --job-name=distill-full-v2
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=120
#SBATCH --reservation=rres_liuyz_2026-04-21_k59vzqd4
#SBATCH --qos=rres_qos_liuyz_2026-04-21_k59vzqd4
#SBATCH --account=rres_acc_liuyz_2026-04-21_k59vzqd4
#SBATCH --time=36:00:00

module load miniforge
source activate distill

mkdir -p logs

BATCH_SIZE=32   # per GPU; 32 × 1 accum × 8 GPUs = 256 sequences = 0.5M tokens
GRAD_ACCUM=1

echo "=== Run 10: KL diff WD ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type kl \
    --teacher_temperature 0.1 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 6e-4 \
    --warmup_steps 1430 \
    --max_steps 79000 \
    --log_interval 100 \
    --lr_decay_steps 143000 \
    --lm_head_wd 0.01 \
    --out_dir checkpoints/kl_01_v2 \
    --dtype bfloat16 \
    --run_name kl01v2 \
    --use_wandb

echo "=== Run 11: KL diff WD ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type kl \
    --teacher_temperature 0.1 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 6e-4 \
    --warmup_steps 1430 \
    --max_steps 79000 \
    --log_interval 100 \
    --lr_decay_steps 143000 \
    --weight_decay 0.01 \
    --out_dir checkpoints/kl_01_v3 \
    --dtype bfloat16 \
    --run_name kl01v3 \
    --use_wandb

echo "=== Run 12: MSE diff WD ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type mse \
    --teacher_temperature 0.1 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 6e-4 \
    --warmup_steps 1430 \
    --max_steps 79000 \
    --log_interval 100 \
    --lr_decay_steps 143000 \
    --lm_head_wd 0.01 \
    --out_dir checkpoints/mse_01_v1 \
    --dtype bfloat16 \
    --run_name mse01v1 \
    --use_wandb

echo "=== Run 13: MSE diff WD ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type mse \
    --teacher_temperature 0.1 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 6e-4 \
    --warmup_steps 1430 \
    --max_steps 79000 \
    --log_interval 100 \
    --lr_decay_steps 143000 \
    --weight_decay 0.01 \
    --out_dir checkpoints/mse_01_v2 \
    --dtype bfloat16 \
    --run_name mse01v2 \
    --use_wandb

echo "=== All done ==="