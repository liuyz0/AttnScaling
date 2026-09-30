#!/bin/bash
#SBATCH --job-name=distill-ddp-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --reservation=rres_liuyz_2026-04-21_k59vzqd4
#SBATCH --qos=rres_qos_liuyz_2026-04-21_k59vzqd4
#SBATCH --account=rres_acc_liuyz_2026-04-21_k59vzqd4

module load miniforge
source activate distill

mkdir -p logs

echo "=== batch size 32 ==="

BATCH_SIZE=32   # per GPU; 64 × 2 accum × 8 GPUs = 1024 sequences = 2M tokens
GRAD_ACCUM=4

# ---- Smoke test 1: CE loss, random data, single GPU ----
echo "=== Smoke test 1: CE ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --loss_type ce \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ddp_ce \
    --dtype bfloat16 \
    --run_name test_ddp_batch \
    --use_wandb

# ---- Smoke test 2: KL distillation ----
echo "=== Smoke test 2: KL ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type kl \
    --teacher_temperature 1.0 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ddp_kl \
    --dtype bfloat16

# ---- Smoke test 3: MSE distillation ----
echo "=== Smoke test 3: MSE ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type mse \
    --teacher_temperature 1.0 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ddp_mse \
    --dtype bfloat16

echo "=== batch size 64 ==="

BATCH_SIZE=64   # per GPU; 64 × 2 accum × 8 GPUs = 1024 sequences = 2M tokens
GRAD_ACCUM=2

# ---- Smoke test 1: CE loss, random data, single GPU ----
echo "=== Smoke test 1: CE ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --loss_type ce \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ddp_ce \
    --dtype bfloat16 \
    --run_name test_ddp_batch \
    --use_wandb

# ---- Smoke test 2: KL distillation ----
echo "=== Smoke test 2: KL ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type kl \
    --teacher_temperature 1.0 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ddp_kl \
    --dtype bfloat16

# ---- Smoke test 3: MSE distillation ----
echo "=== Smoke test 3: MSE ==="
torchrun --nproc_per_node=8 train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type mse \
    --teacher_temperature 1.0 \
    --data_source local \
    --local_data_path ./data/pythia_deduped_pile_idxmaps/ \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ddp_mse \
    --dtype bfloat16

echo "=== All DDP smoke tests done ==="
