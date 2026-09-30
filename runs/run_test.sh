#!/bin/bash
#SBATCH --job-name=distill-test
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

module load miniforge
source activate distill

mkdir -p logs

# ---- Smoke test 1: CE loss, random data, single GPU ----
echo "=== Smoke test 1: CE ==="
python train.py \
    --student_model EleutherAI/pythia-160m \
    --loss_type ce \
    --data_source random \
    --batch_size 2 \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_ce \
    --dtype bfloat16

# ---- Smoke test 2: KL distillation ----
echo "=== Smoke test 2: KL ==="
python train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type kl \
    --teacher_temperature 1.0 \
    --data_source random \
    --batch_size 2 \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_kl \
    --dtype bfloat16

# ---- Smoke test 3: MSE distillation ----
echo "=== Smoke test 3: MSE ==="
python train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type mse \
    --teacher_temperature 1.0 \
    --data_source random \
    --batch_size 2 \
    --max_steps 50 \
    --log_interval 10 \
    --save_interval 50 \
    --out_dir checkpoints/test_mse \
    --dtype bfloat16

echo "=== All smoke tests done ==="
