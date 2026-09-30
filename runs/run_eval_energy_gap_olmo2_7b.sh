#!/bin/bash
#SBATCH --job-name=distill-energy-gap-olmo2-7b
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_energy_gap_olmo2_7b_%j.out
#SBATCH --error=logs/eval_energy_gap_olmo2_7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liuyz@mit.edu

module load miniforge
source activate olmo2
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

mkdir -p results/energy_gap/olmo2_7b

echo "=== OLMo-2-7B: energy gap histogram (final checkpoint) ==="

python olmo/eval_energy_gap.py \
    --model allenai/OLMo-2-1124-7B \
    --revisions \
        stage1-step928646-tokens3896B \
    --n_eval_tokens 2000000 \
    --context_length 2048 \
    --batch_size 16 \
    --gap_max 5.0 \
    --n_bins 1000 \
    --out_dir results/energy_gap/olmo2_7b

echo "Done. Results at results/energy_gap/olmo2_7b/"
