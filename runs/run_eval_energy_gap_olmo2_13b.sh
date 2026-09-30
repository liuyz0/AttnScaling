#!/bin/bash
#SBATCH --job-name=distill-energy-gap-olmo2-13b
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_energy_gap_olmo2_13b_%j.out
#SBATCH --error=logs/eval_energy_gap_olmo2_13b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liuyz@mit.edu

module load miniforge
source activate olmo2
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

mkdir -p results/energy_gap/olmo2_13b

echo "=== OLMo-2-13B: energy gap histograms across training checkpoints ==="

python olmo/eval_energy_gap.py \
    --model allenai/OLMo-2-1124-13B \
    --revisions \
        stage1-step1100-tokens10B \
        stage1-step5000-tokens42B \
        stage1-step7000-tokens59B \
        stage1-step12000-tokens101B \
        stage1-step19850-tokens167B \
        stage1-step31000-tokens261B \
        stage1-step48000-tokens403B \
        stage1-step72000-tokens604B \
        stage1-step107500-tokens902B \
        stage1-step167000-tokens1401B \
        stage1-step239000-tokens2005B \
        stage1-step358000-tokens3004B \
        stage1-step596057-tokens5001B \
    --n_eval_tokens 2000000 \
    --context_length 2048 \
    --batch_size 4 \
    --gap_max 5.0 \
    --n_bins 1000 \
    --out_dir results/energy_gap/olmo2_13b

echo "Done. Results at results/energy_gap/olmo2_13b/"
