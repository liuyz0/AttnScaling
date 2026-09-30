#!/bin/bash
#SBATCH --job-name=distill-attn-std-olmo2-1b
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_attn_logit_std_olmo2_1b_%j.out
#SBATCH --error=logs/eval_attn_logit_std_olmo2_1b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liuyz@mit.edu

module load miniforge
source activate olmo2
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

mkdir -p results

echo "=== OLMo-2-1B: per-head attention logit std across training checkpoints ==="

python olmo/eval_attn_logit_std.py \
    --model allenai/OLMo-2-0425-1B \
    --revisions \
        stage1-step300-tokens1B \
        stage1-step10000-tokens21B \
        stage1-step23100-tokens49B \
        stage1-step40000-tokens84B \
        stage1-step60000-tokens126B \
        stage1-step90000-tokens189B \
        stage1-step140000-tokens294B \
        stage1-step210000-tokens441B \
        stage1-step330000-tokens693B \
        stage1-step510000-tokens1070B \
        stage1-step790000-tokens1657B \
        stage1-step1230000-tokens2580B \
        stage1-step1907359-tokens4001B \
    --n_eval_tokens 2000000 \
    --context_length 2048 \
    --batch_size 16 \
    --out_file results/olmo2_1b_attn_logit_std.jsonl

echo "Done. Results at results/olmo2_1b_attn_logit_std.jsonl"
