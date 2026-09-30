#!/bin/bash
#SBATCH --job-name=distill-attn-std-olmo2-7b
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_attn_logit_std_olmo2_7b_%j.out
#SBATCH --error=logs/eval_attn_logit_std_olmo2_7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liuyz@mit.edu

module load miniforge
source activate olmo2
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

mkdir -p results

echo "=== OLMo-2-7B: per-head attention logit std across training checkpoints ==="

python olmo/eval_attn_logit_std.py \
    --model allenai/OLMo-2-1124-7B \
    --revisions \
        stage1-step150-tokens1B \
        stage1-step5000-tokens21B \
        stage1-step11000-tokens47B \
        stage1-step20000-tokens84B \
        stage1-step30000-tokens126B \
        stage1-step45000-tokens189B \
        stage1-step70000-tokens294B \
        stage1-step105000-tokens441B \
        stage1-step165000-tokens693B \
        stage1-step255000-tokens1070B \
        stage1-step395000-tokens1657B \
        stage1-step615000-tokens2580B \
        stage1-step928646-tokens3896B \
    --n_eval_tokens 200000 \
    --context_length 2048 \
    --batch_size 16 \
    --out_file results/olmo2_7b_attn_logit_std.jsonl

echo "Done. Results at results/olmo2_7b_attn_logit_std.jsonl"
