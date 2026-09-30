#!/bin/bash
#SBATCH --job-name=distill-attn-std-olmo2-13b-resume
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_attn_logit_std_olmo2_13b_resume_%j.out
#SBATCH --error=logs/eval_attn_logit_std_olmo2_13b_resume_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liuyz@mit.edu

module load miniforge
source activate olmo2
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

mkdir -p results

echo "=== OLMo-2-13B: attn logit std — resuming from step19850 (appending to existing jsonl) ==="

# Resumes after OOM at step19850 in the original run.
# step1100 / step5000 / step7000 / step12000 are already in the jsonl.
python olmo/eval_attn_logit_std.py \
    --model allenai/OLMo-2-1124-13B \
    --revisions \
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
    --append \
    --out_file results/olmo2_13b_attn_logit_std.jsonl

echo "Done. Results at results/olmo2_13b_attn_logit_std.jsonl"
