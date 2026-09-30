#!/bin/bash
#SBATCH --job-name=distill-attn-std-12b
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_attn_logit_std_12b_%j.out
#SBATCH --error=logs/eval_attn_logit_std_12b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=8:00:00

module load miniforge
source activate distill

mkdir -p results

echo "=== Pythia-12b: per-head attention logit std across training steps ==="

python eval_attn_logit_std.py \
    --model EleutherAI/pythia-12b \
    --revisions step128 step256 step512 step1000 step2000 step3000 \
               step6000 step12000 step16000 step32000 step48000 step79000 step143000 \
    --data_path data/pythia_deduped_pile_idxmaps/ \
    --clean_eval \
    --n_eval_tokens 2000000 \
    --batch_size 16 \
    --out_file results/eval_attn_logit_std_12b.jsonl

echo "Done. Results at results/eval_attn_logit_std_12b.jsonl"
