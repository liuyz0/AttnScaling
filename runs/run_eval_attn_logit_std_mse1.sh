#!/bin/bash
#SBATCH --job-name=distill-attn-std-mse1
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_attn_logit_std_mse1_%j.out
#SBATCH --error=logs/eval_attn_logit_std_mse1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=2:00:00

module load miniforge
source activate distill

mkdir -p results

CKPT=checkpoints/mse_1
OUT=results/eval_attn_logit_std_mse1.jsonl

echo "=== mse_1: per-head attention logit std across training steps ==="

FIRST=true
for step_dir in $(ls -d ${CKPT}/step_* 2>/dev/null | sort -V); do
    APPEND_FLAG=""
    [ "$FIRST" = false ] && APPEND_FLAG="--append"
    FIRST=false

    echo "  Processing ${step_dir}"
    python eval_attn_logit_std.py \
        --model ${step_dir} \
        --revisions none \
        --data_path data/pythia_deduped_pile_idxmaps/ \
        --clean_eval \
        --n_eval_tokens 2000000 \
        --batch_size 16 \
        --out_file ${OUT} \
        ${APPEND_FLAG}
done

echo "Done. Results at ${OUT}"
