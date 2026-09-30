#!/bin/bash
#SBATCH --job-name=distill-bos-norms
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_bos_norms_%j.out
#SBATCH --error=logs/eval_bos_norms_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=1:00:00

module load miniforge
source activate distill

mkdir -p results

echo "=== Pythia-160m: BOS norms across training steps ==="

python eval_bos_norms.py \
    --model EleutherAI/pythia-160m \
    --revisions step128 step256 step512 step1000 step2000 step3000 \
               step6000 step12000 step16000 step32000 step48000 step79000 \
    --out_file results/eval_bos_norms.jsonl

echo "Done. Results at results/eval_bos_norms.jsonl"
