#!/bin/bash
#SBATCH --job-name=distill-energy-gap
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_energy_gap_%j.out
#SBATCH --error=logs/eval_energy_gap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -p mit_normal_gpu
#SBATCH --time=4:00:00

module load miniforge
source activate distill

mkdir -p results/energy_gap/410m

echo "=== Pythia-410m: energy gap histograms (final checkpoint) ==="

python eval_energy_gap.py \
    --model EleutherAI/pythia-410m \
    --revisions none \
    --data_path data/pythia_deduped_pile_idxmaps/ \
    --clean_eval \
    --train_steps 79000 \
    --n_eval_tokens 2000000 \
    --batch_size 16 \
    --gap_max 5.0 \
    --n_bins 1000 \
    --out_dir results/energy_gap/410m

echo "Done. Results at results/energy_gap/410m/"
