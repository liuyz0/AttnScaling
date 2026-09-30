#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH -o logs/exp-1.log-%A-%a
#SBATCH -a 0-63
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu

# Load Anaconda Module
module load miniforge/24.3.0-0

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

python -u toy/exp/exp-1.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT