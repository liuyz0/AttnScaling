#!/bin/bash
#SBATCH --job-name=distill-stats
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_stats_%j.out
#SBATCH --error=logs/eval_stats_%j.err
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

CKPT=checkpoints
OUT=results/eval_stats.jsonl
DATA=data/pythia_deduped_pile_idxmaps/

mkdir -p results

COMMON_ARGS=(
    --data_path "${DATA}"
    --clean_eval
    --train_steps 79000
    --train_world_size 8
    --train_batch_size 32
    --n_eval_tokens 5000000
    --batch_size 32
    --dtype bfloat16
    --out_file "${OUT}"
)

# --- Teacher: 4 temperatures in one forward pass (overwrites output file) ---
echo "=== Teacher: EleutherAI/pythia-160m (step79000) ==="
python eval_stats.py \
    --model EleutherAI/pythia-160m \
    --revision step79000 \
    --temperatures 0.01 0.1 1.0 10.0 \
    "${COMMON_ARGS[@]}"

# --- Students: T=1 only, one checkpoint at a time (append) ---
STUDENT_RUNS=(
    kl_001
    kl_01
    kl_01_v1
    kl_01_v2
    kl_01_v3
    kl_01_v4
    kl_1
    kl_10
    kl_10_v1
    mse_01
    mse_01_v1
    mse_01_v2
    mse_1
    mse_10
)

for run in "${STUDENT_RUNS[@]}"; do
    CKPTS=()
    for ckpt in $(ls -d ${CKPT}/${run}/step_* 2>/dev/null | sort); do
        CKPTS+=("$ckpt")
    done

    if [ ${#CKPTS[@]} -eq 0 ]; then
        echo "No checkpoints found for ${run}, skipping."
        continue
    fi

    echo "=== ${run}: ${#CKPTS[@]} checkpoints ==="
    for ckpt in "${CKPTS[@]}"; do
        python eval_stats.py \
            --model "${ckpt}" \
            --temperatures 1.0 \
            "${COMMON_ARGS[@]}" \
            --append
    done
done

echo "Done. Results at ${OUT}"
