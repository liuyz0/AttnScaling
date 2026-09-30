#!/bin/bash
#SBATCH --job-name=distill-eval
#SBATCH --chdir=/orcd/compute/gore/001/DistillScaling
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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
OUT=results/eval.jsonl

mkdir -p results

# (run_dir, kl_temperature) pairs — temperature matches training
RUNS=(
    "kl_001  0.01"
    "kl_01   0.1"
    "kl_1    1.0"
    "kl_10   10.0"
    "mse_1   1.0"
)

FIRST=true
for entry in "${RUNS[@]}"; do
    run=$(echo $entry | awk '{print $1}')
    temp=$(echo $entry | awk '{print $2}')

    MODELS=()
    for ckpt in $(ls -d ${CKPT}/${run}/step_* 2>/dev/null | sort); do
        MODELS+=("$ckpt")
    done

    if [ ${#MODELS[@]} -eq 0 ]; then
        echo "No checkpoints found for ${run}, skipping."
        continue
    fi

    echo "=== ${run} (temp=${temp}): ${#MODELS[@]} checkpoints ==="

    # First run overwrites; subsequent runs append
    APPEND_FLAG=""
    [ "$FIRST" = false ] && APPEND_FLAG="--append"
    FIRST=false

    python eval.py \
        --models "${MODELS[@]}" \
        --loss_types kl \
        --teacher_model EleutherAI/pythia-160m \
        --teacher_revision step79000 \
        --teacher_temperatures ${temp} \
        --data_path data/pythia_deduped_pile_idxmaps/ \
        --clean_eval \
        --train_steps 79000 \
        --train_world_size 8 \
        --train_batch_size 32 \
        --n_eval_tokens 5000000 \
        --batch_size 32 \
        --dtype bfloat16 \
        --out_file ${OUT} \
        ${APPEND_FLAG}
done

echo "Done. Results at ${OUT}"
