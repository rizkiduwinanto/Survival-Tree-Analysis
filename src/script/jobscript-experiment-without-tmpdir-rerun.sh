#!/bin/bash
#SBATCH --job-name=Forest-experiment
#SBATCH --output=Forest-experiment_%j_%a.out
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --array=1-10

module load CUDA/12.4.0
source /home4/$USER/venvs/umcg_env/bin/activate

export WANDB_API_KEY=<WANDB_API_KEY>

wandb login

CONFIGURATIONS=(
    # "extreme --no-is_custom_dist --no-is_bootstrap"
    # "normal --no-is_custom_dist --no-is_bootstrap"
    # "logistic --no-is_custom_dist --no-is_bootstrap"
    "extreme --is_custom_dist --no-is_bootstrap"
    "normal --is_custom_dist --no-is_bootstrap"
    "logistic --is_custom_dist --no-is_bootstrap"
    "weibull --is_custom_dist --no-is_bootstrap"
    "gmm --is_custom_dist --no-is_bootstrap"
    "extreme --is_custom_dist --is_bootstrap"
    "normal --is_custom_dist --is_bootstrap"
    "logistic --is_custom_dist --is_bootstrap"
    "weibull --is_custom_dist --is_bootstrap"
    "gmm --is_custom_dist --is_bootstrap"
)

IFS=' ' read -r FUNCTION FLAGS <<< "${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID - 1]}"

PATH="results_${FUNCTION}_${SLURM_ARRAY_TASK_ID}/models"
OUTPUT_FILE="results_${FUNCTION}_${SLURM_ARRAY_TASK_ID}.csv"
SAFE_FLAGS=$(echo "$FLAGS" | tr ' ' '_')

CMD="python3 src/main_experiment.py \
    --parameter=\"aftforest\" \
    --dataset=\"support\" \
    --function=\"$FUNCTION\" \
    --no-is_grid \
    --is_cv \
    --n_tries=10 \
    --n_models=5 \
    --n_splits=5 \
    --path=\"${PATH}\" \
    --path-res=\"results/${OUTPUT_FILE}\" \
    --aggregator=\"median\" \
    --no-is_split_fitting
    $FLAGS"

echo "Running configuration $SLURM_ARRAY_TASK_ID: $FUNCTION with $FLAGS"
echo "Command: $CMD"
eval $CMD &
