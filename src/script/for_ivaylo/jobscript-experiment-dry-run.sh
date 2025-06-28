#!/bin/bash
#SBATCH --job-name=Forest-experiment-dry_run
#SBATCH --output=Forest-experimentdry_run_%j_%a.out
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --array=1

# module load CUDA/12.4.0
# source /home4/$USER/venvs/umcg_env/bin/activate

module load Python
module load CUDA
# source /home4/$USER/venvs/umcg_env/bin/activate
source ../proj_env/bin/activate .

export WANDB_API_KEY=<WANDB_API_KEY>

wandb login

CONFIGURATIONS=(
    "normal --no-is_custom_dist --no-is_bootstrap"
)

DATASET="veteran"

IFS=' ' read -r FUNCTION FLAGS <<< "${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID - 1]}"

mkdir -p "results"
PATH_SAVE="results/results_${DATASET}_${FUNCTION}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_FILE="results/results_${DATASET}_${FUNCTION}_${SLURM_ARRAY_TASK_ID}.csv"
SAFE_FLAGS=$(echo "$FLAGS" | tr ' ' '_')

echo "Saving in path: ' $PATH_SAVE"
echo "Output file: ' $OUTPUT_FILE"

CMD="python3 src/main_experiment.py \
    --parameter=\"aftforest\" \
    --dataset=\"$DATASET\" \
    --function=\"$FUNCTION\" \
    --no-is_grid \
    --is_cv \
    --n_tries=10 \
    --n_models=5 \
    --n_splits=5 \
    --path=\"${PATH_SAVE}\" \
    --path-res=\"results/${OUTPUT_FILE}\" \
    --aggregator=\"mean\" \
    --is_split_fitting
    $FLAGS"

echo "Running configuration $SLURM_ARRAY_TASK_ID: $FUNCTION with $FLAGS"
echo "Command: $CMD"
eval $CMD &
