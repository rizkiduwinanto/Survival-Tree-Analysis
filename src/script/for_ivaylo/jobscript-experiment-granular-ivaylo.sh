#!/bin/bash
#SBATCH --job-name=Forest-experiment-granular-ivaylo
#SBATCH --output=logs/Forest-experiment-granular-ivaylo_%j_%a.out
#SBATCH --time=23:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --array=1-60

# module load CUDA/12.4.0
# source /home4/$USER/venvs/umcg_env/bin/activate

module load Python
module load CUDA
# source /home4/$USER/venvs/umcg_env/bin/activate
source ../proj_env/bin/activate .

export WANDB_API_KEY=<WANDB_API_KEY>

wandb login

mkdir -p results
INDEX=$SLURM_ARRAY_TASK_ID
HYPERPARAM_FILE="hyperparam/params_$INDEX.json"

CMD="python3 src/main_granular.py \
    --index=$INDEX \
    --n_models=5 \
    --n_splits=5 \
    --path_to_read=\"$HYPERPARAM_FILE\" \
    --path_models=\"results\" \
    --path_to_save=\"results\" \
    --path_to_image=\"results\" \
    "

echo "Running configuration $SLURM_ARRAY_TASK_ID: Hyperparam $INDEX"
echo "Command: $CMD"
eval $CMD
