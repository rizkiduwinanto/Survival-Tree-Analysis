#!/bin/bash
#SBATCH --job-name=tree-experiment
#SBATCH --output=tree-experiment_%j_%a.out
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --array=1-13

module load CUDA/12.4.0
source /home4/$USER/venvs/umcg_env/bin/activate

export WANDB_API_KEY=<WANDB_API_KEY>

wandb login

mkdir -p $TMPDIR/results
cp -R /home4/$USER/Survival-Tree-Analysis $TMPDIR/Survival-Tree-Analysis
cd $TMPDIR/Survival-Tree-Analysis

trap 'mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/models.tar.gz $TMPDIR/models
cp $TMPDIR/results/${OUTPUT_FILE} /home4/$USER/job_${SLURM_JOBID}/' 12

CONFIGURATIONS=(
    "extreme --no-is_custom_dist --no-is_bootstrap"
    "norm --no-is_custom_dist --no-is_bootstrap"
    "logistic --no-is_custom_dist --no-is_bootstrap"
    
    "extreme --is_custom_dist --no-is_bootstrap"
    "norm --is_custom_dist --no-is_bootstrap"
    "logistic --is_custom_dist --no-is_bootstrap"
    "weibull --is_custom_dist --no-is_bootstrap"
    "gmm --is_custom_dist --no-is_bootstrap"
    
    "extreme --is_custom_dist --is_bootstrap"
    "norm --is_custom_dist --is_bootstrap"
    "logistic --is_custom_dist --is_bootstrap"
    "weibull --is_custom_dist --is_bootstrap"
    "gmm --is_custom_dist --is_bootstrap"
)

IFS=' ' read -r FUNCTION FLAGS <<< "${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID - 1]}"

OUTPUT_FILE="results_${FUNCTION}_${SLURM_ARRAY_TASK_ID}.csv"
SAFE_FLAGS=$(echo "$FLAGS" | tr ' ' '_')

CMD="python3 src/main_experiment.py \
    --parameter=\"afttree\" \
    --dataset=\"nhanes\" \
    --function=\"$FUNCTION\" \
    --no-is_grid \
    --is_cv \
    --n_tries=10 \
    --n_models=5 \
    --n_splits=5 \
    --path=\"$TMPDIR/results/models\" \
    --path-res=\"$TMPDIR/results/${OUTPUT_FILE}\" \
    --aggregator=\"mean\" \
    --is_split_fitting
    $FLAGS"

echo "Running configuration $SLURM_ARRAY_TASK_ID: $FUNCTION with $FLAGS"
echo "Command: $CMD"
eval $CMD &
wait

mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/models.tar.gz $TMPDIR/results/models
cp $TMPDIR/results/${OUTPUT_FILE} /home4/$USER/job_${SLURM_JOBID}/
