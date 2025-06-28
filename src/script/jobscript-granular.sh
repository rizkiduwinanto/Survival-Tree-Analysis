#!/bin/bash
#SBATCH --job-name=Forest-experiment-granular
#SBATCH --output=logs/Forest-experiment-granular_%j_%a.out
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --array=1-60

module load CUDA/12.4.0
source /home4/$USER/venvs/umcg_env/bin/activate

export WANDB_API_KEY=792c7896914a8e8e361566d5bddddf821e40fc53

wandb login

mkdir -p $TMPDIR/results
cp -R /home4/$USER/Survival-Tree-Analysis $TMPDIR/Survival-Tree-Analysis
cd $TMPDIR/Survival-Tree-Analysis

trap 'mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results' 12

INDEX=$SLURM_ARRAY_TASK_ID
HYPERPARAM_FILE="hyperparam/params_$INDEX.json"

CMD="python3 src/main_granular.py \
    --index=$INDEX \
    --n_models=5 \
    --n_splits=5 \
    --path_to_read=\"$HYPERPARAM_FILE\" \
    --path_models=\"$TMPDIR/results/models\" \
    --path_to_save=\"$TMPDIR/results\" \
    --path_to_image=\"$TMPDIR/results\" \
    "

echo "Running configuration $SLURM_ARRAY_TASK_ID: Hyperparam $INDEX"
echo "Command: $CMD"
eval $CMD &
wait

mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results
