#!/bin/bash
#SBATCH --job-name=RSForest-experiment-support-%A_%a
#SBATCH --output=logs/RSForest-experiment-%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=24GB
#SBATCH --array=1-2

source /home4/$USER/venvs/umcg_env/bin/activate

export WANDB_API_KEY=792c7896914a8e8e361566d5bddddf821e40fc53

wandb login

DATASETS=("support" "nhanes")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID-1]}

OUTPUT_FILE="results_rsf_${DATASET}.csv"

mkdir -p $TMPDIR/results
cp -R /home4/$USER/Survival-Tree-Analysis $TMPDIR/Survival-Tree-Analysis
cd $TMPDIR/Survival-Tree-Analysis

trap 'mkdir -p /home4/$USER/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID};
tar czvf /home4/$USER/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/models.tar.gz $TMPDIR/results/models;
cp $TMPDIR/results/${OUTPUT_FILE} /home4/$USER/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/' 12

CMD="python3 src/main_experiment.py \
    --parameter=\"randomsurvivalforest\" \
    --dataset=\"${DATASET}\" \
    --no-is_grid \
    --is_cv \
    --n_tries=10 \
    --n_models=5 \
    --n_splits=5 \
    --path=\"$TMPDIR/results\" \
    --path-res=\"$TMPDIR/results/${OUTPUT_FILE}\" \
    "

echo "Command: $CMD"
eval $CMD &
wait

mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/models.tar.gz $TMPDIR/results
cp $TMPDIR/results/${OUTPUT_FILE} /home4/$USER/job_${SLURM_JOBID}/
