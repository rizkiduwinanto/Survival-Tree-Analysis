#!/bin/bash
#SBATCH --job-name=RF
#SBATCH --output=RF.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10GB

mkdir -p $TMPDIR/results

# Compress and save the results if the timelimit is close
trap 'mkdir -p /home4/$USER/job_${SLURM_JOBID}; tar czvf /home4/$USER/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results' 12

# Copy to TPMDIR
cp -R /home4/$USER/UDL-Project $TMPDIR/UDL-Project
cd $TMPDIR/UDL-Project

# Run the training
/home4/$USER/venvs/udl_envs/bin/python3 $TMPDIR/UDL-Project/src/main.py &
wait

mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results