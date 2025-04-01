#!/bin/bash
#SBATCH --job-name=Forest
#SBATCH --output=Forest.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10GB

/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support --n_trees=50 --function=random --is_bootstrap=False --is_custom_dist=False