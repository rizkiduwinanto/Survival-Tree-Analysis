#!/bin/bash
#SBATCH --job-name=Tree
#SBATCH --output=Tree.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10GB

/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTForest support forest-wei.json 50 random False False