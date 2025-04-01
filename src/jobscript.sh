#!/bin/bash
#SBATCH --job-name=Tree
#SBATCH --output=Tree.out
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10GB

/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-nff.json 0 normal False False
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-eff.json 0 extreme False False
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-lff.json 0 logistic False False
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-nft.json 0 normal False True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-eft.json 0 extreme False True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-lft.json 0 logistic False True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-ntt.json 0 normal True True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-ett.json 0 extreme True True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-ltt.json 0 logistic True True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-wft.json 0 weibull False True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-wtt.json 0 weibull True True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-gft.json 0 gmm False True
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTSurvivalTree support tree-gtt.json 0 gmm True True

# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTForest support forest-wei.json 20 weibull False True
# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTForest support forest-wei.json 20 weibull False True
# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTForest support forest-wei.json 20 weibull False True
# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py AFTForest support forest-wei.json 20 weibull False True