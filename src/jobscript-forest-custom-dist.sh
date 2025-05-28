#!/bin/bash
#SBATCH --job-name=Forest-custom-dist-6
#SBATCH --output=Forest-custom-dist-6.out
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10GB

# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-random --n_trees=50 --function=random --no-is_bootstrap --no-is_custom_dist
# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-norm --n_trees=50 --function=norm --no-is_bootstrap --no-is_custom_dist
# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-logistic --n_trees=50 --function=logistic --no-is_bootstrap --no-is_custom_dist
# /home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-extreme --n_trees=50 --function=extreme --no-is_bootstrap --no-is_custom_dist

/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-extreme-c-run-100 --n_trees=100 --function=extreme --no-is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-gmm-c-run-100 --n_trees=100 --function=gmm --no-is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-weibull-c-run-100 --n_trees=100 --function=weibull --no-is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-random-c-run-100 --n_trees=100 --function=random --no-is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-norm-c-run-100 --n_trees=100 --function=norm --no-is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-logistic-c-run-100 --n_trees=100 --function=logistic --no-is_bootstrap --is_custom_dist --no-is_feature_subsample
