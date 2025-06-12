#!/bin/bash
#SBATCH --job-name=Forest-boostrap-5
#SBATCH --output=Forest-boostrap-5.out
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10GB

/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-random --n_trees=50 --function=random --is_bootstrap --no-is_custom_dist
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-norm --n_trees=50 --function=norm --is_bootstrap --no-is_custom_dist
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-logistic --n_trees=50 --function=logistic --is_bootstrap --no-is_custom_dist
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-extreme --n_trees=50 --function=extreme --is_bootstrap --no-is_custom_dist

/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-extreme-c-b-run-100 --n_trees=100 --function=extreme --is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-gmm-c-b-run-100 --n_trees=100 --function=gmm --is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-weibull-b-run-100 --n_trees=100 --function=weibull --is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-random-c-b-run-100 --n_trees=100 --function=random --is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-norm-c-b-run-100 --n_trees=100 --function=norm --is_bootstrap --is_custom_dist --no-is_feature_subsample
/home4/$USER/venvs/umcg_env/bin/python3 /home4/$USER/Survival-Tree-Analysis/src/main.py --parameter=AFTForest --dataset=support --path=forest-support-logistic-c-b-run-100 --n_trees=100 --function=logistic --is_bootstrap --is_custom_dist --no-is_feature_subsample
