# AFT Random Forest 

This repository contains the code and scripts for the Thesis "Survival Tree Analysis
for Predicting Absolute Time-to-event".

The repository contains the implementation of Random Forest with Acceleaate Failure Time Loss Function.

## Installation
To set up the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Project
To execute the algorithm scripts, use the provided job scripts:

```python
python3 src/main.py --parameter=aftforest --dataset=veteran --path=tree --function=extreme  --no-is_bootstrap --no-is_custom_dist --no-is_feature_subsample --aggregator=median --no-is_split_fitting
```

To execute the experiment scripts, use the provided job scripts:

```python
python3 src/main_experiment.py --parameter="afttree" --dataset="NHANES" --function=logistic --no-is_grid --is_cv --n_splits=2 --n_tries=2 --n_models=1 --path-res=results.csv  
```

The parameter reference can be seen in:

| Argument          | Options                          | Default     | Description                  |
|-------------------|----------------------------------|-------------|------------------------------|
| `--parameter`     | `aftforest`, `afttree` | *Required* | Model type |
| `--dataset`       | `veteran`, `NHANES`,`support`| *Required* | Dataset |
| `--function`      | `extreme`, `logistic`, `norm`, `weibull`. `gmm` | *Required* | Splitting function |
| `--aggregator`    | `median`, `mean` | `mean` | Prediction aggregation |

The flag parameter can be seen in:

| Flag                      | Effect                             |
|---------------------------|------------------------------------|
| `--is_bootstrap`          | Enable bootstrap sampling          |
| `--is_custom_dist`        | Enable custom dist                 |
| `--is_feature_subsample`  | Enable random feature selection    |
| `--is_cv`                 | Enable cross-validation (only experiments) |
| `--is_grid`               | Enable grid (only experiments) |

to disable each flag replace `--` with `--no-`. 

## Outputs
* Models are saved in `\models`
* Results: CSV files (e.g `results.csv`)

## Credits








