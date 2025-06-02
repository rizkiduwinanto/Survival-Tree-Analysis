import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sns

def plot_survival_trees(y_true, y_pred, censored):
    """
    Plot the survival curves for the predicted and true survival times.
    Parameters:
    y_true: list of tuples (censored, time)
        The true survival times and censoring information.
    y_pred: list of tuples (censored, time)
        The predicted survival times and censoring information.
    """
    print(y_true)
    print(y_pred)
    print(censored)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, label='Predicted')
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axline(xy1=(0, 0), slope=1, color='r', linestyle='--', label='Perfect Prediction')

    ax.set_title('True vs Predicted Survival Times')
    ax.set_xlabel('True survival time (years)')
    ax.set_ylabel('Predicted survival time (years)')
    ax.legend()

def dump_results_to_csv(results, path="results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


