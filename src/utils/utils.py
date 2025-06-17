import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import cupy as cp

def gpu_train_test_split(X, y_death, y_time, test_size=0.2, random_state=None):
    n_samples = X.shape[0]
    cp.random.seed(random_state)

    indices = cp.random.permutation(n_samples)
    test_count = int(n_samples * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], y_death[train_idx], y_time[train_idx], X[test_idx], y_death[test_idx], y_time[test_idx]

def stratified_gpu_train_test_split(X, y_death, y_time, test_size=0.2, random_state=None):
    cp.random.seed(random_state)

    uncensored = cp.where(y_death == 1)[0]
    censored = cp.where(y_death == 0)[0]

    uncensored_count = uncensored.shape[0]
    censored_count = censored.shape[0]

    uncensored_indices = cp.random.permutation(uncensored)
    censored_indices = cp.random.permutation(censored)

    uncensored_test_count = int(uncensored_count * test_size)
    censored_test_count = int(censored_count * test_size)

    uncensored_test_idx = uncensored_indices[:uncensored_test_count]
    uncensored_train_idx = uncensored_indices[uncensored_test_count:]
    censored_test_idx = censored_indices[:censored_test_count]
    censored_train_idx = censored_indices[censored_test_count:]

    train_idx = cp.concatenate((uncensored_train_idx, censored_train_idx))
    test_idx = cp.concatenate((uncensored_test_idx, censored_test_idx))

    return X[train_idx], y_death[train_idx], y_time[train_idx], X[test_idx], y_death[test_idx], y_time[test_idx]

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)

    uncensored = np.where(y[:, 0] == 1)[0]
    censored = np.where(y[:, 0] == 0)[0]

    uncensored_count = uncensored.shape[0]
    censored_count = censored.shape[0]

    uncensored_indices = np.random.permutation(uncensored)
    censored_indices = np.random.permutation(censored)

    uncensored_test_count = int(uncensored_count * test_size)
    censored_test_count = int(censored_count * test_size)

    uncensored_test_idx = uncensored_indices[:uncensored_test_count]
    uncensored_train_idx = uncensored_indices[uncensored_test_count:]
    censored_test_idx = censored_indices[:censored_test_count]
    censored_train_idx = censored_indices[censored_test_count:]

    train_idx = np.concatenate((uncensored_train_idx, censored_train_idx))
    test_idx = np.concatenate((uncensored_test_idx, censored_test_idx))

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    
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


