import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cupy as cp
import os
from .metrics.metrics import mae

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

def stratified_time_gpu_train_test_split(X, y_death, y_time, test_size=0.2, random_state=None):
    cp.random.seed(random_state)
    
    uncensored_mask = (y_death == 1)
    uncensored_times = y_time[uncensored_mask]
    
    time_bins = cp.quantile(uncensored_times, q=[0.25, 0.5, 0.75])
    
    strata = cp.zeros_like(y_death, dtype=cp.int32)
    
    strata[uncensored_mask] = cp.digitize(uncensored_times, bins=time_bins) + 1
    
    strata[~uncensored_mask] = 0
    
    unique_strata = cp.unique(strata)
    train_idx = []
    test_idx = []
    
    for stratum in unique_strata:
        stratum_indices = cp.where(strata == stratum)[0]
        stratum_count = len(stratum_indices)
        test_count = int(stratum_count * test_size)
        
        shuffled_indices = cp.random.permutation(stratum_indices)
        
        test_idx.append(shuffled_indices[:test_count])
        train_idx.append(shuffled_indices[test_count:])
    
    train_idx = cp.concatenate(train_idx)
    test_idx = cp.concatenate(test_idx)
    
    train_idx = cp.random.permutation(train_idx)
    test_idx = cp.random.permutation(test_idx)
    
    return X[train_idx], y_death[train_idx], y_time[train_idx], X[test_idx], y_death[test_idx], y_time[test_idx]
    
def plot_survival_trees(pred_times, y, dataset=None, function=None, model=None, save=False, path=None, index=None):
    """
    Plot the survival curves for the predicted and true survival times.
    Parameters:
    y_true: list of tuples (censored, time)
        The true survival times and censoring information.
    y_pred: list of tuples (censored, time)
        The predicted survival times and censoring information.
    """
    if model == "XGBoostAFT":
        y_true = np.array(y['d.time'])
        censored = np.array(y['death'] == 0)
    else:
        y_true = np.array([time for _, time in y])
        censored = np.array([not death for death, _ in y])

    y_pred = np.array(pred_times)

    uncensored = ~censored

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_true[uncensored], y_pred[uncensored], alpha=0.6, label='Uncensored', color='blue', edgecolor='k')
    ax.scatter(y_true[censored], y_pred[censored], alpha=0.6, label='Censored', color='red', marker='x')

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', label='Perfect Prediction')

    if model == "XGBoostAFT":
        y = np.array(list(zip(censored, y_true)), dtype=[('censored', bool), ('time', float)])

    mae_ = mae(y_pred, y)

    if dataset is not None and function is not None and model is not None:
        ax.set_title(f'True vs Predicted Survival Times\n in {dataset} with {model} - {function}\nMAE: {mae_:.2f}')
    ax.set_title(f'True vs Predicted Survival Times\nMAE: {mae_:.2f}')

    if dataset is not None and dataset == "nhanes":
        ax.set_xlabel('True survival time (years)')
        ax.set_ylabel('Predicted survival time (years)')
    else:
        ax.set_xlabel('True survival time (days)')
        ax.set_ylabel('Predicted survival time (days)')
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    if save:
        if path is not None:
            os.makedirs(path, exist_ok=True)
            if index is not None:
                plt.savefig(os.path.join(path, f'survival_tree_{model}_{function}_{index}.png'))
            else:
                plt.savefig(os.path.join(path, f'survival_tree_{model}_{function}.png'))
        else:
            plt.savefig('calibration.png')
    else:
        plt.show()

def save_model(model, path, name):
    if model == "AFTForest":
        path_dir = os.path.join(path, name)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        model.save(path_dir)
    elif model == "AFTSurvivalTree":
        path_dir = os.path.join(path, f"{name}.json")
        model.save(path_dir)
    else:
        path_dir = os.path.join(path, f"{name}.pkl")
        model.save(path_dir)

def dump_results_to_csv(results, path="results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


