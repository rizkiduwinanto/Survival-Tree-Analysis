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