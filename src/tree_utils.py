import cupy as cp

def gpu_train_test_split(X, y_death, y_time, test_size=0.2, random_state=None):
    n_samples = X.shape[0]
    cp.random.seed(random_state)

    indices = cp.random.permutation(n_samples)
    test_count = int(n_samples * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], y_death[train_idx], y_time[train_idx], X[test_idx], y_death[test_idx], y_time[test_idx]