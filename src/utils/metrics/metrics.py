import numpy as np

from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.metrics import mean_absolute_error
from lifelines.utils import concordance_index

def c_index(pred_times, y):
    """
        Compute the concordance index (C-index).
        param: pred_times: list of predicted survival times
        param: y: list of tuples, where each tuple is (censored, time)
        return: float, the concordance index
    """
    event_true = [0 if not censored else 1 for censored, _ in y]
    times_true = [time for _, time in y]

    c_index = concordance_index(times_true, pred_times, event_true)
    return c_index

def brier(pred_times, y):
    """
        Compute the Integrated Brier Score (IBS).
        param: pred_times: list of predicted survival times
        param: y: list of tuples, where each tuple is (censored, time)
        return: float, the integrated brier score
    """
    y_structured = np.array([(bool(not censor), float(time)) for censor, time in y], dtype=[('event', bool), ('time', float)])
        
    times_true = [time for _, time in y]

    min_time = min(times_true) 
    max_time = max(times_true)
    time_points = np.linspace(min_time, max_time * 0.999, 100)

    survival_probs = np.array([[1.0 if t < pred_time else 0.0 for t in time_points] 
                            for pred_time in pred_times])

    ibs = integrated_brier_score(y_structured, y_structured, survival_probs, time_points)
    return ibs

def auc(pred_times, y):
    """
        Compute the Cumulative Dynamic AUC.
        param: pred_times: list of predicted survival times
        param: y: list of tuples, where each tuple is (censored, time)
        return: float, the cumulative dynamic AUC
        return: float, the mean cumulative dynamic AUC
    """
    y_structured = np.array([(bool(not censor), float(time)) for censor, time in y], dtype=[('event', bool), ('time', float)])

    times_true = [time for _, time in y]
    min_time = min(times_true) 
    max_time = max(times_true)
    time_points = np.linspace(min_time, max_time * 0.999, 100)

    survival_probs = np.array([[1.0 if t < pred_time else 0.0 for t in time_points] 
                            for pred_time in pred_times])

    auc, mean_auc = cumulative_dynamic_auc(y_structured, y_structured, survival_probs, time_points)
    return auc, mean_auc

def mae(pred_times, y):
    """
        Compute the Mean Absolute Error (MAE).
        param: pred_times: list of predicted survival times
        param: y: list of tuples, where each tuple is (censored, time)
        return: float, the mean absolute error
    """
    event_mask = np.array([not censored for censored, _ in y])
    true_times = np.array([time for _, time in y])
    pred_times = np.array(pred_times)

    finite_mask = event_mask & np.isfinite(true_times) & np.isfinite(pred_times)
    if np.sum(finite_mask) == 0:
        raise ValueError("No valid predictions to compute MAE.")

    mae = mean_absolute_error(true_times[finite_mask], pred_times[finite_mask])
    return mae
