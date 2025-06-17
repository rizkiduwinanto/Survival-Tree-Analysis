from sksurv.ensemble import RandomSurvivalForest
from utils.metrics.metrics import c_index, brier, auc, mae
import numpy as np

class RandomSurvivalForestWrapper:
    def __init__(
        self,
        n_trees=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
    ):
        self.model = RandomSurvivalForest(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            n_jobs=1
        )

    def fit(self, X, y):
        y = y.copy()
        y['death'] = y['death'].astype(bool)
        y_struct = np.array(list(zip(y['death'], y['d.time'])), dtype=[('death', bool), ('d.time', float)])
        self.model.fit(X, y_struct)

    def predict(self, X):
        return self.model.predict_survival_function(X)

    def predict_risk(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y = y.copy()
        y['death'] = y['death'].astype(bool)
        y_struct = np.array(list(zip(y['death'], y['d.time'])), dtype=[('death', bool), ('d.time', float)])
        return self.model.score(X, y_struct)

    def predict_median(self, X):
        surv_funcs = self.model.predict_survival_function(X)
        medians = []
        for fn in surv_funcs:
            times = fn.x
            probs = fn.y
            if np.any(probs < 0.5):
                median_time = times[np.where(probs < 0.5)[0][0]]
            else:
                median_time = times[-1]
            medians.append(median_time)
        return np.array(medians)

    def _score(self, X, y):
        pred_times = self.predict_median(X)
        return c_index(pred_times, y)

    def _brier(self, X, y):
        pred_times = self.predict_median(X)
        return brier(pred_times, y)

    def _auc(self, X, y):
        pred_times = self.predict_median(X)
        return auc(pred_times, y)

    def _mae(self, X, y):
        pred_times = self.predict_median(X)
        return mae(pred_times, y)