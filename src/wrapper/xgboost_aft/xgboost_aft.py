import xgboost as xgb
from wrapper.wrapper import Wrapper
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.metrics.metrics import c_index, brier, auc, mae
import pickle

class XGBoostAFTWrapper(Wrapper):
    """
    A wrapper class for XGBoost AFT (Accelerated Failure Time) model.
    This class is designed to facilitate the use of XGBoost for survival analysis tasks.
    """

    def __init__(
        self,
        max_depth=6,
        function='normal',
        sigma=0.1,
        learning_rate=0.05,
        lambda_=0.01,
        alpha=0.02,
        num_boost_round=1000,
        early_stopping_rounds=10
    ):
        self.params = {
            'max_depth': max_depth,
            'objective': 'survival:aft',
            'aft_loss_distribution': function,
            'aft_loss_distribution_scale': sigma,
            'learning_rate': learning_rate,
            'lambda': lambda_,
            'alpha': alpha,
            'eval_metric': 'aft-nloglik',
        }

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y):
        """
        Fit the XGBoost AFT model to the training data.
        
        Parameters:
        - x_train: Training features.
        - y_train: Training labels (survival times and event indicators).
        """
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dtrain = xgb.DMatrix(x_train)
        dtrain.set_float_info('label_lower_bound', y_train['Survival_label_lower_bound'])
        dtrain.set_float_info('label_upper_bound', y_train['Survival_label_upper_bound'])

        dvalid = xgb.DMatrix(x_valid)
        dvalid.set_float_info('label_lower_bound', y_valid['Survival_label_lower_bound'])
        dvalid.set_float_info('label_upper_bound', y_valid['Survival_label_upper_bound'])
        
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train'), (dvalid, 'eval')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self._predict(dtest)

    def _predict(self, dtest):
        predictions = self.model.predict(dtest)
        return predictions

    def bin_y(self, y):
        """
        Convert survival labels into a format suitable for evaluation metrics.
        This function extracts the lower and upper bounds of survival times and event indicators.
        Parameters:
        - y: DataFrame containing survival labels with 'Survival_label_lower_bound' and 'Survival_label_upper_bound'.
        Returns:
        - A tuple of two numpy arrays: (times_true, event_true)
          where times_true contains the lower bounds of survival times,
          and event_true contains binary indicators for events (1 if event occurred, 0 otherwise).
        """
        times_true = []
        event_true = []
        for _, rows in y.iterrows():
            if rows['Survival_label_upper_bound'] == np.inf or not rows['Survival_label_upper_bound']:
                event = 0
            else:
                event = 1 
            event_true.append(event)
            times = rows['Survival_label_lower_bound']
            times_true.append(times)
        times_true = np.array(times_true)
        event_true = np.array(event_true)

        y_new = np.array(list(zip(event_true, times_true)), dtype=[('death', bool), ('d.time', float)])

        return y_new

    def _score(self, X, y):
        pred_times = self.predict(X)
        y_new = self.bin_y(y)
        return c_index(pred_times, y_new)

    def _brier(self, X, y):
        pred_times = self.predict(X)
        y_new = self.bin_y(y)
        return brier(pred_times, y_new)

    def _auc(self, X, y):
        pred_times = self.predict(X)
        y_new = self.bin_y(y)
        return auc(pred_times, y_new)
        
    def _mae(self, X, y):
        pred_times = self.predict(X)
        y_new = self.bin_y(y)
        return mae(pred_times, y_new)

    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        model = pickle.load(open(path, 'rb'))
        wrapper = cls()
        wrapper.model = model
        return wrapper