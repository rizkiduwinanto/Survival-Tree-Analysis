import pandas as pd
import numpy as np

from .dataset import Dataset

from sklearn.model_selection import train_test_split

class VeteranLungDataset(Dataset):
    """
    Veteran Lung dataset for survival analysis.
    This dataset contains information about veterans with lung cancer.
    It includes features such as demographics, clinical measurements, and survival outcomes.
    """
    def __init__(self, data):
        self.data = data.copy()
        self.label = self.create_label()
        self.xgboost_label = self.create_xgboost_label()

        self.data.drop(["Survival_label_lower_bound", "Survival_label_upper_bound"], axis=1, inplace=True)

        self.data = self.data.to_numpy()

    def create_label(self):
        label = pd.DataFrame()
        label['death'] = [0 if x < np.inf else 1 for x in self.data['Survival_label_upper_bound']]
        label['d.time'] = self.data['Survival_label_lower_bound']
        record = label.to_records(index=False)
        structured_arr = np.stack(record, axis=0)

        return structured_arr

    def preprocess(self, impute_rest=True, convert_bool=True):
        pass

    def create_xgboost_label(self):
        label = pd.DataFrame()
        label['Survival_label_lower_bound'] = self.data['Survival_label_lower_bound']
        label['Survival_label_upper_bound'] = self.data['Survival_label_upper_bound']
        return label
        
    def get_label(self):
        return self.label

    def get_xgboost_label(self):
        return self.xgboost_label

    def get_train_test(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.xgboost_label, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_data(self):
        return self.data
