import pandas as pd
import numpy as np

from .dataset import Dataset

import shap
from sklearn.model_selection import train_test_split

class NHANESDataset(Dataset):
    """
    NHANES dataset for survival analysis.
    This dataset contains health and nutrition data from the National Health and Nutrition Examination Survey (NHANES).
    It includes information on mortality and survival times, which can be used for survival analysis tasks.
    """
    def __init__(self, convert_bool=True):
        self.data, self.label_shap = shap.datasets.nhanesi()

        self.preprocess(convert_bool)

        self.label = self.create_label()
        self.xgboost_label = self.create_xgboost_label()

        self.data = self.data.to_numpy()

    def preprocess(self, convert_bool):
        self.data.fillna(self.data.median())

        if convert_bool:
            self.convert_bool_to_int()
    
    def create_label(self):
        label = pd.DataFrame()
        label['death'] = [0 if x < 0 else 1 for x in self.label_shap]
        label['d.time'] = abs(self.label_shap)
        record = label.to_records(index=False)
        structured_arr = np.stack(record, axis=0)

        return structured_arr

    def create_xgboost_label(self):
        label = pd.DataFrame()
        label['Survival_label_lower_bound'] = abs(self.label_shap)
        label['Survival_label_upper_bound'] = np.where(self.label_shap > 0, self.label_shap, np.inf)
        
        return label

    def get_label(self):
        return self.label
    
    def get_xgboost_label(self):
        return self.xgboost_label

    def get_shap_label(self):
        return self.label_shap

    def get_data(self):
        return self.data

    def get_train_test(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
        
    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.xgboost_label, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def convert_bool_to_int(self):
        self.data.replace({False: 0, True: 1}, inplace=True)