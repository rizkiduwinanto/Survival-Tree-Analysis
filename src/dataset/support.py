import pandas as pd
import numpy as np

from .dataset import Dataset

from sklearn.model_selection import train_test_split

class SupportDataset(Dataset):
    """
    Support dataset for survival analysis.
    This dataset contains information about patients in a medical support setting.
    It includes features such as demographics, clinical measurements, and survival outcomes.
    The dataset is used for survival analysis tasks, particularly in the context of medical research.
    """
    def __init__(self, data, impute_rest=True, convert_bool=True):
        self.data = data.copy()

        self.preprocess(impute_rest, convert_bool)
        self.label = self.create_label()
        self.xgboost_label = self.create_xgboost_label()
        
        self.data.drop(["death", "d.time"], axis=1, inplace=True)

        self.data = self.data.to_numpy()

    def create_label(self):
        self.data["death"] = self.data["death"].astype('bool')
        label = self.data[["death", "d.time"]]
        record = label.to_records(index=False)
        structured_arr = np.stack(record, axis=0)

        return structured_arr

    def create_xgboost_label(self):
        self.temp = pd.DataFrame()
        self.temp["Survival_label_lower_bound"] = self.data["d.time"]
        self.temp["Survival_label_upper_bound"] = self.data.apply(
            lambda row: row["d.time"] if row["death"] == 0 else np.inf, axis=1
        )
        self.temp["death"] = self.data["death"].astype('int')

        return self.temp

    def preprocess(self, impute_rest, convert_bool):
        self.impute_values(impute_rest)
        self.drop_values()
        self.data.dropna(axis=1, inplace=True)
        self.convert_to_one_hot()

        if convert_bool:
            self.convert_bool_to_int()

        # self.data = self.data[self.data["death"] ==1]

    def impute_values(self, impute_rest):
        VALUE = {
            "alb": 3.5,
            "pafi": 333.3,
            "bili": 1.01,
            "crea": 1.01,
            "bun": 6.51,
            "wblc": 9.0,
            "urine": 2502.0,
        }

        self.data.fillna(value=VALUE, inplace=True)

    def drop_values(self):
        TO_DROP = [
            "aps",
            "sps",
            "surv2m",
            "surv6m",
            "prg2m",
            "prg6m",
            "dnr",
            "dnrday",
            "sfdm2",
            "hospdead",
            "slos",
            "charges",
            "totcst",
            "totmcst",
        ]

        self.data.drop(TO_DROP, axis=1, inplace=True)

    def convert_to_one_hot(self):
        TO_CONVERT = [
            "sex",
            "dzgroup",
            "dzclass",
            # "race",
            "ca",
            # "adlp",
            # "edu",
            # "income"
        ]
        self.data = pd.get_dummies(self.data, columns=TO_CONVERT)

    def convert_bool_to_int(self):
        self.data.replace({False: 0, True: 1}, inplace=True)
    
    def get_label(self):
        return self.label
    
    def get_xgboost_label(self):
        return self.xgboost_label

    def get_train_test(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=test_size, stratify=self.label["death"], random_state=random_state)
        return X_train, X_test, y_train, y_test

    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.xgboost_label, test_size=test_size, stratify=self.xgboost_label["death"],  random_state=random_state)
        return X_train, X_test, y_train, y_test

    def get_data(self):
        return self.data
