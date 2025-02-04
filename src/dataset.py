import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

np.random.seed(42) 

class Dataset:
    def __init__(self, data, impute_rest=True):
        self.data = data
        self.preprocess(impute_rest)
        self.label = self.create_label()

    def create_label(self):
        self.data["death"] = self.data["death"].astype('bool')
        label = self.data[["death", "d.time"]]
        record = label.to_records(index=False)
        structured_arr = np.stack(record, axis=0)

        self.data.drop(["death", "d.time"], axis=1, inplace=True)

        return structured_arr

    def preprocess(self, impute_rest):
        self.impute_values(impute_rest)
        self.drop_values()
        self.data.dropna(axis=1, inplace=True)
        self.convert_to_one_hot()

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
    
    def get_label(self):
        return self.label

    def get_train_test(self, test_size=0.2):
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=42)
        return x_train, y_train, x_test, y_test

    def get_data(self):
        return self.data

class SyntheticDataset:
    def __init__(self, n_censored, n_uncensored, n_feature=2):
        self.n_feature = n_feature
        self.n_samples = n_censored + n_uncensored
        self.n_censored = n_censored
        self.n_uncensored = n_uncensored
    
    def create_data(self):
        X = np.random.rand(self.n_samples, self.n_feature)
        y_censored = [(False, np.random.randint(1, 100)) for _ in range(self.n_censored)]
        y_uncensored = [(True, np.random.randint(1, 100)) for _ in range(self.n_uncensored)]

        y = np.concatenate([y_censored, y_uncensored])
        return X, y

    def create_one_data(self):
        X = np.random.rand(1, self.n_feature)
        y = [(True, np.random.randint(1, 100))]
        return X, y
