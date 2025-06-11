import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
import shap

class SupportDataset():
    def __init__(self, data, impute_rest=True, convert_bool=True, is_scikit=False):
        self.data = data.copy()

        self.preprocess(impute_rest, convert_bool)
        self.label = self.create_label(is_scikit)
        self.xgboost_label = self.create_xgboost_label()
        
        self.data.drop(["death", "d.time"], axis=1, inplace=True)

        self.data = self.data.to_numpy()

    def create_label(self, is_scikit):
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
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=42)
        return x_train, y_train, x_test, y_test

    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.xgboost_label, test_size=test_size, random_state=42)
        return x_train, y_train, x_test, y_test

    def get_data(self):
        return self.data

class VeteranLungDataset():
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
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=42)
        return x_train, y_train, x_test, y_test

    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.xgboost_label, test_size=test_size, random_state=42)
        return x_train, y_train, x_test, y_test

    def get_data(self):
        return self.data

class NHANESDataset():
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
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=42)

        print("Type of x_train:", type(x_train))
        return x_train, y_train, x_test, y_test

    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        x_train, y_train, x_test, y_test = train_test_split(self.data, self.xgboost_label, test_size=test_size, random_state=42)
        return x_train, y_train, x_test, y_test
    
    def convert_bool_to_int(self):
        self.data.replace({False: 0, True: 1}, inplace=True)

class SyntheticDataset():
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