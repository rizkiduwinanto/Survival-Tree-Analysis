from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    Abstract base class for datasets.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def create_label(self):
        """
        Create the label for the dataset.
        """
        pass

    @abstractmethod
    def create_xgboost_label(self):
        """
        Create the label for XGBoost.
        """
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        Preprocess the dataset.
        """
        pass

    @abstractmethod
    def get_train_test(self, test_size=0.2, random_state=42):
        """
        Get the train and test splits of the dataset.
        """
        pass

    @abstractmethod
    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        """
        Get the train and test splits of the dataset for XGBoost.
        """
        pass

    @abstractmethod
    def get_data(self):
        """
        Get the dataset.
        """
        pass

    @abstractmethod
    def get_label(self):
        """
        Get the label of the dataset.
        """
        pass

    @abstractmethod
    def get_xgboost_label(self):
        """
        Get the XGBoost label of the dataset.
        """
        pass





