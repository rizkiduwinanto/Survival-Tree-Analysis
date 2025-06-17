from abc import ABC, abstractmethod

class Wrapper(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        """
        Fit the model to the training data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict survival times or event indicators for the test data.
        """
        pass

    @abstractmethod
    def _score(self, X, y):
        """
        Evaluate the model on the test data and return a score.
        """
        pass

    @abstractmethod
    def _brier(self, X, y):
        """
        Calculate the Brier score for the model predictions.
        """
        pass

    @abstractmethod
    def _auc(self, X, y):
        """
        Calculate the Area Under the Curve (AUC) for the model predictions.
        """
        pass

    @abstractmethod
    def _mae(self, X, y):
        """
        Calculate the Mean Absolute Error (MAE) for the model predictions.
        """
        pass