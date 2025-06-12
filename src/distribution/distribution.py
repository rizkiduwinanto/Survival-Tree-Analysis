import numpy as np
import cupy as cp
from cupyx.scipy.special import erf
from scipy.stats import weibull_min
from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter
from sklearn.mixture import GaussianMixture
from abc import abstractmethod
from scipy.stats import norm, lognorm, fisk, gumbel_r
import warnings
from lifelines.exceptions import ApproximationWarning

warnings.filterwarnings("ignore", category=ApproximationWarning)

class Distribution():
    """
    Class to define the distribution of the target variable
    """
    def __init__(self):
        self.y = None

    @staticmethod
    def unpack_data(y):
        """
        Load the data
        """
        times = list(list(zip(*y))[1])
        events = list(list(zip(*y))[0])

        return times, events

    @abstractmethod
    def fit(self, data):
        """
        Fit the distribution to the data
        """
        pass

    @abstractmethod
    def fit_bootstrap(self, n_samples=1000, percentage=0.5):
        """
        Fit the distribution to the data
        """
        pass

    @abstractmethod
    def _fit(self, times, events):
        """
        Fit the distribution to the data
        """
        pass

    @abstractmethod
    def pdf(self, x):
        """
        Compute the probability density function
        """ 
        pass

    @abstractmethod
    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Get the parameters of the distribution
        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        pass 


