import numpy as np
import cupy as cp
from cupyx.scipy.special import erf
from scipy.stats import weibull_min
from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter
from sklearn.mixture import GaussianMixture
from abc import abstractmethod
from scipy.stats import norm, lognorm, fisk, gumbel_r
import warnings
from .distribution import Distribution
from lifelines.exceptions import ApproximationWarning

class LogLogistic(Distribution):
    """
    Class to define the log-logistic distribution
    """
    def __init__(self):
        super().__init__()
        self.alpha_ = None
        self.beta_ = None
        self.fitter = LogLogisticFitter()
        self.epsilon = 1e-10

    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        llf = self._fit(times, events)

        self.alpha_ = llf.alpha_
        self.beta_ = llf.beta_

    def fit_bootstrap(self, y, n_samples=1000, percentage=0.5):
        """
        Fit the distribution to the data
        """
        self.y = y

        bootstrap_shapes = []
        bootstrap_scales = []

        for _ in range(n_samples):
            sample_indices = np.random.choice(range(len(self.y)), int(percentage * len(self.y)), replace=True)
            
            resampled_y = self.y[sample_indices]

            resampled_times, resampled_events = self.unpack_data(resampled_y)

            llf = self._fit(resampled_times, resampled_events)

            bootstrap_shapes.append(llf.alpha_)
            bootstrap_scales.append(llf.beta_)

        mean_shape = np.mean(bootstrap_shapes)
        mean_scale = np.mean(bootstrap_scales)

        self.alpha_ = mean_shape
        self.beta_ = mean_scale

    def _fit(self, times, events):
        """
        Fit the distribution to the data
        """
        llf = self.fitter.fit(times, events)
        return llf

    def pdf(self, x):
        """
        Compute the probability density function
        """
        return fisk.pdf(x, self.alpha_, loc=0, scale=self.beta_)

    def pdf_gpu(self, x):
        """
        Compute the probability density function using GPU
        """
        x = cp.maximum(x, self.epsilon)
        z = x / self.beta_
        pdf = (self.alpha_ / self.beta_) * z ** (self.alpha_ - 1) / (1 + z ** self.alpha_) ** 2
        return pdf

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return fisk.cdf(x, self.alpha_, loc=0, scale=self.beta_)

    def cdf_gpu(self, x):
        """
        Compute the cumulative density function
        """
        x = cp.maximum(x, self.epsilon)
        z = x / self.beta_
        cdf = 1 / (1 + z ** self.alpha_)
        return cdf

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'alpha': self.alpha_,
            'beta': self.beta_
        }

        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.alpha_ = params['alpha']
        self.beta_ = params['beta']