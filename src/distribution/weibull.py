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

class Weibull(Distribution):
    """
    Class to define the Weibull distribution
    """
    def __init__(self):
        super().__init__()
        self.rho_ = None
        self.lambda_ = None
        self.fitter = WeibullFitter()
        
    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        wf = self._fit(times, events)

        self.rho_ = wf.rho_
        self.lambda_ = wf.lambda_

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

            wf = self._fit(resampled_times, resampled_events)

            bootstrap_shapes.append(wf.rho_)
            bootstrap_scales.append(wf.lambda_)

        mean_shape = np.mean(bootstrap_shapes)
        mean_scale = np.mean(bootstrap_scales)

        self.rho_ = mean_shape
        self.lambda_ = mean_scale

    def _fit(self, times, events):
        """
        Fit the distribution to the data
        """
        wf = self.fitter.fit(times, events)
        return wf

    def pdf(self, x):
        """
        Compute the probability density function
        """
        return weibull_min.pdf(x, self.rho_, loc=0, scale=self.lambda_)

    def pdf_gpu(self, x):
        """
        Compute the probability density function using GPU
        """
        pdf = (self.rho_ / self.lambda_) * (x / self.lambda_) ** (self.rho_ - 1) * cp.exp(- (x / self.lambda_) ** self.rho_)
        return pdf

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return weibull_min.cdf(x, self.rho_, loc=0, scale=self.lambda_)

    def cdf_gpu(self, x):
        """
        Compute the cumulative density function
        """
        cdf = 1 - cp.exp(- (x / self.lambda_) ** self.rho_)
        return cdf

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'rho': self.rho_,
            'lambda': self.lambda_
        }

        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.rho_ = params['rho']
        self.lambda_ = params['lambda']