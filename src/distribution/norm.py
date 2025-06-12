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

class LogNormal(Distribution):
    """
    Class to define the log-normal  
    """
    def __init__(self):
        super().__init__()
        self.mu_ = None
        self.sigma_ = None
        self.fitter = LogNormalFitter()

    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        lnf = self._fit(times, events)

        self.mu_ = lnf.mu_
        self.sigma_ = lnf.sigma_

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

            lnf = self._fit(resampled_times, resampled_events)

            bootstrap_shapes.append(lnf.mu_)
            bootstrap_scales.append(lnf.sigma_)

        mean_shape = np.mean(bootstrap_shapes)
        mean_scale = np.mean(bootstrap_scales)

        self.mu_ = mean_shape
        self.sigma_ = mean_scale

    def _fit(self, times, events):
        """
        Fit the distribution to the data
        """
        lnf = self.fitter.fit(times, events)
        return lnf

    def pdf(self, x):
        """
        Compute the probability density function
        """
        return lognorm.pdf(x, self.sigma_, loc=0, scale=np.exp(self.mu_))

    def pdf_gpu(self, x):
        """
        Compute the probability density function using GPU
        """
        pdf = (1 / (x * self.sigma_ * cp.sqrt(2 * np.pi))) * cp.exp(-0.5 * ((cp.log(x) - self.mu_) / self.sigma_) ** 2)
        return pdf

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return lognorm.cdf(x, self.sigma_, loc=0, scale=np.exp(self.mu_))

    def cdf_gpu(self, x):
        """
        Compute the cumulative density function
        """
        cdf = 0.5 * (1 + erf((cp.log(x) - self.mu_) / (self.sigma_ * cp.sqrt(2))))
        return cdf

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'mu': self.mu_,
            'sigma': self.sigma_
        }

        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.mu_ = params['mu']
        self.sigma_ = params['sigma']