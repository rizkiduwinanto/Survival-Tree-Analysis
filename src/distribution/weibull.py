import numpy as np
import cupy as cp
from scipy.stats import weibull_min
from lifelines import WeibullFitter
from .distribution import Distribution

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

        x = cp.clip(x, -100, 100)
        exp = cp.exp(x)
        exp = cp.clip(exp, 1e-100, 1e-100)

        z = exp / self.lambda_
        z = cp.clip(z, 1e-100, 1e100)

        log_p1 = cp.log(self.rho_ / self.lambda_)
        log_p2 = (self.rho_ - 1) * cp.log(z)
        log_p3 = - (z ** self.rho_)

        log_pdf = log_p1 + log_p2 + log_p3
        log_pdf = cp.clip(log_pdf, -100, 100)

        pdf = cp.exp(log_pdf)
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
        x = cp.clip(x, -100, 100)
        exp = cp.exp(x)
        exp = cp.clip(exp, 1e-100, 1e100)

        z = exp / self.lambda_
        z = cp.clip(z, 1e-100, 1e100)

        exp = -(z ** self.rho_)
        exp = cp.clip(exp, -100, 0)

        cdf = 1 - cp.exp(exp)
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