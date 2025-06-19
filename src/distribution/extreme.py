import numpy as np
import cupy as cp
from scipy.stats import gumbel_l
from .distribution import Distribution

class LogExtremeNew(Distribution):
    """
    Class to define the log-extreme distribution (not with lifelines)
    """
    def __init__(self):
        super().__init__()
        self.scale_ = None

        self.scale_cdf_ = None

    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        uncensored_times = np.array(times)[np.array(events) == 1]
        loc, scale = self._fit(uncensored_times)

        times = np.array(times)
        loc_cdf, scale_cdf = self._fit(times)

        self.scale_ = scale
        self.scale_cdf_ = scale_cdf

    def _fit(self, times):
        """
        Fit the distribution to the data
        """
        loc, scale = gumbel_l.fit(times)
        return loc, scale

    def fit_bootstrap(self, y, n_samples=1000, percentage=0.5):
        """
        Fit the distribution to the data using bootstrap sampling
        """
        self.y = y

        bootstrap_scales = []
        bootstrap_scales_cdf = []

        for _ in range(n_samples):
            sample_indices = np.random.choice(range(len(self.y)), int(percentage * len(self.y)), replace=True)

            resampled_y = self.y[sample_indices]

            resampled_times, resampled_events = self.unpack_data(resampled_y)

            loc, scale = self._fit(resampled_times[resampled_events == 1])
            loc_cdf, scale_cdf = self._fit(resampled_times)

            bootstrap_scales.append(scale)
            bootstrap_scales_cdf.append(scale_cdf)

        mean_scale = np.mean(bootstrap_scales)
        mean_scale_cdf = np.mean(bootstrap_scales_cdf)

        self.scale_ = mean_scale
        self.scale_cdf_ = mean_scale_cdf

    def pdf(self, x):
        """
        Compute the probability density function
        """
        return gumbel_l.pdf(x, loc=0, scale=self.scale_)

    def pdf_gpu(self, x):
        """
        Compute the probability density function on GPU
        """
        z = x / self.scale_
        exp_z = cp.exp(z)
        pdf = (1/self.scale_) * exp_z * cp.exp(-exp_z)
        return pdf

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return gumbel_l.cdf(x, loc=0, scale=self.scale_cdf_)

    def cdf_gpu(self, x):
        """
        Compute the cumulative density function
        """
        z = x / self.scale_cdf_
        exp_z = cp.exp(z)
        cdf = 1 - cp.exp(-exp_z)
        return cdf

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'scale': self.scale_,
            'scale_cdf': self.scale_cdf_
        }

        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.scale_ = params['scale']
        self.scale_cdf_ = params['scale_cdf']