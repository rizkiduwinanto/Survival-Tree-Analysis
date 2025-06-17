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

class GMM_New(Distribution):
    """
    Class to define the Gaussian Mixture Model
    """
    def __init__(self, n_components=10):
        super().__init__()
        self.n_components = n_components

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

        self.means_cdf_ = None
        self.covariances_cdf_ = None
        self.weights_cdf_ = None
        
        self.fitter = GaussianMixture(n_components=n_components)
        self.fitter_cdf = GaussianMixture(n_components=n_components)
        self.gmm = None

    def fit(self, y):
        """
        Fit the distribution to the data
        """

        self.y = y
        times, events = self.unpack_data(y)

        uncensored_times = np.array(times)[np.array(events) == 1]
        gmm = self.fitter.fit(uncensored_times.reshape(-1, 1))

        self.means_ = cp.array(gmm.means_)
        self.covariances_ = cp.array(gmm.covariances_)
        self.weights_ = cp.array(gmm.weights_)

        times = np.array(times)
        gmm_cdf = self.fitter_cdf.fit(times.reshape(-1, 1))
        self.means_cdf_ = cp.array(gmm_cdf.means_)
        self.covariances_cdf_ = cp.array(gmm_cdf.covariances_)
        self.weights_cdf_ = cp.array(gmm_cdf.weights_)

    def fit_bootstrap(self, y, n_samples=1000, percentage=0.5):
        """
        Fit the distribution to the data
        """
        self.y = y

        bootstrap_pdf_means = []
        bootstrap_pdf_covs = []
        bootstrap_pdf_weights = []

        bootstrap_cdf_means = []
        bootstrap_cdf_covs = []
        bootstrap_cdf_weights = []

        for _ in range(n_samples):
            sample_indices = np.random.choice(range(len(self.y)), int(percentage * len(self.y)), replace=True)
            
            resampled_y = self.y[sample_indices]

            resampled_times, resampled_events = self.unpack_data(resampled_y)

            uncensored_times = np.array(resampled_times)[np.array(resampled_events) == 1]
             
            if len(uncensored_times) > 0:
                gmm_pdf = self.fitter.fit(uncensored_times.reshape(-1, 1))
                bootstrap_pdf_means.append(gmm_pdf.means_)
                bootstrap_pdf_covs.append(gmm_pdf.covariances_)
                bootstrap_pdf_weights.append(gmm_pdf.weights_)

            resampled_times = np.array(resampled_times)
            gmm_cdf = self.fitter_cdf.fit(resampled_times.reshape(-1, 1))
            bootstrap_cdf_means.append(gmm_cdf.means_)
            bootstrap_cdf_covs.append(gmm_cdf.covariances_)
            bootstrap_cdf_weights.append(gmm_cdf.weights_)

        mean_pdf_means = np.mean(bootstrap_pdf_means, axis=0)
        mean_pdf_covs = np.mean(bootstrap_pdf_covs, axis=0)
        mean_pdf_weights = np.mean(bootstrap_pdf_weights, axis=0)

        mean_cdf_means = np.mean(bootstrap_cdf_means, axis=0)
        mean_cdf_covs = np.mean(bootstrap_cdf_covs, axis=0)
        mean_cdf_weights = np.mean(bootstrap_cdf_weights, axis=0)

        self.means_ = cp.array(mean_pdf_means)
        self.covariances_ = cp.array(mean_pdf_covs)
        self.weights_ = cp.array(mean_pdf_weights)

        self.means_cdf_ = cp.array(mean_cdf_means)
        self.covariances_cdf_ = cp.array(mean_cdf_covs)
        self.weights_cdf_ = cp.array(mean_cdf_weights)

    @staticmethod
    def norm_pdf_gpu(x, mean, cov):
        """
        Compute the probability density function
        """
        cov = cov.reshape(-1)
        std_dev = cp.sqrt(cov)

        x = x.reshape(-1, 1)
        mean = mean.reshape(1, -1)
        std_dev = std_dev.reshape(1, -1)

        pdf = (1 / (std_dev * cp.sqrt(2 * np.pi))) * cp.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        return pdf

    def pdf_gpu(self, x):
        """
        Compute the probability density function using GPU
        """
        means = self.means_
        covs = self.covariances_
        weights = self.weights_.reshape(1, -1)

        weighted_pdfs = self.norm_pdf_gpu(x, means, covs) * weights
        pdf = cp.sum(weighted_pdfs, axis=1)
        return pdf

    @staticmethod
    def norm_cdf_gpu(x, mean, cov):
        """
        Compute the cumulative density function
        """
        cov = cov.reshape(-1)
        std_dev = cp.sqrt(cov)

        x = x.reshape(-1, 1)
        mean = mean.reshape(1, -1)
        std_dev = std_dev.reshape(1, -1)

        cdf = 0.5 * (1 + erf((x - mean) / (std_dev * cp.sqrt(2))))
        return cdf

    def cdf_gpu(self, x):
        """
        Compute the cumulative density function using GPU
        """
        means = self.means_cdf_
        covs = self.covariances_cdf_
        weights = self.weights_cdf_.reshape(1, -1)

        weighted_cdfs = self.norm_cdf_gpu(x, means, covs) * weights
        cdf = cp.sum(weighted_cdfs, axis=1)
        return cdf

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'means': self.means_.tolist(),
            'covariances': self.covariances_.tolist(),
            'weights': self.weights_.tolist(),
            'n_components': self.n_components
        }
        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.means_ = np.array(params['means'])
        self.covariances_ = np.array(params['covariances'])
        self.weights_ = np.array(params['weights'])
        self.n_components = params['n_components']

class GMM(Distribution):
    """
    Class to define the Gaussian Mixture Model
    """
    def __init__(self, n_components=10):
        super().__init__()
        self.n_components = n_components

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

        self.means_gpu_ = None
        self.covariances_gpu_ = None
        self.weights_gpu_ = None
        
        self.fitter = GaussianMixture(n_components=n_components)
        self.gmm = None

    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        gmm = self._fit(times, events)

        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.weights_ = gmm.weights_

        self.means_gpu_ = cp.array(gmm.means_)
        self.covariances_gpu_ = cp.array(gmm.covariances_)
        self.weights_gpu_ = cp.array(gmm.weights_)

    def fit_bootstrap(self, y, n_samples=1000, percentage=0.5):
        """
        Fit the distribution to the data
        """
        self.y = y

        bootstrap_means = []
        bootstrap_covs = []
        bootstrap_weights = []

        for _ in range(n_samples):
            sample_indices = np.random.choice(range(len(self.y)), int(percentage * len(self.y)), replace=True)
            
            resampled_y = self.y[sample_indices]

            resampled_times, resampled_events = self.unpack_data(resampled_y)

            gmm = self._fit(resampled_times, resampled_events)

            bootstrap_means.append(gmm.means_)
            bootstrap_covs.append(gmm.covariances_)
            bootstrap_weights.append(gmm.weights_)

        mean_means = np.mean(bootstrap_means, axis=0)
        mean_covs = np.mean(bootstrap_covs, axis=0)
        mean_weights = np.mean(bootstrap_weights, axis=0)

        self.means_ = mean_means
        self.covariances_ = mean_covs   
        self.weights_ = mean_weights

        self.means_gpu_ = cp.array(mean_means)
        self.covariances_gpu_ = cp.array(mean_covs)
        self.weights_gpu_ = cp.array(mean_weights)

    def _fit(self, times, events):
        """
        Fit the distribution to the data
        """
        times = np.array(times)
        return self.fitter.fit(times.reshape(-1, 1))

    def mix_norm_pdf(self, x):
        """
        Compute the probability density function
        """
        pdf_value = 0.0
        for mean, cov, weight in zip(self.means_, self.covariances_, self.weights_):
            std_dev = np.sqrt(cov)
            pdf_value += weight * norm.pdf(x, loc=mean, scale=std_dev)
        return pdf_value

    def pdf(self, x):
        """
        Compute the probability density function
        """
        return self.mix_norm_pdf(x)

    def _pdf(self, x):
        """
        Compute the probability density function (only for non bootstrap)
        """
        return np.exp(self.gmm.score_samples(x.reshape(-1, 1)))

    @staticmethod
    def norm_pdf_gpu(x, mean, cov):
        """
        Compute the probability density function
        """
        cov = cov.reshape(-1)
        std_dev = cp.sqrt(cov)

        x = x.reshape(-1, 1)
        mean = mean.reshape(1, -1)
        std_dev = std_dev.reshape(1, -1)

        pdf = (1 / (std_dev * cp.sqrt(2 * np.pi))) * cp.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        return pdf

    def pdf_gpu(self, x):
        """
        Compute the probability density function using GPU
        """
        means = self.means_gpu_
        covs = self.covariances_gpu_
        weights = self.weights_gpu_.reshape(1, -1)

        weighted_pdfs = self.norm_pdf_gpu(x, means, covs) * weights
        pdf = cp.sum(weighted_pdfs, axis=1)
        return pdf

    def mix_norm_cdf(self, x):
        """
        Compute the cumulative density function
        """
        cdf_value = 0.0
        for mean, cov, weight in zip(self.means_, self.covariances_, self.weights_):
            std_dev = np.sqrt(cov)
            cdf_value += weight * norm.cdf(x, loc=mean, scale=std_dev)
        return cdf_value

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return self.mix_norm_cdf(x)

    @staticmethod
    def norm_cdf_gpu(x, mean, cov):
        """
        Compute the cumulative density function
        """
        cov = cov.reshape(-1)
        std_dev = cp.sqrt(cov)

        x = x.reshape(-1, 1)
        mean = mean.reshape(1, -1)
        std_dev = std_dev.reshape(1, -1)

        cdf = 0.5 * (1 + erf((x - mean) / (std_dev * cp.sqrt(2))))
        return cdf

    def cdf_gpu(self, x):
        """
        Compute the cumulative density function using GPU
        """
        means = self.means_gpu_
        covs = self.covariances_gpu_
        weights = self.weights_gpu_.reshape(1, -1)

        weighted_cdfs = self.norm_cdf_gpu(x, means, covs) * weights
        cdf = cp.sum(weighted_cdfs, axis=1)
        return cdf

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'means': self.means_.tolist(),
            'covariances': self.covariances_.tolist(),
            'weights': self.weights_.tolist(),
            'n_components': self.n_components
        }
        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.means_ = np.array(params['means'])
        self.covariances_ = np.array(params['covariances'])
        self.weights_ = np.array(params['weights'])
        self.n_components = params['n_components']