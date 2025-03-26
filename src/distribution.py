import numpy as np
from scipy.stats import weibull_min
from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter
from sklearn.mixture import GaussianMixture
from abc import abstractmethod
from scipy.stats import norm, lognorm, fisk, gumbel_r

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
    def fit_bootstrap(self, n_samples=100, percentage=0.8):
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

    def fit_bootstrap(self, y, n_samples=1000, percentage=1):
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

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return weibull_min.cdf(x, self.rho_, loc=0, scale=self.lambda_)

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


    def fit_bootstrap(self, y, n_samples=1000, percentage=1):
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

    def fit_bootstrap(self, y, n_samples=100, percentage=0.8):
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

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return lognorm.cdf(x, self.sigma_, loc=0, scale=np.exp(self.mu_))

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

class LogLogistic(Distribution):
    """
    Class to define the log-logistic distribution
    """
    def __init__(self):
        super().__init__()
        self.alpha_ = None
        self.beta_ = None
        self.fitter = LogLogisticFitter()

    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        llf = self._fit(times, events)

        self.alpha_ = llf.alpha_
        self.beta_ = llf.beta_

    def fit_bootstrap(self, y, n_samples=100, percentage=0.5):
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

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return fisk.cdf(x, self.alpha_, loc=0, scale=self.beta_)

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

class LogExtreme(Distribution):
    """
    Class to define the log-extreme (not with lifelines)
    """
    def __init__(self):
        super().__init__()
        self.scale_ = None

    def fit(self, y):
        """
        Fit the distribution to the data
        """
        self.y = y
        times, events = self.unpack_data(y)

        loc, scale = self._fit(times, events)

        self.scale_ = scale

    def fit_bootstrap(self, y, n_samples=100, percentage=0.8):
        """
        Fit the distribution to the data
        """
        self.y = y

        bootstrap_scales = []

        for _ in range(n_samples):
            sample_indices = np.random.choice(range(len(self.y)), int(percentage * len(self.y)), replace=True)

            resampled_y = self.y[sample_indices]

            resampled_times, resampled_events = self.unpack_data(resampled_y)

            loc, scale = self._fit(resampled_times, resampled_events)

            bootstrap_scales.append(scale)

        mean_scale = np.mean(bootstrap_scales)

        self.scale_ = mean_scale

    def _fit(self, times, events):
        """
        Fit the distribution to the data
        """
        loc, scale = gumbel_r.fit(times)
        return loc, scale

    def pdf(self, x):
        """
        Compute the probability density function
        """
        return gumbel_r.pdf(x, loc=0, scale=self.scale_)

    def cdf(self, x):
        """
        Compute the cumulative density function
        """
        return gumbel_r.cdf(x, loc=0, scale=self.scale_)

    def get_params(self):
        """
        Get the parameters of the distribution
        """
        params = {
            'scale': self.scale_
        }

        return params

    def set_params(self, params):
        """
        Set the parameters of the distribution
        """
        self.scale_ = params['scale']




