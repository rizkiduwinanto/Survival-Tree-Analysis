# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as cnp
cnp.import_array()

from abc import ABC, abstractmethod
from scipy.stats import weibull_min
from lifelines import WeibullFitter

cdef class Distribution():
    cdef:
        public int[:,:] y

    def __init__(self):
        pass

    @staticmethod
    cdef unpack_data(int[:,:] y):
        cdef int n = len(y)
        cdef int[:] times = np.zeros(n, dtype=np.int32)
        cdef int[:] events = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            times[i] = y[i,0]
            events[i] = y[i,1]
        
        return times, events

    @abstractmethod
    def fit(self, int[:,:] y):
        pass

    @abstractmethod
    cdef fit_bootstrap(self, int[:,:] y, int n_samples, double percentage):
        pass

    @abstractmethod
    cdef object _fit(self, int[:] times, int[:] events):
        pass

    @abstractmethod
    cdef double pdf(self, cnp.ndarray x):
        pass

    @abstractmethod
    cdef double cdf(self, cnp.ndarray x):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

cdef class Weibull(Distribution):
    cdef:
        double rho_
        double lambda_
        public object weibull_fitter

    def __init__(self):
        super().__init__()
        self.rho_ = 0
        self.lambda_ = 0
        self.fitter = WeibullFitter()

    def fit(self, int[:,:] y):
        cdef int[:] times, events = self.unpack_data(y)
        cdef object wf

        self.y = y
        wf = self._fit(times, events)
        self.rho_ = wf.rho_
        self.lambda_ = wf.lambda_

    cdef fit_bootstrap(self, int[:,:] y, int n_samples=100, double percentage=0.8):
        self.y = y

        cdef int[:] bootstrap_shape = np.zeros(n_samples, dtype=np.int32)
        cdef int[:] bootstrap_scale = np.zeros(n_samples, dtype=np.int32)
        cdef int[:] bootstrap_data, times, events

        cdef object wf

        for i in range(n_samples):
            bootstrap_data = np.random.choice(self.y, int(len(self.y) * percentage), replace=True)
            times, events = self.unpack_data(y)
            wf = self._fit(times, events)
            bootstrap_shape[i] = wf.rho_
            bootstrap_scale[i] = wf.lambda_
        
    cdef object _fit(self, int[:] times, int[:] events):
        return self.fitter.fit(times, events)

    cdef double pdf(self, cnp.ndarray x):
        return weibull_min.pdf(x, self.rho_, scale=self.lambda_)

    cdef double cdf(self, cnp.ndarray x):
        return weibull_min.cdf(x, self.rho_, scale=self.lambda_)

    def get_params(self):
        params = {
            'rho': self.rho_,
            'lambda': self.lambda_
        }
        return params

    def set_params(self, params):
        self.rho_ = params['rho']
        self.lambda_ = params['lambda']