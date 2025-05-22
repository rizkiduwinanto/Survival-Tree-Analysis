from cupy import cupy as cp

class DistributionFitter():
    def __init__(self, data):
        self.data = data

    def fit(self, distribution):
        # Fit the distribution to the data
        params = distribution.fit(self.data)
        return params

    def pdf(self, x, distribution, params):
        # Calculate the probability density function
        return distribution.pdf(x, *params)

    def cdf(self, x, distribution, params):
        # Calculate the cumulative distribution function
        return distribution.cdf(x, *params)


