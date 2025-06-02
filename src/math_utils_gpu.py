import cupy as cp
from cupyx.scipy.special import erf

def norm_pdf(y, pred, sigma):
    link_function = (cp.log(y) - pred)/sigma
    norm_pdf = cp.exp(-0.5 * link_function**2) / (cp.sqrt(2 * cp.pi))
    return norm_pdf * 1/(sigma * y)

def norm_cdf(y, pred, sigma):
    link_function = (cp.log(y) - pred)/sigma
    norm_cdf = 0.5 * (1 + (erf(link_function / cp.sqrt(2))))
    return norm_cdf

def logistic_pdf(y, pred, sigma):
    link_function = (cp.log(y) - pred)/sigma
    logistic_pdf = cp.exp(link_function)/(1+cp.exp(link_function))**2
    return logistic_pdf

def logistic_cdf(y, pred, sigma):
    link_function = (cp.log(y) - pred)/sigma
    logistic_cdf = cp.exp(link_function)/(1+cp.exp(link_function))
    return logistic_cdf

def extreme_pdf(y, pred, sigma):
    link_function = (cp.log(y) - pred)/sigma
    extreme_pdf = cp.exp(link_function) * cp.exp(-cp.exp(link_function))
    return extreme_pdf

def extreme_cdf(y, pred, sigma):
    link_function = (cp.log(y) - pred)/sigma
    extreme_cdf = 1 - cp.exp(-cp.exp(link_function))
    return extreme_cdf

    