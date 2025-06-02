import numpy as np
import cupy as cp
import math
import numba as nb

@nb.jit(fastmath=True)
def norm_pdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    norm_pdf = np.exp(-0.5 * link_function**2) / (np.sqrt(2 * np.pi))
    return norm_pdf * 1/(sigma * y)

@nb.jit(fastmath=True)
def norm_cdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    norm_cdf = 0.5 * (1 + (math.erf(link_function / np.sqrt(2))))
    return norm_cdf

@nb.jit(fastmath=True)
def logistic_pdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    logistic_pdf = np.exp(link_function)/(1+np.exp(link_function))**2
    return logistic_pdf

@nb.jit(fastmath=True)
def logistic_cdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    logistic_cdf = np.exp(link_function)/(1+np.exp(link_function))
    return logistic_cdf

@nb.jit(fastmath=True)
def extreme_pdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    extreme_pdf = np.exp(link_function) * np.exp(-np.exp(link_function))
    return extreme_pdf

@nb.jit(fastmath=True)
def extreme_cdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    extreme_cdf = 1 - np.exp(-np.exp(link_function))
    return extreme_cdf
    

    