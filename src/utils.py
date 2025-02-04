import numpy as np
import math

def norm_pdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    norm_pdf = np.exp(-0.5 * link_function**2) / (np.sqrt(2 * np.pi))
    return norm_pdf * 1/(sigma * y)

def norm_cdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    norm_cdf = 0.5 * (1 + (math.erf(link_function / np.sqrt(2))))
    return norm_cdf

def logistic_pdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    logistic_pdf = np.exp(link_function)/(1+np.exp(link_function))**2
    return logistic_pdf

def logistic_cdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    logistic_cdf = np.exp(link_function)/(1+np.exp(link_function))
    return logistic_cdf

def extreme_pdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    extreme_pdf = np.exp(link_function) * np.exp(-np.exp(link_function))
    return extreme_pdf

def extreme_cdf(y, pred, sigma):
    link_function = (np.log(y) - pred)/sigma
    extreme_cdf = 1 - np.exp(-np.exp(link_function))
    return extreme_cdf
    

    