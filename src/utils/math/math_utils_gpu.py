import cupy as cp
from cupyx.scipy.special import erf

"""
This module contains GPU-accelerated mathematical functions for survival analysis.
It includes implementations of the normal, logistic, and extreme value distributions,
as well as their probability density functions (PDF) and cumulative distribution functions (CDF).
"""

def norm_pdf(y, pred, sigma):
    """
    Normal probability density function for survival analysis.
    Args:
        y (cupy.ndarray): The survival times.
        pred (cupy.ndarray): The predicted log survival times.
        sigma (float): The scale parameter of the normal distribution.
    Returns:
        cupy.ndarray: The probability density function values.
    """

    z  = (cp.log(y) - pred)/sigma
    norm_pdf = cp.exp(-0.5 * (z **2)) / (sigma * y * cp.sqrt(2 * cp.pi))
    return norm_pdf

def norm_cdf(y, pred, sigma):
    """
    Normal cumulative distribution function for survival analysis.
    Args:
        y (cupy.ndarray): The survival times.
        pred (cupy.ndarray): The predicted log survival times.
        sigma (float): The scale parameter of the normal distribution.
    Returns:
        cupy.ndarray: The cumulative distribution function values.
    """

    z  = (cp.log(y) - pred)/sigma
    norm_cdf = 0.5 * (1 + (erf(z / cp.sqrt(2))))
    return norm_cdf

def logistic_pdf(y, pred, sigma):
    """
    Logistic probability density function for survival analysis.
    Args:
        y (cupy.ndarray): The survival times.
        pred (cupy.ndarray): The predicted log survival times.
        sigma (float): The scale parameter of the logistic distribution.
    Returns:
        cupy.ndarray: The probability density function values.
    """
    z = (cp.log(y) - pred)/sigma
    exp_z = cp.exp(z)
    denom = (1 + exp_z)
    logistic_pdf = (exp_z / (sigma * y * denom ** 2))
    return logistic_pdf

def logistic_cdf(y, pred, sigma):
    """
    Logistic cumulative distribution function for survival analysis.
    Args:   
        y (cupy.ndarray): The survival times.
        pred (cupy.ndarray): The predicted log survival times.
        sigma (float): The scale parameter of the logistic distribution.
    Returns:
        cupy.ndarray: The cumulative distribution function values.
    """
    z = (cp.log(y) - pred)/sigma
    logistic_cdf = cp.where(
        z >= 0,
        1 / (1 + cp.exp(-z)),
        cp.exp(z) / (1 + cp.exp(z))
    )
    return logistic_cdf

def extreme_pdf(y, pred, sigma):
    """
    Extreme value probability density function for survival analysis.
    Args:
        y (cupy.ndarray): The survival times.
        pred (cupy.ndarray): The predicted log survival times.
        sigma (float): The scale parameter of the extreme value distribution.
    Returns:
        cupy.ndarray: The probability density function values.
    """
    z = (cp.log(y) - pred)/sigma
    exp_z = cp.exp(z)
    extreme_pdf = exp_z * cp.exp(-exp_z) / (sigma * y)
    return extreme_pdf

def extreme_cdf(y, pred, sigma):
    """
    Extreme value cumulative distribution function for survival analysis.
    Args:
        y (cupy.ndarray): The survival times.
        pred (cupy.ndarray): The predicted log survival times.
        sigma (float): The scale parameter of the extreme value distribution.
    Returns:
        cupy.ndarray: The cumulative distribution function values.
    """
    z = (cp.log(y) - pred)/sigma
    exp_z = cp.exp(z)
    extreme_cdf = 1 - cp.exp(-exp_z)
    return extreme_cdf

    