#!/usr/bin/python

import numpy as np


def func_gm1(dist, C0, d0):
    '''
    First-order Gauss-Markov process
    '''
    return C0 * np.exp(-1 * dist / d0)

def func_gm2(dist, C0, d0):
    '''
    Second-order Gauss-Markov process
    '''
    return C0 * np.exp(-1 * dist**2 / d0**2)

def func_reilly(dist, C0, d0):
    '''
    Reilly covariance function
    '''
    return C0 * (1 - 0.5 * (dist / d0)**2) * np.exp(-0.5 * (dist / d0)**2)

def func_markov1(dist, C0, d0):
    '''
    First-order Markov covariance function
    '''
    return C0 * (1 + (dist / d0)) * np.exp(-1 * dist / d0)

def func_markov2(dist, C0, d0):
    '''
    Second-order Markov covariance function
    '''
    return C0 * (1 + (dist / d0) + (dist**2 / (3 * d0**2))) * np.exp(-1 * dist / d0)

def func_tri(dist, C0, d0):
    '''
    Triangular covariance function
    '''
    return C0 * (1 - (dist / (2 * d0)))


def func_lauer(dist, C0, d0):
    '''
    Lauer covariance function
    '''
    return C0 / (dist**d0)


# Other with additional parameters

def func_gauss(dist, C0, alpha):
    '''
    Gaussian function with the factor alpha
    '''
    return C0 * np.exp(-1 * alpha**2 * dist**2)

def func_log(dist, C0, d0, m):
    '''
    Logarithmic covariance function (m=2 is Hirvonen function)
    '''
    return C0 * (d0**m / (d0**m + dist**m))

def func_vestol(dist, C0, a, b):
    '''
    Covariance function used by Olav Vestol with a=10/400^2 and b=8/400
    '''
    return C0 * ((a * dist**2) + (b * dist) + 1)