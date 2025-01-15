#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:56:38 2022

@author: govorcin

Model references:
    [1] pykrige: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html
    [2] sckit-gstat: https://scikit-gstat.readthedocs.io/en/latest/_modules/skgstat/Variogram.html#Variogram
      https://scikit-gstat.readthedocs.io/en/latest/reference/models.html?highlight=models#variogram-models



"""

import numpy as np

'''
Functions to correct uncertainity maps for the reference effect

SILL - value at which the model first flattens out
RANGE - the distance at which the model first falttens out
NUGGET - the value at which the semi-variogram (almost) intercepts the y-value

https://aegis4048.github.io/spatial-simulation-1-basics-of-variograms


'''
def get_model(model_name: str):
    models = ['spherical',
              'exponential',
              'gaussian',
              'cubic',
              'hole_effect',
              'power',
              'linear']

    # Select the model for variogram fit
    if model_name == 'spherical':
        model = spherical
    elif model_name == 'exponential':
        model = exponential
    elif model_name == 'gaussian':
        model = gaussian
    elif model_name == 'cubic':
        model = cubic
    elif model_name == 'power':
        model = power
    elif model_name == 'linear':
        model = linear
    elif model_name == 'hole_effect':
        model = hole_effect
    else:
        raise ValueError('Select one of available models: ', models)
    return model


def spherical(d, p, n, r):
    """
    Compute spherical variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: spherical variogram model
    """

    a = r / 1.

    if r>d.max():
        print(' Range is larger then max distance!')
        r=d.max()-1

    return np.where(d > r, p + n, n + p * (3/2 * (d / a) - (1/2 * ((d/a)**3))))


def exponential(d, p, n, r):
    """
    Compute Exponential variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: spherical variogram model
    """

    a = r / 3.

    if r>d.max():
        print(' Range is larger then max distance!')
        r=d.max()-1

    return n + p * (1 - np.exp(-d/a))

def gaussian(d, p, n, r):
    """
    Compute gausian variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: spherical variogram model
    """

    a = r / 2.

    if r>d.max():
        print(' Range is larger then max distance!')
        r=d.max()-1

    return  n + p * (1 - np.exp(-np.square(d) / a**2))


def cubic(d, p, n, r):
    """
    Compute Cubic variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: spherical variogram model
    """

    a = r / 1.

    if r>d.max():
        print(' Range is larger then max distance!')
        r=d.max()-1

    return np.where(d >= r, n + p, n + p * ((7 - ( d**2 / a**2))
                                    - ((35/4) * (d**3 / a**3))
                                    + ((7/2) * (d**5 / a**5))
                                    - ((3/4) * (d**7 / a ** 7))))

def hole_effect(d, p, n, r):
    """Hole Effect model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: Hole effect variogram model
    """
    return n + p * (1.0 - (1.0 - d / (r/ 3.0)) * np.exp(-d / (r / 3.0)))

def power(d, s, n, r):
    """Power model,
    @param d: 1D distance array
    @param s: scale
    @param n: nugget
    @param e: exponent
    @return: Power variogram model
    """
    a = r / 1000. #exponent

    return n + s * d ** a

def linear(d, s, n, r=0):
    """Linear model,
    @param d: 1D distance array
    @param s: slope
    @param n: nugget
    @return: Linear variogram model
    """
    return n + s * d

