#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:56:38 2022

@author: govorcin
"""

import numpy as np

'''
Functions to correct uncertainity maps for the reference effect

SILL - value at which the model first flattens out
RANGE - the distance at which the model first falttens out
NUGGET - the value at which the semi-variogram (almost) intercepts the y-value

https://aegis4048.github.io/spatial-simulation-1-basics-of-variograms


pykrige
def linear_variogram_model(m, d):
    """Linear model, m is [slope, nugget]"""
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def power_variogram_model(m, d):
    """Power model, m is [scale, exponent, nugget]"""
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d ** exponent + nugget
def hole_effect_variogram_model(m, d):
    """Hole Effect model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return (
        psill * (1.0 - (1.0 - d / (range_ / 3.0)) * np.exp(-d / (range_ / 3.0)))
        + nugget
    )

'''

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
        
    return np.where(d > r, p + n, n + p * (3/2 * (d / a) - (1/2 * ((d / a) ** 3 ))))


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

    return  n + p * (1 - np.exp(-np.square(d) / a ** 2))


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

    return np.where(d >= r, n + p, n + p * ((7 - ( d ** 2 / a ** 2)) -
                                   ((35 / 4) * (d ** 3 / a ** 3)) + 
                                   ((7 /2) * (d ** 5 / a ** 5)) - 
                                   ((3 /4) * (d ** 7 / a ** 7))))