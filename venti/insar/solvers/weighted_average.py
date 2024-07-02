#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:40:54 2022

@author: govorcin

Need a littlbe bit more of cleaning

"""

import numpy as np

def weighted_average(data:np.array,
                     weights:np.array,
                     axis:np.int32 = 0) -> (np.array, np.array):
    '''
    Input:
        data:  data to average
        weights: weights 1/ var
        axis: average along axis
    Return:
        weighted_average
        weighted_std
    '''
    # multiply the data with weights, and sum along the axis
    wsum_data = np.nansum(data * weights, axis=axis)

    # remove 0 data
    wsum_data[wsum_data == 0.0] = np.nan

    weighted_average = wsum_data / np.nansum(weights, axis=axis)

    # Get count per data
    count = np.float32(np.count_nonzero(~np.isnan(data), axis=axis))
    ncount = count - 1
    # Set -1 values as 0
    ncount[ncount==-1] = 0

    # Get residuals
    residuals = data - weighted_average

    # Check if wsum_data is the same as se
    wsum_data = np.nansum((residuals)**2 * weights, axis=axis)
    wsum_data[wsum_data == 0.0] = np.nan

    sum_weights = np.nansum(weights, axis=0)
    sum_weights[sum_weights == 0.0] = np.nan

    weighted_var = wsum_data / (ncount * sum_weights)

    #need to do this as nansum give 0 for [nan, nan] change conj with power **2 as it is faster
    se = np.nansum(((residuals) * np.sqrt(weights)) * np.conjugate((residuals) * np.sqrt(weights)), axis=axis)
    #se = np.nansum(np.power(residuals * np.sqrt(weights),2), axis=axis) )

    se[se==0.0] = np.nan

    #Get mean square error
    mse = se / ncount

    weighted_std = np.sqrt(weighted_var / mse)

    return weighted_average, weighted_std