#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:56:38 2022

@author: govorcin
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, optimize

def diagnostic_scatter_plots(residuals, fitted_obs, mse, hii, n_x=None):
    #### Diagnostic scatter plots

    ## Residuals vs Fitted values
    '''
    The plot is used to detect non-linearity, unequal error variances, and outliers
    
    
    '''
    fig, ax = plt.subplots(2,2, figsize=(8,6))
    ax[0,0].scatter(fitted_obs, residuals, s=10, c='black',marker='o', alpha=0.1)
    ax[0,0].set_xlabel('Fitted values')
    ax[0,0].set_ylabel('Residuals')
    ax[0,0].set_title('Residuals vs Fitted')

    ## Stand. residuals vs fitted  - this is acctually studenized residuals
    stand_res = np.squeeze(residuals) / np.sqrt(mse * (1 - hii))
    
    ## cook distance 
    ## Eubank, R. L. (1999). Nonparametric regression and spline smoothing. CRC press., p 93,94
    if n_x is not None:
        cook_distance = (stand_res**2) / n_x
        cook_distance *= hii / (1 - hii)
        pvals = stats.f.sf(cook_distance, n_x, residuals.shape[0] - n_x)
    '''
    the scale-location plot is a more sensitive approach to looking for deviations 
    from the constant variance assumption. If you see significant trends in the red line on this plot, 
    it tells you that the residuals (and hence errors) have non-constant variance. 
    '''
    ax[0,1].scatter(fitted_obs, np.sqrt(np.abs(stand_res)), s=10, c='blue',marker='o', alpha=0.1)
    ax[0,1].set_xlabel('Fitted values')
    ax[0,1].set_ylabel('Sqrt. Stand. Residuals')
    ax[0,1].set_title('Scale - Location')

    ## Resdiuals vs Leverage
    ax[1,0].scatter(hii, stand_res, s=10, c='black',marker='o', alpha=0.1)
    ax[1,0].set_xlabel('Leverage')
    ax[1,0].set_ylabel('Stand. Residuals')
    ax[1,0].set_title('Residual vs Leverage')

    ## Normal QQ Plot
    '''
    The Normal QQ plot helps us to assess whether the residuals are roughly normally distributed.
    theoretical percentiles of the normal distribution versus the observed sample 
    percentiles of the residuals should be approximately linear. If a normal probability plot of the residuals 
    is approximately linear, we proceed assuming that the error terms are normally distributed
    '''
    stats.probplot(np.squeeze(residuals), plot=ax[1,1])
    fig.tight_layout()
    
    # Variance of residuals
    res_var = mse * (1 - hii)


'''
https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
L-1 norm : sum(abs(y - y_fit)) sum of absolute residualts 
L-2 norm : (y-y_fit).T * W * (y-y_fit) v.T * P v
normalized L2-norm : L2 / (y.T * W * y)

rms = sqrt( ((y-y_fit).T  * (y-y_fit)) / n)

'''