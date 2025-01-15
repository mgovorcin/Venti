#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:40:54 2022

@author: govorcin

NOTES:
    Power, Linear, Hole_effect models need a little bit more work to be fully
    included in general way, avoid using them for now (especially power)

"""



import numpy as np
from typing import Tuple
from pyproj import Geod
from scipy import stats, optimize
from matplotlib import pyplot as plt
from . import models


def calc_variogram(data: np.array,
                   lats: np.array,
                   lons: np.array,
                   ref_lat: np.float32,
                   ref_lon: np.float32,
                   bin_size: np.int32 = 100,
                   max_range: np.float32 = None,
                   variogram_model: str = 'exponential',
                   debug_plot: bool = False,
                   print_msg: bool = True) -> Tuple[float, float, float]:

    vprint = print if print_msg else lambda *args, **kwargs: None

    # Create reference lat. lon arrays
    ref_lats = np.ones(lats.shape) * np.float32(ref_lat)
    ref_lons = np.ones(lons.shape) * np.float32(ref_lon)

    # Caculate the distance array from ref lat, lon
    geod = Geod(ellps="WGS84")
    distance_array = geod.inv(ref_lons, ref_lats, lons, lats)[2] / 1000 # to get in km

    #Find the nan in data array
    mask = ~np.isnan(data)

    #Calculate binned statistics (median and std) for n bins along both x and y
    # to find the initial values for variogram fitting
    # partial sill :: last value of bins_median array
    # nugget       :: first value of bins_median array
    # range        :: mid point of bins_center bincenter[len(bincenter)//2]

    bins_median, bin_edges = stats.binned_statistic(distance_array[mask],
                                                    data[mask],
                                                    'median',
                                                    bin_size)[:-1]

    bins_std = stats.binned_statistic(distance_array[mask],
                                      data[mask],
                                      'std',
                                      bin_size)[0]

    #Get the center of binedegs
    bincenter_array = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create var that correlates with distance
    sigma = bins_std + bincenter_array / max(bincenter_array)

    # Find the best bit variogram
    # p0 - inital values for parameters p, n, r
    if max_range == None or bincenter_array[-1] < max_range:
        max_range = bincenter_array[-1]

    # To limit the max_range as if it too small the optimization will fail
    #elif bincenter_array[-1] > max_range * 1.5:
    #    max_range = bincenter_array[-1] / 1.5
    msg = f'Variogram Model: {variogram_model}'
    msg += f'\nMax range: {max_range:.2f}'
    msg += f'\nBin size: {bin_size}'
    vprint(msg)

    #Get the variogram model
    model = models.get_model(variogram_model)

    #Fit Variogram to binned data
    coef, cov = optimize.curve_fit(model,
                                   bincenter_array,
                                   bins_median,
                                   p0=(bins_median[-1],
                                       bins_median[0],
                                       bincenter_array[len(bincenter_array)//2]),
                                   sigma=sigma,
                                   bounds=(0, max_range))

    #get the sill value from the coeffs: partial sill + nugget
    sill = coef[0] + coef[1]

    # Predicted values with rmse and r-square of the fit
    bin_hat = model(bincenter_array, coef[0], coef[1], coef[2])
    # if weights are used than they need to be included in the estimate of rmse
    # 1/n - n_u * sum(w * (predicted -actual)**2 / sum(w))
    # w = (1/sigma**2)
    rmse = np.sqrt(np.sum((bin_hat - bins_median)**2 / (len(bin_hat)-3)))
    sst = np.sum((bins_median - bins_median.mean())**2)
    ssr = np.sum((bin_hat - bins_median.mean())**2)
    r_square = ssr / sst
    vprint(f'Variogram fit R2={r_square:.2f}, RMSE={rmse:.2f}')

    if debug_plot:
        import random

        #mask nan in distance and data array
        ma_distance = distance_array[mask]
        ma_data = data[mask]

        subset = random.sample(range(1, len(ma_distance)), 50000)

        fig, ax = plt.subplots(1,1)
        ax.scatter(ma_distance[subset], ma_data[subset], c=ma_data[subset],
                   s=0.1, vmin=0, vmax=np.quantile(ma_data[subset], 0.75), cmap='GnBu')
        ax.plot(bincenter_array, bins_median, linewidth=2, c="gold")
        ax.plot(bincenter_array, bins_median + bins_std, linewidth=1, c="gold")
        ax.plot(bincenter_array, bins_median - bins_std, linewidth=1, c="gold")

        ax.set_xlim((0, bincenter_array[-1]))
        ax.set_ylim((0, max(bins_median + 3*bins_std)))
        ax.set_ylabel("Data")
        ax.set_xlabel("Distance km")
        ax.set_title(f'Variogram Profile, model {variogram_model}')
        plot_label = f'nugget={coef[0]:.1f}, sill={sill:.1f}, range={coef[2]:.0f}'
        ax.plot(bincenter_array, model(bincenter_array, coef[0], coef[1], coef[2]),
                linewidth=2, c="red", label=plot_label)
        ax.legend(loc=4)

    return coef, cov

