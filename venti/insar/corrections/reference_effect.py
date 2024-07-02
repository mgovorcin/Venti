#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:07:24 2022

@author: govorcin
"""

import numpy as np
import time
from pyproj import Geod
from venti import variogram
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def insar_std_reference_effect(insar_std: np.array,
                               snwe: list,
                               latlon_step: list,
                               ref_xy: list,
                               variogram_model: str = 'spherical',
                               debug_plot:bool = False,
                               print_msg = True,
                               **variogram_kwargs) -> (np.array, np.array):

    vprint = print if print_msg else lambda *args, **kwargs: None

    # Time the function execution
    start_time = time.time()

    # Find optional parameters if exist
    if 'bin_size' in variogram_kwargs:
        bin_size = variogram_kwargs['bin_size']
    else:
        bin_size = 100

    if 'max_range' in variogram_kwargs:
        max_range = variogram_kwargs['max_range']
    else:
        max_range = None

    # Construct the grid of latitudes and longitudes for the data
    length, width = insar_std.shape
    d_lats, d_lons = np.mgrid[snwe[1] + latlon_step[0]/2.0:snwe[0] + latlon_step[0]/2.0:(length)*1j,
                              snwe[2] + latlon_step[1]/2.0:snwe[3] + latlon_step[1]/2.0:(width)*1j]

    # Create array of reference latitude and longitude
    [x, y] = ref_xy
    ref_lat = d_lats[int(y), int(x)]
    ref_lon = d_lons[int(y), int(x)]

    # Create reference lat. lon arrays
    ref_lats = np.ones(d_lats.shape) * np.float32(ref_lat)
    ref_lons = np.ones(d_lons.shape) * np.float32(ref_lon)

    # Caculate the distance array from ref lat, lon
    geod = Geod(ellps="WGS84")
    distance = geod.inv(ref_lons, ref_lats, d_lons, d_lats)[2] / 1000 # to get in km

    variogram_coef, _ = variogram.calc_variogram(insar_std,
                                                 d_lats,
                                                 d_lons,
                                                 ref_lat,
                                                 ref_lon,
                                                 bin_size,
                                                 max_range,
                                                 variogram_model = variogram_model,
                                                 debug_plot = debug_plot)

    #get the sill value from the coeffs: partial sill + nugget
    sill = variogram_coef[0] + variogram_coef[1]

    model = variogram.models.get_model(variogram_model)

    #Fit to all data
    y_hat = model(distance, variogram_coef[0], variogram_coef[1], variogram_coef[2])

    #Find the scaling factor
    scaling = sill / y_hat

    #Correct the insar uncertainty map
    corrected_std = insar_std * scaling

    if debug_plot:
        fig, ax = plt.subplots(1,3, sharey=True)
        # plot uncertainty map with reference location

        # Color range values
        cvalue_min = np.nanquantile(insar_std, 0)
        cvalue_max = np.nanquantile(insar_std, 0.95)


        im = ax[0].imshow(insar_std, vmin=cvalue_min, vmax=cvalue_max, cmap='GnBu')
        ax[0].plot(int(x), int(y), marker="s", markersize=5, c='black')
        ax[0].set_title("Uncertainty Map")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        im = ax[1].imshow(scaling, vmin=cvalue_min, vmax=cvalue_max, cmap='GnBu')
        ax[1].set_title("Scaling Map")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # plot scaled uncertainty map
        im = ax[2].imshow(corrected_std, vmin=cvalue_min, vmax=cvalue_max, cmap='GnBu')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label="mm/yr")
        ax[2].set_title("Scaled Uncertainty Map")
        ax[2].set_xticks([])
        ax[2].set_yticks([])
    vprint(f'Time elapsed for running: {time.time() - start_time:.2f}s')

    return corrected_std, scaling


