#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:49:29 2022

@author: govorcin
"""
from insar import read_insar
from geometry import geometry
from inversion import lsq_elements, inversion
import numpy as np
from matplotlib import pyplot as plt
 
def insar_tracks(insar_data, insar_data_std, insar_attr, 
                 insar_track_names,  n_resample=1, polynomial_order=0,
                 plot_result=True, debug_plot=False):
    
    
    ###########################################################################
    #### Project it to vertical with assumption of no-horizontal
    vproj_data, vproj_stds = [], []
    for i, ([ifg_data, _, ifg_inc, ifg_azi, _], ifg_std, ifg_track_name) in enumerate(
            zip(insar_data, insar_data_std, insar_track_names)):
        txt = 'Projecting T{} to quasi-vertical'.format(ifg_track_name)
        print(str(i+1), txt, end="")
        
        dn, de, dv = read_insar.disp_unit_vector(ifg_inc, ifg_azi)  
        vproj_data.append(ifg_data * dv)
        vproj_stds.append(ifg_std * dv)
        print(' - DONE!')
        
        
    snwe_list = [np.array(atr['snwe']) for atr in insar_attr]
    tcoherence = [d[4] for d in insar_data]
    latlon_list = [latlon_step['latlon_step'] for latlon_step in insar_attr]
        
    #############################   Mosaic  ###################################
    [A, x, W, lonlat, n_x] = lsq_elements.multiple_insar_tracks_overlaps(vproj_data, 
                                                                         vproj_stds,
                                                                         tcoherence,
                                                                         snwe_list, 
                                                                         latlon_list, 
                                                                         True, 
                                                                         n_resamp=n_resample, 
                                                                         polynomial_order=polynomial_order)


    [x, stdx, mse, Qxx, res, wres, obs_h, Q] = inversion.weighted_lstq(A, x, W)
    inversion.diagnostic_scatter_plots(res, obs_h, mse, np.diag(Q))
    
    if plot_result:   
        fig, axs = plt.subplots(1, sharey=True)
        im1 = axs.scatter(lonlat[:,0],lonlat[:,1],6, res, cmap='coolwarm')
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1, ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
        
        fig, axs = plt.subplots(1, sharey=True)
        im1 = axs.scatter(lonlat[:,0],lonlat[:,1],6, wres, cmap='coolwarm')
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1, ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
    

    plane_coef, plane_qxx = [], []

    for i in range(len(insar_data)):
        track_range = range(i*n_x,(i+1)*n_x)
        plane_coef.append(x[track_range])
        plane_qxx.append(np.take(np.take(Qxx, track_range, axis=0), track_range, axis=1))
        
    insar_corr, insar_std, insar_plane, insar_plane_std = [], [], [], []

    for i, [coef, qxx] in enumerate(zip(plane_coef, plane_qxx)):
        lalo_step = insar_attr[i]['latlon_step']
        length = int(np.rint((insar_attr[i]['snwe'][0] - insar_attr[i]['snwe'][1]) / lalo_step[0]))
        width  = int(np.rint((insar_attr[i]['snwe'][3] - insar_attr[i]['snwe'][2]) / lalo_step[1]))
        d_lats, d_lons = np.mgrid[insar_attr[i]['snwe'][1]+lalo_step[0]/2.0:insar_attr[i]['snwe'][0]+lalo_step[0]/2.0:(length)*1j,
                                  insar_attr[i]['snwe'][2]+lalo_step[1]/2.0:insar_attr[i]['snwe'][3]+lalo_step[1]/2.0:(width)*1j]

        plane = inversion.calc_plane_data(d_lons, d_lats, coef, polynomial_order)
        _, plane_std = inversion.calc_plane_cov(d_lons, d_lats, qxx, polynomial_order)

        insar_plane.append(plane)
        insar_plane_std.append(plane_std)
        insar_corr.append(vproj_data[i] + plane)  
        insar_std.append(np.sqrt(vproj_stds[i]**2 + plane_std**2))

    if plot_result:   
        fig, axs = plt.subplots(1, sharey=True)
        im1 = [axs.imshow(data, extent=geometry.snwe_to_extent(attr['snwe']), clim=[-15,15], 
                      cmap='coolwarm', interpolation='nearest') for data, attr in zip(insar_corr, insar_attr)]
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1[0], ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
    
        fig, axs = plt.subplots(1, sharey=True)
        im1 = [axs.imshow(data, extent=geometry.snwe_to_extent(attr['snwe']), clim=[0,5], 
                      cmap='OrRd', interpolation='nearest') for data, attr in zip(insar_std, insar_attr)]
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1[0], ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
    
        fig, axs = plt.subplots(1, sharey=True)
        im1 = [axs.imshow(data, extent=geometry.snwe_to_extent(attr['snwe']), clim=[-15,15], 
                      cmap='coolwarm', interpolation='nearest') for data, attr in zip(insar_plane, insar_attr)]
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1[0], ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
        
    return insar_plane, insar_plane_std