#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:19:28 2022

@author: govorcin
"""
import numpy as np
from matplotlib import pyplot as plt


from insar import read_insar
from inversion import inversion
from inversion import lsq_elements
from geometry import geometry


def projectGNSS2LOS(gnss_latitude, gnss_longitude, 
                    gnss_NS, gnss_EW, gnss_UD,
                    gnss_sNS, gnss_sEW, gnss_sUD,
                    insar_incidenceAngle, insar_azimuthAngle, 
                    insar_snwe, insar_latlonStep):
    # Get the InSAR displacement unit vector
    dn, de, dv = read_insar.disp_unit_vector(insar_incidenceAngle, insar_azimuthAngle)
    
    #Find all GNSS stations within the InSAR scene
    #find  GPS within the insar scene
    idx = ((gnss_latitude >= insar_snwe[0]) * (gnss_latitude <= insar_snwe[1]) *
           (gnss_longitude >= insar_snwe[2]) * (gnss_longitude <= insar_snwe[3]))
    
    #number of used GNSS stations
    n = len(gnss_latitude[idx])
    
    #find the insar raster x,y  collocated with GNNS
    insar_idx = np.empty((n, 2), dtype=np.int32)
    
    for i, [lon, lat] in enumerate(zip(gnss_longitude[idx], gnss_latitude[idx])):
        insar_idx[i] = geometry.lalo2xy(lat, lon, insar_snwe, insar_latlonStep)
            
    gnss_disp_los = dn[insar_idx[:,1], insar_idx[:,0]] * gnss_NS[idx] \
                    + de[insar_idx[:,1], insar_idx[:,0]] * gnss_EW[idx] \
                    + dv[insar_idx[:,1], insar_idx[:,0]] * gnss_UD[idx]
                    
    gnss_std_los = dn[insar_idx[:,1], insar_idx[:,0]] * gnss_sNS[idx] \
                    + de[insar_idx[:,1], insar_idx[:,0]] * gnss_sEW[idx] \
                    + dv[insar_idx[:,1], insar_idx[:,0]] * gnss_sUD[idx]
                    
    return gnss_disp_los, gnss_std_los
                    
    
def insar_reref(insar_disp, insar_stds, insar_incidenceAngle, insar_azimuthAngle,
                    insar_snwe, insar_latlonStep,
                    gnss_latitude, gnss_longitude, 
                    gnss_NS, gnss_EW, gnss_UD,
                    gnss_sNS, gnss_sEW, gnss_sUD,
                    poly_order=1, plot_result=True):
    
    A_gps, b_gps, W_gps, lonlat_gps = [], [], [], []
    np_gps = []
    projected_gps_obs = []
    
    for i, [data, std, inc, azi, snwe, latlonStep], in enumerate(zip(insar_disp, 
                                                                     insar_stds, 
                                                                     insar_incidenceAngle, 
                                                                     insar_azimuthAngle,
                                                                     insar_snwe, 
                                                                     insar_latlonStep)):
        
        gnss_disp_los, gnss_std_los = projectGNSS2LOS(gnss_latitude, gnss_longitude, 
                                                      gnss_NS, gnss_EW, gnss_UD,
                                                      gnss_sNS, gnss_sEW, gnss_sUD,
                                                      inc, azi, snwe, latlonStep)
        
        [a, b_gps12, w_gps, n_x, lonlat] = inversion.lstq_elements_gps_insar(gnss_longitude, 
                                                                             gnss_latitude, 
                                                                             gnss_disp_los, 
                                                                             gnss_std_los, 
                                                                             data, 
                                                                             std, 
                                                                             snwe, 
                                                                             latlonStep,
                                                                             poly_order = poly_order)
        
        A_gps.append(a)
        b_gps.append(b_gps12)
        W_gps.append(w_gps)
        np_gps.append(b_gps12.shape[0])
        lonlat_gps.append(lonlat)
        projected_gps_obs.append(gnss_disp_los)
        
    
    # Organize lstq elements
    A_gps = np.concatenate(A_gps) #use block_diag for multiple correction planes
    b_gps = np.concatenate(b_gps)
    W_gps = np.concatenate(W_gps)
    lonlat_gps= np.concatenate(lonlat_gps)
    
    # invert
    [x, stdx, mse, Qxx, res, wres, obs_h, Q] = inversion.weighted_lstq(A_gps, 
                                                                       np.squeeze(b_gps), 
                                                                       np.squeeze(W_gps))
    if plot_result:   
        fig, axs = plt.subplots(1, sharey=True)
        im1 = axs.scatter(lonlat_gps[:,0],lonlat_gps[:,1],6, res, cmap='coolwarm')
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1, ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
        
        fig, axs = plt.subplots(1, sharey=True)
        im1 = axs.scatter(lonlat_gps[:,0],lonlat_gps[:,1],6, wres, cmap='coolwarm')
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1, ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
    
    
    inversion.diagnostic_scatter_plots(res, obs_h, mse, np.diag(Q))

    plane_coef, plane_qxx = [], []
    
    n_x = lsq_elements.coeff_poly[poly_order]

    for i in range(len(insar_disp)):
        track_range = range(0*n_x,(0+1)*n_x)
        plane_coef.append(x[track_range])
        plane_qxx.append(np.take(np.take(Qxx, track_range, axis=0), track_range, axis=1))
        
        
    corr, std, cplane = [], [], []
    for i, [coef, qxx, snwe, latlonStep] in enumerate(zip(plane_coef, plane_qxx, insar_snwe, insar_latlonStep)):

        length = int(np.rint((snwe[0] - snwe[1]) / latlonStep[0]))
        width  = int(np.rint((snwe[3] - snwe[2]) / latlonStep[1]))
        d_lats, d_lons = np.mgrid[snwe[1]+latlonStep[0]/2.0:snwe[0]+latlonStep[0]/2.0:(length)*1j,
                                  snwe[2]+latlonStep[1]/2.0:snwe[3]+latlonStep[1]/2.0:(width)*1j]

        plane = inversion.calc_plane_data(d_lons, d_lats, coef, poly_order)
        _, plane_std = inversion.calc_plane_cov(d_lons, d_lats, qxx, poly_order)
        
        cplane.append(plane)
        corr.append(insar_disp[i] + plane)
        std.append(np.sqrt(insar_stds[i]**2 + plane_std**2))
        
    if plot_result:
        quantile = np.mean([np.nanquantile(d, [0.05, 0.95]) for d in corr], axis=0)
        print(quantile)
        fig, axs = plt.subplots(1, sharey=True)
        im1 = [axs.imshow(data, extent=geometry.snwe_to_extent(snwe), clim=[quantile[0],quantile[1]], 
                      cmap='coolwarm', interpolation='nearest') for data, snwe in zip(corr, insar_snwe)]
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1[0], ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
    
        fig, axs = plt.subplots(1, sharey=True)
        im1 = [axs.imshow(data, extent=geometry.snwe_to_extent(snwe), clim=[0,5], 
                      cmap='OrRd', interpolation='nearest') for data, snwe in zip(std, insar_snwe)]
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1[0], ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
    
        fig, axs = plt.subplots(1, sharey=True)
        im1 = [axs.imshow(data, extent=geometry.snwe_to_extent(snwe), clim=[-15,15], 
                      cmap='coolwarm', interpolation='nearest') for data, snwe in zip(cplane, insar_snwe)]
        axs.set_xlim([-125,-113]), axs.set_ylim([32,44])
        cbar = fig.colorbar(im1[0], ax=axs, location='bottom', pad=0.10)
        cbar.set_label('[mm/yr]')
        
    
    return corr, std, cplane
        
        
        
    
