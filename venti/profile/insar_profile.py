#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:12:10 2022

@author: govorcin
"""
'''
check if I can replace geometry.llh2local with pyproj
'''


import shapefile as shp
import numpy as np
from geometry import geometry
from scipy import stats
from shapely.geometry import LineString
from matplotlib import pyplot as plt

def load_shapefile(filename):
    data = shp.Reader(filename)
    # Get coordinates of all points
    points = np.squeeze([s.points for s in data.shapes()])
    
    n_points = points.shape[0]
    lines = []
    for i in range(n_points-1):
        lines.append(LineString([points[i,:], points[i+1,:]]))
    
    return points

def create_lonlat_grid(snwe, latlon_step, length, width):
    #shift half a pixel size to get latitudes and longitude of the pixel center
    lats, lons = np.mgrid[snwe[1] + latlon_step[0]/2.0:snwe[0] + latlon_step[0]/2.0: length*1j,
                          snwe[2] + latlon_step[1]/2.0:snwe[3] + latlon_step[1]/2.0: width*1j]
    
    return lons, lats
    
def profile(lons, lats, data, profile_point1, profile_point2, n_bins, radius, debug=False):
    
    #Get the data extent around profile
    profile_points = np.vstack((profile_point1, profile_point2))
    
    # FInd the origin as begining of the profile
    
    min_lat, max_lat = [np.min(profile_points[:,1]), np.max(profile_points[:,1])] 
    min_lon, max_lon = [np.min(profile_points[:,0]), np.max(profile_points[:,0])] 
    
    if lons.ndim == 1:
        lons = np.atleast_2d(lons)
    if lats.ndim == 1:
        lats = np.atleast_2d(lats)
    if data.ndim == 1:
        data = np.atleast_2d(data)
        
    ix, iy = np.where((lons >= min_lon - 0.01) &
                      (lons <= max_lon + 0.01) &
                      (lats >= min_lat - 0.01) &
                      (lats <= max_lat + 0.01))
    
    #Transform to local reference frame, using first point of line as origin
    point1 = geometry.llh2local(profile_point1[0], profile_point1[1], 
                                profile_point1[0], profile_point1[1])
    point2 = geometry.llh2local(profile_point2[0], profile_point2[1], 
                                profile_point1[0], profile_point1[1])
    
    # Rotate the local such as the line becomes horizontal, angle in [rad]
    alpha = np.squeeze(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
    rotation_matrix = np.vstack(([np.cos(alpha), -np.sin(alpha)],
                                 [np.sin(alpha),  np.cos(alpha)]))
    
    rotated_point1 = np.linalg.solve(rotation_matrix, np.concatenate(point1)) 
    rotated_point2 = np.linalg.solve(rotation_matrix, np.concatenate(point2))
    
    segment_length = rotated_point2[0]
    
    if ix.size==0 or iy.size==0:
        print('No data around profile!')
        return segment_length
    else:
        # Get the data around the profile
        lonlat = np.vstack((lons[ix, iy], lats[ix, iy]))
        data = data[ix, iy]
        
        if np.sum(np.isnan(data)) == data.size:
            print('All NaNs around profile!')
            return segment_length
        else:
            # Rotate data
            xy = geometry.llh2local(lonlat[0,:], lonlat[1,:], 
                                    profile_point1[0], profile_point1[1])
            
            xy_rotated = np.linalg.solve(rotation_matrix, np.vstack(xy))
            
            # Search for points within the buffer distance R
            ix_data = np.where((np.abs(xy_rotated[1,:]) <= radius/1000) & 
                               (xy_rotated[0,:] >= rotated_point1[0]-0.001) & 
                               (xy_rotated[0,:] <= rotated_point2[0]+0.001))[0]
            
            if ix_data.size == 0:
                print('No Data within the defined radius around profile!')
                return segment_length
            else:
                #update vectors - return xy, lonlat to original location
                new_xy = rotation_matrix @ xy_rotated[:, ix_data]
                profile_lonlat = np.vstack(geometry.local2llh(new_xy[0,:], new_xy[1,:], 
                                                              profile_point1[0], profile_point1[1]))
                profile_data = data[ix_data]
                
                # Profile xy: 1 row: along profile, 2 row: perpendicular to profile 
                profile_xy = xy_rotated[:, ix_data]
                
                if debug:
                    #Find sampling factor for faster plotting
                    if data.size > 1000000:
                        sample = np.int32(data.size // 1e6)
                    else:
                        sample = 1
                    
                    
                    fig, ax = plt.subplots(1,2)
                    #No-Rotated
                    ax[0].plot([point1[0][0], point2[0][0]], [point1[1][0], point2[1][0]],'-ok', zorder=3)
                    ax[0].scatter(np.vstack(xy)[0,::sample], np.vstack(xy)[1,::sample], s=0.1, c=data[::sample], alpha=0.1, zorder=1)
                    ax[0].scatter(new_xy[0,:], new_xy[1,:], s=4, c=profile_data, zorder=2, cmap='bwr')
                    ax[0].set_xlabel('[km]')
                    ax[0].set_ylabel('[km]')
                    ax[0].set_title('Not-Rotated')
                    #Rotated
                    ax[1].plot([rotated_point1[0][0], rotated_point2[0][0]], [rotated_point1[1][0], rotated_point2[1][0]],'-ok', zorder=3)
                    ax[1].scatter(xy_rotated[0,::sample], xy_rotated[1,::sample], s=0.1, c=data[::sample], alpha=0.1,zorder=1)
                    ax[1].scatter(profile_xy[0,:], profile_xy[1,:], s=4, c=profile_data,zorder=2, cmap='bwr')
                    ax[1].set_xlabel('[km]')
                    ax[1].set_title('Rotated')
                
                #Remove nans
                no_nan_ix = np.where(~np.isnan(profile_data))[0]
                if no_nan_ix.size ==0:
                    print('All NaNs within the defined radius around profile!')
                    return segment_length
                else:
                    n = np.count_nonzero(no_nan_ix)
                    print(f'Number of points along the profile: {n}')
                    # Binning the results
                    bins_median, bin_edges = stats.binned_statistic(np.squeeze(profile_xy[0, no_nan_ix]), 
                                                                    np.squeeze(profile_data[no_nan_ix]),
                                                                    'mean',
                                                                    n_bins)[:-1]
                    
                    bins_std = stats.binned_statistic(np.squeeze(profile_xy[0, no_nan_ix]), 
                                                      np.squeeze(profile_data[no_nan_ix]), 
                                                      'std', 
                                                      n_bins)[0]
                    
                    bins_center = (bin_edges[:-1] + bin_edges[1:]) / 2
                    #Remove binsif number of points is less than 10
                    #if n < 10:
                    #    bins_median *= np.nan
                    
                    return segment_length, bins_median, bins_std, bins_center, profile_data[no_nan_ix], profile_xy[:, no_nan_ix], profile_lonlat[:, no_nan_ix]
            
            
#Function profile2 does not work good, total length of profile is wrong due to rotation            
def profile2(lons, lats, data, profile_points, n_bins, radius, debug=False):
    '''
    Does not work due to rotation origin, which shifts x axis along profiles
    
    '''

    #Get the data extent around profile
    min_lat, max_lat = [np.min(profile_points[:,1]), np.max(profile_points[:,1])] 
    min_lon, max_lon = [np.min(profile_points[:,0]), np.max(profile_points[:,0])] 
    
    ix, iy = np.where((lons >= min_lon - 0.001) &
                      (lons <= max_lon + 0.001) &
                      (lats >= min_lat - 0.001) &
                      (lats <= max_lat + 0.001))
    
    #Select first profile point as transformation origin
    origin = profile_points[0]
    
    if ix.size==0 or iy.size==0:
        print('No data around profile!')
        return 
    else:
        # Get the data around the profile
        lonlat = np.vstack((lons[ix, iy], lats[ix, iy]))
        data = data[ix, iy]
        
        #Transform to local reference frame
        xy = np.vstack(geometry.llh2local(lonlat[0,:], lonlat[1,:], 
                                          origin[0], origin[1]))
        
        #Generate empty profile variables
        profile_bins_median = np.empty((1,0), np.float64)
        profile_bins_std = np.empty((1,0), np.float64)
        profile_bins_center = np.empty((1,0), np.float64)
        profile_data_all = np.empty((1,0), np.float64)
        profile_xy_all = np.empty((2,0), np.float64)
        profile_lonlat_all = np.empty((2,0), np.float64)
        
        # Loop trough all profile segements
        for (profile_point1, profile_point2) in zip(profile_points[:-1,:], profile_points[1:,:]):
            #Transform to local reference frame, using first point of line as origin
            point1 = geometry.llh2local(profile_point1[0], profile_point1[1], 
                                        origin[0], origin[1])
            point2 = geometry.llh2local(profile_point2[0], profile_point2[1], 
                                        origin[0], origin[1])
            
            # Rotate the local such as the line becomes horizontal, angle in [rad]
            alpha = np.squeeze(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
            rotation_matrix = np.vstack(([np.cos(alpha), -np.sin(alpha)],
                                         [np.sin(alpha),  np.cos(alpha)]))
            
            
            #Rotate profile line 
            rotated_point1 = np.linalg.solve(rotation_matrix, np.concatenate(point1)) 
            rotated_point2 = np.linalg.solve(rotation_matrix, np.concatenate(point2))
            
            # Rotate data 
            xy_rotated = np.linalg.solve(rotation_matrix, xy)
            
            # Search for points within the buffer distance R
            ix_data = np.where((np.abs(xy_rotated[1,:] - rotated_point1[1]) <= radius/1000) & 
                               (xy_rotated[0,:] >= rotated_point1[0]-0.001) & 
                               (xy_rotated[0,:] <= rotated_point2[0]+0.001))[0]
            
            if ix_data.size==0:
                print('No data around profile!')
            else:
                #update vectors - return xy, lonlat to original location
                new_xy = rotation_matrix @ xy_rotated[:, ix_data]
                profile_lonlat = np.vstack(geometry.local2llh(new_xy[0,:], new_xy[1,:], 
                                                              profile_point1[0], profile_point1[1]))
                profile_data = data[ix_data]
                
                # Profile xy: 1 row: along profile, 2 row: perpendicular to profile 
                profile_xy = xy_rotated[:, ix_data]
                
                #Debug plots
                if debug:
                    #Find sampling factor for faster plotting
                    if data.size > 1000000:
                        sample = np.int32(data.size // 1e6)
                    else:
                        sample = 1
                    
                    
                    fig, ax = plt.subplots(1,2)
                    #No-Rotated
                    ax[0].plot([point1[0][0], point2[0][0]], [point1[1][0], point2[1][0]],'-ok', zorder=3)
                    ax[0].scatter(xy[0,::sample], xy[1,::sample], s=0.1, c=data[::sample], alpha=0.1, zorder=1)
                    ax[0].scatter(new_xy[0,:], new_xy[1,:], s=6, c=profile_data, zorder=2, cmap='bwr')
                    ax[0].set_xlabel('[km]')
                    ax[0].set_ylabel('[km]')
                    ax[0].set_title('Not-Rotated')
                    #Rotated
                    ax[1].plot([rotated_point1[0][0], rotated_point2[0][0]], [rotated_point1[1][0], rotated_point2[1][0]],'-ok', zorder=3)
                    ax[1].scatter(xy_rotated[0,::sample], xy_rotated[1,::sample], s=0.1, c=data[::sample], alpha=0.1,zorder=1)
                    ax[1].scatter(profile_xy[0,:], profile_xy[1,:], s=6, c=profile_data,zorder=2, cmap='bwr')
                    ax[1].set_xlabel('[km]')
                    ax[1].set_title('Rotated')
                
                #Remove nans
                no_nan_ix = np.where(~np.isnan(profile_data))[0]
                if no_nan_ix.size == 0:
                    print('All NaNs within the defined radius around profile!')
                else:
                    n = np.count_nonzero(no_nan_ix)
                    print(f'Number of points along the profile: {n}')
                    # Binning the results
                    bins_median, bin_edges = stats.binned_statistic(np.squeeze(profile_xy[0, no_nan_ix]), 
                                                                    np.squeeze(profile_data[no_nan_ix]),
                                                                    'mean',
                                                                    n_bins)[:-1]
                    
                    bins_std = stats.binned_statistic(np.squeeze(profile_xy[0, no_nan_ix]), 
                                                      np.squeeze(profile_data[no_nan_ix]), 
                                                      'std', 
                                                      n_bins)[0]
                    
                    bins_center = (bin_edges[:-1] + bin_edges[1:]) / 2
                    

                    #Append the results
                    profile_bins_median = np.append(profile_bins_median, bins_median)
                    profile_bins_std = np.append(profile_bins_std, bin_edges)
                    profile_data_all = np.append(profile_data_all, profile_data[no_nan_ix])
                    profile_bins_center = np.append(profile_bins_center, bins_center)
                    profile_xy_all = np.append(profile_xy_all, profile_xy[:, no_nan_ix], axis=1)
                    profile_lonlat_all = np.append(profile_lonlat_all,  profile_lonlat[:, no_nan_ix], axis=1)
                
        return profile_bins_median, profile_bins_std, profile_bins_center, profile_lonlat_all, profile_xy_all, profile_data_all
                

def profile_multi_segement(lons, lats, data, points,  n_bins, radius):
    
    profile_bins_median = np.empty((1,0), np.float64)
    profile_bins_std = np.empty((1,0), np.float64)
    profile_bins_center = np.empty((1,0), np.float64)
    profile_profile_data = np.empty((1,0), np.float64)
    profile_profile_xy = np.empty((2,0), np.float64)
    profile_profile_lonlat = np.empty((2,0), np.float64)
    profile_profile_length = np.zeros((1,1), np.float64)
    profile_length =0
    for (point1, point2) in zip(points[:-1,:], points[1:,:]):
        profile_data = profile(lons, lats, data, point1, point2, n_bins, radius)

        if type(profile_data) is not tuple:
            profile_length += profile_data[0]
            #bins
            profile_bins_median = np.append(profile_bins_median, np.nan)
            profile_bins_std = np.append(profile_bins_std, np.nan)
            profile_bins_center = np.append(profile_bins_center, profile_length)
            #All data along profile
            profile_profile_data = np.append(profile_profile_data, np.nan)
            profile_profile_xy = np.append(profile_profile_xy, np.vstack((profile_length, 0)), axis=1)
            profile_profile_lonlat = np.append(profile_profile_lonlat, np.vstack((np.nan, np.nan)), axis=1)
            profile_profile_length = np.append(profile_profile_length, profile_length)
        else:
            #bins
            profile_bins_median = np.append(profile_bins_median, profile_data[1])
            profile_bins_std = np.append(profile_bins_std, profile_data[2])
            profile_bins_center = np.append(profile_bins_center, profile_data[3] + profile_length)
            #All data along profile
            profile_profile_data = np.append(profile_profile_data, profile_data[4])
            profile_profile_xy = np.append(profile_profile_xy, profile_data[5] + np.vstack((profile_length, 0)), axis=1)
            profile_profile_lonlat = np.append(profile_profile_lonlat, profile_data[6], axis=1)
            
            #Add segment length
            profile_length += profile_data[0]
            profile_profile_length = np.append(profile_profile_length, profile_length)
        print(profile_length)
        
    return profile_bins_median, profile_bins_std, profile_bins_center, profile_profile_data, profile_profile_xy, profile_profile_lonlat, profile_profile_length


        
        
        
        
        
        
    
        

