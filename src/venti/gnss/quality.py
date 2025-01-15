#!/usr/bin/env python3
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from venti.solvers.midas import midas
from venti.gnss.interpolation import hvlsc
from matplotlib import pyplot as plt

def calculate_temporal_variability(dates, data, time_increment, steps=None):
    # Get moving windows
    date1 = dates[0]
    moving_window = np.c_[True, True]
    moving_dates = [date1]

    while moving_window.any():
        moving_window = dates > date1 + \
            np.timedelta64(365, 'D') * (time_increment / 2)
        if ~moving_window.any():
            break
        if np.timedelta64((np.max(dates) - dates[moving_window][0]), 'D') > np.timedelta64(365, 'D') * (time_increment1):
            moving_dates.append(dates[moving_window][0])
        date1 = dates[moving_window][0]

    # Get variable velocities
    vels, vel_stds = [], []
    for mov_date in moving_dates:
        condition1 = dates > mov_date
        condition2 = dates < mov_date + \
            np.timedelta64(365, 'D') * time_increment
        mask = condition1 * condition2

        # Get velocity
        vel, vel_std, _, _ = midas(dates[mask],
                                   data[mask, :],
                                   steps=steps)
        if vel is not None:
            vels.append(vel*1e3)
            vel_stds.append(vel_std*1e3)

    return moving_dates, vels, vel_stds


def compute_delaunay_network(lon, lat, visualize=False):
    """
    Computes the Delaunay triangulation of a set of points.

    Parameters:
        lon (array-like): Longitudes of points.
        lat (array-like): Latitudes of points.
        visualize (bool): If True, plot the triangulation.

    Returns:
        triangles (ndarray): Array of triangle vertex indices.
    """
    # Combine longitude and latitude into a single array of 2D points
    points = np.vstack((lon, lat)).T

    # Compute the Delaunay triangulation
    tri = Delaunay(points)
    
    if visualize:
        # Plot the points and the Delaunay triangulation
        plt.figure(figsize=(8, 6))
        plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue', alpha=0.6)
        plt.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=3)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Delaunay Triangulation')
        plt.grid(True)
        plt.show()
    
    # Return the indices of the triangles
    return tri.simplices


def calculate_ssf(lon, lat, velocity, uncertainty, 
                  dvmax=10, bin_spacing=0.25, distance_km=False, 
                  visualize=False):
    """
    Calculate the Spatial Structure Function (SSF) on vertices of a Delaunay triangulation.

    Parameters:
        lon (array-like): Longitudes of points.
        lat (array-like): Latitudes of points.
        velocity (array-like): Velocities at the points.
        uncertainty (array-like): Uncertainties of the velocities.
        dvmax (float): Maximum allowable velocity difference.
        bins (array-like): Bin edges for the distance histogram (log scale).
        visualize (bool): Whether to visualize the results.

    Returns:
        ssf (ndarray): Spatial structure function as an array with distance bins and SSF values.
    Note: add weighted median using uncertainties
    """

    # Convert inputs to numpy arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    velocity = np.asarray(velocity)

    # Combine longitude and latitude into 2D points
    points = np.vstack((lon, lat)).T

    # Calculate pairwise distances and velocity differences
    if distance_km:
        # Get only upper tringular matrix to avoid duplicates
        distances = np.triu(hvlsc.utils.get_distance_matrix(lon,lat,
                                                            lon,lat))
        dist_label ='km'
    else:
        distances = np.triu(cdist(points, points, metric='euclidean'))
        dist_label ='degrees'

    distances[distances==0]=np.nan

    velocity_diff = np.abs(velocity[:, None] - velocity[None, :])

    # Mask pairs exceeding the dvmax or NaNs in input
    flag_nonan = (~np.isnan(velocity_diff)) & (~np.isnan(distances)) 
    valid_pairs =  flag_nonan & (velocity_diff < dvmax)
    distances = distances[valid_pairs]
    velocity_diff = velocity_diff[valid_pairs]

    # Get bins
    dmax = np.nanmax(distances)

    if dmax < bin_spacing:
        bin_spacing = np.round(np.min(distances),1)
        print('Update_bin_spacing', bin_spacing)

    if distance_km:
        bins = np.arange(0, dmax, bin_spacing)
    else:
        bins = 10**np.arange(-2, dmax, bin_spacing)  # Default log-spaced bins

    # Bin distances and compute median absolute velocity difference per bin
    ssf = []
    for i in range(len(bins) - 1):
        in_bin = (distances >= bins[i]) & (distances < bins[i + 1])
        if np.any(in_bin):
            median_diff = np.median(np.abs(velocity_diff[in_bin]))
        else:
            median_diff = np.nan
        ssf.append([np.sqrt(bins[i] * bins[i + 1]), median_diff])

    ssf = np.array(ssf)

    if ssf.any():
        # Normalize SSF
        ssf[:, 1] = 1.0 / ssf[:, 1]
        ssf[:, 1] /= np.nanmax(ssf[:, 1])
        ssf = np.vstack([[0, 1], ssf])

        if visualize:
            # Visualize the SSF
            plt.figure(figsize=(10, 6))
            plt.semilogx(ssf[:, 0], ssf[:, 1], 'k-', linewidth=2)
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.xlabel(f'Distance Between Station Pairs ({dist_label})')
            plt.ylabel('Normalized Spatial Structure Function (SSF)')
            plt.title('Spatial Structure Function')
            plt.show()
    else:
        ssf = np.array([np.nan, np.nan])

    return ssf


def calculate_spatial_variability(lon, lat, values):
    """
    Calculate spatial variability for each node in a Delaunay network.

    Parameters:
    - points: ndarray of shape (N, 2)
        Array of points (lon, lat) for the network.
    - values: ndarray of shape (N,)
        Array of values (e.g., velocities) at each point.
    - triangulation: scipy.spatial.Delaunay (optional)
        Precomputed Delaunay triangulation. If None, it will be computed.

    Returns:
    - spatial_variability: ndarray of shape (N,)
        Spatial variability values for each node.
    """
    tri = compute_delaunay_network(lon, lat)

    neighbors = {i: set() for i in range(len(lon))}
    
    # Get neighbors from the Delaunay simplices
    for simplex in tri:
        for i, j in zip(simplex, np.roll(simplex, -1)):
            neighbors[i].add(j)
            neighbors[j].add(i)

    spatial_variability = np.zeros((len(lon), 2))  # Two columns: RMS and MAD

    # Compute spatial variability for each node
    for i, point_neighbors in neighbors.items():
        if len(point_neighbors) == 0:
            spatial_variability[i] = [np.nan, np.nan]
            continue
        
        neighbor_values = values[list(point_neighbors)]
        diff = neighbor_values - values[i]
        
        # RMS of differences
        rms = np.sqrt(np.mean(diff**2))
        # MAD of differences
        mad = np.median(np.abs(diff - np.median(diff)))
        
        spatial_variability[i] = [rms, mad]
    
    return spatial_variability

def calculate_network_ssf(sites, lon, lat, velocity, uncertainty, 
                          dvmax=10, bin_spacing=25, distance_km=True):
    
    # Create a dictionary to store connections for each point
    connections = defaultdict(set)

    # Iterate through the simplices to build connectivity
    for simplex in compute_delaunay_network(lon, lat):
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                # Add bidirectional connections
                connections[simplex[i]].add(simplex[j])
                connections[simplex[j]].add(simplex[i])

    # Convert to a more readable format
    connections_dict = {point: list(neighbors) for point, neighbors in connections.items()}

    ssf_dict=dict(site=None, ssf=None, ssf_npoints=None)

    ssf_sites = []
    for site_ix, tri_ix in connections_dict.items():
        ssf_dict['site'] = sites[site_ix]
        ssf_dict['ssf_npoints'] = len(tri_ix)

        ssf_dict['ssf'] = np.nanmedian(calculate_ssf(
                        np.r_[lon[site_ix], lon[tri_ix]], 
                        np.r_[lat[site_ix], lat[tri_ix]],
                        np.r_[velocity[site_ix], velocity[tri_ix]],
                        np.r_[uncertainty[site_ix], uncertainty[tri_ix]],
                        dvmax=dvmax, bin_spacing=bin_spacing,
                        distance_km=distance_km)[:,1])
        ssf_sites.append(ssf_dict.copy())
    return pd.DataFrame(ssf_sites)

def get_network_distance_stat(sites, lon, lat):
    # Create a dictionary to store connections for each point
    connections = defaultdict(set)

    # Iterate through the simplices to build connectivity
    for simplex in compute_delaunay_network(lon, lat):
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                # Add bidirectional connections
                connections[simplex[i]].add(simplex[j])
                connections[simplex[j]].add(simplex[i])

    # Convert to a more readable format
    connections_dict = {point: list(neighbors) for point, neighbors in connections.items()}
    
    dist_dict=dict(site=None, dmin=None,
                   dmax=None, dmean=None,
                   dmedian=None, npoints=None)

    dist = []
    for site_ix, tri_ix in connections_dict.items():
        distances = hvlsc.utils.get_distance_matrix(np.atleast_1d(lon[site_ix]), 
                                                            np.atleast_1d(lat[site_ix]),
                                                            lon[tri_ix], lat[tri_ix])
        distances[distances==0]=np.nan
        dist_dict['site'] = sites[site_ix]
        dist_dict['dmin'] = np.nanmin(distances)
        dist_dict['dmax'] = np.nanmax(distances)   
        dist_dict['dmean'] = np.nanmean(distances)
        dist_dict['dmedian'] = np.nanmedian(distances)
        dist_dict['npoints'] = len(tri_ix)
        dist.append(dist_dict.copy())
    return pd.DataFrame(dist)

