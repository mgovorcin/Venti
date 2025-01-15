#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:05:09 2022

@author: govorcin
"""
import numpy as np

def xyz2llh(x, y, z):
    """
    Converts Cartesian X, Y, Z coordinate to Geographic Latitude, Longitude and
    Ellipsoid Height. Default Ellipsoid parameters used are GRS80.
    :param x: Cartesian X Coordinate (metres)
    :param y: Cartesian Y Coordinate (metres)
    :param z: Cartesian Z Coordinate (metres)
    :param ellipsoid: Ellipsoid Object
    :type ellipsoid: Ellipsoid
    :return: Geographic Latitude (Decimal Degrees), Longitude (Decimal Degrees)
    and Ellipsoid Height (metres)
    :rtype: tuple
    """
    wgs84_semimaj =  6378137.0
    wgs84_f = (1 / 298.257223563)
    wgs84_semimin =  wgs84_semimaj - (1 - wgs84_f)
    wgs84_ecc1sq = wgs84_f * (2 - wgs84_f)
    wgs84_ecc2sq = wgs84_ecc1sq / (1 - wgs84_ecc1sq) 
    
    # Calculate Longitude
    lon = np.atan2(y, x)
    # Calculate Latitude
    p = np.sqrt(x**2 + y**2)
    latinit = np.atan((z*(1 + wgs84_ecc2sq)) / p)
    lat = latinit
    itercheck = 1
    while abs(itercheck) > 1e-10:
        nu = wgs84_semimaj / (np.sqrt(1 - wgs84_ecc1sq * (np.sin(lat))**2))
        itercheck = lat - np.atan((z + nu * wgs84_ecc1sq * np.sin(lat))/p)
        lat = np.atan((z + nu * wgs84_ecc1sq  * np.sin(lat))/p)
    nu = wgs84_semimaj/(np.sqrt(1 - wgs84_ecc1sq * (np.sin(lat))**2))
    ellht = p/(np.cos(lat)) - nu
    # Convert Latitude and Longitude to Degrees
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return lat, lon, ellht


def llh2xyz(lat, lon, ellht=0):
    """
    Converts Geographic Latitude, Longitude and Ellipsoid Height to Cartesian
    X, Y and Z Coordinates. Default Ellipsoid parameters used are GRS80.
    :param lat: Geographic Latitude
    :type lat: Float (Decimal Degrees), DMSAngle or DDMAngle
    :param lon: Geographic Longitude
    :type lon: Float (Decimal Degrees), DMSAngle or DDMAngle
    :param ellht: Ellipsoid Height (metres, default is 0m)
    :param ellipsoid: Ellipsoid Object
    :type ellipsoid: Ellipsoid
    :return: Cartesian X, Y, Z Coordinate in metres
    :rtype: tuple
    """
    wgs84_semimaj =  6378137.0
    wgs84_f = (1 / 298.257223563)
    wgs84_semimin =  wgs84_semimaj * (1 - wgs84_f)
    wgs84_ecc1sq = wgs84_f * (2 - wgs84_f)
    wgs84_ecc2sq = wgs84_ecc1sq / (1 - wgs84_ecc1sq) 
    
    # Convert lat & long to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    
    # Calculate Ellipsoid Radius of Curvature in the Prime Vertical - nu
    if lat == 0: #this line will not work with arrays, use np.where to find 0.0 
        nu = wgs84_semimaj
    else:
        nu = wgs84_semimaj / (np.sqrt(1 - wgs84_ecc1sq * (np.sin(lat)**2)))
    # Calculate x, y, z
    x = (nu + ellht) * np.cos(lat) * np.cos(lon)
    y = (nu + ellht) * np.cos(lat) * np.sin(lon)
    z = ((wgs84_semimin**2 / wgs84_semimaj**2) * nu + ellht) * np.sin(lat)
    return x, y, z    
    
#######
###############################################################################
'''
Python version of StamPS/TRAIN matlab llh2local and local2llh 
https://github.com/dbekaert/TRAIN/blob/master/matlab/llh2local.m

check if this is possible to do with 
pargs = proj.Proj(proj="aeqd", lat_0=gps_lat, lon_0=gps_long, datum="WGS84", units="m")

'''
 
def llh2local(lon, lat, lon_x0, lat_x0):
     # WGS84 ellipsoid constants 
     a = 6378137.0
     e = 0.08209443794970 #Do not know where this number comes from
     
     # Convert degrees to radians
     lat = np.deg2rad(np.atleast_2d(lat))
     lon = np.deg2rad(np.atleast_2d(lon))
     lat_x0 = np.deg2rad(lat_x0)
     lon_x0 = np.deg2rad(lon_x0)
     
     # Do the projection
     m1 = 1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256
     m2 = 3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024
     m3 = 15 * e**4 / 256 + 45 * e**6 /1024
     m4 = 35 * e**6 / 3072

     dlambda = np.array(lon - lon_x0)
    
     M = a *(m1 * lat - m2 * np.sin(2 * lat) + \
             m3 * np.sin(4 * lat) - m4 * np.sin(6 * lat))
    
     M0 = a *(m1 * lat_x0 - m2 * np.sin(2 * lat_x0) + \
             m3 * np.sin(4 * lat_x0) - m4 * np.sin(6 * lat_x0))
         
     N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
     E = dlambda * np.sin(lat)
     
     x = (N * (1 / np.tan(lat)) * np.sin(E)) / 1000
     y = (M - M0 + N * (1 / np.tan(lat)) * (1 - np.cos(E))) / 1000
     
     # Special case if latitude is 0
     if lat.size == 1:
         ix = np.where(lat == 0.0)[0]
         if ix.size != 0:
            x[ix] = (a * dlambda[ix]) / 1000
            y[ix] = -M0 / 1000
     else:
         ix, iy = np.where(lat == 0.0)
         x[ix, iy] = (a * dlambda[ix, iy]) / 1000
         y[ix, iy] = -M0 / 1000
     
     return x, y 
 
def local2llh(x, y, lon_x0, lat_x0):
     # WGS84 ellipsoid constants 
     a = 6378137.0
     e = 0.08209443794970
     
     # Convert degrees to radians and xy to m
     x = np.atleast_2d(x) * 1000
     y = np.atleast_2d(y) * 1000
     lat_x0 = np.deg2rad(lat_x0)
     lon_x0 = np.deg2rad(lon_x0)
     
     # Do the inverse  projection
     m1 = 1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256
     m2 = 3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024
     m3 = 15 * e**4 / 256 + 45 * e**6 /1024
     m4 = 35 * e**6 / 3072
     
     M0 = a *(m1 * lat_x0 - m2 * np.sin(2 * lat_x0) + \
             m3 * np.sin(4 * lat_x0) - m4 * np.sin(6 * lat_x0))
     
     A = (M0 + y) / a
     B = x**2 / a**2 + A**2
     
     lat = A
     delta = 1
     c = 0 
     
     while np.nanmax(np.abs(delta)) > 1e-8:
         C = np.sqrt((1 - e**2 * np.sin(lat)**2)) * np.tan(lat)
         
         M = a *(m1 * lat - m2 * np.sin(2 * lat) + \
                 m3 * np.sin(4 * lat) - m4 * np.sin(6 * lat))
             
         Mn = m1 - 2 * m2 * np.cos(2 * lat) + \
              4 * m3 * np.cos(4 * lat) - 6 * m4 * np.cos(6 * lat)
              
         Ma = M/a
        
         delta = - (A * (C * Ma + 1) - Ma - 0.5 * (Ma**2 + B)*C) / \
                  (e**2 * np.sin(2 * lat) * (Ma**2 + B - 2 * A * Ma) / \
                  (4 * C) + (A - Ma) * (C * Mn - 2 / np.sin(2 * lat)) - Mn)
         
         lat = lat + delta
        
         c += 1

         if c > 100:
             raise ValueError('Convergence failure')
             break
     lon = (np.arcsin(x * C / a) / np.sin(lat)) + lon_x0  
     
     # if lat is 0
     ix, iy = np.where(y == -M0)
     lon[ix, iy] = x[ix, iy] / a + lon_x0
     lat[ix, iy] = 0.0
     
     #Convert radians to degrees
     lat = np.rad2deg(lat)
     lon = np.rad2deg(lon)
    
     return lon, lat