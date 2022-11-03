#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:05:09 2022

@author: govorcin
"""
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    '''
    Use Haversine equation to calculate distance between two points on a sphere
    
    lat1, lon1, lat2, lon2 : define in degrees
    
    '''
    cos_lat1 = np.cos(np.deg2rad(lat1))
    cos_lat2 = np.cos(np.deg2rad(lat2))
    cos_lon_d = np.cos(np.deg2rad(lon2 - lon1))
    cos_lat_d = np.cos(np.deg2rad(lat2 - lat1))
    

    # Earth radius at sea level at the equator (WGS84)
    r = 6378137.0
      
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def vincenty(lat1, lon1, lat2, lon2):
    '''
    Set source reference to Australian github repo
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    geod.inv(15, 45, 16, 46)
    
    check also 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    
    when working with arrays gives slightly different result than when is just one point
    '''
    
    # Earth radius at sea level at the equator ( semi-major axis of the ellipsoid WGS84)
    r_equator = 6378137
    # Earth radius at the poles  ( semi-minor axis of the ellipsoid WGS84)
    r_pole = 6356752.314245179
    
    # flattening of the ellipsoid
    f =  -(r_pole / r_equator) + 1
    
    ## Reduced latitudes
    U1 = np.arctan((1 - f) * np.tan(np.deg2rad(lat1)))
    U2 = np.arctan((1 - f) * np.tan(np.deg2rad(lat2)))
    
    # Difference in longitude on the sphere
    L = np.deg2rad(lon2 - lon1)
    omega = L
    
    #initialize variable
    delta = 1

    while np.abs(delta).all() >  1e-12:
        sin_sigma = np.sqrt((np.cos(U2)*np.sin(L))**2
                            + (np.cos(U1)*np.sin(U2) - np.sin(U1)*np.cos(U2)*np.cos(L))**2)
        
        cos_sigma = np.sin(U1)*np.sin(U2) + np.cos(U1)*np.cos(U2)*np.cos(L)
        
        sigma = np.arctan2(sin_sigma, cos_sigma)
        alpha = np.arcsin((np.cos(U1)*np.cos(U2)*np.sin(L)) / sin_sigma)
        cos_two_sigma_m = np.cos(sigma) - (2*np.sin(U1)*np.sin(U2) / np.cos(alpha)**2)
        
        c = (f / 16) * np.cos(alpha)**2 * (4 + f * (4 - 3 * np.cos(alpha)**2))
        
        L1 = omega + (1 - c) * f * np.sin(alpha) * (sigma + c*np.sin(sigma) 
                                * (cos_two_sigma_m + c * np.cos(sigma)
                                * (-1 + 2*(cos_two_sigma_m**2))))
        
        delta = L1 - L
        L = L1
     
    
    u_squared = np.cos(alpha)**2 * (r_equator**2 - r_pole**2) / r_pole**2
    
    A = 1 + (u_squared / 16384) * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))
    B = (u_squared / 1024) * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))

    delta_sigma = B*np.sin(sigma) * (cos_two_sigma_m + (B / 4) * (np.cos(sigma) * (-1 + 2*cos_two_sigma_m**2)
                              - (B / 6)*cos_two_sigma_m * (-3 + 4*np.sin(sigma)**2)
                              * (-3 + 4*cos_two_sigma_m**2)))
    
    # Calculate the ellipsoidal distance
    ell_dist = r_pole*A * (sigma - delta_sigma)
    
    # Calculate the azimuth from point 1 to point 2
    azimuth12 = np.rad2deg(np.arctan2(np.cos(U2)*np.sin(L),(np.cos(U1)*np.sin(U2)
                              - np.sin(U1)*np.cos(U2)*np.cos(L))))
    
    if np.mean(azimuth12) < 0:
        azimuth12 += 360
    
    azimuth21 = np.rad2deg(np.arctan2(np.cos(U1)*np.sin(L),(-np.sin(U1)*np.cos(U2)
                              + np.cos(U1)*np.sin(U2)*np.cos(L)))) + 180
    
    return ell_dist, azimuth12, azimuth21


'''
def vincdir(lat1, lon1, azimuth1to2, ell_dist, ellipsoid=grs80):
    """
    Vincenty's Direct Formula
    :param lat1: Latitude of Point 1 (decimal degrees)
    :type lat1: float (decimal degrees), DMSAngle or DDMAngle
    :param lon1: Longitude of Point 1 (decimal degrees)
    :type lon1: float (decimal degrees), DMSAngle or DDMAngle
    :param azimuth1to2: Azimuth from Point 1 to 2 (decimal degrees)
    :type azimuth1to2: float (decimal degrees), DMSAngle or DDMAngle
    :param ell_dist: Ellipsoidal Distance between Points 1 and 2 (metres)
    :param ellipsoid: Ellipsoid Object
    :return: lat2: Latitude of Point 2 (Decimal Degrees),
             lon2: Longitude of Point 2 (Decimal Degrees),
             azimuth2to1: Azimuth from Point 2 to 1 (Decimal Degrees)
    Code review: 14-08-2018 Craig Harrison
    """

    # Convert Angles to Decimal Degrees (if required)
    lat1 = angular_typecheck(lat1)
    lon1 = angular_typecheck(lon1)
    azimuth1to2 = radians(angular_typecheck(azimuth1to2))

    # Equation numbering is from the GDA2020 Tech Manual v1.0

    # Eq. 88
    u1 = atan((1 - ellipsoid.f) * tan(radians(lat1)))

    # Eq. 89
    sigma1 = atan2(tan(u1), cos(azimuth1to2))

    # Eq. 90
    alpha = asin(cos(u1) * sin(azimuth1to2))

    # Eq. 91
    u_squared = cos(alpha)**2 \
        * (ellipsoid.semimaj**2 - ellipsoid.semimin**2) \
        / ellipsoid.semimin**2

    # Eq. 92
    a = 1 + (u_squared / 16384) \
        * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))

    # Eq. 93
    b = (u_squared / 1024) \
        * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))

    # Eq. 94
    sigma = ell_dist / (ellipsoid.semimin * a)

    # Iterate until the change in sigma, delta_sigma, is insignificant (< 1e-9)
    # or after 1000 iterations have been completed
    two_sigma_m = 0
    for i in range(1000):

        # Eq. 95
        two_sigma_m = 2*sigma1 + sigma

        # Eq. 96
        delta_sigma = b * sin(sigma) * (cos(two_sigma_m) + (b/4)
                                        * (cos(sigma)
                                           * (-1 + 2 * cos(two_sigma_m)**2)
                                           - (b/6) * cos(two_sigma_m)
                                           * (-3 + 4 * sin(sigma)**2)
                                           * (-3 + 4 * cos(two_sigma_m)**2)))
        new_sigma = (ell_dist / (ellipsoid.semimin * a)) + delta_sigma
        sigma_change = new_sigma - sigma
        sigma = new_sigma

        if abs(sigma_change) < 1e-12:
            break

    # Calculate the Latitude of Point 2
    # Eq. 98
    lat2 = atan2(sin(u1)*cos(sigma) + cos(u1)*sin(sigma)*cos(azimuth1to2),
                 (1 - ellipsoid.f)
                 * sqrt(sin(alpha)**2 + (sin(u1)*sin(sigma)
                        - cos(u1)*cos(sigma)*cos(azimuth1to2))**2))
    lat2 = degrees(lat2)

    # Calculate the Longitude of Point 2
    # Eq. 99
    lon = atan2(sin(sigma)*sin(azimuth1to2),
                cos(u1)*cos(sigma) - sin(u1)*sin(sigma)*cos(azimuth1to2))

    # Eq. 100
    c = (ellipsoid.f/16)*cos(alpha)**2 \
        * (4 + ellipsoid.f*(4 - 3*cos(alpha)**2))

    # Eq. 101
    omega = lon - (1-c)*ellipsoid.f*sin(alpha) \
        * (sigma + c*sin(sigma)*(cos(two_sigma_m) + c*cos(sigma)
                                 * (-1 + 2*cos(two_sigma_m)**2)))

    # Eq. 102
    lon2 = float(lon1) + degrees(omega)

    # Calculate the Reverse Azimuth
    azimuth2to1 = degrees(atan2(sin(alpha), -sin(u1)*sin(sigma)
                          + cos(u1)*cos(sigma)*cos(azimuth1to2))) + 180

    return round(lat2, 11), round(lon2, 11), round(azimuth2to1, 9)
'''