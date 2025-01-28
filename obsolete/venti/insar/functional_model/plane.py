#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:40:54 2022

@author: govorcin

"""

import numpy as np
from typing import Tuple


coeff_poly = {
    0   : 1, # resolve for constant
    1   : 3, # linear coeff a,b,c :  ax + by + c
    1.5 : 4, # coeff a,b,c,d : ax + bx + cxy + d
    2   : 6, # quadratic coeff, a,b,c,d,e,f : ax^2 + by^2 + cxy + dx + ey + f
    3   : 8} # cubic coeff, a,b,c,d,e,f,g,h : ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h


def design_matrix(lon: np.array, 
                  lat: np.array, 
                  c:np.int32 = 1, 
                  poly_order:np.int32 = 1) -> np.array:
    '''
    Input:
        lon: longitudes as x-axis values
        lat: latitudes as y-axis values
        c:   constant [default=1]
        poly_order: polynomial order of plane

    Return:
        A: design matrix for plane

    '''

    # construct design based on polynomial degree 
    if poly_order == 0:
        # constant:  c
        A = np.array([np.ones(len(lon)) * c])
    elif poly_order == 1:
        # linear: ax + by + c
        A = np.array([lon, lat, np.ones(len(lon)) * c])
    elif poly_order == 1.5:
        # ax + bx + cxy + d
        A = np.array([lon, lat, lon*lat, np.ones(len(lon)) * c])
    elif poly_order == 2: 
        # quadratic: ax^2 + by^2 + cxy + dx + ey + f
        A = np.array([lon**2, lat**2, lon*lat, lon, lat, np.ones(len(lon)) * c])
    elif poly_order == 3: 
        # cubic: ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        A = np.array([lon**3, lat**3, lon**2, lat**2, lon*lat, lon, lat, np.ones(len(lon))* c])
    
    return A.T

def calc_plane_data(lon: np.array, 
                    lat: np.array,  
                    coef: np.array,
                    poly_order:np.int32 = 1) -> np.array:
    '''
    Input:
        lon: longitudes as x-axis values
        lat: latitudes as y-axis values
        coef:  estimated plane coeeficients from lsq
        poly_order: polynomial order of plane

    Return:
        plane: polynomial_plane

    '''


    # construct based on polynomial degree 
    if poly_order == 0:
        # constant: c
        plane = np.ones(lon.shape) * coef
    elif poly_order == 1:
        # linear: ax + by + c
        plane = lon * coef[0] + lat * coef[1] + coef[2]
    elif poly_order == 1.5:
        # ax + bx + cxy + d
        plane = lon * coef[0] + lat * coef[1] + lon * lat *coef[2] + coef[3]
    elif poly_order == 2: 
        # quadratic: ax^2 + by^2 + cxy + dx + ey + f
        plane = lon**2 * coef[0] + lat**2 * coef[1] + lon * lat *coef[2] + lon * coef[3] + lat * coef[4] + coef[5]
    elif poly_order == 3: 
        # cubic: ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        plane = lon**3 * coef[0] + lat**3 * coef[1] + lon**2 * lat *coef[2] + lon * lat**2 * coef[3] + lon**2 * coef[4] + + lat**2 * coef[5] + coef[6]
    return plane

def calc_plane_cov(lon: np.array, 
                   lat: np.array,
                   Qxx: np.array,
                   poly_order:np.int32 = 1) -> (np.array, np.array):
    '''
    Input:
        lon: longitudes as x-axis values
        lat: latitudes as y-axis values
        Qxx:  covariance matrix of unknowns from lsq
        poly_order: polynomial order of plane

    Return:
        plane_var: polynomial plane of propageted variance
        plane_std: polynomial plane of propageted standard deviations 
    '''

    # matrix or vector of ones
    e =  np.ones(lon.shape)
    
    if poly_order == 0:
        # constant: c
        plane_var = np.ones(lon.shape) * Qxx
        
    elif poly_order == 1:
        # linear: ax + by + c
        [x_var, y_var, z_var] = np.diag(Qxx)
        #covariance
        [cxy_var, cxz_var] = Qxx[0, 1:]
        cyz_var = Qxx[1,2:]
    
        plane_var = lon**2 * x_var + lat**2 *y_var + e**2 * z_var + \
            2 * lon * (lat*cxy_var + e*cxz_var) + \
            2 * lat * (e*cyz_var)
        
    elif poly_order == 1.5:
        # ax + bx + cxy + d
        [x_var, y_var, xy_var, z_var] = np.diag(Qxx)
        #covariance
        [cxy_var, cxxy_var, cxz_var] = Qxx[0, 1:]
        [cyxy_var, cyz_var] = Qxx[1,2:]
        cxyz_var = Qxx[2,3:]
        
        plane_var = lon**2 * x_var + lat**2 * y_var + (lon*lat)**2 * xy_var +  e**2 * z_var + \
            2 * lon* (lat*cxy_var + lon*lat*cxxy_var + e*cxz_var) + \
            2 * lat * (lon*lat*cyxy_var + e*cyz_var) + \
            2 * lon*lat * (e*cxyz_var)
            
    elif poly_order == 2: 
        # quadratic: ax^2 + by^2 + cxy + dx + ey + f
        [xx_var, yy_var, xy_var, x_var, y_var, z_var] = np.diag(Qxx)
        #covariance
        [cxxyy_var, cxxxy_var, cxxx_var, cxxy_var, cxxz_var] = Qxx[0, 1:]
        [cyyxy_var, cyyx_var, cyyy_var, cyyz_var] = Qxx[1,2:]
        [cxyx_var, cxyy_var, cxyz_var] = Qxx[2,3:]
        [cxy_var, cxz_var] = Qxx[3,4:]
        cyz_var = Qxx[4,5:]
        
        plane_var = lon**4 * xx_var + lat**4 * yy_var + (lon*lat)**2 * xy_var + lon**2 * x_var + lat**2 * y_var + e**2 * z_var + \
            2 * lon**2 * (lat**2*cxxyy_var + lon*lat*cxxxy_var + lon*cxxx_var +  lat*cxxy_var + e*cxxz_var) + \
            2 * lat**2 * (lat*lon*cyyxy_var + lon*cyyx_var + lat*cyyy_var + e*cyyz_var) + \
            2 * lon*lat * (lon* cxyx_var + lat*cxyy_var + e*cxyz_var) + \
            2 * lon * (lat * cxy_var + e*cxz_var) + \
            2 * lat * (e*cyz_var)
            
    elif poly_order == 3: 
        # cubic: ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        [xxx_var, yyy_var, xxy_var, yyx_var, xx_var, yy_var, xy_var, z_var] = np.diag(Qxx)

        #covariance
        [cxxxyyy_var, cxxxxxy_var, cxxxyyx_var, cxxxxx_var, cxxxyy_var, cxxxxy_var, cxxxz_var] = Qxx[0, 1:]
        [cyyyxxy_var, cyyyyyx_var, cyyyxx_var, cyyyyy_var, cyyyxy, cyyyz_var] = Qxx[1,2:]
        [cxxyyyx_var, cxxyxx_var, cxxyyy_var, cxxyxy_var, cxxyz_var] = Qxx[2,3:]
        [cyyxxx_var, cyyxyy_var, cyyxxy_var, cyyxz_var] = Qxx[3,4:]
        [cxxyy_var, cxxxy_var, cxxz_var] = Qxx[4,5:]
        [cyyxy_var, cyyz_var] = Qxx[5,6:]
        cxyz_var = Qxx[6,7:]

        plane_var = lon**5 * xxx_var + lat**5 * yyy_var + (lon**2 * lat)**2 * xxy_var + (lon*lat**2)**2 * yyx_var + \
                    lon**4 * xx_var + lat**4*yy_var + (lon*lat)**2 * xy_var + e**2 * z_var + \
            2 * lon**3 * (lat**3*cxxxyyy_var + lon**2*lat*cxxxxxy_var + lat*2*lon*cxxxyyx_var +  lon*2*cxxxxx_var + \
                          lat*2*cxxxyy_var + lon*lat*cxxxxy_var + e*cxxxz_var) + \
            2 * lat**3 * (lon**2*lat*cyyyxxy_var + lat**2*lon*cyyyyyx_var + lon**2*cyyyxx_var + lat**2*cyyyyy_var + \
                          lon*lat*cyyyxy + e*cyyyz_var) + \
            2 * lon**2*lat * (lat**2*lon*cxxyyyx_var + lon**2*cxxyxx_var + lat**2*cxxyyy_var + lon*lat*cxxyxy_var + e*cxxyz_var) + \
            2 * lat**2*lon * (lon**2 * cyyxxx_var + lat**2*cyyxyy_var + lon*lat*cyyxxy_var + e*cyyxz_var) + \
            2 * lon**2 * (lat**2*cxxyy_var + lon*lat*cxxxy_var + e*cxxz_var) + \
            2 * lat**2 * (lon*lat*cyyxy_var + e*cyyz_var) + \
            2 * lon*lat * (e*cxyz_var)
         
    plane_std = np.sqrt(plane_var)
    
    return plane_var, plane_std


