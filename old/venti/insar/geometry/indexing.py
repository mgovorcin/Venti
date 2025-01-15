#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:05:09 2022

@author: govorcin
"""
import numpy as np

DEG2METER =  """
degrees     --> meters on equator
0.000925926 --> 100
0.000833334 --> 90
0.000555556 --> 60
0.000462963 --> 50
0.000277778 --> 30
0.000185185 --> 20
0.000092593 --> 10
"""
METER2DEG = {
    100 : 0.000925926,
    90  : 0.000833334,
    60  : 0.000555556,
    50  : 0.000462963,
    30  : 0.000277778, 
    20  : 0.000185185,
    10  : 0.000092593}


def lalo2xy(lat, lon, data_snwe, latlon_step, mround = 'floor'):
    # np.floor works better with points and raster - Need to check way
    # but with two raster sometimes one pixel is missing or is redundant
    if mround == 'floor':
        x = int(np.floor((lon - data_snwe[2]) / latlon_step[1] + 0.01))
        y = int(np.floor((lat - data_snwe[1]) / latlon_step[0] + 0.01))
    #np.around works better with two rasters
    # or should I use floor for min(lat,lon) and np.ceil fo max(lat,lon)
    # test it out
    elif mround == 'around': 
        x = int(np.around((lon - data_snwe[2]) / latlon_step[1] + 0.01))
        y = int(np.around((lat - data_snwe[1]) / latlon_step[0] + 0.01))
    
    return x, y

def lalo_box2xy_box(lalo_box, snwe, lalo_step):
    x1, y1 = lalo2xy(lalo_box[1], lalo_box[2], snwe, lalo_step)
    x2, y2 = lalo2xy(lalo_box[0], lalo_box[3], snwe, lalo_step)
    box = [y1, y2, x1, x2]
    width =  x2 - x1
    length = y2 - y1
    
    return box, width, length

def overlap(snwe1, snwe2, latlon_step1, latlon_step2):
    S = max(snwe1[0], snwe2[0])
    N = min(snwe1[1], snwe2[1])
    W = max(snwe1[2], snwe2[2])
    E = min(snwe1[3], snwe2[3])
    SNWE = [S, N, W, E]
    
    #subset length and width
    #find
    mm = pixel_spacing_deg2meter(latlon_step1[1])

    latlon_step = [-METER2DEG[mm], METER2DEG[mm]]
    length = int(round((S - N) / latlon_step[0]))  #need to add 1 to include the lat pixel
    width  = int(round((E - W) / latlon_step[1]))
    
    #update S and E
    SNWE = np.array([SNWE[1]+latlon_step[0]*(length), SNWE[1],
                     SNWE[2], SNWE[2]+latlon_step[1]*(width)])

    x1, y1 = lalo2xy(N, W, snwe1, latlon_step1)
    x2, y2 = lalo2xy(N, W, snwe2, latlon_step2)        

    box1 = [y1, y1 + length, x1, x1 + width]
    box2 = [y2, y2 + length, x2, x2 + width]

    grid_lats, grid_lons = np.mgrid[SNWE[1]:SNWE[0]+latlon_step[0]:(length)*1j,
                                    SNWE[2]:SNWE[3]+latlon_step[1]:(width)*1j]

    return box1, box2, SNWE, grid_lats, grid_lons