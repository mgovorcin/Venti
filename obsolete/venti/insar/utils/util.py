#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:49:16 2022

@author: govorcin
"""

import os
import numpy as np
from osgeo import gdal, osr


def geotiff_to_array(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    trans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    
    snwe = [trans[3] + ysize*trans[5], trans[3], trans[0], trans[0] + xsize * trans[1]]
    
    ds = None
    
    return array, snwe


def array_to_geotiff(filename, array, snwe, epsg=4326):
    if array.dtype == np.float32:
        array_type = gdal.GDT_Float32
    elif array.dtype == np.float64:
        array_type = gdal.GDT_Float64   
    else:
        array_type = gdal.GDT_Int32
        
    x_step = (snwe[3] - snwe[2]) / array.shape[1]
    y_step = (snwe[0] - snwe[1]) / array.shape[0]
               
    geo = (snwe[2], x_step, 0, snwe[1], 0, y_step)   
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, array.shape[1], array.shape[0], 1, array_type)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds.SetGeoTransform(geo)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    band.ComputeStatistics(False)