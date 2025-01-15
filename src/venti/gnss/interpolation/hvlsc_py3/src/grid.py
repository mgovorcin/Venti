#!/usr/bin/python
import numpy as np
from pyproj import Geod, CRS, Transformer
g = Geod(ellps='WGS84')
# Earth radius
R = g.a 

def ceil_to_exponent(value, rounding_exponent=0):
    # rounding exponent on the logarithm base 10
    exponent = np.ceil(np.log10(np.abs(value)))
    exponent -= rounding_exponent 
    return np.ceil(value / 10**exponent) * 10**exponent

def floor_to_exponent(value, rounding_exponent=0):
    # rounding exponent on the logarithm base 10
    exponent = np.ceil(np.log10(np.abs(value)))
    exponent -= rounding_exponent 
    return np.floor(value / 10**exponent) * 10**exponent

def get_obs_lalon0_extent(lon, lat):
    # find center of the aoi
    lon0 = np.round((np.nanmax(lon) - np.nanmin(lon)) / 2. + np.nanmin(lon), 1)
    lat0 = np.round((np.nanmax(lat) - np.nanmin(lat)) / 2. + np.nanmin(lat), 1)
    max_dist_lon = (g.inv(np.nanmin(lon), lat0, np.nanmax(lon), lat0)[2]) / 1e3 # to km
    max_dist_lat = (g.inv(lon0, np.nanmin(lat), lon0, np.nanmax(lat))[2]) / 1e3 # to km

    # round the max distance between observations
    rounded_dist_lon = ceil_to_exponent(max_dist_lon, 2)
    rounded_dist_lat = ceil_to_exponent(max_dist_lat, 2) 

    return lon0, lat0, rounded_dist_lon, rounded_dist_lat

def create_regular_grid(lon0, lat0, grid_width, grid_height, 
                        dx=50, dy=50, buffer_x=0, buffer_y=0, unit='km'):
                        
    # Another option EPSG:4087 
    crs_aeqd = CRS(proj='aeqd', lon_0=lon0, lat_0=lat0, datum="WGS84", units=unit)
    crs_wgs84 = CRS(proj='latlong', datum='WGS84')
    transformer = Transformer.from_crs(crs_wgs84, crs_aeqd, always_xy=True)

    # Add buffer around extent
    grid_height += buffer_y
    grid_width += buffer_x

    # define the regular grid
    xs = np.arange(-grid_width//2 - dx, grid_width//2 + dx, dx)
    ys = np.arange(-grid_height//2 - dy, grid_height//2 + dy, dy)
    xi, yi = np.meshgrid(xs, ys)

    loni, lati = transformer.transform(xi.ravel(), yi.ravel(),
                                       direction = 'INVERSE')
    return loni, lati