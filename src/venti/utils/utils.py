import numpy as np
import pyproj
from osgeo import gdal
from mintpy.utils import utils as ud

# Interpolation
def fill_gaps(array: np.ndarray, fill_value:int=0, **kwargs) -> np.ndarray:
    # Check kwargs
    default_kwargs = dict(maxSearchDist = int(np.max(array.shape) // 4),
                          maskBand = None, 
                          smoothingIterations=0)
    kwargs = { **default_kwargs, **kwargs }
    # Create a temporary in-memory raster using GDAL
    driver = gdal.GetDriverByName('MEM')
    rows, cols = array.shape
    dataset = driver.Create('', cols, rows, 1, gdal.GDT_Float32)

    # Write the NumPy array to the GDAL raster band
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)

    # Set nodata value for the band
    band.SetNoDataValue(fill_value) 
    # Interpolate gaps
    gdal.FillNodata(targetBand = band, **kwargs)
    interpolated_array = band.ReadAsArray()
    dataset = None

    return interpolated_array

## Subseting
def get_sliding_window(length:int, win_size:int,
                       win_overlap:int, first:int=0, 
                       end:int=0):
    stop = win_size
    ix = 1
    windows = []
    if end ==0: end = length
    while stop < end:
        start = first + (ix-1) * win_size - win_overlap * (ix-1)
        stop = first + ix * win_size - win_overlap * (ix-1) 
        ix += 1
        windows.append(np.s_[start:stop])
    windows[-1] = np.s_[windows[-1].start:end]
    return windows

def find_start_stop(input:np.ndarray, axis:int=0):
    data_count = np.count_nonzero(input, axis=axis)
    start = np.min(np.where(data_count != 0))
    stop = np.max(np.where(data_count != 0))
    return start, stop

def get_window_snwe(attr:dict, win_x:list, win_y:list):
    # From the upper right corner
    coord = ud.coordinate(attr)
    box = [win_x[0], win_y[0], win_x[1], win_y[1]]
    w, n, e, s = coord.bbox_radar2geo(box)
    return np.r_[s, n, w, e]

def get_win_lalo_grid(attr:dict, 
                       win_x:list, win_y:list) -> [np.ndarray, np.ndarray]:
    snwe = get_window_snwe(attr, win_x, win_y)
    lons = np.linspace(snwe[2], snwe[3],win_x[1]- win_x[0])#np.float64(attr['X_STEP']))[:-1]
    lats = np.linspace(snwe[1], snwe[0],win_y[1]- win_y[0])#, win_x1[1]-win_x1[0])np.float64(attr['Y_STEP']))[:-1]
    return np.meshgrid(lons, lats)

def extend_window(slice_y:tuple, slice_x:tuple,
                  extend_y:int, extend_x:int,
                  length:int, width:int):
    
    # axis 0 - y direction
    if slice_y.start - extend_y < 0:
        y_start = 0 
    else:
        y_start = slice_y.start - extend_y 

    if slice_y.stop + extend_y > length:
        
        y_stop = length
    else:
        y_stop = slice_y.stop + extend_y

    # axis 1 - x direction
    if slice_x.start - extend_x < 0:
        x_start = 0 
    else:
        x_start = slice_x.start - extend_x 

    if slice_x.stop + extend_x > width:
        x_stop = width
    else:
        x_stop = slice_x.stop + extend_x

    extended_win = np.s_[y_start:y_stop, x_start:x_stop]
    nlength = y_stop - y_start
    nwidth = x_stop - x_start
    padding = np.s_[slice_y.start - y_start:nlength-(y_stop - slice_y.stop),
                    slice_x.start - x_start:nwidth -(x_stop - slice_x.stop)]

    return extended_win, padding 

def get_moving_windows(insar_data,
                        win_xsize, win_ysize,
                        win_overlap_x, win_overlap_y):
    length, width = insar_data.shape

    # Get all mov. windows
    # find win along axis-y
    y_start, y_stop = find_start_stop(insar_data, axis=1)
    win_ys = get_sliding_window(length, win_ysize,
                                win_overlap_y, y_start, y_stop)
    
    wins = []
    for win_y in win_ys:
        x_start, x_stop = find_start_stop(insar_data[win_y, :], axis=0)
        win_xs = get_sliding_window(x_stop - x_start, 
                                    win_xsize, win_overlap_x,
                                    x_start, x_stop)
        wins.append([(win_y, win_x) for win_x in win_xs])
    wins = np.vstack(wins)
    
    return wins

## Other

def snwe_to_extent(snwe):
    return [snwe[2], snwe[3], snwe[0], snwe[1]]


def degrees_to_meters(lat_degrees, lon_degrees,
                      lat_spacing_degrees, lon_spacing_degrees):
    # Define the geographic coordinate system using WGS84 datum (standard for GPS)
    wgs84 = pyproj.Geod(ellps='WGS84')
    
    # Compute the length of one degree of latitude and longitude at the given latitude
    lat_spacing_meters = wgs84.inv(lat_degrees, lon_degrees, lat_degrees + lat_spacing_degrees, lon_degrees)[2]
    lon_spacing_meters = wgs84.inv(lat_degrees, lon_degrees, lat_degrees, lon_degrees + lon_spacing_degrees)[2]

    return lat_spacing_meters, lon_spacing_meters 
  