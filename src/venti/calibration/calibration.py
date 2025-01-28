#!/usr/bin/env python3
import numpy as np
from scipy import interpolate
from skimage.filters import gaussian
from tqdm import tqdm

from .utils import get_moving_windows, fill_gaps, extend_window, get_win_lalo_grid, degrees_to_meters
from venti.solvers.solvers import fit_plane, get_distance_kernel, hamming2d_filter

def insar_referencing(insar_data:np.ndarray,
                       insar_std:np.ndarray,
                       inc_angle:np.ndarray,
                       az_angle:np.ndarray,
                       gnss_data:list,
                       gnss_latlon:tuple,
                       win_xsize:int,
                       win_ysize:int,
                       win_overlap_x:int,
                       win_overlap_y:int,
                       win_extend_x:int,
                       win_extend_y:int,
                       attr:dict,
                       mask:np.ndarray=None,
                       decimate:int=1,
                       poly_order:float=1.5,
                       filter_residuals:bool=False):
    
    # gnns_data = [ew, ns, v, se, sn, sv]
    # Get shape
    length, width = insar_data.shape

    # Get all mov. windows
    # find win along axis-y
    wins = get_moving_windows(insar_data, 
                              win_xsize, win_ysize,
                              win_overlap_x, win_overlap_y)

    # Initialize
    m = np.zeros(insar_data.shape)
    m_std = np.zeros(insar_data.shape)

    if mask is not None:
        print('Use mask and fill the gaps')
        insar_data = np.ma.masked_array(insar_data, mask=mask)
        insar_data = fill_gaps(insar_data.filled(fill_value=0),
                                smoothingIterations=10)
        
        # NOTE run it on interpolated grid to avoid gross outliers
        if filter_residuals:
            print('Filtering gap-filled residuals!')
            insar_data = gaussian(insar_data, (5,5))

    for ik, ix in tqdm(enumerate(wins),total=len(wins)):
        win, pad = extend_window(ix[0], ix[1],
                                  win_extend_y, win_extend_x,
                                  length, width)
        # Get windows
        win_y = [win[0].start, win[0].stop]
        win_x = [win[1].start, win[1].stop]  
        win_lons, win_lats = get_win_lalo_grid(attr,
                                               win_y=win_y,
                                               win_x=win_x)
        
        dn, de, dv = disp_unit_vector(inc_angle[win],
                                      az_angle[win])

        # Interpolate
        gps_interpolated = dict(ew=None, ns=None, v=None,
                                se=None, sn=None, sv=None)
        
        for gnss, key in zip(gnss_data, gps_interpolated.keys()):  
            gps_interpolated[key] = (interpolate.griddata(gnss_latlon,
                                                          gnss,
                                                          (win_lats, win_lons),
                                                          method='linear', 
                                                          fill_value=np.nan))
            
        # Project GNSS to InSAR LOS
        gnss_los = np.array(np.multiply(gps_interpolated['ew'], de)
                          + np.multiply(gps_interpolated['ns'], dn) 
                          + np.multiply(gps_interpolated['v'], dv))

        gnss_los_std = np.sqrt(np.multiply(gps_interpolated['se'], np.abs(de))**2
                             + np.multiply(gps_interpolated['sn'], np.abs(dn))**2 
                             + np.multiply(gps_interpolated['sv'], np.abs(dv)))**2
        
        # Get residuals
        res = insar_data[win] - gnss_los
        res_std = np.sqrt((insar_std[win])**2 + gnss_los_std**2)
        mask = np.ma.masked_invalid(res).mask

        # Skip if data covers less than 1% of bin
        data_percentage = res[pad][np.isnan(res[pad])].size / res[pad].size
        if data_percentage > 0.99:
            print(f'Skip window: {ik}, Nodata: {data_percentage*100}%')
            continue

        # Distance kernel: adds some time, do not see too much difference
        # turn off by now
        #TODO refine the get_distance_kernel code and make it faster
        #dist_kernel = get_distance_kernel(win, pad, penalize_pad=True)
        dist_kernel = None 
        
        # Fitting plane
        plane, plane_std = fit_plane(res, res_std,
                                     win_lons, win_lats,
                                     order=poly_order, decimate=decimate,
                                     dist_weights=dist_kernel)

        m[tuple(ix)] = np.ma.masked_array(plane[pad],
                                mask=mask[pad]).filled(fill_value=0)
        m_std[tuple(ix)] = np.ma.masked_array(plane_std[pad],
                                mask=mask[pad]).filled(fill_value=0)
        
    return m, m_std

def get_calibration_plane_fft(insar_data:np.ndarray,
                              gnss_los:np.ndarray,
                              fft_xsize:int,
                              fft_ysize:int,
                              fft_cutoff:float=1.,
                              mask:np.ndarray=None,
                              gap_fill:bool=True,
                              rotate_kernel:float=0.,
                              nodata=None):
    
    print('Referencing InSAR with respect to GNSS') 
    # Get the residual plane between InSAR and GNSS
    insar_gnss_res = insar_data - gnss_los

    if nodata is not None:
        # Change nodata to nan to avoid interpolation outside of track
        print(f' ## Mask pixels with {nodata}')
        insar_gnss_res = np.ma.masked_equal(insar_gnss_res,
                                            nodata)
        insar_gnss_res = insar_gnss_res.filled(fill_value=np.nan)

    # Mask the residual plane
    if mask.any() == None:
        print(' ## Mask invalid pixels!')
        mask = np.ma.masked_invalid(insar_gnss_res)
    else:
        print(' ## Use mask input!')
        insar_gnss_res = np.ma.masked_array(insar_gnss_res, mask=mask)

    # Interpolate gaps
    # NOTE: Smooth interation add some time
    if gap_fill:
        print(' ## Fill gaps using gdal!')
        insar_gnss_res = fill_gaps(insar_gnss_res.filled(fill_value=0),
                                    smoothingIterations=10,
                                    maxSearchDist=int(np.max(insar_gnss_res.shape)))
    else:
        # Remove nan from data
        insar_gnss_res = insar_gnss_res.filled(np.nanmean(insar_gnss_res))

    
    # FFT Low pass filtering
    if nodata is not None:
        insar_gnss_res = np.ma.masked_invalid(insar_gnss_res)
        res_mean = np.nanmean(insar_gnss_res)
        # Fill with mean 
        insar_gnss_res = insar_gnss_res.filled(fill_value=res_mean)

    print(' ## Run FFT filtering:!')
    print(f' ## Kernel: x_size: {fft_xsize},  y_size: {fft_ysize}')
    print(f' ## Kernel: cutoff: {fft_cutoff}, rotation: {rotate_kernel} deg')

    filtered_res = hamming2d_filter(insar_gnss_res,
                                    kernel_x=fft_xsize,
                                    kernel_y=fft_ysize,
                                    angle=rotate_kernel, 
                                    cut_off=fft_cutoff)[0]
    
    return filtered_res, insar_gnss_res

def disp_unit_vector(incidenceAngle, azimuthAngle):
    '''
    azimuthAngle  - 0 at east, as it is in ISCE2 convention 

    '''
    dn = np.multiply(np.sin(np.deg2rad(np.float64(incidenceAngle))), np.cos(np.deg2rad(np.float64(azimuthAngle))))
    de = np.multiply(np.sin(np.deg2rad(np.float64(incidenceAngle))), -np.sin(np.deg2rad(np.float64(azimuthAngle))))
    dv = np.cos(np.deg2rad(np.float64(incidenceAngle)))
    
    return dn, de, dv

def get_fft_size(attr, filter_wavelength):
    # Get pixel spacing along axes
    y_spacing, x_spacing = degrees_to_meters(np.float64(attr['REF_LAT']),
                                             np.float64(attr['REF_LON']),
                                             np.float64(attr['Y_STEP']),
                                             np.float64(attr['X_STEP']))

    print(f'Pixel spacing along axis-y [lat]: {y_spacing:.2f} m')
    print(f'Pixel spacing along axis-x [lon]: {x_spacing:.2f} m')

    # Get fft kernel size for low-pass masking
    input_shape = np.int16([attr['LENGTH'], attr['WIDTH']])
    yx_km = input_shape * np.array([y_spacing, x_spacing])
    win_y, win_x = np.ceil(yx_km / filter_wavelength)

    win_y = round(win_y / 2) * 2
    win_x = round(win_x / 2) * 2

    print(f'Filter kernel size along axis-y [lat]: {win_y:.0f} px')
    print(f'Filter kernel size along axis-x [lon]: {win_x:.0f} px')
    return win_x, win_y
