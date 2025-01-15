#!/usr/bin/env python3
import os
import joblib
import tqdm
import numpy as np
import hectorp_wrapper
from hectorp.calendar import compute_mjd
from pathlib import Path
import time
from mintpy import subset

from mintpy.objects import HDFEOS, cluster, giantTimeseries
from mintpy.utils import plot as pp, ptime, readfile, time_func, utils as ut, writefile
from mintpy.objects import timeseries

def dt2mjd(x):
    return compute_mjd(x.year, x.month, x.day, 0, 0, 0)

DATA_TYPE = np.float32
# key configuration parameter name
key_prefix = 'mintpy.timeFunc.'
config_keys = [
    # date
    'startDate',
    'endDate',
    'excludeDate',
    # time functions
    'polynomial',
    'periodic',
    'stepDate',
    'exp',
    'log',
    # uncertainty quantification
    'uncertaintyQuantification',
    'timeSeriesCovFile',
    'bootstrapCount',
]


def read_date_info(inps):
    """Read dates used in the estimation and its related info.

    Parameters: inps - Namespace
    Returns:    inps - Namespace, adding the following new fields:
                       date_list - list of str, dates used for estimation
                       dropDate  - 1D np.ndarray in bool in size of all available dates
    """
    # initiate and open time-series file object
    ftype = readfile.read_attribute(inps.timeseries_file)['FILE_TYPE']
    print(ftype)
    if ftype == 'timeseries':
        ts_obj = timeseries(inps.timeseries_file)
    elif ftype == 'giantTimeseries':
        ts_obj = giantTimeseries(inps.timeseries_file)
    elif ftype == 'HDFEOS':
        ts_obj = HDFEOS(inps.timeseries_file)
    else:
        raise ValueError(f'Un-recognized time-series type: {ftype}')
    ts_obj.open()

    # exclude dates - user inputs
    ex_date_list = ptime.get_exclude_date_list(
        date_list=ts_obj.dateList,
        start_date=inps.startDate,
        end_date=inps.endDate,
        exclude_date=inps.excludeDate)

    # exclude dates - no obs data [for offset time-series only for now]
    if os.path.basename(inps.timeseries_file).startswith('timeseriesRg'):
        data, atr = readfile.read(inps.timeseries_file)
        flag = np.nansum(data, axis=(1,2)) == 0
        flag[ts_obj.dateList.index(atr['REF_DATE'])] = 0
        if np.sum(flag) > 0:
            print(f'number of empty dates to exclude: {np.sum(flag)}')
            ex_date_list += np.array(ts_obj.dateList)[flag].tolist()
            ex_date_list = sorted(list(set(ex_date_list)))

    # dates used for estimation - inps.date_list
    inps.date_list = [i for i in ts_obj.dateList if i not in ex_date_list]

    # flag array for ts data reading
    inps.dropDate = np.array([i not in ex_date_list for i in ts_obj.dateList], dtype=np.bool_)

    # print out msg
    print('-'*50)
    print(f'dates from input file: {ts_obj.numDate}\n{ts_obj.dateList}')
    print('-'*50)
    if len(inps.date_list) == len(ts_obj.dateList):
        print('using all dates to calculate the time function')
    else:
        print(f'dates used to estimate the time function: {len(inps.date_list)}\n{inps.date_list}')
    print('-'*50)

    return inps

def get_mintpy_data_subset(mintpy_ts: str, lon_lim : list, lat_lim: list):
    atr_dict = dict(bbox=None, geo_bbox=None, date_list=None, ref_date=None, 
                    ref_point=None, seconds=None, extent=None)

    ## Get dates
    obj = timeseries(str(mintpy_ts))
    obj.open(print_msg=True)
    atr_dict['date_list'] = obj.dateList
    obj.close()

    # Get subset
    atr = readfile.read_attribute(str(mintpy_ts))
    atr_dict['ref_date'] = atr['REF_DATE']
    atr_dict['ref_point'] = [np.float64(atr['REF_LAT']), np.float64(atr['REF_LON'])]

    if lon_lim is None:
        lon_lim= [np.float64(atr['X_FIRST']), np.float64(atr['X_FIRST']) + np.float64(atr['WIDTH']) * np.float64(atr['X_STEP'])]
    if lat_lim is None:
        lat_lim= [np.float64(atr['Y_FIRST']), np.float64(atr['Y_FIRST']) + np.float64(atr['LENGTH']) * np.float64(atr['Y_STEP'])]
    print(lon_lim, lat_lim)
    bbox_dict = dict(subset_lon=lon_lim,
                subset_lat=lat_lim)
    atr_dict['bbox'], atr_dict['geo_bbox'] = subset.subset_input_dict2box(bbox_dict, atr)

    # Get seconds from UTC time
    #seconds = np.float32(atr['UTCTime (HH:MM:SS.ss)'].split(':')[0]) * 3600
    #seconds += np.float32(atr['UTCTime (HH:MM:SS.ss)'].split(':')[1]) * 60
    #seconds += np.float32(atr['UTCTime (HH:MM:SS.ss)'].split(':')[2])
    #atr_dict['seconds'] = seconds
    atr_dict['extent'] = [atr_dict['geo_bbox'][0], atr_dict['geo_bbox'][2],
                          atr_dict['geo_bbox'][3], atr_dict['geo_bbox'][1]]
    atr_dict['heading'] = np.float32(atr['HEADING'])

    return atr_dict

##########################################


# Path to files 
work_dir = Path('/u/trappist-r0/govorcin/01_OPERA/VLM/Houston/opera_disp/MINTPY')
insar_ts = work_dir / 'timeseries_cor_ERA5.h5'


# prepare input
class Inps:
    pass
atr = readfile.read_attribute(insar_ts)
inps = Inps()
inps.timeseries_file = insar_ts
inps.startDate = None
inps.endDate = None
inps.excludeDate = []
inps.polynomial = 1
inps.periodic = [1., 0.5]
inps.stepDate = []
inps.polyline = []
inps.exp = [] # 20181026 60 date, tau
inps.log = []
inps.outfile = 'velocity.h5'
inps.save_res = True
inps.maxMemory = 0.1
inps.res_file = 'timeseriesResidual.h5'
inps.timeSeriesCovFile = 'cov'
inps.bootstrapCount = 400
inps.ref_yx = [int(atr['REF_Y']), int(atr['REF_X'])]
inps.ref_date = atr['REF_DATE'] 
inps.uncertaintyQuantification = 'residue' 

# basic file info
atr = readfile.read_attribute(insar_ts)
length, width = int(atr['LENGTH']), int(atr['WIDTH'])

ds_name_dict = {
    "velocity" : [np.float32, (length, width), None],
    "velocity_std" : [np.float32, (length, width), None]
}

atrV = dict(atr)
atrV['FILE_TYPE'] = 'velocity'
atrV['UNIT'] = 'm/year'

writefile.layout_hdf5('vel_hector.h5',
                      ds_name_dict=ds_name_dict,
                      metadata=atrV)

# read date info
inps = read_date_info(inps)
num_date = len(inps.date_list)
dates = np.array(inps.date_list)
seconds = atr.get('CENTER_LINE_UTC', 0)

# use the 1st date as reference if not found, e.g. timeseriesResidual.h5 file
if "REF_DATE" not in atr.keys() and not inps.ref_date:
    inps.ref_date = inps.date_list[0]
    print('WARNING: No REF_DATE found in time-series file or input in command line.')
    print(f'  Set "--ref-date {inps.date_list[0]}" and continue.')

# get deformation model from inputs
model = time_func.inps2model(inps, date_list=inps.date_list)
num_param = time_func.get_num_param(model)

memoryAll = (num_date + num_param * 2 + 2) * length * width * 4
num_box = int(np.ceil(memoryAll * 3 / (inps.maxMemory * 1024**3)))
num_box=1
box_list, num_box = cluster.split_box2sub_boxes(
    box=(0, 0, width, length),
    num_split=num_box,
    dimension='y',
    print_msg=True,
)

# Dict
mintpy_dict = get_mintpy_data_subset(insar_ts,
                                     lon_lim=None,
                                     lat_lim=None)

# Read
i=0
box = box_list[i]
box_wid = box[2] - box[0]
box_len = box[3] - box[1]
num_pixel = box_len * box_wid
if num_box > 1:
    print(f'\n------- processing patch {i+1} out of {num_box} --------------')
    print(f'box width:  {box_wid}')
    print(f'box length: {box_len}')

# initiate output
m = np.zeros((num_param, num_pixel), dtype=DATA_TYPE)
m_std = np.zeros((num_param, num_pixel), dtype=DATA_TYPE)

# read input
print(f'reading data from file {inps.timeseries_file} ...')
ts_data = readfile.read(inps.timeseries_file, box=box)[0]

# mask invalid pixels
print('skip pixels with zero/nan value in all acquisitions')
ts_stack = np.nanmean(ts_data, axis=0)
mask = np.multiply(~np.isnan(ts_stack), ts_stack!=0.)
del ts_stack

ts_data = ts_data[:, mask]
num_pixel2inv = int(np.sum(mask))
idx_pixel2inv = np.where(mask)[0]
print('number of pixels to invert: {} out of {} ({:.1f}%)'.format(
    num_pixel2inv, num_pixel, num_pixel2inv/num_pixel*100))

# Get mjd
mjd = list(map(dt2mjd, ptime.date_list2vector(mintpy_dict['date_list'])[0]))

def process_column(column, ix):
    result = hectorp_wrapper.estimate_trend(mjd, column,
                                            fname=f'data_{ix}.mom',
                                            ctl_name=f'control_{ix}.ctl',
                                            sampling_period=12)

    return np.c_[result[0]['trend'], result[0]['trend_sigma']]

# Assuming ts_data is a NumPy array
ts_data_T = ts_data.T

# Set the number of parallel jobs, adjust as needed
num_jobs = 150  # Use all available cores

# Use joblib to parallelize the loop
vel_parallel = np.array(
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(process_column)(obs, ix) 
        for ix, obs in enumerate(tqdm.tqdm(ts_data_T))
    )
)

mshape = (box_len, box_wid)
vel = np.zeros(mshape)
vel_std = np.zeros(mshape)
vel[mask] = np.vstack(vel_parallel[:])[:,0]
vel_std[mask] = np.vstack(vel_parallel[:])[:,1]
vel = vel.reshape(mshape)
vel_std = vel_std.reshape(mshape)

block = [box[1], box[3], box[0], box[2]]
ds_dict = {'velocity': vel,
           'velocity_std': vel_std}

for ds_name, data in ds_dict.items():
    writefile.write_hdf5_block('vel_hector.h5',
                                data=data.reshape(box_len, box_wid),
                                datasetName=ds_name,
                                block=block)
