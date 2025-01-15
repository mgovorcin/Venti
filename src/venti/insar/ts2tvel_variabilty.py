#!/usr/bin/env python3
import argparse
import numpy as np
from scipy import linalg
import joblib
from pathlib import Path
from scipy import stats
from datetime import datetime

# MIDAS
import sys
sys.path.insert(0, '/home/govorcin/aux/dev')
import midas

# Mintpy
from mintpy.timeseries2velocity import run_or_skip, read_date_info
from mintpy.cli.timeseries2velocity import DROP_DATE_TXT
from mintpy.utils import ptime, readfile, writefile, arg_utils, time_func
from mintpy.objects import timeseries, cluster


config_keys = [
    # date
    'startDate',
    'endDate',
    'excludeDate']

MODEL = {
        'polynomial' : 1,                    
        'periodic'   : [1.0, 0.5],           
        'stepDate'   : [],
        'polyline'   : [],        
        'exp'        : {},
        'log'        : {}}

def create_parser():
    parser = argparse.ArgumentParser(
        description='Velocity estimation form ts with Hector')
    parser.add_argument('timeseries_file',
                        help='Time series file for time function estimation.')
    parser = arg_utils.add_reference_argument(parser, plot=False)

    date = parser.add_argument_group('Dates of interest')
    date.add_argument('-s', '--start-date', dest='startDate', default=None,
                      help='start date for time function estimation')
    date.add_argument('-e', '--end-date', dest='endDate', default=None,
                      help='end date for time function estimation')
    date.add_argument('--ex', '--ex-date', dest='excludeDate', nargs='+', default=[],
                      help='date(s) not included in time function estimation, i.e.:\n' +
                           '--exclude 20040502 20060708 20090103\n' +
                           '--exclude exclude_date.txt\n'+DROP_DATE_TXT)
    variability = parser.add_argument_group('Velocity variability')
    variability.add_argument('--step', '--step-date', dest='stepDate',
                             type=str, nargs='+', default=[],
                             help='step function(s) at YYYYMMDD (default: %(default)s). E.g.:\n'
                             '--step 20061014          # coseismic step  at 2006-10-14T00:00')
    variability.add_argument('--periodic', '--period', '--peri', dest='periodic', type=float, 
                             nargs='+', default=[1.],
                             help='periodic function(s) with period in decimal years (default: %(default)s). E.g.:\n'
                             '--periodic 1.0                         # an annual cycle\n'
                             '--periodic 1.0 0.5                     # an annual cycle plus a semi-annual cycle\n')
    variability.add_argument('--min_span', dest='min_window',
                             type=int, default=2,
                             help='Minimum window size for moving window velocity estimation')
    variability.add_argument('--max_span', dest='max_window',
                             type=int, default=None,
                             help='Maximum window size for moving window velocity estimation')
    variability.add_argument('--window_offset', dest='offset',
                             type=float, default=None,
                             help='Offset in yr for moving window velocity estimation' \
                                  + ' if None, offset is half the moving window')
    variability.add_argument('--method', dest='method', type=str, default='mad',
                             choices=['mad', 'median', 'wrms'],
                             help='Method to estimate temp velocity variability')
    variability.add_argument('--inv_method', dest='inv_method', type=str, default='ols',
                             choices=['midas', 'ols'],
                             help='Method for velocity estimation')
    parser.add_argument('-o', '--output', dest='outfile',
                        help='output file name')
    parser.add_argument('--n_jobs', default=1,
                        help='Number of jobs for parallel run',
                        type=int, dest='n_jobs')
    parser.add_argument('--update', dest='update_mode', action='store_true',
                        help='Enable update mode, and skip estimation if:\n' +
                             '1) output file already exists, readable ' +
                             'and newer than input file\n' +
                             '2) all configuration parameters are the same.')
    return parser


def str2datetime(date):
    # Convert string date to datetime
    dformat = ptime.get_date_str_format(date)
    return datetime.strptime(date, dformat)


def cmd_line_parse(iargs=None):
    """Command line parser."""
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # check
    atr = readfile.read_attribute(inps.timeseries_file)

    # check: input file type (time series is required)
    ftype = atr['FILE_TYPE']
    if ftype not in ['timeseries', 'giantTimeseries', 'HDFEOS']:
        raise Exception(f'input file is {ftype}, NOT timeseries!')

    # Convert string to datetime
    inps.stepDate = list(map(str2datetime, inps.stepDate))
    inps.stepDate = sorted(inps.stepDate)

    # default: --output option
    if not inps.outfile:
        # compose default output filename
        inps.outfile = f'dV_{inps.method}.h5'

    return inps

# MINTPY utils
def read_timeseries(infile, date_list=None,
                    box=None, ref_date=None,
                    ref_yx=None):
    # Read and re-reference timeseries data from mintpy timeserie.h5
    if box:
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]
        # num_pixel = box_len * box_wid
        print(f'box width:  {box_wid}')
        print(f'box length: {box_len}')

    # read input
    print(f'reading data from file {infile} ...')
    ts_data = readfile.read(infile, box=box)[0]

    if date_list is None:
        date_list = timeseries(infile).get_date_list()

    date_list2 = np.array(ptime.date_list2vector(date_list)[0])
    dates_np = date_list2.astype('datetime64[s]')

    # referencing in time and space
    if ref_date:
        print(f'referencing to date: {ref_date}')
        ref_ind = date_list.index(ref_date)
        ts_data -= np.tile(ts_data[ref_ind, :, :],
                           (ts_data.shape[0], 1, 1))

    if ref_yx:
        print(f'referencing to point (y, x): ({ref_yx[0]}, {ref_yx[1]})')
        ref_box = (ref_yx[1], ref_yx[0], ref_yx[1]+1, ref_yx[0]+1)
        ref_val = readfile.read(infile, box=ref_box)[0]
        ts_data -= np.tile(ref_val.reshape(ts_data.shape[0], 1, 1),
                           (1, ts_data.shape[1], ts_data.shape[2]))

    return ts_data, dates_np


def get_box_list(length, width,
                 num_date, num_param=1,
                 max_memory=4,
                 print_msg=True):
    # Get subset boxlist based on maximum memory

    memoryall = (num_date + num_param * 2 + 2)
    memoryall *= (length * width * 4)

    num_box = int(np.ceil(memoryall * 3 / (max_memory * 1024**3)))
    box_list, num_box = cluster.split_box2sub_boxes(
        box=(0, 0, width, length),
        num_split=num_box,
        dimension='y',
        print_msg=print_msg,
    )
    return box_list

def filter_steps(dates, steps):
    step_list = []
    for step in steps:
        if (step > dates.min()) & (step < dates.max()):
            step_list.append(datetime.strftime(step, '%Y%m%d'))

    return step_list

def get_velocity_variability(dates, data, time_increment,
                             method='midas', steps=None,
                             periodic=[1.0, 0.5],
                             seconds=None, offset=None):
    # time increment in yr
    time_increment_dt = np.timedelta64(365, 'D') * time_increment

    # Set moving windows offset, in yr
    if offset is None:
        offset = np.timedelta64(365, 'D') * (time_increment / 2)
    else:
        offset *= np.timedelta64(365, 'D')

    # Get moving windows
    date1 = dates[0]
    moving_window = np.c_[True, True]
    moving_dates = [date1]

    while moving_window.any():
        moving_window = dates > date1 + offset

        if ~moving_window.any():
            break

        tdiff = np.timedelta64(
            (np.max(dates) - dates[moving_window][0]), 'D')
        if tdiff > time_increment_dt:
            moving_dates.append(dates[moving_window][0])
        date1 = dates[moving_window][0]

    # Get variable velocities
    vels, vel_stds = [], []

    # Modify model
    model = MODEL
    model['periodic'] = periodic
    model['stepDate'] = steps

    if method == 'midas':
        steps = list(map(str2datetime, steps))

    for mov_date in moving_dates:
        condition1 = dates > mov_date
        condition2 = dates < mov_date + time_increment_dt
        mask = condition1 * condition2

        # Get velocity
        if method == 'midas':
            vel, vel_std, _, _ = midas.midas(dates[mask],
                                             data[mask, :],
                                             steps=steps)
        elif method == 'ols':
            # Make sure step date is in the window
            msteps = filter_steps(dates[mask], steps)
            model['stepDate'] = msteps

            m, m_std = mintpy_ols(dates[mask],
                                  data[mask, :],
                                  model,
                                  seconds)

            # Velocity is index 1
            vel = m[1]
            vel_std = m_std[1]
        else:
            raise ValueError('Unkown method specified!!')

        if vel is not None:
            vels.append(vel)
            vel_stds.append(vel_std)

    return moving_dates, vels, vel_stds

def mintpy_ols(date_list, data, model, seconds):
    if isinstance(date_list, np.ndarray):
        date_list = list(map(lambda x: datetime.strftime(x, '%Y%m%d'),
                             date_list.tolist()))

    num_param = time_func.get_num_param(model)
    num_date = len(date_list)

    # Get design matric
    G = time_func.get_design_matrix4time_func(date_list,
                                              model,
                                              seconds=seconds)
    # Invert
    m, e2 = linalg.lstsq(G, data, cond=None)[:2]

    # Get variables std
    if linalg.det(np.dot(G.T, G)) == 0.:
        msg = 'G_inv is singular, return zeros for var stds'
        print('\033[93m' + msg + '\033[0m')
        m_std = np.zeros(m.shape) 
    else:
        G_inv = linalg.inv(np.dot(G.T, G))
        m_var = e2.reshape(1, -1) / (num_date - num_param)
        m_std = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), m_var))

    return m, m_std


def _weighted_rms(predict, target, weights=None, axis=0):
    if weights is None:
        weights = np.ones(predict.shape)
    ssum = np.nansum(weights * (predict - target)**2, axis=axis)
    wsum = np.nansum(weights, axis=axis)
    return np.sqrt(ssum / wsum)

def run_temporal_variability(dates,
                             data,
                             min_window=2,
                             max_window=None,
                             method='mad',
                             inv_method='midas',
                             periodic=[1.0, 0.5],
                             steps=None,
                             seconds=None,
                             offset=None):

    # Get duration
    print(f'Estimate velocities with: {inv_method}')
    duration = (np.max(dates) - np.min(dates))
    duration /= (np.timedelta64(1, 'D') * 365.25)
    if max_window is None:
        max_window = np.int16(duration - duration / 4)

    # Get MIDAS velocities in moving time windows
    # Initialize
    mdates_list, mvels_list, mvstds_list = [], [], []

    # Loop through moving windows
    for dt in np.arange(min_window, max_window):
        # Offset window if it is 1yr to avoid midas failing
        if dt == 1:
            dt += 0.1

        mdates, mvels, mvstds = get_velocity_variability(dates,
                                                         data,
                                                         dt,
                                                         inv_method,
                                                         steps,
                                                         periodic,
                                                         seconds,
                                                         offset)
        if mdates:
            mdates_list.append(np.array(mdates))
            mvels_list.append(np.vstack(mvels))
            mvstds_list.append(np.vstack(mvstds))

    # Convert to array
    vv = np.vstack(mvels_list)
    vs = np.vstack(mvstds_list)

    # Get MIDAS for whole time period
    if inv_method == 'midas':
        v = midas.midas(dates, data, steps=steps)[0]
    elif inv_method == 'ols':
        # Modify model
        model = MODEL
        model['periodic'] = periodic
        if steps:
            model['stepDate'] = [datetime.strftime(step, '%Y%m%d')
                                 for step in steps]
        v = mintpy_ols(dates, data, model, seconds=seconds)[0][1]

    # Estimate temporal velocity variabilty
    if method == 'mad':
        d_vel = stats.median_abs_deviation(vv - v, axis=0)
    elif method == 'median':
        # Root square median error
        d_vel = np.sqrt(np.nanmedian((vv - v)**2, axis=0))
    elif method == 'wrms':
        # get weights from vel stds
        w = 1 / (vs**2)
        d_vel = _weighted_rms(vv, v, w)
    else:
        raise ValueError('Unknown option to calculate velocity variability!')

    return d_vel

def run_temp_variabilty_patch(inps, atr, box, box_ix):
    print(f'\n------- processing patch {box_ix[0]} of {box_ix[1]} --------------')
    data, dates = read_timeseries(inps.timeseries_file, box=box)
    dates = dates[inps.dropDate]
    num_date, dwid, dlen = data.shape
    num_date -= np.sum(~inps.dropDate)
    num_pixel = dlen * dwid

    if atr['UNIT'] == 'mm':
        data *= 1./1000.

    ts_data = data[inps.dropDate, :, :].reshape(num_date, -1)
    # Prepare data
    print('skip pixels with zero/nan value in all acquisitions')
    ts_stack = np.nanmean(ts_data, axis=0)
    mask = np.multiply(~np.isnan(ts_stack), ts_stack != 0.)
    del ts_stack

    # include the reference point
    ry = int(atr['REF_Y']) - box[1]
    rx = int(atr['REF_X']) - box[0]

    if 0 <= rx < dwid and 0 <= ry < dlen:
        mask[ry * dwid + rx] = 1

    ts_data = ts_data[:, mask]
    num_pixel2inv = int(np.sum(mask))
    # idx_pixel2inv = np.where(mask)[0]
    print('number of pixels to invert: {} out of {} ({:.1f}%)'.format(
        num_pixel2inv, num_pixel, num_pixel2inv/num_pixel*100))

    if num_pixel2inv != 0:
        dV = run_temporal_variability(dates, ts_data,
                                      min_window=inps.min_window,
                                      max_window=inps.max_window,
                                      steps=inps.stepDate,
                                      method=inps.method,
                                      inv_method=inps.inv_method,
                                      periodic=inps.periodic,
                                      seconds=atr['CENTER_LINE_UTC'],
                                      offset=inps.offset)
        vel_dstd = np.zeros(data.shape[1:]).ravel()
        vel_dstd[mask] = dV
        vel_dstd = vel_dstd.reshape(data.shape[1:])
    else:
        vel_dstd = np.zeros(data.shape[1:])

    # Write to local file
    block = [box[1], box[3], box[0], box[2]]
    writefile.write_hdf5_block(inps.outfile,
                               data=vel_dstd,
                               datasetName='dV',
                               block=block)
    print(f'\n------- finished processing patch {box_ix[0]} of {box_ix[1]} --------------')

def ts2temporal_velvar(inps):
    atr = readfile.read_attribute(inps.timeseries_file)
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])

    # Read Date info 
    num_date = len(inps.date_list)

    # use the 1st date as reference if not found, e.g. timeseriesResidual.h5 file
    if "REF_DATE" not in atr.keys() and not inps.ref_date:
        inps.ref_date = inps.date_list[0]
        print('WARNING: No REF_DATE found in time-series file or input in command line.')
        print(f'  Set "--ref-date {inps.date_list[0]}" and continue.')

    # Output preparation
    date0, date1 = inps.date_list[0], inps.date_list[-1]
    atrV = dict(atr)
    atrV['FILE_TYPE'] = 'velocity'
    atrV['UNIT'] = 'm/year'
    atrV['START_DATE'] = date0
    atrV['END_DATE'] = date1
    atrV['DATE12'] = f'{date0}_{date1}'
    if inps.ref_yx:
        atrV['REF_Y'] = inps.ref_yx[0]
        atrV['REF_X'] = inps.ref_yx[1]
    if inps.ref_date:
        atrV['REF_DATE'] = inps.ref_date

    # Add processing attributes
    if inps.offset:
        atrV['window_offset'] = inps.offset
    atrV['min_window'] = inps.min_window
    atrV['max_window'] = inps.max_window
    atrV['method'] = inps.method
    atrV['inv_method'] = inps.inv_method

    # time_func_param: config parameter
    print(f'add/update the following configuration metadata:\n{config_keys}')
    for key in config_keys:
        atrV['mintpy.timeFunc.'+key] = str(vars(inps)[key])

    # Init output
    out_dict = dict(dV=[np.float32, (length, width), None])
    writefile.layout_hdf5(inps.outfile,
                          metadata=atrV,
                          ds_name_dict=out_dict)

    # Get boxes
    box_list = get_box_list(length, width, num_date)

    if inps.n_jobs > 1:
        joblib.Parallel(n_jobs=inps.n_jobs)(
        joblib.delayed(run_temp_variabilty_patch)(
            inps, atr, box=box, box_ix=[ix+1, len(box_list)]) 
            for ix, box in enumerate(box_list))
    else:
        for ix, box in enumerate(box_list):
            run_temp_variabilty_patch(inps, atr, box=box,
                                      box_ix=[ix+1, len(box_list)])   

def main(iargs=None):
    inps = cmd_line_parse(iargs)

    # Run of Skip
    if inps.update_mode and run_or_skip(inps) == 'skip':
        return

    # Update datelist if start/end date specified
    inps = read_date_info(inps)

    # Run
    ts2temporal_velvar(inps) 

if __name__ == '__main__':
    main(sys.argv[1:])