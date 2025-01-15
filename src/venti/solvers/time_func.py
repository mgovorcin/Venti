#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg, stats
from datetime import datetime
from . import midas

MODEL_EXAMPLE = """time function configuration:
    model = {
        'polynomial' : 2,                    # int, polynomial degree with 1 (linear), 2 (quadratic), 3 (cubic), etc.
        'periodic'   : [1.0, 0.5],           # list(float), period(s) in years. 1.0 (annual), 0.5 (semiannual), etc.
        'stepDate'   : ['20061014'],         # list(str), date(s) for the onset of step in YYYYMMDD.
        'polyline'   : ['20190101'],         # list(str), date(s) for the onset of extra line segments in YYYYMMDD.
        'exp'        : {'20181026': [60],    # dict, key for onset time in YYYYMMDD(THHMM) and value for char times in integer days.
                        ...
                        },
        'log'        : {'20161231': [80],    # dict, key for onset time in YYYYMMDD(THHMM) and value for char times in integer days.
                        '20190125': [100, 200],
                        ...
                        },
        ...
    }
"""


def date2decimalyr(date):
    d = datetime.utcfromtimestamp(np.squeeze(date.astype(int)))
    return (d.year + (d.timetuple().tm_yday - 1) / 365.25 +
            d.hour / (365.25 * 24) +
            d.minute / (365.25 * 24 * 60) +
            d.second / (365.25 * 24 * 60 * 60))


def decimalyr2date(decimal_yr):
    year = np.floor(decimal_yr).astype(int)
    yday = np.floor((decimal_yr - year) * 365.25).astype(int) + 1
    return datetime.strptime(f'{year:d}-{yday:d}', "%Y-%j")


# from Mintpy
def get_polynomial_design_matrix(yr_diff, degree):
    A = np.zeros([len(yr_diff), degree + 1], dtype=np.float32)
    for i in range(degree+1):
        A[:, i] = (yr_diff**i) / np.math.factorial(i)
    return A


def get_periodic_design_matrix(yr_diff, periods):
    num_date = len(yr_diff)
    num_period = len(periods)
    A = np.zeros((num_date, 2*num_period), dtype=np.float32)
    for i, period in enumerate(periods):
        c0, c1 = 2*i, 2*i+1
        A[:, c0] = np.cos(2*np.pi/period * yr_diff)
        A[:, c1] = np.sin(2*np.pi/period * yr_diff)
    return A


def get_step_design_matrix(date_list, step_date_list, seconds=0):
    num_date = len(date_list)
    num_step = len(step_date_list)
    A = np.zeros((num_date, num_step), dtype=np.float32)

    t = np.apply_along_axis(date2decimalyr, 0, np.atleast_2d(date_list))
    t_steps = np.apply_along_axis(date2decimalyr, 0, np.atleast_2d(
        step_date_list).astype('datetime64[s]'))
    for i, t_step in enumerate(t_steps):
        A[:, i] = np.array(t > t_step).flatten()
    return A


def get_log_exp_design_matrix(date_list, func_dict, seconds=0, func='log'):
    num_date = len(date_list)
    num_param = sum(len(val) for key, val in func_dict.items())
    A = np.zeros((num_date, num_param), dtype=np.float32)

    t = np.apply_along_axis(date2decimalyr, 0, np.atleast_2d(date_list))
    # loop for onset time(s)
    i = 0
    for onset in func_dict.keys():
        # convert string to float in years
        T = date2decimalyr(onset)

        # loop for charateristic time(s)
        for tau in func_dict[onset]:
            # convert time from days to years
            tau /= 365.25
            if func == 'exp':
                A[:, i] = np.array(t > T).flatten() * \
                    (1 - np.exp(-1 * (t - T) / tau))
            elif func == 'log':
                olderr = np.seterr(invalid='ignore', divide='ignore')
                A[:, i] = np.array(t > T).flatten() * np.nan_to_num(
                    np.log(1 + (t - T) / tau),
                    nan=0,
                    neginf=0,
                )
                np.seterr(**olderr)

            i += 1

    return A


def get_design_matrix(date_list, poly_deg=1, periods=[], steps=[],
                      exps=dict(), logs=dict(), ref_idx=None):
    # convert list of date into array of years in float
    # yr_diff = np.apply_along_axis(date2decimalyr, 0, np.atleast_2d(date_list))

    # reference date
    if ref_idx is None:
        ref_idx = 0
    # yr_diff -= yr_diff[ref_idx]
    yr_diff = np.squeeze(midas.date_difference_ydec(
        date_list[ref_idx], date_list))

    # construct design matrix A
    # read the models
    num_period = len(periods)
    num_step = len(steps)
    num_exp = sum(len(val) for key, val in exps.items())
    num_log = sum(len(val) for key, val in logs.items())

    num_param = (poly_deg + 1) + (2 * num_period) + \
        num_step + num_exp + num_log
    if num_param <= 1:
        raise ValueError('NO time functions specified!')

    # initialize the design matrix
    num_date = len(yr_diff)
    A = np.zeros((num_date, num_param), dtype=np.float32)
    c0 = 0

    # update linear/polynomial term(s)
    # poly_deg of 0 --> offset
    # poly_deg of 1 --> velocity
    # ...
    c1 = c0 + poly_deg + 1
    A[:, c0:c1] = get_polynomial_design_matrix(yr_diff, poly_deg)
    c0 = c1

    # update periodic term(s)
    if num_period > 0:
        c1 = c0 + 2 * num_period
        A[:, c0:c1] = get_periodic_design_matrix(yr_diff, periods)
        c0 = c1

    # update step term(s)
    if num_step > 0:
        c1 = c0 + num_step
        A[:, c0:c1] = get_step_design_matrix(date_list, steps)
        c0 = c1

    # update polyline term(s)
    # if num_pline > 0:
    #    c1 = c0 + num_pline
    #    A[:, c0:c1] = get_design_matrix4polyline(date_list, polylines, seconds=seconds)
    #    c0 = c1

    # update exponential term(s)
    if num_exp > 0:
        c1 = c0 + num_exp
        A[:, c0:c1] = get_log_exp_design_matrix(
            date_list, exps, seconds=seconds, func='exp')
        c0 = c1

    # update logarithmic term(s)
    if num_log > 0:
        c1 = c0 + num_log
        A[:, c0:c1] = get_log_exp_design_matrix(
            date_list, logs, seconds=seconds, func='log')
        c0 = c1

    return A


def fit_function(dates, disp, poly_deg=1, periods=[], steps=[],
                 ref_date=None, conf_level=0.95, display=False,
                 fig_out_name=None):

    # Get the design matrix
    A = get_design_matrix(dates, poly_deg=poly_deg,
                          periods=periods, steps=steps)

    # Invert
    m, e2 = linalg.lstsq(A, disp, cond=None)[:2]
    if len(e2)==0:
        e2 = np.zeros(m.shape)

    # Get the uncertainty
    num_obs = len(dates)
    num_param = A.shape[1]

    try:
        A_inv = linalg.inv(np.dot(A.T, A))
    except:
        print('Warning A_inv singular')
        A_inv = np.zeros(A.shape)
    m_var_sum = e2.flatten() / (num_obs - num_param)
    m_std = np.sqrt(np.dot(np.diag(A_inv).reshape(-1, 1),
                    np.atleast_2d(m_var_sum)))


    if display:
        plot_tsfit(dates, disp*1000, A, m*1000, m_std*1000,
                   steps=steps, fig_out_name=fig_out_name)

    return m*1000, m_std*1000


def plot_tsfit(dates, disp, A, m, m_std, steps=None, conf_level=0.95, fig_out_name=None):
    from matplotlib import pyplot as plt

    n_disp = m.shape[1]

    fig, axs = plt.subplots(n_disp, 1, figsize=(14, 10), sharex=True)
    if n_disp == 1:
        axs = [axs]
        disp = disp[:, np.newaxis]
        m = m[:, np.newaxis]
        m_std = m_std[:, np.newaxis]

    # fit the model
    conf_level = 0.95
    ts_fit = np.matmul(A, m)
    # TODO fix this for inversion of all 3 components at once
    # ts_fit_std = np.sqrt(np.diag(A.dot(np.diag(m_std**2)).dot(A.T)))
    alpha = 1 - conf_level                                # level of significance
    # scaling factor for confidence interval
    conf_int_scale = stats.norm.ppf(1 - alpha / 2)
    # ts_fit_lim = [ts_fit - conf_int_scale * ts_fit_std,
    #              ts_fit + conf_int_scale * ts_fit_std]

    txt = ['East (mm)', 'North (mm)', 'UP (mm)']
    for i, (ax, tx) in enumerate(zip(axs, txt)):
        ax.plot(dates, (disp[:, i] - np.mean(disp[:, i])), 'b.')
        ax.plot(dates, ts_fit[:, i] - np.mean(disp[:, i]), 'r-',
            label=f'v {m[1, i]:.2f} \u00B1 {m_std[1, i]:.2f} mm/yr')
        if steps is not None:
            ax.vlines(steps,
                    np.min(disp[:, i] - np.mean(disp[:, i])),
                    np.max(disp[:, i] - np.mean(disp[:, i])),
                    linestyles='dashed')
            
        ax.set_ylabel(tx)
        ax.legend(loc='upper left') 

    if fig_out_name:
        fig.suptitle(fig_out_name.split('.')[0])
        plt.savefig(fig_out_name)
        plt.close()
