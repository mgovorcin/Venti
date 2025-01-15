#!/usr/bin/python
import numpy as np
from .models import func_gm1
from .utils import get_pair_matrix, get_distance_matrix
from .covariance import create_Css, create_Cnn, get_moving_Cps


# Collocation at know observations


def collocation_signal_known(obs, Css, Cnn):
    # Numb of observation components
    num_obs = obs.shape[1]

    # Combine covariance signal and noise
    Czz = Css + Cnn
    # Get the covariance inverse
    Czz_inv = np.linalg.inv(Czz)

    # Reformat obs array
    l_new = np.atleast_2d(obs.ravel(order='F')).T

    # Calculate s (signals)
    signal_xy = np.dot(np.dot(Css, Czz_inv), l_new)

    # Calculate error covariance matrix of the signal s and the mean error ds
    sigmas = Css - np.dot(np.dot(Css, Czz_inv), Css)
    signal_xy_error = np.sqrt(np.abs(np.diag(sigmas))
                              ).reshape(signal_xy.shape)  # ds

    # Calculate n (noise)
    noise_xy = np.dot(np.dot(Cnn, Czz_inv), l_new)

    # Reformat
    signal_xy_new = signal_xy.reshape((-1, num_obs), order='F')
    signal_xy_error_new = signal_xy_error.reshape((-1, num_obs), order='F')
    noise_xy_new = noise_xy.reshape((-1, num_obs), order='F')
    print('Collocation at the observation points is done.')

    return signal_xy_new, signal_xy_error_new, noise_xy_new, Czz_inv


# Collocation at unknown points
def collocation_signal_unknown(obs, Cps, Cpp, C_inv):
    # Numb of observation components
    num_obs = obs.shape[1]

    # Reformat obs array
    l_new = np.atleast_2d(obs.ravel(order='F')).T

    # Calculate sp (signal at unknown points)
    signal_xy = np.dot(np.dot(Cps, C_inv), l_new)

    # Calculate error covariance matrix of the signal sp and the mean error ds
    sigmas = Cpp - np.dot(np.dot(Cps, C_inv), Cps.T)
    signal_xy_error = np.sqrt(np.abs(np.diag(sigmas)))
    signal_xy_error = signal_xy_error.reshape(
        signal_xy.shape)  # ds - uncertainty

    # Reformat
    signal_xy_new = signal_xy.reshape((-1, num_obs), order='F')
    signal_xy_error_new = signal_xy_error.reshape((-1, num_obs), order='F')
    print('Collocation at the interpolation points is done.')
    return signal_xy_new, signal_xy_error_new


def calc_signal_xiyi(x, y, xi, yi,
                     obs, C_inv,
                     empirical_fparameters,
                     moving_variance=False,
                     movvar_parameters=[1, 850, None, 7],
                     constrain_flag=False,
                     distance_penalty=1500,
                     iteration_order=['ee', 'en', 'ne', 'nn']):

    # Calculate signal at unknown points
    if constrain_flag:
        # TODO
        print('TODO: Not supported at the moment!')
        print('Calculation of intersection between points of grid/input takes lot of time')
        constrain_mask_ps = None
        constrain_mask_pp = None

    else:
        constrain_mask_ps = None
        constrain_mask_pp = None

    Cps = create_Css(x, y, xi, yi,
                     func_gm1, empirical_fparameters,
                     constrain_mask=constrain_mask_ps,
                     iteration_order=iteration_order, 
                     distance_penalty=distance_penalty)

    Cpp = create_Css(xi, yi, xi, yi,
                     func_gm1, empirical_fparameters,
                     constrain_mask=constrain_mask_pp,
                     iteration_order=iteration_order, 
                     distance_penalty=distance_penalty)

    if moving_variance is True:
        # replace empirical parameter first col with this
        c0_mov = movvar_parameters[0]
        empirical_fparameters = empirical_fparameters.copy()
        empirical_fparameters[:, 0] = np.ones(len(empirical_fparameters))* c0_mov
        #print(empirical_fparameters)
        delta_mov = movvar_parameters[1]
        fill_value = movvar_parameters[2]
        min_num = movvar_parameters[3]
        # For vertical
        if len(iteration_order) == 1:
            obs = np.c_[obs,obs]
        Cps_mov, Cpp_mov = get_moving_Cps(x, y, xi, yi,
                                          obs[:, 0], obs[:, 1],
                                          empirical_fparameters,
                                          delta_mov=delta_mov, min_number=min_num,
                                          fill_value=fill_value,
                                          iteration_order=iteration_order)
        Cps *= Cps_mov
        Cpp *= Cpp_mov

    signal_xiyi, signal_xiyi_error = collocation_signal_unknown(
        obs, Cps, Cpp, C_inv)
    return signal_xiyi, signal_xiyi_error
