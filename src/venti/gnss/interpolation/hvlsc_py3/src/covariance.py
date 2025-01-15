#!/usr/bin/python
import numpy as np
from .models import func_gm1
from .utils import get_pair_matrix, get_distance_matrix

# CSS -  Estimate the Css matrix for the collocation (covariance function of the signal
#         at the known points)


def create_Css(lon1, lat1, lon2, lat2,
               function, empirical_covariances,
               iteration_order=['ee', 'en', 'ne', 'nn'],
               cross_corelation=True,
               constrain_mask=None,
               distance_penalty=1500):
    '''
    Estimate the Css matrix for the collocation (covariance function of the signal)
    '''
    # for up component
    if len(iteration_order) == 1:
        cross_corelation = False
    if empirical_covariances.ndim == 1:
        empirical_covariances = np.atleast_2d(empirical_covariances)

    component_dict = dict(ee=calc_fee, en=calc_fen,
                          ne=calc_fne, nn=calc_fnn)

    # Get distance array in km
    dist = get_distance_matrix(lon1, lat1, lon2, lat2)

    if constrain_mask is not None:
        print('Using a constrained solution!')
        dist[constrain_mask] += distance_penalty

    if cross_corelation:
        lon1, lon2 = get_pair_matrix(lon1, lon2)
        lat1, lat2 = get_pair_matrix(lat1, lat2)

    # Estimate signal covariance, d0 in km
    css = dict.fromkeys(iteration_order)
    for ix, parameters in enumerate(empirical_covariances):
        function_parameters = dict(C0=parameters[0],
                                   d0=parameters[1])

        # Function: Gauss-markov model 1st order
        # Note: later include others
        Css = func_gm1(dist.ravel(), **function_parameters)
        Css = Css.reshape(dist.shape)
        if cross_corelation:
            print(f'Getting covariance component: "{iteration_order[ix]}"')
            angular_function = component_dict[iteration_order[ix]]
            Css = np.multiply(Css, angular_function(lon2, lat2, lon1, lat1))
            # note if reverse order from lat2, lat2 to lat1, lat2 getting diff results
            # for fee, fne

        # Add it to dic
        css[iteration_order[ix]] = Css

    # Create Css matrix
    if len(iteration_order) == 1:
        return Css
    else:
        if cross_corelation:
            return np.block([[css['ee'], css['en']],
                            [css['ne'], css['nn']]])
        else:
            zero_array = np.zeros((len(lon1), len(lon2)))
            return np.block([[css['ee'], zero_array],
                            [zero_array, css['nn']]])

# CNN -  Estimate the Cnn matrix for the collocation (covariance function of the noise)
# NOTE: lat, lon need to be same dimensionions as n - measurement uncertainty


def create_Cnn(noise):
    '''
    Estimate the Cnn matrix for the collocation (covariance function of the noise)
    '''
    # Flatten noise array, use order F (column-wise)
    # if there more columns, e.g. ew std; nw-std
    noise = noise.ravel(order='F')
    print('Noise-covariance matrix Cnn created')
    return np.diag(noise**2)

# Cross-corelation component function taking into account angular velocity field
# Needed for calculation on the sphere

# Covariance of angular velocity field
# eq 48


def calc_fen(lon1, lat1, lon2, lat2):
    '''
    Calculate fen component
    '''
    lat1r = np.radians(np.float64(lat1))
    lon12r = np.radians(np.float64(lon1 - lon2))
    fen = np.sin(lon12r) * np.sin(lat1r)

    return fen


def calc_fee(lon1, lat1, lon2, lat2):
    '''
    Calculate fee component
    '''

    lat1r = np.radians(np.float64(lat1))
    lat2r = np.radians(np.float64(lat2))
    lon12r = np.radians(np.float64(lon1 - lon2))
    fee = np.sin(lat1r) * np.sin(lat2r) * np.cos(lon12r)
    fee += np.cos(lat1r) * np.cos(lat2r)
    return fee


def calc_fne(lon1, lat1, lon2, lat2):
    '''
    Calculate fne component
    '''
    lat2r = np.radians(np.float64(lat2))
    lon21r = np.radians(np.float64(lon2 - lon1))
    fne = np.sin(lon21r) * np.sin(lat2r)
    return fne


def calc_fnn(lon1, lat1, lon2, lat2):
    '''
    Calculate fnn component
    '''
    lon12r = np.radians(np.float64(lon1 - lon2))
    fnn = np.cos(lon12r)
    return fnn


# Moving variance depending on  delta
def get_moving_Css(lon1, lat1, lon2, lat2,
                   data, empirical_covariances,
                   delta_mov, min_number=2, fill_value=None,
                   iteration_order=['ee', 'en', 'ne', 'nn'],
                   cross_corelation=True,
                   constrain_mask=None,
                   distance_penalty=1500):

    if len(iteration_order) == 1:
        cross_corelation = False
        ss_mode = False

        c01 = np.sum(data**2) / data.shape[0]
        c02 = c01

        iteration_data = dict(data1=[data],
                              data2=[data])

    else:
        # Assume this order
        ew = data[:, 0]
        ns = data[:, 1]
        ss_mode = True

        c01 = np.sum(ew**2) / ew.shape[0]
        c02 = np.sum(ns**2) / ns.shape[0]

        iteration_data = dict(data1=[ew, ew, ns, ns],
                              data2=[ew, ns, ew, ns])

    if empirical_covariances.ndim == 1:
        empirical_covariances = np.atleast_2d(empirical_covariances)

    # max_data normaalize CHECK!!!
    if abs(data.min()) > abs(data.max()):
        max_data = abs(data.min())
    else:
        max_data = abs(data.max())

    if empirical_covariances[:, 0].max() <= 1:
        max_data = 1.

    # Initialize c0 array
    C0_movvar = np.ones(len(empirical_covariances))

    if empirical_covariances[:, 0].all() == 0:
        C0_movvar *= (c01 * 0.5 + c02 * 0.5)
    else:
        C0_movvar *= empirical_covariances[:, 0]
 
    # Get the distance array in km
    distance_array = get_distance_matrix(lon1, lat1, lon2, lat2)

    if constrain_mask is not None:
        print('Using a constrained solution!')
        distance_array[constrain_mask] += distance_penalty

    # Get points below the delta
    distance_mask = np.ma.masked_greater_equal(
        np.abs(distance_array), delta_mov).mask

    c_movvar = dict.fromkeys(iteration_order)
    for ix, order in enumerate(iteration_order):
        print('Estimating moving variance for: '
              f'{order} component')

        d1 = iteration_data['data1'][ix]
        d2 = iteration_data['data2'][ix]

        css1, css2 = _get_moving_variance(d1 / max_data, d2 / max_data,
                                          distance_mask, min_number=min_number,
                                          c0=C0_movvar[ix], fill_value=fill_value,
                                          ss_mode=ss_mode)
        c_movvar[order] = np.outer(css1, css2)

    if len(iteration_order) == 1:
        return c_movvar[order]

    else:
        if cross_corelation:
            C_movvar = np.block([[c_movvar['ee'], c_movvar['en']],
                                 [c_movvar['ne'], c_movvar['nn']]])
        else:
            zero_array = np.zeros((len(lon1), len(lon2)))
            C_movvar = np.block([[c_movvar['ee'], zero_array],
                                 [zero_array, c_movvar['nn']]])

        return C_movvar


def get_moving_Cps(lon1, lat1, lon2, lat2,
                   ew, ns, empirical_covariances,
                   delta_mov, min_number=2, fill_value=None,
                   iteration_order=['ee', 'en', 'ne', 'nn'],
                   cross_corelation=True):

    iteration_data = dict(data1=[ew, ew, ns, ns],
                          data2=[ew, ns, ew, ns])

    # max_data normaalize CHECK!!!
    l = np.c_[ew, ns]
    if abs(l.min()) > abs(l.max()):
        max_l = abs(l.min())
    else:
        max_l = abs(l.max())
    if empirical_covariances[:, 0].max() <= 1:
        max_l = 1.

    # Initialize c0 array
    C0_movvar = np.ones(len(empirical_covariances))

    if empirical_covariances[:, 0].all() == 0:
        c01 = np.sum(ew**2) / ew.shape[0]
        c02 = np.sum(ns**2) / ns.shape[0]
        C0_movvar *= (c01 * 0.5 + c02 * 0.5)
    else:
        C0_movvar *= empirical_covariances[:, 0]

    # Get distance matrices
    # Assume lon1/lat1 are input,
    # where lon2/lat2 coord for interpolatin
    distance_array_ss = get_distance_matrix(lon1, lat1, lon1, lat1)
    distance_array_ps = get_distance_matrix(lon1, lat1, lon2, lat2)

    # Get distance mask
    distance_mask_ss = np.ma.masked_greater_equal(
        np.abs(distance_array_ss), delta_mov).mask
    distance_mask_ps = np.ma.masked_greater_equal(
        np.abs(distance_array_ps), delta_mov).mask

    c_movvar_ps = dict.fromkeys(iteration_order)
    c_movvar_pp = dict.fromkeys(iteration_order)
    cps_dict = dict.fromkeys(iteration_order)

    # Initialize empty data for interpolation points
    d3 = np.zeros(lon2.shape)

    for ix, order in enumerate(iteration_order):
        print('Estimating moving variance for: '
              f'{order} component')

        d1 = iteration_data['data1'][ix]
        d2 = iteration_data['data2'][ix]

        cps, _ = _get_moving_variance(d1, d3,
                                      distance_mask_ps, min_number=min_number,
                                      c0=C0_movvar[ix], fill_value=fill_value,
                                      ss_mode=False) 

        _, c2_ss = _get_moving_variance(d1 / max_l, d2 / max_l,
                                        distance_mask_ss, min_number=min_number,
                                        c0=C0_movvar[ix], fill_value=fill_value,
                                        ss_mode=True)

        c_movvar_ps[order] = np.outer(cps, c2_ss)
        cps_dict[order] = cps

    # Create moving Cpp
    # TODO: missing option to do mov var for vv
    c_movvar_pp['ee'] = np.outer(cps_dict['ee'], cps_dict['ee'])
    c_movvar_pp['en'] = np.outer(cps_dict['ee'], cps_dict['nn'])
    c_movvar_pp['ne'] = np.outer(cps_dict['nn'], cps_dict['ee'])
    c_movvar_pp['nn'] = np.outer(cps_dict['nn'], cps_dict['nn'])

    if cross_corelation:
        C_movvar_ps = np.block([[c_movvar_ps['ee'], c_movvar_ps['en']],
                                [c_movvar_ps['ne'], c_movvar_ps['nn']]])
        C_movvar_pp = np.block([[c_movvar_pp['ee'], c_movvar_pp['en']],
                                [c_movvar_pp['ne'], c_movvar_pp['nn']]])
    else:
        zero_array = np.zeros((len(lon1), len(lon2)))
        C_movvar_ps = np.block([[c_movvar_ps['ee'], zero_array],
                                [zero_array, c_movvar_ps['nn']]])
        C_movvar_pp = np.block([[c_movvar_pp['ee'], zero_array],
                                [zero_array, c_movvar_pp['nn']]])

    return C_movvar_ps, C_movvar_pp

# code


def _get_moving_variance(data1, data2, distance_mask,
                         min_number=7, c0=0, fill_value=None,
                         ss_mode=True):
    # Get data matrix
    data1, data2 = get_pair_matrix(data1, data2)

    # Get moving variance per each point (data1 -row wise, data2 - column wise)
    count = np.count_nonzero(~distance_mask, axis=1)

    # Masked data1
    masked_data1 = np.ma.masked_array(data1, mask=distance_mask)

    # get moving variance c1
    c_d1 = np.sqrt(np.sum(masked_data1**2, axis=1) / (count-1))

    # Fill values where point does not have any pair or the number
    # of pair is below min_number after masking
    # 1. - zero pairs
    delta_data0 = np.array([np.sqrt(c0 / 2.), np.sqrt(c0 / 2.)])
    c_d1[count == 0] = np.sqrt(np.sum(delta_data0**2) / 1)

    # 2 - below min number
    ix = np.where((count > 0) & (count < min_number))[0]

    add = min_number - count[ix]

    if fill_value:
        fill_value1 = fill_value
    else:
        # if fill value is not specified, calc the mean and add it
        # number of times to reach min_number
        fill_value1 = (np.mean(masked_data1[ix, :], axis=1) + c0) / 2.
        fill_value1 = add * fill_value1**2
        fill_value1 = np.sum(masked_data1[ix, :]**2, axis=1) + fill_value1
        fill_value1 = np.sqrt(fill_value1 / (min_number - 1))

    c_d1[ix] = fill_value1

    # Moving variance with signal
    if ss_mode:
        # Maskeddata2
        masked_data2 = np.ma.masked_array(data2, mask=distance_mask)

        # get moving variance c1 data2
        c_d2 = np.sqrt(np.sum(masked_data2**2, axis=0) / (count-1))

        # Fill values data2
        c_d2[count == 0] = np.sqrt(np.sum(delta_data0**2) / 1)

        if fill_value:
            fill_value2 = fill_value
        else:
            fill_value2 = (np.mean(masked_data2[:, ix], axis=0) + c0) / 2.
            fill_value2 = add * fill_value2**2
            fill_value2 = np.sum(masked_data2[:, ix]**2, axis=0) + fill_value2
            fill_value2 = np.sqrt(fill_value2 / (min_number - 1))

        c_d2[ix] = fill_value2
        return c_d1, c_d2

    else:
        return c_d1, c_d1
