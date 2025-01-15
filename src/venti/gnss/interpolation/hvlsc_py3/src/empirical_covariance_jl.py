#!/usr/bin/python
import numpy as np
from requests import get
import scipy
import time
from sys import getsizeof
from scipy import stats
from pyproj import Geod
from .utils import get_pair_matrix, get_distance_matrix
from .empirical_covariance import plot_empirical_covariance, find_outliers
from .covariance import calc_fee, calc_fen, calc_fne, calc_fnn
from .models import func_gm1
g = Geod(ellps='WGS84')


# Calculate empirical covariance of the input data following Juliette Legrand (2007)
def get_empirical_covariance_jl(x, y,
                                data1, data2,
                                noise1, noise2,
                                custom_covar,
                                bin_spacing=50e3,
                                constrain_mask=None,
                                distance_penalty=1500,
                                correlation_analysis=False,
                                plot_covariance=True,
                                max_range=None):

    # Initialiaze cov parameters, for only for gm1
    c0 = custom_covar[0]
    d0 = custom_covar[1]

    # 1.Outlier removal
    outlier_ix1 = find_outliers(data1, sigma=3)

    if outlier_ix1.size != 0:
        x = np.delete(x, outlier_ix1)
        y = np.delete(y, outlier_ix1)
        data1 = np.delete(data1, outlier_ix1)
        data2 = np.delete(data2, outlier_ix1)
        noise1 = np.delete(noise1, outlier_ix1)
        noise2 = np.delete(noise2, outlier_ix1)
        print('Outlier identified and removed')

    outlier_ix2 = find_outliers(data2, sigma=3)

    if outlier_ix2.size != 0:
        x = np.delete(x, outlier_ix2)
        y = np.delete(y, outlier_ix2)
        data1 = np.delete(data1, outlier_ix2)
        data2 = np.delete(data2, outlier_ix2)
        noise1 = np.delete(noise1, outlier_ix2)
        noise2 = np.delete(noise2, outlier_ix2)
        print('Outlier identified and removed')

    # distance and cross-data product matrixes
    distance_array = get_distance_matrix(x, y, x, y)
    dmax = g.inv(np.amin(x), np.amin(y),
                 np.amax(x), np.amax(y))[2]
    dmax *= 1e-3  # m to km

    if constrain_mask is not None:
        print('Using a constrained solution!')
        distance_array[constrain_mask] += distance_penalty

    # Get bins
    n = distance_array.shape[0]
    bin_edges = np.arange(bin_spacing,
                          dmax,
                          bin_spacing*2)
    bin_edges = np.r_[[0, 1e-10], bin_edges]
    if max_range:
        bin_edges= bin_edges[bin_edges < max_range]

    bins = np.digitize(distance_array, bin_edges[1:])
    if max_range:
        bins[bins == np.max(bins)] = 9999
    bins[np.tril_indices(n)] = 9999
    bins[np.diag_indices(n)] = 0

    # Remove bins with less than n points
    bin_counts = np.bincount(bins.ravel())

    # Remove empty bins
    bin_counts = bin_counts[bin_counts != 0]

    # Get center of bins
    bincenter_array = (bin_edges[:-1] + bin_edges[1:]) / 2
    bincenter_array[0] = 0.

    # Get empirial covariance based on JL 2007
    emp_k, emp_kvar = [], []
    bins_center = []

    n_sites = len(x)
    for ix, count, bincenter in zip(np.unique(bins)[:-1],
                                    bin_counts[:-1],
                                    bincenter_array):
        print(f'Bin: {ix}, Count: {count}, N_Bin: {len(np.unique(bins))}')
        # Skip bin if it has less points than input
        if count >= n_sites:

            k, kv, _ = get_empirical_K(x, y,
                                       data1, data2,
                                       noise1, noise2,
                                       bins, ix,
                                       c0, d0,
                                       correlation_analysis)
            emp_k.append(k)
            emp_kvar.append(kv)
            bins_center.append(bincenter)

    # Convert lists to arrays
    emp_k = np.squeeze(np.vstack(emp_k))
    emp_kvar = np.squeeze(np.vstack(emp_kvar))
    bincenter_array = np.squeeze(np.vstack(bins_center))

    # Get fitting bounds
    corr_bounds1 = [0, dmax]
    for ecov, bin_center in zip(emp_k[1:], bincenter_array[1:]):
        if ecov < emp_k[0] / 2.:
            corr_bounds1 = [bin_center - bin_spacing,
                            bin_center + bin_spacing]
            break

    corr_bounds = [corr_bounds1, [0, bin_edges[-1]]]

    # Change c0 based on new empirical
    # c0 at d0
    c0 = emp_k[0]
    bins_mean = emp_k
    bins_std = emp_kvar

    # define func with fixed value at d0, for now only gm1
    def fixed_gm1_func(cova_dist, d1):
        return func_gm1(cova_dist, c0, d1)

    # Fit
    # initalize variables
    popt, perr, misfit, misfit_1st, pearson_cor = [], [], [], [], []
    fit_model = []
    for ix, bounds in enumerate(corr_bounds):
        coef, cov = scipy.optimize.curve_fit(fixed_gm1_func,
                                             bincenter_array, bins_mean,
                                             bounds=(bounds[0], bounds[1]))  # sigma=sigma

        perr1 = np.sqrt(np.diag(cov))
        sd0 = perr1[0]
        popt.append(np.array([c0, coef[0]]))  # d0 = coef[0]
        perr.append(np.array([0, sd0]))

        # Fit the function using estimated parameters
        fitted_model = func_gm1(bincenter_array, popt[-1][0], popt[-1][1])
        fit_model.append(fitted_model)

        misfit.append(
            np.sqrt(sum((fitted_model - bins_mean)**2) / float(len(bins_mean))) / c0)
        misfit_1st.append(np.sqrt(
            sum((fitted_model[:3] - bins_mean[:3])**2) / float(len(bins_mean[:3]))) / c0)
        pearson_cor.append(stats.pearsonr(bins_mean, fitted_model)[0])

    # Get final result - note clean code below
    if (misfit[1] <= misfit[0]) and (misfit_1st[1] <= misfit_1st[0]):
        select_ix = 1
    elif (misfit[1] > misfit[0]) and (misfit_1st[1] <= misfit_1st[0]) \
            and np.round(pearson_cor[1], 2) >= np.round(pearson_cor[0], 2):
        select_ix = 1 if popt[1][1] < popt[0][1] else 0
    elif (misfit[1] <= misfit[0]) and (misfit_1st[1] > misfit_1st[0]) \
            and np.round(pearson_cor[1], 2) >= np.round(pearson_cor[0], 2):
        select_ix = 1 if popt[1][1] < popt[0][1] else 0
    else:
        select_ix = 0

    # do not understand why adding 0 leave for now
    popt = popt[select_ix] + [0]
    perr = perr[select_ix] + [0]
    misfit = misfit[select_ix] + 0
    misfit_1st = misfit_1st[select_ix]
    pearson_cor = pearson_cor[select_ix]
    print('Final values:')
    print('C0 = %.5f +/- %.5f, d0 = %d +/- %d' %
          (popt[0], perr[0], popt[1], perr[1]))
    print('Misfit is: %.5f' % (misfit))
    print('Misfit of the first three points is: %.5f' % (misfit_1st))
    print('Pearsons correlation: %.3f \n' % (pearson_cor))

    if plot_covariance:
        plot_empirical_covariance(bincenter_array, bins_mean, bins_std,
                                  fit_model[select_ix], popt[0], popt[1])

    return popt, perr

# Needed functions
def get_empirical_K(x, y, data1, data2,
                    noise1, noise2, bins, bin_ix,
                    c0, d0, correlation=False):
    # Get Beta
    print('Get BETA')
    beta, norm = get_beta(x, y, data1, data2, noise1,
                          noise2, bins, bin_ix, correlation)

    # Get covariance Beta
    print('Get COV BETA')
    cov_beta = get_cov_beta4_v2(x, y, noise1, noise2, bins, bin_ix, c0, d0)
    print(f' Cov Beta shape: {cov_beta.shape}')
    cov_beta_inv = np.linalg.inv(cov_beta * np.identity(cov_beta.shape[0]))

    # Get alpha
    id_m = np.ones(beta.shape)
    etce = np.dot(np.dot(id_m.T, cov_beta_inv.T), id_m)
    alpha = np.dot(cov_beta_inv, id_m) / etce

    # Get empirical K and its var
    k = np.squeeze(np.dot(alpha.T, beta)) / norm
    k_var = np.squeeze(1. / etce)

    return k, k_var, np.sum(alpha)

# BETA functions
def calc_beta(x1, x2, y1, y2, l1, l2, n1, n2, f_function):
    '''
    Calculate beta (equation (83))
    '''
    f = f_function(x1, y1, x2, y2)
    beta = (l1 * l2 - np.diag(np.cov(np.c_[n1, n2]))) / f
    return beta


def get_beta(x, y, data1, data2, noise1, noise2, bins, bin_ix, correlation=False):
    # NOTE: differs on 15 decimal due to numpy array precision
    # Get data in needed structure
    # coords, square matrices and their transpose
    x1, x2 = get_pair_matrix(x, x)
    y1, y2 = get_pair_matrix(y, y)

    # Data
    data_e1, data_e2 = get_pair_matrix(data1, data1)
    data_n1, data_n2 = get_pair_matrix(data2, data2)

    # Data uncertainty: vel_std
    noise_e1, noise_e2 = get_pair_matrix(noise1, noise1)
    noise_n1, noise_n2 = get_pair_matrix(noise2, noise2)

    # Calc beta, for the bin
    mask = np.where(bins == bin_ix)

    # Select only needed data
    x1, x2, y1, y2 = [x1[mask], x2[mask], y1[mask], y2[mask]]
    de1, de2, dn1, dn2 = [data_e1[mask], data_e2[mask],
                          data_n1[mask], data_n2[mask]]
    ne1, ne2, nn1, nn2 = [noise_e1[mask], noise_e2[mask],
                          noise_n1[mask], noise_n2[mask]]

    # Stdo flag
    if correlation is True:
        mean_e = np.mean(data1[np.unique(mask[0])])
        mean_n = np.mean(data2[np.unique(mask[1])])
        std_e = np.std(data1[np.unique(mask[0])])
        std_n = np.std(data2[np.unique(mask[1])])

        # Demean
        de1 -= mean_e
        de2 -= mean_e
        dn1 -= mean_n
        dn2 -= mean_n

        # get norm
        norm = std_e * std_n
    else:
        norm = 1

    beta_ee = calc_beta(x1, x2, y1, y2,
                        de1, de2, ne1, ne2,
                        calc_fee)

    beta_nn = calc_beta(x1, x2, y1, y2,
                        dn1, dn2, nn1, nn2,
                        calc_fnn)
    # Bin0, distance = 0, auto-estimation on sites
    #   only ee, nn components
    if bin_ix == 0:
        beta = np.c_[beta_ee, beta_nn].ravel(order='C')

    # Other bins, components ee, nn, en, ne
    else:
        beta_ne = calc_beta(x2, x1, y2, y1,
                            dn2, de1, nn2, ne1,
                            calc_fne)
        beta_en = calc_beta(x2, x1, y2, y1,
                            de2, dn1, ne2, nn1,
                            calc_fen)

        beta = np.c_[beta_ee, beta_nn,
                     beta_en, beta_ne].ravel(order='C')

    return np.atleast_2d(beta).T, norm


def get_cov_beta4(x, y, noise1, noise2, bins, bin_ix, c0, d0):
    start = time.time()
    # Get F matrices
    fst, fsu, fsv, ftu, ftv, fuv = get_F4_matrix(x, y, bins, bin_ix)
    #print(f' F matrix: {fst.shape}')

    # Get K matrices
    Ktv, Ktu, Ksv, Ksu = get_K_functions(x, y, bins, bin_ix, c0, d0)
    #print(f' K matrix: {Ktv.shape}')

    # Get Cov matrices
    Csu, Csv, Ctu, Ctv = get_cov_matrics(noise1, noise2, bins, bin_ix)
    #print(f' C matrix: {Csu.shape}')

    # Calculate covariance of beta (equation (86))
    cov = (fsu * ftv * Ksu * Ktv)
    cov += (fsv * ftu * Ksv * Ktu)
    cov += (ftv * Ktv * Csu)
    cov += (fsv * Ksv * Ctu)
    cov += (ftu * Ktu * Csv)
    cov += (ftu * Ktu * Csv)
    cov += (fsu * Ksu * Ctv)
    cov += (Csu * Ctv)
    cov += (Csv * Ctu)

    cov_beta = cov / (fst * fuv)
    end = time.time()
    print((end - start) / 60, ' min')

    return cov_beta

def get_cov_beta4_v2(x, y, noise1, noise2, bins, bin_ix, c0, d0):
    start = time.time()
    
    n_points = np.sum(bins == bin_ix) 

    if (bin_ix == 0) or (n_points < 1000):
        # Get F matrices
        fst, fsu, fsv, ftu, ftv, fuv = get_F4_matrix(x, y, bins, bin_ix)
        #print(f' F matrix: {fst.shape}')

        # Get K matrices
        Ktv, Ktu, Ksv, Ksu = get_K_functions(x, y, bins, bin_ix, c0, d0)
        #print(f' K matrix: {Ktv.shape}')

        Csu, Csv, Ctu, Ctv = get_cov_matrics(noise1, noise2, bins, bin_ix)
        # Calculate covariance of beta (equation (86))
        cov = (fsu * ftv * Ksu * Ktv) #
        cov += (fsv * ftu * Ksv * Ktu)
        cov += (ftv * Ktv * Csu)
        cov += (fsv * Ksv * Ctu)
        cov += (ftu * Ktu * Csv)
        cov += (fsu * Ksu * Ctv)
        cov += (Csu * Ctv)
        cov += (Csv * Ctu)

        cov_beta = cov / (fst * fuv)

    else:
        # Option to keep memory in check
        # Note: get all Fs at once for now
        # Get F matrices
        fst, fsu, fsv, ftu, ftv, fuv = get_F4_matrix(x, y, bins, bin_ix)
        #print(f' F matrix: {fst.shape}: {(time.time() - start)/60:.2f}')

        # Get K matrices
        # Get site pair matrix
        x1, x2 = get_pair_matrix(x, x)
        y1, y2 = get_pair_matrix(y, y)

        # Select only pairs within defined bin
        x1 = x1[np.where(bins == bin_ix)]
        x2 = x2[np.where(bins == bin_ix)]
        y1 = y1[np.where(bins == bin_ix)]
        y2 = y2[np.where(bins == bin_ix)]

        # Get noise
        noise_e, noise_et = get_pair_matrix(noise1, noise1)
        noise_n, noise_nt = get_pair_matrix(noise2, noise2)

        # Get bin
        bin_ixs = np.where(bins == bin_ix)

        # Get noise matrices for cov calculations
        n1 = get_noise_matrices(noise_et[bin_ixs], noise_nt[bin_ixs])[0]
        n2 = get_noise_matrices(noise_e[bin_ixs], noise_n[bin_ixs])[1]
        n3 = n1.T
        n4 = n2.T

        # Cov1: fsu, ftv, Ksu, Ktv
        #print(f' Get Cov-1: {(time.time() - start)/60:.2f} min')
        Ksu = get_K_function(x2, y2, x2, y2, c0, d0, 4)
        Ktv = get_K_function(x1, y1, x1, y1, c0, d0, 4)

        cov = (fsu * ftv * Ksu * Ktv)

        # Cov2: fsu, Ksu, Ctv
        #print(f' Get Cov-2: {(time.time() - start)/60:.2f} min')
        Ctv = covariance_along2dim(n2, n4)
        cov += (fsu * Ksu * Ctv)
        del fsu, Ksu

        # Cov3: Csu, Ctv
        #print(f' Get Cov-3: {(time.time() - start)/60:.2f} min')
        Csu = covariance_along2dim(n1, n3)
        cov += (Csu * Ctv)
        del Ctv

        # Cov4 : ftv * Ktv * Csu
        #print(f' Get Cov-4: {(time.time() - start)/60:.2f} min')
        cov += (ftv * Ktv * Csu)
        del Ktv, Csu, ftv

        # Cov5 : fsv * ftu * Ksv * Ktu
        #print(f' Get Cov-5: {(time.time() - start)/60:.2f} min')
        Ksv = get_K_function(x2, y2, x1, y1, c0, d0, 4) 
        Ktu = get_K_function(x1, y1, x2, y2, c0, d0, 4) 
        cov += (fsv * ftu * Ksv * Ktu) 
        del x1, y1, x2, y2

        # Cov6: fsv * Ksv * Ctu
        #print(f' Get Cov-6: {(time.time() - start)/60:.2f} min')
        Ctu = covariance_along2dim(n2, n3) 
        cov += (fsv * Ksv * Ctu)
        del Ksv, fsv

        # Cov7: Csv * Ctu
        #print(f' Get Cov-7: {(time.time() - start)/60:.2f} min')
        Csv = covariance_along2dim(n1, n4) 
        cov += (Csv * Ctu)
        del Ctu, n1, n2, n3, n4

        # Cov8: ftu * Ktu * Csv
        #print(f' Get Cov-8: {(time.time() - start)/60:.2f} min')
        cov += (ftu * Ktu * Csv) 
        del ftu, Ktu, Csv
        
        # CovBeta
        #print(f' Get CovBeta: {(time.time() - start)/60:.2f} min')
        cov_beta = cov / (fst * fuv)
        del fst, fuv, cov

    end = time.time()
    print(f'Finished calculating covBeta: {(end - start) / 60:.2f} min')

    return cov_beta


# FST, FSU, FSV, FTU, FTV, FUV

def get_F4_matrix(x, y, bins, bin_ix, get_matrix=None):
    def _get_bin_coords(x, y, bins, bin_ix):
        x1, x2 = get_pair_matrix(x, x)
        y1, y2 = get_pair_matrix(y, y)

        bin_ixs = np.where(bins == bin_ix)

        x11, x22 = x1[bin_ixs], x2[bin_ixs]
        y11, y22 = y1[bin_ixs], y2[bin_ixs]

        # Get coordinates for:
        #  FST, FSU, FSV, FTU, FTV, FUV
        X11, X22 = get_pair_matrix(x22, x11)
        Y11, Y22 = get_pair_matrix(y22, y11)

        #     XY1   XY2   XY3    XY4
        xx = [X11, X22.T, X11.T, X22]
        yy = [Y11, Y22.T, Y11.T, Y22]

        return xx, yy

    def _get_angular_mat(x1, y1, x2, y2):
        fee = calc_fee(x1, y1, x2, y2)
        fnn = calc_fnn(x1, y1, x2, y2)
        fen = calc_fen(x1, y1, x2, y2)
        fne = calc_fne(x1, y1, x2, y2)

        return fee, fnn, fen, fne

    def _get_f4matrix(n, submatrices, index):
        F = np.zeros((n*index, n*index))
        for i in range(index):
            for j in range(index):
                F[i::index, j::index] = submatrices[i][j]
        return F

    def _get_fmatrix_bin0(X1, Y1, X2, Y2, X3, Y3):
        n = X1.shape[0]

        [fee, fnn,
         fen, fne] = _get_angular_mat(X1, Y1,
                                      X2, Y2)
        # Get FST, FUV
        sub = [[fee, fnn, fee, fnn],
               [fee, fnn, fee, fnn],
               [fee, fnn, fee, fnn],
               [fee, fnn, fee, fnn]]

        f1 = _get_f4matrix(n, sub, 2)

        # Get FSU, FSV, FTU, FTV
        [fee, fnn,
         fen, fne] = _get_angular_mat(X1, Y1,
                                      X3, Y3)

        sub = [[fee, fne, fee, fne],
               [fen, fnn, fen, fnn],
               [fee, fne, fee, fne],
               [fen, fnn, fen, fnn]]

        f2 = _get_f4matrix(n, sub, 2)

        return f1, f2

    def _get_FST_FUV(X1, Y1, X2, Y2):
        # FST = FUV.T
        n = X1.shape[0]

        [fee, fnn,
         fen, fne] = _get_angular_mat(X1, Y1,
                                      X2, Y2)

        # Create sub-matrix formating
        sub = [[fee, fnn, fen, fne],
               [fee, fnn, fen, fne],
               [fee, fnn, fen, fne],
               [fee, fnn, fen, fne]]

        return _get_f4matrix(n, sub, 4)

    def _get_FSU(X1, Y1, X3, Y3):
        n = X1.shape[0]

        [fee, fnn,
         fen, fne] = _get_angular_mat(X1, Y1,
                                      X3, Y3)

        # Create sub-matrix formating
        sub = [[fee, fne, fee, fne],
               [fen, fnn, fen, fnn],
               [fee, fne, fee, fne],
               [fen, fnn, fen, fnn]]

        return _get_f4matrix(n, sub, 4)

    def _get_FSV(X1, Y1, X4, Y4):
        n = X1.shape[0]

        [fee, fnn,
         fen, fne] = _get_angular_mat(X1, Y1,
                                      X4, Y4)

        # Create sub-matrix formating
        sub = [[fee, fne, fee, fne],
               [fen, fnn, fen, fnn],
               [fen, fnn, fen, fnn],
               [fee, fne, fee, fne]]

        return _get_f4matrix(n, sub, 4)

    def _get_FTU(X2, Y2, X3, Y3):
        n = X2.shape[0]

        [fee, fnn,
         fen, fne] = _get_angular_mat(X2, Y2,
                                      X3, Y3)

        # Create sub-matrix formating
        sub = [[fee, fne, fne, fee],
               [fen, fnn, fnn, fen],
               [fee, fne, fne, fee],
               [fen, fnn, fnn, fen]]

        return _get_f4matrix(n, sub, 4)

    def _get_FTV(X2, Y2, X4, Y4):
        n = X2.shape[0]

        [fee, fnn,
         fen, fne] = _get_angular_mat(X2, Y2,
                                      X4, Y4)

        # Create sub-matrix formating
        sub = [[fee, fne, fne, fee],
               [fen, fnn, fnn, fen],
               [fen, fnn, fnn, fen],
               [fee, fne, fne, fee]]

        return _get_f4matrix(n, sub, 4)

    # MAIN FUNCTION
    [xx, yy] = _get_bin_coords(x, y, bins, bin_ix)

    if bin_ix == 0:
        fst, fsu = _get_fmatrix_bin0(xx[0], yy[0],
                                     xx[1], yy[1],
                                     xx[2], yy[2])
        fuv = fst.T
        fsv, ftu, ftv = fsu, fsu, fsu
        return fst, fsu, fsv, ftu, ftv, fuv

    else:
        # NOTE Added get_matrix option to avoid memory overflow
        #  with big arrays
        f_matrices = ['fst', 'fuv', 'fsu', 'fsv', 'ftu', 'ftv']
        if (get_matrix is None) or (get_matrix not in f_matrices):
            fst = _get_FST_FUV(xx[0], yy[0], xx[1], yy[1])
            fuv = fst.T
            fsu = _get_FSU(xx[0], yy[0], xx[2], yy[2])
            fsv = _get_FSV(xx[0], yy[0], xx[3], yy[3])
            ftu = _get_FTU(xx[1], yy[1], xx[2], yy[2])
            ftv = _get_FTV(xx[1], yy[1], xx[3], yy[3])
            return fst, fsu, fsv, ftu, ftv, fuv
        elif (get_matrix == 'fst') or (get_matrix == 'fuv'):
            fst = _get_FST_FUV(xx[0], yy[0], xx[1], yy[1])
            if get_matrix == 'fuv': 
                return fst.T
            else: 
                return fst
        elif get_matrix == 'fsu':
            fsu = _get_FSU(xx[0], yy[0], xx[2], yy[2])
            return fsu
        elif get_matrix == 'fsv':
            fsv = _get_FSV(xx[0], yy[0], xx[3], yy[3])
            return fsv
        elif get_matrix == 'ftu':
            ftu = _get_FTU(xx[1], yy[1], xx[2], yy[2])
            return ftu
        elif get_matrix == 'ftv':
            ftv = _get_FTV(xx[1], yy[1], xx[3], yy[3])
            return ftv 

# KTV, KTU, KSV, KSU

# NOTE ; getting different values for bin !=0, difference on the level 1e-16
#         precision issue, ignore it!

def get_K_functions(x, y, bins, bin_ix, c0, d0):
    # Get site pair matrix
    x1, x2 = get_pair_matrix(x, x)
    y1, y2 = get_pair_matrix(y, y)

    # Select only pairs within defined bin
    x1 = x1[np.where(bins == bin_ix)]
    x2 = x2[np.where(bins == bin_ix)]
    y1 = y1[np.where(bins == bin_ix)]
    y2 = y2[np.where(bins == bin_ix)]

    # Get distances
    dtv = get_distance_matrix(x1, y1, x1, y1)
    dtu = get_distance_matrix(x1, y1, x2, y2)
    dsv = get_distance_matrix(x2, y2, x1, y1)
    dsu = get_distance_matrix(x2, y2, x2, y2)

    # Define fitting function- gm1 for now
    function_parameters = dict(C0=c0, d0=d0)

    # Get K matrices, only gm1 for now
    ktv = func_gm1(dtv.ravel(), **function_parameters)
    ktu = func_gm1(dtu.ravel(), **function_parameters)
    ksv = func_gm1(dsv.ravel(), **function_parameters)
    ksu = func_gm1(dsu.ravel(), **function_parameters)

    # Reshape
    ktv = ktv.reshape(dtv.shape)
    ktu = ktu.reshape(dtu.shape)
    ksv = ksv.reshape(dsv.shape)
    ksu = ksu.reshape(dsu.shape)

    if bin_ix == 0:
        repeat_n = np.ones((2, 2))
    else:
        repeat_n = np.ones((4, 4))

    # Repeat 2/4 times depending on bin
    ktv = np.kron(ktv, repeat_n)
    ktu = np.kron(ktu, repeat_n)
    ksv = np.kron(ksv, repeat_n)
    ksu = np.kron(ksu, repeat_n)

    return ktv, ktu, ksv, ksu

def get_K_function(x1, y1, x2, y2, c0, d0, n):
    # Get distances
    d = get_distance_matrix(x1, y1, x2, y2)

    # Define fitting function- gm1 for now
    function_parameters = dict(C0=c0, d0=d0)

    # Get K matrices, only gm1 for now
    k = func_gm1(d.ravel(), **function_parameters)

    # Reshape
    k = k.reshape(d.shape)
    repeat_n = np.ones((n, n))

    # Repeat 2/4 times depending on bin
    kf = np.kron(k, repeat_n)
    return kf


# CSU, CSV, CTU, CTV

def covariance_along2dim(data1, data2):
    # NOTE: this function is taking lot of mem
    #      bin count ~5k takes 7gb
    #                ~10k takes 25gb
    #      Need to reformat this to run in chunks
    data3d = np.dstack([data1, data2])
    print(data1.shape, data3d.shape)

    # Estimate covariance along axis 2
    avg = np.average(data3d, axis=2)
    data3d -= np.dstack([avg, avg])

    print(f'  Cov mem: {round(getsizeof(data3d) / 1024 / 1024,2)} MB')

    n = data3d.shape[2]
    ddof = n - 1
    cov = np.sum(data3d * data3d, axis=2) / ddof
    return cov


def _get_cov_bin0(noise1, noise2):
    noise_e, _ = get_pair_matrix(noise1, noise1)
    noise_n, _ = get_pair_matrix(noise2, noise2)

    n = noise1.shape[0]

    # Get
    combined_noise = np.zeros((n*2, n*2))
    combined_noise[:, ::2] = np.r_[noise_e, noise_e]
    combined_noise[:, 1::2] = np.r_[noise_n, noise_n]

    # Calculate covariance
    cov = covariance_along2dim(combined_noise, combined_noise.T)

    return cov


def get_noise_matrices(noise1, noise2):
    noise_e, noise_n = get_pair_matrix(noise1, noise2)
    noise_n = noise_n.T
    n = noise1.shape[0]

    # Get N1, N3
    n1 = np.zeros((n*4, n*4))
    n1[:, 0::4] = np.r_[noise_e, noise_e, noise_e, noise_e]
    n1[:, 1::4] = np.r_[noise_n, noise_n, noise_n, noise_n]
    n1[:, 2::4] = np.r_[noise_e, noise_e, noise_e, noise_e]
    n1[:, 3::4] = np.r_[noise_n, noise_n, noise_n, noise_n]

    # Get N2, N4
    n2 = np.zeros((n*4, n*4))
    n2[:, 0::4] = np.r_[noise_e, noise_e, noise_e, noise_e]
    n2[:, 1::4] = np.r_[noise_n, noise_n, noise_n, noise_n]
    n2[:, 2::4] = np.r_[noise_n, noise_n, noise_n, noise_n]
    n2[:, 3::4] = np.r_[noise_e, noise_e, noise_e, noise_e]
    return n1, n2


def get_cov_matrics(noise1, noise2, bins, bin_ix):
    if bin_ix == 0:
        # CSU, CSV, CTU, CTV are the same for bin0
        # autocovariance
        c = _get_cov_bin0(noise1, noise2)
        return c, c, c, c

    else:
        # Get pair matrix, e.g noise_e1 and its transpose
        noise_e, noise_et = get_pair_matrix(noise1, noise1)
        noise_n, noise_nt = get_pair_matrix(noise2, noise2)

        # Get bin
        bin_ixs = np.where(bins == bin_ix)

        # Get noise matrices for cov calculations
        n1 = get_noise_matrices(noise_et[bin_ixs],
                                noise_nt[bin_ixs])[0]
        n2 = get_noise_matrices(noise_e[bin_ixs],
                                noise_n[bin_ixs])[1]
        n3 = n1.T
        n4 = n2.T

        # Get covariances
        csu = covariance_along2dim(n1, n3)
        csv = covariance_along2dim(n1, n4)
        ctu = covariance_along2dim(n2, n3)
        ctv = covariance_along2dim(n2, n4)

        return csu, csv, ctu, ctv  
    