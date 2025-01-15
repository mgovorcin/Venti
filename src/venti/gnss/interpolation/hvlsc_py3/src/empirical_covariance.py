#!/usr/bin/python
import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt
from .utils import get_pair_matrix, get_distance_matrix
from .models import func_gm1
from pyproj import Geod
g = Geod(ellps='WGS84')


def get_empirical_covariance(x, y,
                             data1, data2,
                             noise1, noise2,
                             bin_spacing=50e3,
                             custom_covar=None,
                             constrain_mask=None,
                             distance_penalty=1500,
                             correlation_analysis=False,
                             plot_covariance=True,
                             max_range=None):

    # 1.Outlier removal
    outlier_ix1 = find_outliers(data1, sigma=3)

    if outlier_ix1.size != 0:
        data1 = np.delete(data1, outlier_ix1)
        x = np.delete(x, outlier_ix1)
        y = np.delete(y, outlier_ix1)
        data2 = np.delete(data2, outlier_ix1)
        noise1 = np.delete(noise1, outlier_ix1)
        noise2 = np.delete(noise2, outlier_ix1)
        print('Outlier identified and removed')

    outlier_ix2 = find_outliers(data2, sigma=3)

    if outlier_ix2.size != 0:
        data2 = np.delete(data2, outlier_ix2)
        x = np.delete(x, outlier_ix2)
        y = np.delete(y, outlier_ix2)
        data1 = np.delete(data1, outlier_ix2)
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

    bin_edges = np.arange(bin_spacing,
                          dmax,
                          bin_spacing*2)
    bin_edges = np.r_[[0, 0.001], bin_edges]

    if max_range:
        distance_array[distance_array > max_range] = -1


    # Get bin_stats
    if correlation_analysis:
        data1, data2 = get_pair_matrix(data1, data2)
        bins_mean, bins_std, bincenter_array = bin_normalized_stats(distance_array,
                                                                    data1, data2,
                                                                    bin_edges)
        # Run optimization twice to find better misfit
        corr_bounds = [[bin_edges[2], bin_edges[3]],
                       [0, bin_edges[3]]]
    else:
        bins_mean, bins_std, bincenter_array = bin_stats(distance_array,
                                                         data1, data2,
                                                         noise1, noise2,
                                                         bin_edges)
        # Run optimization twice to find better misfit
        # if there are no bins, set bounds to be
        # mean(distance) - max(distance) - add it??
        corr_bounds = [[bin_edges[2], bin_edges[3]], [0, bin_edges[-1]]]

    # c0 at d0
    c0 = bins_mean[0]
 
    # Fit function
    # fix the variance at d0
    # TODO add here custom function
    if custom_covar is not None:
        assert np.asarray(
            custom_covar).shape[0] == 2, 'Custom covariance must have this formating: [c0, d0]!'

        def fixed_gm1_func(cova_dist, d0):
            return func_gm1(cova_dist, custom_covar[0], custom_covar[1])
    else:
        def fixed_gm1_func(cova_dist, d0):
            return func_gm1(cova_dist, c0, d0)

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

    # Get final result - note clean below
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

    out_results = dict(misfit=misfit, misfit_1st=misfit_1st, pearson=pearson_cor)    
    if plot_covariance:
        plot_empirical_covariance(bincenter_array, bins_mean, bins_std,
                                  fit_model[select_ix], popt[0], popt[1])

    return popt, perr, out_results


# for empirical_covariance_std
def bin_normalized_stats(distance_matrix, data1, data2, bin_edges):
    # Use triangular matrix to avoid duplicates
    distance_matrix = np.tril(distance_matrix)
    n = distance_matrix.shape[0]

    # Mask negative values - constrain mask
    dist_mask = distance_matrix < 0

    # Get bins
    bins = np.digitize(distance_matrix, bin_edges)
    # Get counts per bin
    # bin_counts = np.bincount(bins.ravel())[2:]

    # Prepare bin for d0, keep only diag elements
    bins[np.triu_indices(n)] = 9999
    bins[dist_mask] = 9999
    bins[np.diag_indices(n)] = 0

    # Get counts per bin
    bin_counts = np.bincount(bins.ravel())

    # Remove empty bins
    bin_counts = bin_counts[bin_counts != 0]

    # Update count for bin d0
    # bin_counts = np.r_[n, bin_counts]

    # Initialize
    bin_means = []
    bin_stds = []

    # Loop through bins, skip the last one 9999
    for ix, count in zip(np.unique(bins)[:-1], bin_counts[:-1]):
        mask = bins == ix
        bin_d1 = data1[mask]
        bin_d2 = data2[mask]

        mean1 = np.mean(np.unique(bin_d1))
        mean2 = np.mean(np.unique(bin_d2))
        std1 = np.std(np.unique(bin_d1))
        std2 = np.std(np.unique(bin_d2))

        bin_data = (bin_d1 - mean1) * (bin_d2 - mean2)
        norm = std1 * std2

        # Get normalized bin mean and std
        bin_means.append(np.sum(bin_data) / (count - 1) / norm)
        bin_stds.append(np.std(bin_data / norm))

    bins = np.c_[bin_counts[:-1], bin_means, bin_stds]

    # Get center of bins
    bincenter_array = (bin_edges[:-1] + bin_edges[1:]) / 2
    bincenter_array = bincenter_array[np.where(bins[:, 0] >= n)]
    bincenter_array[0] = 0

    # filter bins with less than obs points
    bins = bins[np.where(bins[:, 0] >= n)]

    return bins[:, 1], bins[:, 2], bincenter_array


def bin_stats(distance_matrix, data1, data2, noise1, noise2, bin_edges):
    # num of points
    n = data1.size

    # 2. Get variance C0 at d0=0, distance matrix, and cross-data matrix
    # eq 11
    c01 = np.mean(data1**2) - np.mean(noise1**2)
    c02 = np.mean(data2**2) - np.mean(noise2**2)

    # variance and covariance at d0=0
    c0 = c01 * 0.5 + c02 * 0.5
    s0 = np.std(np.c_[data1, data2])
    bin_d0 = np.r_[n*2, c0, s0]

    # cross-data product matrix
    data1, data2 = get_pair_matrix(data1, data2)
    data_array = data1 * data2

    # Flatten the array
    dist = distance_matrix.ravel()
    data = data_array.ravel()

    # Remove zero values
    data = data[dist > 0]
    dist = dist[dist > 0]

    # Get stats
    bins_sum = stats.binned_statistic(dist, data, 'sum', bins=bin_edges)[0]
    count = stats.binned_statistic(dist, data, 'count', bins=bin_edges)[0]

    # Get adjusted mean for sample variance
    bins_mean = bins_sum / (count - 1)
    bins_std = stats.binned_statistic(dist, data, 'std', bins=bin_edges)[0]

    bins = np.c_[count, bins_mean, bins_std]

    # Get center of bins
    bincenter_array = (bin_edges[:-1] + bin_edges[1:]) / 2
    bincenter_array = bincenter_array[np.where(bins[:, 0] >= n * 2)]

    # Add bin d0
    if bins_mean[1] < 0 : bin_d0[1] *= -1
    bins = np.vstack([bin_d0, bins])
    bincenter_array = np.r_[0, bincenter_array]

    # filter bins with less than 2x obs points (as here is use full dist matrix)
    bins = bins[np.where(bins[:, 0] >= n * 2)]

    return bins[:, 1], bins[:, 2], bincenter_array


def find_outliers(data, sigma=3):
    rms = sigma * np.sqrt(np.mean(data**2))
    return np.where(abs(data) > rms)[0]


def plot_empirical_covariance(bincenter_array, bins_mean, bins_std, fitted_values, C0, d0):
    fig = plt.figure(figsize=(8, 4))
    plt.errorbar(bincenter_array, bins_mean,
                 xerr=0, yerr=bins_std, fmt='+',
                 ecolor='darkgrey', elinewidth=1, capsize=6, capthick=1,
                 ms=10, mfc='black', mec='black', label='Estimated covariogram')
    plt.scatter(bincenter_array, bins_mean, s=100,
                c='black', marker='+', lw=2.0, zorder=2)

    plt.xlabel('Distance [km]', fontsize=10)
    plt.ylabel(r'Covariance [$\mathregular{\frac{mm^2}{a^2}}$]', fontsize=10)
    plt.title('Covariance function determination using a Gauss-Markov 1st order  function',
              fontsize=12, y=1.05)

    plt.plot(bincenter_array, fitted_values[:],
             c='red', lw=2.5, zorder=3,
             label=r'$\mathregular{C_0}$ = %.3f, $\mathregular{d_0}$ = %d' % (C0, d0))
    plt.legend(loc='upper right')
