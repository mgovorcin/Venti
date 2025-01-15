import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from multiprocessing import Pool
from datetime import datetime as dt

# hectrop functions
from hectorp.control import Control, SingletonMeta
from hectorp.designmatrix import DesignMatrix
from hectorp.datasnooping import DataSnooping
from hectorp.covariance import Covariance
from hectorp.observations import Observations
from hectorp.mle import MLE

# TODO - write it locally to enable paralellization and then load it as dataframe
#           with inversion results


def estimate_trend(t, obs, sampling_period=1,
                   poly_deg=1, periodics=[1., 0.5],  steps=None,
                   postseismic_params=None, postseismic_function='log',
                   sstanh=None, noise_model='PLWN', do_filtering=False,
                   filtering_iq=3.0, path='./', fname='data.mom', ctl_name='control.ctl',
                   useRMLE=True, display=False, debug_display=False, verbose=False):

    # t - dates need to be in MJD
    # obs - need to be mm
    # offsets need to be list of MJD
    # postseismic need to be list of [MJD_T]
    # REF: https://gitlab.com/machielsimonbos/hectorp

    # Initialize hectorp control file
    ctl_file = create_control_file(path=path, ctl_name=ctl_name, poly_deg=poly_deg,
                                   periodics=periodics, filtering_iq=filtering_iq,
                                   noise_model=noise_model, useRMLE=useRMLE,
                                   verbose=verbose)
    control = Control(str(ctl_file))

    if postseismic_function == 'log':
        postseismic_log = postseismic_params
    else:
        postseismic_log = None

    if postseismic_function == 'exp':
        postseismic_exp = postseismic_params
    else:
        postseismic_exp = None

    # Initialize temp observations
    # TODO - convert this to function, no need for class
    temp_obs = wObservations(t, obs, sampling_period,
                             offsets=steps,
                             postseismic_log=postseismic_log,
                             postseismic_exp=postseismic_exp,
                             sstanh=sstanh,
                             verbose=verbose)

    # Create mom file
    obs_dir = Path(path) / fname
    temp_obs.momwrite(str(obs_dir))

    # Update control file
    control.params["DataDirectory"] = str(obs_dir.parent.resolve())
    control.params["DataFile"] = obs_dir.name

    # Get design matrix
    designmatrix = DesignMatrix()

    # Filter the data - remove outliers
    if do_filtering is True:
        # Do the filtering
        datasnooping = DataSnooping()
        hp_obs = Observations()
        output_filt = {}
        datasnooping.run(output_filt)

        if debug_display is True:
            # Get mean obs

            mean_obs = np.nanmean(np.hstack([obs, hp_obs.data.obs]))
            fig = plt.figure(figsize=(6, 4), dpi=150)
            plt.plot(mjd2decimalyr(t), obs - mean_obs, 'b-', label='observed')
            plt.plot(mjd2decimalyr(hp_obs.data.index),
                     hp_obs.data - mean_obs, 'r-', label='filtered')
            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('[{0:s}]'.format(control.params['PhysicalUnit']))
    else:
        hp_obs = Observations()

    # Estimate the trend
    covariance = Covariance()
    mle = MLE()

    # --- run MLE (least-squares + nelder-mead cycle to find minimum)
    [theta, C_theta, noise_params, sigma_eta] = mle.estimate_parameters()
    error = np.sqrt(np.diagonal(C_theta))

    # Get results
    output = {}
    hp_obs.show_results(output)
    mle.show_results(output)
    covariance.show_results(output, noise_params, sigma_eta)
    designmatrix.show_results(output, theta, error)

    # Add theta to obs
    designmatrix.add_mod(theta)
    hp_obs.add_mod(designmatrix.ts.data['mod'].values)

    if display:
        # --- Get data
        mjd = hp_obs.data.index.to_numpy()
        t1 = mjd2decimalyr(mjd)

        x = hp_obs.data['obs'].to_numpy()
        if 'mod' in hp_obs.data.columns:
            xhat = hp_obs.data['mod'].to_numpy()

        # Plot results
        mean_obs = np.nanmean(np.hstack([obs, hp_obs.data.obs]))

        fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(t1, x - mean_obs, 'b-', label='observed')
        # plt.errorbar(t, x, yerr=6.72, label='observed')
        if 'mod' in hp_obs.data.columns:
            plt.plot(t1, xhat - mean_obs, 'r-', label='model')

        plt.xlabel('Year')
        msg = f"v: {output['trend']:.2f} +/-"
        msg += f"{output['trend_sigma']:.2f} mm/yr"

        plt.plot(
            [], [], ' ', label=msg)
        plt.legend()
        plt.ylabel('[mm]')

    # Clear control file and mom file
    ctl_file.unlink()
    obs_dir.unlink()

    out_data = hp_obs.data

    # Clean metaclasses
    SingletonMeta.clear_all()

    return output, out_data

# Added for parallel processing


def _parallel_trend(kwargs):
    return estimate_trend(**kwargs)


def parallel_run(est_trend_params, n_jobs=3):
    if n_jobs == 1:
        print('run in loop')
        results = list(map(_parallel_trend, est_trend_params))
    else:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_parallel_trend, est_trend_params)
    return results


def estimate_spectrum(data, noisemodels, sampling_period=1):
    from scipy import signal
    from hectorp.estimatespectrum import (compute_G_White, compute_G_Powerlaw,
                                          compute_G_GGM, compute_G_AR1,
                                          compute_G_VA, compute_G_Matern)

    DeltaT = sampling_period

    fs = 1.0/(86400.0*DeltaT)
    T = DeltaT/365.25  # T in yr

    # --- extract parameter values
    if 'White' in noisemodels:
        sigma_w = noisemodels['White']['sigma']
    if 'Powerlaw' in noisemodels:
        sigma_pl = noisemodels['Powerlaw']['sigma']
        kappa = noisemodels['Powerlaw']['kappa']
        d_pl = -kappa/2.0
        sigma_pl *= math.pow(T, 0.5*d_pl)
    if 'FlickerGGM' in noisemodels:
        sigma_fn = noisemodels['FlickerGGM']['sigma']
        sigma_fn *= math.pow(T, 0.5*0.5)
    if 'RandomWalkGGM' in noisemodels:
        sigma_rw = noisemodels['RandomWalkGGM']['sigma']
        sigma_rw *= math.pow(T, 0.5*1.0)
    if 'GGM' in noisemodels:
        sigma_ggm = noisemodels['GGM']['sigma']
        kappa = noisemodels['GGM']['kappa']
        d_ggm = -kappa/2.0
        phi_ggm = noisemodels['GGM']['1-phi']
        sigma_ggm *= math.pow(T, 0.5*d_ggm)
        print('sigma_eta = {0:f}'.format(sigma_ggm))
    if 'VaryingAnnual' in noisemodels:
        sigma_va = noisemodels['VaryingAnnual']['sigma']
        phi_va = noisemodels['VaryingAnnual']['phi']
    if 'AR1' in noisemodels:
        sigma_ar1 = noisemodels['AR1']['sigma']
        phi_ar1 = noisemodels['AR1']['phi']
    if 'Matern' in noisemodels:
        sigma_mt = noisemodels['Matern']['sigma']
        kappa = noisemodels['Matern']['kappa']
        d_mt = -kappa/2.0
        lamba_mt = noisemodels['Matern']['lambda']

    # --- create string with noise model names
    noisemodel_names = ''
    for noisemodel in list(noisemodels):
        if len(noisemodel_names) > 0:
            noisemodel_names += ' + '
        if noisemodel == 'White':
            noisemodel_names += 'WN'
        elif noisemodel == 'Powerlaw':
            noisemodel_names += 'PL'
        elif noisemodel == 'GGM':
            if phi_ggm < 1.0e-5:
                noisemodel_names += 'PL'
            else:
                noisemodel_names += 'GGM'
        elif noisemodel == 'FlickerGGM':
            noisemodel_names += 'FN'
        elif noisemodel == 'RandomwalkGGM':
            noisemodel_names += 'RW'
        elif noisemodel == 'VaryingAnnual':
            noisemodel_names += 'VA'
        elif noisemodel == 'AR1':
            noisemodel_names += 'AR1'
        elif noisemodel == 'Matern':
            noisemodel_names += 'MT'

    # --- Replace NaN's to zero's
    x_clean = np.nan_to_num(data)
    n = len(data)

    # --- Compute PSD with Welch method
    f, Pxx_den = signal.welch(x_clean, fs, window='hann', return_onesided=True,
                              noverlap=n//8, nperseg=n//4)

    tpi = math.pi*2.0
    m = len(f)
    N = 1000
    freq0 = math.log(f[1])
    freq1 = math.log(f[m-1])
    fm = [0.0]*N
    G = [0.0]*N
    for i in range(0, N):
        s = i/float(N)
        fm[i] = math.exp((1.0-s)*freq0 + s*freq1)
        for noisemodel in noisemodels:
            if noisemodel == 'White':
                scale = math.pow(sigma_w, 2.0)/fs  # --- no negative f (2x)
                G[i] += scale*compute_G_White(tpi*fm[i]/fs)
            elif noisemodel == 'Powerlaw':
                scale = math.pow(sigma_pl, 2.0)/fs
                G[i] += scale*compute_G_Powerlaw(tpi*fm[i]/fs, d_pl)
            elif noisemodel == 'FlickerGGM':
                scale = math.pow(sigma_fn, 2.0)/fs
                G[i] += scale*compute_G_Powerlaw(tpi*fm[i]/fs, 0.5)
            elif noisemodel == 'RandomWalkGGM':
                scale = math.pow(sigma_rw, 2.0)/fs
                G[i] += scale*compute_G_Powerlaw(tpi*fm[i]/fs, 1.0)
            elif noisemodel == 'GGM':
                scale = math.pow(sigma_ggm, 2.0)/fs
                G[i] += scale*compute_G_GGM(tpi*fm[i]/fs, d_ggm, phi_ggm)
            elif noisemodel == 'AR1':
                scale = math.pow(sigma_ar1, 2.0)/fs
                G[i] += scale*compute_G_AR1(tpi*fm[i]/fs, phi_ar1)
            elif noisemodel == 'VaryingAnnual':
                scale = math.pow(sigma_va, 2.0)/fs
                G[i] += scale*compute_G_VA(fm[i], fs, phi_va)
            elif noisemodel == 'Matern':
                scale = math.pow(sigma_mt, 2.0)/fs
                G[i] += scale * \
                    compute_G_Matern(tpi*fm[i]/fs, d_mt, lamba_mt)
            else:
                print('Unknown noisemodel: {0:s}'.format(noisemodel))
                sys.exit()

    # PLOT
    fig = plt.figure(figsize=(5, 4), dpi=150)
    plt.loglog(f, Pxx_den, label='observed')
    plt.loglog(fm, G, label='PL +W')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [mm**2/Hz]')
    plt.legend()


# Suporting functions
# Control file
# Additional noise models
# VA - VaryingAnnual, AR1 = AR1, MT- Matern
noise_models = dict(
    FNWN='FlickerGGM White',
    PLWN='GGM White',
    RWFNWN='RandomWalkGGM FlickerGGM White',
    WN='White',
    PL='GGM',
    FL='FlickerGGM')

clr_template = '''DataFile            None
DataDirectory       None
OutputFile          None
PhysicalUnit        mm
TimeUnit            days
DegreePolynomial    {poly_deg}
ScaleFactor         1.0
periodicsignals     {periodics_str}
estimateoffsets     yes
IQ_factor           {filtering_iq}
NoiseModels         {noise_model}
GGM_1mphi           6.9e-06
useRMLE             {useRMLE}
Verbose             {verbose}'''


def create_control_file(path='./', ctl_name='control.ctl', poly_deg=1,
                        periodics=[1., 0.5], filtering_iq=3.0,
                        noise_model='PLWN', useRMLE=True, verbose=False):

    periodics_str_array = np.char.mod('%.3f', np.array(periodics)*365.25)
    periodics_str = ' '.join(periodics_str_array)

    #verbose = 'yes' if verbose is True else 'no'
    useRMLE = 'yes' if useRMLE is True else 'no'

    if noise_model in noise_models.keys():
        n_model = noise_models[noise_model]
    else:
        options = [' - '.join(i) for i in np.c_[list(noise_models.keys()),
                                                list(noise_models.values())]]

        msg = 'Not supported noise model!\n'
        msg += '          Available options:\n'
        for option in options:
            msg += f'            {option}\n'
        raise Warning(msg)

    control = clr_template.format(poly_deg=poly_deg,
                                  periodics_str=periodics_str,
                                  filtering_iq=filtering_iq,
                                  noise_model=n_model,
                                  useRMLE=useRMLE,
                                  verbose=verbose)

    control_file = Path(path) / ctl_name
    control_file.write_text(control)

    return control_file

# Modified hectorp Observations class to allow initializing class
# by adding input parameters and avoid reading mom/mrf file
# Changes only init part and remove some of the unecessary functions
# Original code: https://gitlab.com/machielsimonbos/hectorp


class wObservations():
    def __init__(self,
                 t, obs,
                 sampling_period,
                 offsets=None,
                 postseismic_log=None,
                 postseismic_exp=None,
                 sstanh=None,
                 verbose=True):

        # Postseismics
        # log/exp:  list([MJD, T]) ,e.g [[51994.0, 10.0]]
        if postseismic_exp:
            self.postseismicexp = postseismic_exp
        else:
            self.postseismicexp = []
        if postseismic_log:
            self.postseismicelog = postseismic_log
        else:
            self.postseismiclog = []
        # Offsets
        # list of MJDs
        if offsets:
            self.offsets = offsets
        else:
            self.offsets = []

        # Slow Slip
        # list([MJD, T]) ,e.g [[51994.0, 10.0]]
        if sstanh:
            self.ssetanh = sstanh
        else:
            self.ssetanh = []

        # Get dataframe and needed info
        obs_dict = get_obs_dataframe(t, obs, sampling_period)
        (m, n) = obs_dict['F'].shape

        self.data = obs_dict['data']
        self.sampling_period = sampling_period
        self.F = obs_dict['F']
        self.percentage_gaps = obs_dict['gaps']
        self.m = m
        self.scale_factor = 1.0
        # Fix to mom format
        self.ts_format = 'mom'
        self.verbose = verbose

        if verbose:
            print(f"Number of observations+gaps: {m:d}")
            print(f"Percentage of gaps         : {self.percentage_gaps:5.1f}")

    def set_NaN(self, index):
        """ Set observation at index to NaN and update matrix F

        Args:
            index (int): index of array which needs to be set to NaN
        """

        self.data.iloc[index, 0] = np.nan
        dummy = np.zeros(self.m)
        dummy[index] = 1.0
        self.F = np.c_[self.F, dummy]  # add another column to F

    def add_mod(self, xhat):
        """ Add estimated model as column in DataFrame

        Args:
            xhat (array float) : estimated model
        """

        self.data['mod'] = np.asarray(xhat)

    def add_offset(self, t):
        """ Add time t to list of offsets

        Args:
            t (float): modified julian date or second of day of offset
        """

        EPS = 1.0e-6
        found = False
        i = 0
        while i < len(self.offsets) and found == False:
            if abs(self.offsets[i]-t) < EPS:
                found = True
            i += 1
        if found == False:
            self.offsets.append(t)

    def show_results(self, output):
        """ add info to json-ouput dict
        """

        output['N'] = self.m
        output['gap_percentage'] = self.percentage_gaps
        output['TimeUnit'] = 'days'

    def momwrite(self, fname):
        """Write the momdata to a file called fname

        Args:
            fname (string) : name of file that will be written
        """
        # --- Try to open the file for writing
        try:
            fp = open(fname, 'w')
        except IOError:
            print('Error: File {0:s} cannot be opened for written.'.
                  format(fname))
            sys.exit()
        if self.verbose == True:
            print('--> {0:s}'.format(fname))

        # --- Write header
        fp.write('# sampling period {0:f}\n'.format(self.sampling_period))

        # --- Write header offsets
        for i in range(0, len(self.offsets)):
            fp.write('# offset {0:10.4f}\n'.format(self.offsets[i]))
        # --- Write header exponential decay after seismic event
        for i in range(0, len(self.postseismicexp)):
            [mjd, T] = self.postseismicexp[i]
            fp.write('# exp {0:10.4f} {1:5.1f}\n'.format(mjd, T))
        # --- Write header logarithmic decay after seismic event
        for i in range(0, len(self.postseismiclog)):
            [mjd, T] = self.postseismiclog[i]
            fp.write('# exp {0:10.4f} {1:5.1f}\n'.format(mjd, T))
        # --- Write header slow slip event
        for i in range(0, len(self.ssetanh)):
            [mjd, T] = self.sshtanh[i]
            fp.write('# tanh {0:10.4f} {1:5.1f}\n'.format(mjd, T))

        # --- Write time series
        for i in range(0, len(self.data.index)):
            if not math.isnan(self.data.iloc[i, 0]) == True:
                fp.write('{0:12.6f} {1:13.6f}'.format(self.data.index[i],
                                                      self.data.iloc[i, 0]))
                if len(self.data.columns) == 2:
                    fp.write(' {0:13.6f}\n'.format(self.data.iloc[i, 1]))
                else:
                    fp.write('\n')

        fp.close()


def get_obs_dataframe(t, obs, sampling_period):
    '''
    t - array of times in MJD

    '''
    time_list = []
    obs_list = []
    TINY = 1.0e-6

    for ix, mjd in enumerate(t[:-1]):
        mjd_diff = t[ix + 1] - mjd
        # Fill up gaps with NaNs depending on defined sampling
        if mjd_diff - sampling_period > TINY:
            mjds = np.arange(mjd, t[ix + 1], sampling_period)
            nans = np.ones(mjds.shape[0] - 1) * np.NaN
            obs1 = np.r_[obs[ix], nans]
            time_list.append(mjds)
            obs_list.append(obs1)
        else:
            time_list.append(mjd)
            obs_list.append(obs[ix])

    time_list.append(t[-1])
    obs_list.append(obs[-1])

    # Create dataframe
    data = pd.DataFrame({'obs': np.hstack(obs_list)},
                        index=np.hstack(time_list))

    m = len(data.index)
    n = data['obs'].isna().sum()
    F = np.zeros((m, n))
    j = 0

    for i in range(0, m):
        if np.isnan(data.iloc[i, 0]) == True:
            F[i, j] = 1.0
            j += 1

    percentage_gaps = 100.0 * float(n) / float(m)

    return {'data': data, 'F': F, 'gaps': percentage_gaps}


def mjd2decimalyr(time_mjd):
    return (time_mjd - 51544)/365.25 + 2000


def npdatetime2dt(datetime_array):
    utc_date = np.datetime64('1970-01-01')
    seconds = (datetime_array - utc_date) / np.timedelta64(1, 's')
    dt_list = [dt.utcfromtimestamp(s) for s in seconds]
    return dt_list


def plot_tsfit(dates, disp, ts_fit, m, m_std, steps=None):
    from matplotlib import pyplot as plt

    n_disp = disp.shape[1]

    fig, axs = plt.subplots(n_disp, 1, figsize=(14, 10), sharex=True)
    if n_disp == 1:
        axs = [axs]
        disp = disp[:, np.newaxis]
        m = m[:, np.newaxis]
        m_std = m_std[:, np.newaxis]

    # fit the model

    plot = [(ax.plot(dates, disp[:, i] - np.nanmean(disp[:, i]), 'b.'),
            ax.plot(dates, ts_fit[:, i] - np.nanmean(disp[:, i]),
            'r-', label=f'v {m[i]:.2f} \u00B1 {m_std[i]:.2f} mm/yr'))
            for i, ax in enumerate(axs)]

    if steps is not None:
        [ax.vlines(steps,
                   np.nanmin(disp[:, i] - np.nanmean(disp[:, i])),
                   np.nanmax(disp[:, i] - np.nanmean(disp[:, i])),
                   linestyles='dashed')
         for i, ax in enumerate(axs)]

    txt = ['East (mm)', 'North (mm)', 'UP (mm)']
    labs = [(ax.set_ylabel(txt[i]), ax.legend(loc='upper left'))
            for i, ax in enumerate(axs)]
