#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:49:43 2022

@author: govorcin
"""

import numpy as np
import time
import pickle
import pandas as pd
from pathlib import Path
from pyproj import Geod
from multiprocessing import Pool
import requests
from tqdm import tqdm
from venti.solvers.midas import midas
from venti.gnss.quality import calculate_temporal_variability
from venti.solvers.hector import parallel_run, plot_tsfit, mjd2decimalyr, npdatetime2dt
from venti.solvers import time_func
from itertools import compress, repeat
from urllib.request import urlretrieve
from matplotlib import pyplot as plt
from datetime import datetime as dt
from hectorp.calendar import compute_mjd


GNSS_SOURCE = {
    'jpl': {
        'vel': 'https://sideshow.jpl.nasa.gov/post/tables/table2.html',
        'ts': 'https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/'},
    'unr': {
        'vel': 'http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt',
        'ts': 'http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/'},
    'cors': {
        'vel': 'https://noaa-cors-pds.s3.amazonaws.com/coord/coord_14/itrf2014_geo.comp.txt',
        'ts': None}
}

GNSS_DICT = dict(site=np.str_, lat=np.float32, lon=np.float32,
                 n=np.float64, e=np.float64, v=np.float64,
                 sn=np.float64, se=np.float64, sv=np.float64,
                 n_gaps=np.int16, duration=np.float32,
                 start_date='datetime64[s]', end_date='datetime64[s]',
                 ts_path=np.str_)

# TO DO:
# standardize ts dataframe

JPL_TS = ['decimal_yr', 'e', 'n', 'v', 'sn', 'se', 'sv',
          'en_cor', 'ev_cor', 'nv_cor', 'J2000_time_s',
          'yyyy', 'mm', 'dd', 'h', 'm', 's']
UNR_TS = ['site', 'yyyymmdd', 'decimal_yr', 'MJD', 'GNSS_week', 'day',
          'ref_lon', 'e0', 'east', 'n0', 'north', 'v0', 'v',
          'ant', 'se', 'sn', 'sv', 'en_cor', 'ev_cor', 'nv_cor',
          'lat', 'lon', 'h']


def downloader(url: [str | list], output: [str | list], n_threads: int = 8):
    # Sanity check
    if type(url) != type(output):
        raise ValueError('Input files are different types, both url and '
                         'output need to be either list or str!')

    output = [output] if type(output) is str else output
    url = [url] if type(url) is str else url

    output = [Path(out) for out in output]
    # Crete output directory if does not exist
    for out in output:
        out.parent.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    url, output = np.array(url), np.array(output)
    ix = [Path(out).is_file() == 0 for out in output]
    url, output = url[ix], output[ix]

    # Run download in parallel
    pool = Pool(n_threads)
    pool.starmap(urlretrieve, zip(url, output))
    pool.close()
    pool.join()

    return None


class GNSS():
    """
    Base class for GNSS data

    Parameters
    ----------
    output_dir : str, optional
        Output directory for storing the GNSS data, by default './GNSS'
    """

    def __init__(self, output_dir='./GNSS', **kwargs):
        """
        Initialize the GNSS class
        """
        self.output_dir = Path(output_dir).absolute()
        # Unit of rates in mm/yr
        self.unit_rate = 'mm/yr'
        # Unit of timeseries in meters
        self.unit_ts = 'm'
        # Source of the data
        self.source = None
        # Directory where the timeseries are stored
        self.out_ts_dir = None
        # Path to the station list
        self.station_list = None
        # GNSS data frame
        self.gnss_df = None
        # Path to the timeseries file
        self.ts_path = None
        # Spatial bounding box of the GNSS stations
        self.snwe = None
        # Reference epoch
        self.reference_epoch = None
        # Reference frame
        self.reference_frame = None
        # Reference ellipsoid
        self.reference_ellipsoid = None

    def _defaults(self, source='jpl'):
        self.source = source
        station_list = f'stations_{source}.txt'
        self.out_ts_dir = self.output_dir / 'ts' / self.source
        self.station_list = self.output_dir / station_list

    def _download_station_list(self):
        downloader(GNSS_SOURCE[self.source]['vel'],
                  str(self.station_list), n_threads=1)

    def download_ts_files(self, sites: list = None, n_threads: int = 8, overwrite=False):
        # Get the inputs
        source_url = GNSS_SOURCE[self.source]['ts']

        if sites is None:
            if isinstance(self.gnss_df, pd.DataFrame):
                sites = self.gnss_df.site
            else:
                raise ValueError('Missing station list!'
                                 ' Download it first or'
                                 ' select sites manually!')

        # Define ts extension for different sources
        if self.source == 'jpl':
            extension = '.series'
        elif self.source == 'unr':
            extension = '.tenv3'

        urls = list(source_url + sites + extension)
        outs = list(self.out_ts_dir / sites)

        # Download
        print(f'Downloading time-series of {len(urls)} stations')
        start_time = time.time()
        downloader(urls, outs, n_threads=n_threads)
        end_time = time.time()
        print(f'Finish downloading: {end_time - start_time:.2f}s duration')
        ts_path = {}
        for site in sites:
            ts_path[site] = self.out_ts_dir.absolute() / site

        self.ts_path = ts_path

    def save(self, name):
        self.pickle = self.output_dir / (name + '.pkl')
        with open(str(self.pickle), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        self.pickle = self.output_dir / (name + '.pkl')
        with open(str(self.pickle), 'rb') as inp:
            gnss = pickle.load(inp)
        return gnss


# Meant only for JPL_GNSS class for now

def df_get_duration(df, gap=6):
    # Get TS Duration, number of gaps, start and end date
    duration = df.decimal_yr.iloc[-1] - df.decimal_yr.iloc[0]

    def to_datetime(x):
        return dt(*np.r_[x.iloc[11:-1]].astype(int))

    dates = df.apply(to_datetime, axis=1)
    start_date = dates.min()
    end_date = dates.max()
    n_gaps = np.sum(np.r_[np.diff(dates) // np.timedelta64(1, 'D'), 1] > gap)

    return dict(site=df.site[0], n_gaps=n_gaps, duration=duration,
                start_date=start_date, end_date=end_date)


class JPL(GNSS):

    def __init__(self, output_dir='./GNSS', **kwargs):
        super().__init__(output_dir=output_dir)
        self.reference_epoch = '2023-01-01'
        self.reference_frame = 'IGS14'
        self.reference_ellipsoid = 'GRS80'

        # Get defaults
        self._defaults(source='jpl')

        # Downlad station list
        self._download_station_list()

    def read_station_list(self, snwe=None):
        # Extract position lat, lon from txt file
        latlon = np.loadtxt(open(self.station_list, 'rt').readlines()[7:-1:2],
                            skiprows=1,
                            dtype=str)[:, 2:4]
        # Read velocities
        vel_data = np.loadtxt(open(self.station_list, 'rt').readlines()[8:-1:2],
                              skiprows=1,
                              dtype=str)
        # Dataframe
        df = pd.DataFrame(columns=GNSS_DICT.keys()).astype(GNSS_DICT)
        df.iloc[:, :9] = np.c_[vel_data[:, 0],
                               np.float32(latlon),
                               np.float64(vel_data[:, 2:])]
        df = df.astype(GNSS_DICT, errors='ignore')
        # Filter station based on aoi
        if snwe is not None:
            idx = ((df['lat'] >= snwe[0]) * (df['lat'] <= snwe[1]) *
                   (df['lon'] >= snwe[2]) * (df['lon'] <= snwe[3]))
            df = df[idx]

        self.gnss_df = df
        self.snwe = snwe

        return self.gnss_df

    def read_ts_file(self, site):
        # Get path
        path = self.ts_path[site]
        ts_data = np.loadtxt(open(path, 'rt').readlines(),
                             dtype=np.float64)
        df = pd.DataFrame(ts_data, columns=JPL_TS)
        df['site'] = site

        return df

    def _get_duration(self, site, gap: int = 6):
        # Get ts dataframe
        df = self.read_ts_file(site)

        return df_get_duration(df, gap)

    def _update_df_with_ts(self, gap: int = 6, n_threads: int = 8):
        ts = [self.read_ts_file(site) for site in self.gnss_df.site]
        pool = Pool(n_threads)
        results = pool.starmap(df_get_duration, [(df, gap) for df in ts])
        pool.close()
        pool.join()

        # Update dataframe
        for result in results:
            site = list(result.values())[0]
            data = list(result.values())[1:]
            data.extend([self.ts_path[site]])
            cols = ['n_gaps', 'duration', 'start_date', 'end_date', 'ts_path']
            self.gnss_df.loc[self.gnss_df.site == site, cols] = data
        return self.gnss_df


class UNR(GNSS):
    # TODO: add midas, or parametric function to get velocities
    def __init__(self, output_dir='./GNSS', **kwargs):
        """
        Initialize UNR GNSS class
        """
        super().__init__(output_dir=output_dir)
        self.reference_epoch = None
        self.reference_frame = 'IGS14'
        self.reference_ellipsoid = 'GRS80'

        # Get defaults
        self._defaults(source='unr')

        # Download station list
        self._download_station_list()

    def read_station_list(self, snwe=None):
        # Extract sites name
        sites = np.loadtxt(self.station_list, dtype=bytes,
                           skiprows=1, usecols=(0)).astype(str)

        # Extract position lat, lon from txt file
        latlon = np.loadtxt(self.station_list, dtype=bytes,
                            skiprows=1, usecols=(1, 2)).astype(np.float32)
        latlon[:, 1] -= np.round(latlon[:, 1] / 360.) * 360

        # Dates
        dates = np.loadtxt(self.station_list, dtype=bytes,
                           skiprows=1, usecols=(7, 8)).astype(str)
        start_date = np.array([dt.strptime(i, "%Y-%m-%d")
                              for i in dates[:, 0].astype(str)])
        end_date = np.array([dt.strptime(i, "%Y-%m-%d")
                            for i in dates[:, 1].astype(str)])
        duration = end_date - start_date
        duration = [dur.days / 365.25 for dur in duration]

        # Dataframe
        df = pd.DataFrame(columns=GNSS_DICT.keys()).astype(GNSS_DICT)
        df.iloc[:, :3] = np.c_[sites, np.float32(latlon)]
        df.iloc[:, 11:13] = np.c_[start_date, end_date]
        df.iloc[:, 10] = duration
        df = df.astype(GNSS_DICT, errors='ignore')

        # Filter station based on aoi
        if snwe is not None:
            idx = ((df['lat'] >= snwe[0]) * (df['lat'] <= snwe[1]) *
                   (df['lon'] >= snwe[2]) * (df['lon'] <= snwe[3]))
            df = df[idx]

        self.gnss_df = df
        self.snwe = snwe

        return self.gnss_df

    def read_ts_file(self, site):
        # Get path
        path = self.ts_path[site]
        # ts_data = np.loadtxt(path, dtype=bytes, skiprows=1).astype(str)
        ts_data = pd.read_csv(path, sep='\s+').values

        df = pd.DataFrame(np.atleast_2d(ts_data),
                          columns=UNR_TS)
        df['site'] = site

        return df

    def get_steps(self):
        steps_url = 'http://geodesy.unr.edu/NGLStationPages/steps.txt'
        # Steps readme
        # http://geodesy.unr.edu/NGLStationPages/steps_readme.txt
        steps_list = requests.get(steps_url).text

        # Get steps
        steps = [tuple(line.split()) for line in steps_list.split('\n')][:-1]

        if hasattr(self, 'gnss_df'):
            site_names = self.gnss_df.site.values.tolist()
            steps_flag = [step[0] in site_names for step in steps]
            steps = list(compress(steps, steps_flag))

        # Separate earthquake steps
        eq_steps = [step for step in steps if len(step) > 5]
        steps = list(set(steps).difference(eq_steps))

        cols = ['site', 'date', 'code', 'description']
        eq_cols = ['site', 'date', 'code', 'threshold_distance',
                   'site_distance_from_eq', 'mag', 'description']

        # Dataframe
        self.steps = pd.concat([pd.DataFrame(steps, columns=cols),
                                pd.DataFrame(eq_steps, columns=eq_cols)])

        return self.steps

    def get_station_latlon(self, site):
        """Get station lat/lon - MIintpy"""
        # Note double check the purpose of this!!!
        # Read the site ts data
        ts_data = self.read_ts_file(site)

        ref_lon, ref_lat = float(ts_data.iloc[0, 6]), 0.
        e0, e_off, n0, n_off = ts_data.iloc[0, 7:11].astype(np.float32)
        e0 += e_off
        n0 += n_off
        az = np.arctan2(e0, n0) / np.pi * 180.
        dist = np.sqrt(e0**2 + n0**2)

        # Get the new coordinates with pyproj
        g = Geod(ellps='WGS84')
        site_lon, site_lat = g.fwd(ref_lon, ref_lat, az, dist)[0:2]

        return site_lat, site_lon

    def get_ts_fit(self, site,
                   poly_deg=0, periods=[], ref_date=None,
                   do_midas=True, do_hector=False, start=None, end=None,
                   display=False, do_filtering=False, update=False, save_fig=False):
        def str2datetime(date):
            return dt.strptime(date, '%y%b%d')

        def filter_timespan(dates, disp, steps, start=None, end=None):
            # expect string YYYYMMDD
            if disp.ndim == 1:
                disp = disp[:, np.newaxis]

            if start or end:
                if start:
                    start = dt.strptime(start, "%Y%m%d")
                    flag = dates > start
                    flags = steps > start
                else:
                    start = dates[0]
                    flag = dates >= start
                    flags = steps > start

                if end:
                    end = dt.strptime(end, "%Y%m%d")
                    flag1 = dates < end
                    flags1 = steps < end
                else:
                    end = np.sort(dates)[-1]
                    flag1 = dates < end
                    flags1 = steps <= end

                flag *= flag1
                flags *= flags1

                return dates[flag], disp[flag, :], steps[flags]
            else:
                start = dates[0]
                end = dates[-1]
                flag = (start <= steps) & (steps <= end)

                return dates, disp, steps[flag]

        # Switch midas to parametric fitting
        if poly_deg > 0:
            do_midas = False

        # Get site time-series data
        ts_df = self.read_ts_file(site)
        dates = ts_df.yyyymmdd.apply(str2datetime).values
        dates = dates.astype('datetime64[s]')

        disp = np.c_[ts_df.east.values.astype(np.float32),
                     ts_df.north.values.astype(np.float32),
                     ts_df.v.values.astype(np.float32)]

        # Get steps
        site_steps = self.steps[self.steps.site == site]
        steps = site_steps.date.apply(str2datetime).values
        steps = steps.astype('datetime64[s]')
        # Avoid duplicates in steps
        steps = np.unique(steps)

        # Filter timeseries
        dates, disp, steps = filter_timespan(dates, disp, steps,
                                             start=start, end=end)
        # TS fit
        if do_midas is True:
            # Fit midas
            vel, vel_std, _, _ = midas(dates,
                                       disp,
                                       steps=steps,
                                       display=display)
            # m/yr to mm/yr
            vel *= 1000
            vel_std *= 1000

        elif do_hector is True:
            # prepare input

            # data dict
            data_dict = [dict(obs=disp[:, 0] * 1e3,  fname=f'{site}_e.mom',
                              ctl_name=f'{site}_e.ctl'),
                         dict(obs=disp[:, 1] * 1e3, fname=f'{site}_n.mom',
                              ctl_name=f'{site}_n.ctl'),
                         dict(obs=disp[:, 2] * 1e3, fname=f'{site}_v.mom',
                              ctl_name=f'{site}_v.ctl')]

            # Get dates in MJD
            # TODO make this function compatible with datetime64
            dates_dt = npdatetime2dt(dates)
            t = [compute_mjd(date.year, date.month, date.day, 0, 0, 0)
                 for date in dates_dt]

            # Get steps in MJD
            # First get steps to datetime format
            steps_dt = npdatetime2dt(steps)

            offsets_mjd = []
            for ti in steps_dt:
                offsets_mjd.append(compute_mjd(
                    ti.year, ti.month, ti.day, 0, 0, 0))

            processing_dict = dict(t=t, sampling_period=1, steps=offsets_mjd,
                                   poly_deg=poly_deg, periodics=periods,
                                   postseismic_params=None, postseismic_function='log',
                                   sstanh=None, noise_model='PLWN',
                                   do_filtering=do_filtering, display=False)
            # Combine dicts
            _ = [dix.update(processing_dict) for dix in data_dict]

            # Run Hector estimate_trend on all components in parallel
            results = parallel_run(data_dict, n_jobs=3)

            trend_results = np.vstack(
                [np.c_[r[0]['trend'], r[0]['trend_sigma']] for r in results])
            vel, vel_std = trend_results[:, 0], trend_results[:, 1]

            if display:
                disp1 = np.vstack([r[1]['obs'].values for r in results]).T
                ts_fit = np.vstack([r[1]['mod'].values for r in results]).T
                t1 = np.vstack([mjd2decimalyr(r[1].index) for r in results]).T
                dyears_steps = [mjd2decimalyr(om) for om in offsets_mjd]

                # plot
                plot_tsfit(t1[:, 1], disp1, ts_fit, vel,
                           vel_std, steps=dyears_steps)

        else:
            # Parametric fitting
            print(f'Parametric fit: {site}')
            out_fig_name = self.output_dir / f'{site}.png' if save_fig else None

            m, m_std = time_func.fit_function(dates, disp,
                                              poly_deg=poly_deg, periods=periods,
                                              steps=steps, display=display,
                                              fig_out_name=str(out_fig_name))

            vel, vel_std = m[1], m_std[1]

        if update:
            cols = ['e', 'n', 'v', 'se', 'sn', 'sv']
            data = np.r_[vel.flatten(), vel_std.flatten()]
            self.gnss_df.loc[self.gnss_df.site == site, cols] = data

            if start:
                self.gnss_df.loc[self.gnss_df.site ==
                                 site, 'start_date'] = start
            if end:
                self.gnss_df.loc[self.gnss_df.site == site, 'end_date'] = end

            if start or end:
                # Update duration
                start = self.gnss_df[self.gnss_df.site == site].start_date
                end = self.gnss_df[self.gnss_df.site == site].end_date
                duration = (end - start).item().days / 365.25
                self.gnss_df.loc[self.gnss_df.site ==
                                 site, 'duration'] = duration

        else:
            return vel, vel_std

    def _get_gap_percentage(self, site, sampling_period=1, start=None, end=None, extend=True):
        def str2datetime(date):
            return dt.strptime(date, '%y%b%d')
        TINY = 1.0e-6
        sampling_period = 1
        ts_df = self.read_ts_file(site)
        t = ts_df.MJD.values
        dates = ts_df.yyyymmdd.apply(str2datetime).values
        dates = dates.astype('datetime64[s]')

        # Filter time
        if start or end:
            if start:
                start_date = dt.strptime(start, "%Y%m%d")
                flag = dates > start_date

            if end:
                end_date = dt.strptime(end, "%Y%m%d")
                flag1 = dates < end_date
                if start:
                    flag *= flag1
                else:
                    flag = flag1

        t = t[flag]

        # Add today date
        if extend:
            from hectorp.calendar import compute_mjd
            if start is None:
                mjd_start = t[0]
                t = t[1:]
            else:
                start = dt.strptime(start, "%Y%m%d")
                mjd_start = compute_mjd(
                    start.year, start.month, start.day, 0, 0, 0)

            if end is None:
                today = dt.today()
                mjd_end = compute_mjd(
                    today.year, today.month, today.day, 0, 0, 0)
            else:
                end = dt.strptime(end, "%Y%m%d")
                mjd_end = compute_mjd(end.year, end.month, end.day, 0, 0, 0)
            t = np.r_[mjd_start, t, mjd_end]

        # Find percentage
        time_list = []

        for ix, mjd in enumerate(t[:-1]):
            mjd_diff = t[ix + 1] - mjd
            # Fill up gaps with NaNs depending on defined sampling
            if mjd_diff - sampling_period > TINY:
                mjds = np.arange(mjd, t[ix + 1], sampling_period)
                nans = np.ones(mjds.shape[0] - 1) * np.NaN
                t1 = np.r_[t[ix], nans]
                time_list.append(t1)
            else:
                time_list.append(mjd)

        time_list.append(t[-1])
        t_max = np.hstack(time_list)
        gap = np.sum(np.isnan(t_max))
        percentage_gap = gap / t_max.shape[0] * 100
        return percentage_gap

    def get_dV(self, site, min_data_span=3, start=None):
        df_ts = self.read_ts_file(site)

        def str2datetime(date):
            return dt.strptime(date, '%y%b%d')

        dates = df_ts.yyyymmdd.apply(str2datetime).values
        dates = dates.astype('datetime64[s]')

        disp = np.c_[df_ts.east.values.astype(np.float32),
                     df_ts.north.values.astype(np.float32),
                     df_ts.v.values.astype(np.float32)]
        
        site_steps = self.steps[self.steps.site == site]
        steps = site_steps.date.apply(str2datetime).values
        steps = steps.astype('datetime64[s]')
        # Avoid duplicates in steps
        steps = np.unique(steps)

        if start:
            start = dt.strptime(start, "%Y%m%d") 
            mask = dates > start
            mask2 = steps > start
            dates = dates[mask]
            disp = disp[mask, :]
            steps = steps[mask2]

        # Get maximum duration
        duration = (np.max(dates) - np.min(dates)) / \
                   (np.timedelta64(1, 'D') * 365.25)
        max_window = np.int16(duration - duration / 4)

        mdates_list, mvels_list = [], []
        interval = []
        for dt1 in np.arange(min_data_span, max_window):
            if dt1 == 1:
                dt1 += 0.1
            mdates, mvels, _ = calculate_temporal_variability(dates, disp, dt1, steps)
            if mdates:
                mdates_list.append(np.array(mdates))
                mvels_list.append(np.vstack(mvels))
                interval.append(np.ones(np.vstack(mvels).shape[0]) * dt1)

        if len(mvels_list) > 0:
            veast = np.vstack(mvels_list)[:, 0]
            vnorth = np.vstack(mvels_list)[:, 1]
            vup = np.vstack(mvels_list)[:, 2]

            # Get velocity for whole time series
            v, v_std, _, _ = midas(dates, disp)
            v = v[0] * 1.e3

            # Velocity Variability
            EdV = np.sqrt(np.nanmedian((veast - v[0])**2))
            NdV = np.sqrt(np.nanmedian((vnorth - v[1])**2))
            UdV = np.sqrt(np.nanmedian((vup - v[2])**2))

            return EdV, NdV, UdV, np.vstack(mvels_list), np.hstack(interval), np.hstack(mdates_list), site
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, site

    def _update_df_with_ts(self, poly_deg=0, periods=[],
                           do_midas=True, do_hector=False, start=None, end=None,
                           gap: int = 6, n_threads: int = 8, do_filtering=False):

        kwargs = dict(poly_deg=poly_deg, periods=periods, do_hector=do_hector,
                      do_midas=do_midas, start=start, end=end, update=True,
                      do_filtering=True)

        with tqdm(total=len(self.gnss_df.site.tolist())) as pbar:
            for site in self.gnss_df.site:
                pbar.set_description(f"Processing {site}")
                try:
                    self.get_ts_fit(site, **kwargs)
                except:
                    print(f'Warning, unable to run Hector, check input param!!, Skip {site}.!!')
                pbar.update(1)
