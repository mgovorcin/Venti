#!/usr/bin/env python3

from pyproj import Geod, Transformer
import numpy as np

# Original code: https://github.com/JMNocquet/pyacs36/tree/master


# Conversion matrix R
def get_conversion_matrix(lon, lat):
    # lon, lat in degrees
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    R = np.zeros((3,3), np.float64)
    R[0, 0] = -np.sin(lon_rad)
    R[0, 1] =  np.cos(lon_rad)
    R[1, 0] = -np.sin(lat_rad) * np.cos(lon_rad)
    R[1, 1] = -np.sin(lat_rad) * np.sin(lon_rad)
    R[1, 2] =  np.cos(lat_rad)
    R[2, 0] =  np.cos(lat_rad) * np.cos(lon_rad)
    R[2, 1] =  np.cos(lat_rad) * np.sin(lon_rad)
    R[2, 2] =  np.sin(lat_rad)

    return R

### 
lonlat2xyz_transformer = Transformer.from_crs(
                            "EPSG:4326",
                            {"proj":'geocent', "ellps":'GRS80', "datum":'WGS84'}, 
                            always_xy=True,)


def get_local_frame(lon, lat, height=0):
    # to geocentric cartesian coordinates XYZ (ellipsoid GRS90)
    # height : height above the ellipsoid
    (x, y ,z) = lonlat2xyz_transformer.transform(lon, 
                                                 lat,
                                                 height,
                                                 radians=False)
    # Conversion matrix XYZ to ENU
    R = get_conversion_matrix(lon, lat)

    # Observation equation in local frame
    Ai = np.zeros([3, 3], float)
    Ai[0, 1] =  z
    Ai[0, 2] = -y
    Ai[1, 0] = -z
    Ai[1, 2] =  x
    Ai[2, 0] =  y
    Ai[2, 1] = -x

    Ai = Ai / 1000.0

    RAi = np.dot(R, Ai)
    return RAi

def rotation_rate2euler_pole(wx, wy, wz):
    # Rotation rate vector wx, wy, wz to euler pol (lon, lat , omeag [deg/Myr])
    W = np.sqrt(np.sum(np.array([wx, wy, wz])**2))

    omega = np.rad2deg(W)*1.e6 # anular velocity in decimal degrees per Myr.

    euler_lat = 90.0 - np.rad2deg(np.arccos(wz / W)) # relative to sphere
    euler_lon = np.rad2deg(np.arctan2(wy, wx))

    return euler_lon, euler_lat, omega


# Get it opposite
def euler_pole2rotation_rate(euler_lon, euler_lat, omega):
    omega_rad = np.radians(omega) * 1e-6 
    cos_lon, sin_lon = [np.cos(np.deg2rad(euler_lon)), np.sin(np.deg2rad(euler_lon))]
    cos_lat, sin_lat = [np.cos(np.deg2rad(euler_lat)), np.sin(np.deg2rad(euler_lat))]
    
    wx = cos_lat * cos_lon * omega_rad
    wy = cos_lat * sin_lon * omega_rad
    wz = sin_lat * omega_rad

    return wx, wy, wz

def calc_euler_pole(lon, lat, ve, vn, se, sn, hgt=None):
    nsites = lon.shape[0]

    # Initialize matrices
    A = np.zeros([2 * nsites, 3], float)
    b = np.zeros([2 * nsites, 1], float)
    vcv_b = np.zeros([2 * nsites, 2 * nsites], float)

    if hgt is None:
        hgt = np.zeros(lon.shape)

    # Get conversion matrix
    itter_zip = zip(lon, lat, hgt, ve, vn, se, sn)

    for ix, (ilon, ilat, ihgt, ive, ivn, ise, isn) in enumerate(itter_zip):
        sen = np.sqrt(ise**2 + isn**2)
        Rx = get_local_frame(ilon, ilat, ihgt)
        cov = ise * isn * sen

        A[2 * ix, :] = Rx[0,:]
        A[2 * ix + 1, :] = Rx[1,:]

        # Matrix obs
        b[2 * ix, 0] = ive
        b[2 * ix + 1, 0] = ivn

        vcv_b[2 * ix, 2 * ix] = ise**2
        vcv_b[2 * ix + 1, 2 * ix + 1] = isn**2
        vcv_b[2 * ix + 1, 2 * ix] = cov
        vcv_b[2 * ix, 2 * ix + 1] = cov

    # Solve linear system
    P = np.linalg.linalg.inv(vcv_b)
    ATP = np.dot((A.T), P)

    N = np.dot(ATP, A)
    M = np.dot(ATP, b)

    # Inversion
    Q = np.linalg.linalg.inv(N)
    X = np.dot(Q, M)

    # Model Prediction
    MP = np.dot(A, X)

    # Residuals
    RVen = (b - MP)

    vcv_pole_x = Q * 1e-12
    chi2 = np.dot(np.dot(RVen.T, P), RVen)
    dof = 2 * nsites - 3
    reduced_chi2 = np.sqrt(chi2 / float(dof))

    # Rotation vectors
    wx = X[0, 0] * 1e-6
    wy = X[1, 0] * 1e-6
    wz = X[2, 0] * 1e-6
    
    # Get euler pole 
    euler_lon, euler_lat, omega = rotation_rate2euler_pole(wx, wy, wz)

    #rms
    re = np.hstack(RVen[::2])
    rn = np.hstack(RVen[1::2])
    rms = np.sqrt(np.sum(re**2 + rn**2) / 2 / re.shape[0])

    #wrms
    wrms =  np.sum((re / se)**2 + (rn / sn)**2)
    wrms /= np.sum(1/se**2 + 1/sn**2)

    stats = {}
    stats['rms'] = rms
    stats['wrms'] = np.sqrt(wrms)
    stats['chi2'] = chi2
    stats['reduced_chi2'] = reduced_chi2
    stats['dof'] = dof
    stats['covariance'] = vcv_pole_x

    return euler_lon, euler_lat, omega, stats

def get_euler_pole_uncertainty(euler_lon, euler_lat, omega, euler_cov):
    # Get euler pole uncertainty
    Rp = get_conversion_matrix(euler_lon, euler_lat)
    (wx, wy, wz) = euler_pole2rotation_rate(euler_lon, euler_lat, omega)
    nw = np.sqrt(np.sum(np.array([wx, wy, wz])**2))
    std_pole_enu = np.dot(np.dot(Rp, euler_cov), Rp.T)

    sp11 = std_pole_enu[0, 0]
    sp12 = std_pole_enu[0, 1]
    sp22 = std_pole_enu[1, 1]

    sigma_element = np.sqrt((sp11 - sp22)**2 + 4 * sp12**2) 
    max_sigma = 0.5 * (sp11 + sp22 + sigma_element)
    max_sigma = np.rad2deg(np.arctan(np.sqrt(max_sigma) / nw))

    min_sigma = 0.5 * (sp11 - sp22 + sigma_element)
    min_sigma = np.rad2deg(np.arctan(np.sqrt(min_sigma) / nw))

    azimuth = np.rad2deg(2*np.arctan(2*sp12 / (sp11 - sp22)))
    s_omega = np.rad2deg(np.sqrt(std_pole_enu[2, 2]))*1e6

    return max_sigma, min_sigma, azimuth, s_omega








