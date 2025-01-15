#!/usr/bin/env python3
import numpy as np
import scipy
import scipy.sparse as spu
from skimage.filters import gaussian
from scipy.ndimage import rotate
#from .utils import fill_gaps

# Inversion funcs
coeff_poly = {
    0   : 1, # resolve for constant
    1   : 3, # linear coeff a,b,c :  ax + by + c
    1.5 : 4, # coeff a,b,c,d : ax + bx + cxy + d
    2   : 6, # quadratic coeff, a,b,c,d,e,f : ax^2 + by^2 + cxy + dx + ey + f
    3   : 8} # cubic coeff, a,b,c,d,e,f,g,h : ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h

def design_matrix_plane(x, y, c=1, poly_order=1.):

    # construct design based on polynomial degree 
    if poly_order == 0:
        # constant:  c
        A = np.array([np.ones(len(x)) * c])
    elif poly_order == 1:
        # linear: ax + by + c
        A = np.array([x, y, np.ones(len(x)) * c])
    elif poly_order == 1.5:
        # ax + bx + cxy + d
        A = np.array([x, y, x * y, np.ones(len(x)) * c])
    elif poly_order == 2: 
        # quadratic: ax^2 + by^2 + cxy + dx + ey + f
        A = np.array([x**2, y**2, x * y, x, y, np.ones(len(x)) * c])
    elif poly_order == 3: 
        # cubic: ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        A = np.array([x**3, y**3, x**2, y**2, x * y, x, y, np.ones(len(x))* c])
    
    return A.T

def calc_plane_data(lon, lat, coef, poly_order=1.):

    # construct design based on polynomial degree 
    if poly_order == 0:
        # constant: c
        plane = np.ones(lon.shape) * coef
    elif poly_order == 1:
        # linear: ax + by + c
        plane = lon * coef[0] + lat * coef[1] + coef[2]
    elif poly_order == 1.5:
        # ax + bx + cxy + d
        plane = lon * coef[0] + lat * coef[1] + \
                lon * lat *coef[2] + coef[3]
    elif poly_order == 2: 
        # quadratic: ax^2 + by^2 + cxy + dx + ey + f
        plane = lon**2 * coef[0] + lat**2 * coef[1] + \
                lon * lat *coef[2] + lon * coef[3] + \
                lat * coef[4] + coef[5]
    elif poly_order == 3: 
        # cubic: ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        plane = lon**3 * coef[0] + lat**3 * coef[1] + \
                lon**2 * lat *coef[2] + lon * lat**2 * coef[3] + \
                lon**2 * coef[4] + + lat**2 * coef[5] + coef[6]
    return plane

def calc_plane_cov(lon, lat, Qxx, poly_order=1.):
    # matrix or vector of ones
    e = np.ones(lon.shape)
    
    if poly_order == 0:
        # constant: c
        plane_var = np.ones(lon.shape) * Qxx
        
    elif poly_order == 1:
        # linear: ax + by + c
        [x_var, y_var, z_var] = np.diag(Qxx)
        #covariance
        [cxy_var, cxz_var] = Qxx[0, 1:]
        cyz_var = Qxx[1,2:]
    
        plane_var = lon**2 * x_var + lat**2 *y_var + e**2 * z_var + \
            2 * lon * (lat*cxy_var + e*cxz_var) + \
            2 * lat * (e*cyz_var)
        
    elif poly_order == 1.5:
        # ax + bx + cxy + d
        [x_var, y_var, xy_var, z_var] = np.diag(Qxx)
        #covariance
        [cxy_var, cxxy_var, cxz_var] = Qxx[0, 1:]
        [cyxy_var, cyz_var] = Qxx[1,2:]
        cxyz_var = Qxx[2,3:]
        
        plane_var = lon**2 * x_var + lat**2 * y_var + (lon*lat)**2 * xy_var +  e**2 * z_var + \
            2 * lon* (lat*cxy_var + lon*lat*cxxy_var + e*cxz_var) + \
            2 * lat * (lon*lat*cyxy_var + e*cyz_var) + \
            2 * lon*lat * (e*cxyz_var)
            
    elif poly_order == 2: 
        # quadratic: ax^2 + by^2 + cxy + dx + ey + f
        [xx_var, yy_var, xy_var, x_var, y_var, z_var] = np.diag(Qxx)
        #covariance
        [cxxyy_var, cxxxy_var, cxxx_var, cxxy_var, cxxz_var] = Qxx[0, 1:]
        [cyyxy_var, cyyx_var, cyyy_var, cyyz_var] = Qxx[1,2:]
        [cxyx_var, cxyy_var, cxyz_var] = Qxx[2,3:]
        [cxy_var, cxz_var] = Qxx[3,4:]
        cyz_var = Qxx[4,5:]
        
        plane_var = lon**4 * xx_var + lat**4 * yy_var + (lon*lat)**2 * xy_var + lon**2 * x_var + lat**2 * y_var + e**2 * z_var + \
            2 * lon**2 * (lat**2*cxxyy_var + lon*lat*cxxxy_var + lon*cxxx_var +  lat*cxxy_var + e*cxxz_var) + \
            2 * lat**2 * (lat*lon*cyyxy_var + lon*cyyx_var + lat*cyyy_var + e*cyyz_var) + \
            2 * lon*lat * (lon* cxyx_var + lat*cxyy_var + e*cxyz_var) + \
            2 * lon * (lat * cxy_var + e*cxz_var) + \
            2 * lat * (e*cyz_var)
            
    elif poly_order == 3: 
        # cubic: ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        [xxx_var, yyy_var, xxy_var, yyx_var, xx_var, yy_var, xy_var, z_var] = np.diag(Qxx)

        #covariance
        [cxxxyyy_var, cxxxxxy_var, cxxxyyx_var, cxxxxx_var, cxxxyy_var, cxxxxy_var, cxxxz_var] = Qxx[0, 1:]
        [cyyyxxy_var, cyyyyyx_var, cyyyxx_var, cyyyyy_var, cyyyxy, cyyyz_var] = Qxx[1,2:]
        [cxxyyyx_var, cxxyxx_var, cxxyyy_var, cxxyxy_var, cxxyz_var] = Qxx[2,3:]
        [cyyxxx_var, cyyxyy_var, cyyxxy_var, cyyxz_var] = Qxx[3,4:]
        [cxxyy_var, cxxxy_var, cxxz_var] = Qxx[4,5:]
        [cyyxy_var, cyyz_var] = Qxx[5,6:]
        cxyz_var = Qxx[6,7:]

        plane_var = lon**5 * xxx_var + lat**5 * yyy_var + (lon**2 * lat)**2 * xxy_var + (lon*lat**2)**2 * yyx_var + \
                    lon**4 * xx_var + lat**4*yy_var + (lon*lat)**2 * xy_var + e**2 * z_var + \
            2 * lon**3 * (lat**3*cxxxyyy_var + lon**2*lat*cxxxxxy_var + lat*2*lon*cxxxyyx_var +  lon*2*cxxxxx_var + \
                          lat*2*cxxxyy_var + lon*lat*cxxxxy_var + e*cxxxz_var) + \
            2 * lat**3 * (lon**2*lat*cyyyxxy_var + lat**2*lon*cyyyyyx_var + lon**2*cyyyxx_var + lat**2*cyyyyy_var + \
                          lon*lat*cyyyxy + e*cyyyz_var) + \
            2 * lon**2*lat * (lat**2*lon*cxxyyyx_var + lon**2*cxxyxx_var + lat**2*cxxyyy_var + lon*lat*cxxyxy_var + e*cxxyz_var) + \
            2 * lat**2*lon * (lon**2 * cyyxxx_var + lat**2*cyyxyy_var + lon*lat*cyyxxy_var + e*cyyxz_var) + \
            2 * lon**2 * (lat**2*cxxyy_var + lon*lat*cxxxy_var + e*cxxz_var) + \
            2 * lat**2 * (lon*lat*cyyxy_var + e*cyyz_var) + \
            2 * lon*lat * (e*cxyz_var)
         
    plane_std = np.sqrt(plane_var)
    
    return plane_var, plane_std

def lscov(A, b, weights=None):
    if weights is None:
        # construct weight matrix from observation standard deviation
        # w = np.sqrt(1./(std**2))
        weights = np.ones(b.shape)

    b = b[:, np.newaxis] #to turn it to vector [n_obs x 1]
    [n_obs, n_x] = A.shape
    n_r = n_obs - n_x # redundant obs
    
    # Check if design matrix is sparse
    sparse_flag = spu.issparse(A)

    # Replicate the workflow from matlab lscov
    if sparse_flag:
        #need to figure out how to deal with sparse matrixes in this workflow
        # https://github.com/yig/PySPQR
        A = A.toarray()  

    # Weights given, scale rows of design matrix and response.
    Aw = A * np.sqrt(weights[:, np.newaxis])
    Bw = b * np.sqrt(weights[:, np.newaxis])
    
    # Factor the design matrix, incorporate covariances or weights into the
    # system of equations, and transform the response vector.
    [Q, R, perm] = scipy.linalg.qr(Aw, mode='economic', pivoting=True)
    
    # this line of code qr gives me with some matrices different result than matlab qr
    # [Q,R] = np.linalg.qr(Aw)
    z = np.dot(Q.T, Bw)
    
    # Use the rank-revealing QR to remove dependent columns of A.
    r_diag = np.diag(R)
    keepCols = abs(r_diag) > abs(r_diag[0]) * max(n_obs, n_x)*np.finfo(R.dtype).eps
    rankA = np.sum(keepCols)
    if rankA < n_x:
        print('Warning: A is rank deficient to within machine precision')
        R = R[np.ix_(keepCols, keepCols)]
        z = z[keepCols,:]
        perm = perm[keepCols]
    
    # Compute the LS coefficients
    xx = np.linalg.lstsq(R, z, rcond=None)[0]
    # try to go around the pivoting
    x = np.zeros((n_x, 1))
    x[perm] = xx

    # Compute the MSE, need it for the std. errs. and covs.
    if rankA < n_x:
        Q = Q[:, keepCols]

    # weighted residuals
    wres = Bw - Q.dot(z)
    if n_r > 0:
        mse = np.sum(wres * np.conj(wres)) / n_r
    else:
        mse = np.zeros(1, np.int32(b.shape[1]))
        
    # Compute the covariance matrix of the LS estimates
    Rinv = np.triu(np.linalg.lstsq(R, np.eye(np.linalg.matrix_rank(Aw)), rcond=None)[0])
    # S matrix
    Qxx = np.zeros((n_x, n_x))
    Qxx[np.ix_(perm, perm)] = np.dot(Rinv, Rinv.T) #* mse
    # std
    stdx = np.sqrt(mse * np.diag(Qxx))
    
    # residuals: scale backwards to get unweighted residuals
    res = wres / np.sqrt(weights[:, np.newaxis])
    
    # predicted obs
    obs_hat = Q.dot(z)
    obs_hat /= np.sqrt(weights[:, np.newaxis])

    #Add a new line
    #Qxx = Qxx  / mse
    
    ### A posteriori cofactor matrix of estimates
    if n_r > 0:
        Q = A.dot(Qxx / mse).dot(A.T)
    else:
        Q = np.zeros(Qxx.shape)
    
    return x, stdx, mse, Qxx, res, wres, obs_hat, Q

def fit_plane(data: np.ndarray,
              data_std:np.ndarray,
              lons: np.ndarray,
              lats:np.ndarray,
              order:float=1, 
              decimate:int=1, 
              interpolate:bool=False,
              filter:bool=False,
              filter_size:int=5,
              dist_weights:np.ndarray | None = None):
    
    # Decimation factor
    dec_ix = np.s_[::decimate, ::decimate]

    # Get nans
    index = np.isnan(data[dec_ix].ravel())

    if interpolate:
        data = fill_gaps(np.ma.masked_invalid(data).filled(fill_value=0))

    if filter:
        data = gaussian(data, (filter_size, filter_size))

    # Get design matrix
    A = design_matrix_plane(lons[dec_ix].ravel(),
                            lats[dec_ix].ravel(),
                            poly_order=order)

    # Remove nans
    A = np.delete(A, index, axis=0)
    b = np.delete(data[dec_ix].ravel(), index)
    w = np.delete(data_std[dec_ix].ravel(), index) 
    w = 1./w**2

    # Distance weigthing kernel
    if dist_weights is not None:
        #print('Use distance weighting')
        dist_weights = np.delete(dist_weights[dec_ix].ravel(), index)
        w *= dist_weights

    # Invert
    x, _, _, Qxx, res, _, _, _ = lscov(A,b,w)
    
    # Remove outliers using 2 sigma rule
    #below_2sigma = res < res.mean() - 2 * res.std()
    #above_2sigma = res > res.mean() + 2 * res.std()
    #outliers = np.where(below_2sigma | above_2sigma)

    # Remove data and repeat inversion to get refined planar coefficients
    #A = np.delete(A, outliers[0], axis=0)
    #b = np.delete(b, outliers[0])
    #w = np.delete(w, outliers[0])

    # Invert
    #x, _, _, Qxx, res, _, _, _ = lscov(A, b, w) 
    
    # Get plane and plane_std
    plane = calc_plane_data(lons, lats, x, order)
    plane_std = calc_plane_cov(lons, lats, Qxx, order)[1]
    combined_std = np.sqrt(plane_std**2 + data_std**2)
    return plane, combined_std

# FFT Filter
def hamming2d_filter(array: np.ndarray,
                     kernel_x:int, kernel_y:int,
                     cut_off:float=1., angle:float=0.):
    
    # Create filtering kernal
    kernal = np.zeros(array.shape, dtype=np.float32)
    kernal_center = np.array(kernal.shape) // 2

    # Hamming filter
    ham_x = np.hamming(kernel_x)[:, None]  # 1D Hamming window for x direction
    ham_y = np.hamming(kernel_y)[None, :]  # 1D Hamming window for y direction
    ham2d = np.sqrt(np.dot(ham_x, ham_y)).T ** cut_off
    ham2d = np.flipud(rotate(ham2d, angle=angle, reshape=False, prefilter=False))
    ham2d /= np.max(ham2d)

    half_win_x = kernel_x // 2
    half_win_y = kernel_y // 2

    kernal_mask = np.s_[kernal_center[0] - half_win_y : kernal_center[0] + half_win_y,
                        kernal_center[1] - half_win_x : kernal_center[1] + half_win_x]
    kernal[kernal_mask] = ham2d 

    # Do convolution with fft
    array_fft = np.fft.fftshift(np.fft.fft2(array))
    filt_array = np.fft.ifft2(np.fft.ifftshift(array_fft * kernal))
    return filt_array.real, ham2d  


def get_distance_kernel(window, pad, penalize_pad=True):
    # Kernel size for extended window
    kernel = np.zeros((int(window[0].stop - window[0].start),
                       int(window[1].stop - window[1].start)))
    # Original window
    central_win = np.int16([int(pad[0].stop - pad[0].start) /2,
                            int(pad[1].stop - pad[1].start) /2])
    # Get centroid
    kernel[pad] = 1
    kernel[pad][central_win[0]-1: central_win[0]+1,
                central_win[0]-1: central_win[0]+1] = 2

    # Find distance
    cent = np.vstack(np.where(kernel==2))
    orig_win = np.vstack(np.where(kernel==1))
    ext_win = np.vstack(np.where(kernel==0))

    dist = np.zeros_like(kernel)
    # Original window
    for l in np.arange(orig_win.shape[1]):
        point = np.atleast_2d(orig_win[:,l]).T
        dist[orig_win[0,l], orig_win[1,l]] = np.mean(np.linalg.norm(cent - point, axis=0))

    # Extended window
    for l in np.arange(ext_win.shape[1]):
        point = np.atleast_2d(ext_win[:,l]).T
        dist[ext_win[0,l], ext_win[1,l]] = np.mean(np.linalg.norm(cent - point, axis=0))

    # The Epanechnikov kernel 
    kernel = np.zeros(dist.shape)
    xx_norm = dist / np.max(dist)
    idx = np.where(xx_norm <= 1)
    kernel[idx] = 0.75 * (1 - xx_norm[idx]  ** 2)
    kernel[kernel==0] = 0.001
    # Use orginal weights inside the window, penalize outside
    if penalize_pad:
        kernel[orig_win[0,:], orig_win[1,:]] = 1

    return kernel
 