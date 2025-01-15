
import scipy
import numpy as np
import scipy.sparse as spu


def loop_invert_asc_dsc2(data, std, inc, azi, ns=None, ew=None, box=None):
    if box==None:
        box = np.s_[:,:,:]
        
    dim, length, width = data.shape
    #initialize parameters
    disp = np.empty((3, length, width), dtype = np.float64) * np.nan
    stdx = np.empty((3, length, width), dtype = np.float64) * np.nan
    mse = np.empty((length, width), dtype=np.float64) * np.nan
    n = np.empty((length, width), dtype=np.float64) * np.nan
    
    for iy, ix in np.ndindex(data.shape[1:]):
        if ns is not None:
            ns_subset = ns[:, iy, ix]
        if np.isnan(ns_subset).any():
            ns_subset = [0., 1.]

        if ew is not None:
            ew_subset = ew[:, iy, ix]
        if np.isnan(ew_subset).any():
            ew_subset = [0., 1.]


        # Ensure there is req. data for inversion
        if ~np.isnan(data[:, iy, ix]).size == 0:
            continue
        if ~np.isnan(std[:, iy, ix]).size == 0:
            continue
        if ~np.isnan(azi[:, iy, ix]).size == 0:
            continue
        if ~np.isnan(inc[:, iy, ix]).size == 0:
            continue

        x, stdxi, msei, n_geom = invert_asc_dsc(data[:, iy, ix],
                                                std[:, iy, ix], 
                                                inc[:, iy, ix], 
                                                azi[:, iy, ix],
                                                ns = ns_subset,
                                                ew = ew_subset)
        

        disp[:, iy, ix] = np.atleast_1d(x).T
        stdx[:, iy, ix] = np.atleast_1d(stdxi)
        mse[iy, ix] = msei
        n[iy, ix] = n_geom

    return disp, stdx, mse, n, box

def disp_unit_vector(incidenceAngle:np.float64, azimuthAngle:np.float64):
    '''
    azimuthAngle  - 0 at east, as it is in ISCE2 convention 

    '''
    dn = np.multiply(np.sin(np.deg2rad(incidenceAngle)), np.cos(np.deg2rad(azimuthAngle)))
    de = np.multiply(np.sin(np.deg2rad(incidenceAngle)), -np.sin(np.deg2rad(azimuthAngle)))
    dv = np.cos(np.deg2rad(incidenceAngle))
    
    return dn, de, dv

def invert_asc_dsc(data: np.array, std: np.array,
                    inc: np.array, azi: np.array,
                    ns: np.array=[0., 1.],
                    ew: np.array=[0.,1.]):
    
    # Get number of x unknowns
    n_x = 3 # North-South, East-West, Up-Down

    #Initialize parameters
    container = np.c_[data, std, inc, azi]

    x = np.empty(n_x) * np.nan
    stdx = np.empty(n_x) * np.nan
    mse = np.nan
    n_geom = 0

    # Index bool for disp. components
    rm_index = np.array([False, False, False])

    if np.isnan(ns[0]): ns[0] = 0
    if np.isnan(ns[1]): ns[1] = 1
    if np.isnan(ew[0]): ew[0] = 0
    if np.isnan(ew[1]): ew[1] = 1

    # if there is no valid data, return nans
    if np.sum(~np.isnan(data)) > 0:
        # remove nan
        index = np.isnan(container).any(axis=1)
        container = np.delete(container, index, axis=0)

        #Create InSAR displacement unit vector
        dn, de, dv = disp_unit_vector(container[:, 2],
                                      container[:, 3])
        
        # Both viewing geom. exist, add only external NS constrain
        if np.sum(de < 0) != 0 and np.sum(de > 0) != 0:
            A_ext = np.atleast_2d([1, 0, 0])
            b_ext = np.atleast_2d(ns[0])
            w_ext = 1 / np.atleast_2d(ns[1])**2
            n_geom = 2

            if ns[0] == 0 and ew[0] == 0:
                # Estimate EW and Vertical, assume neglible NS
                rm_index[0] = True 

        # If there is only one viewing geom, add both NS and EW
        else:
            A_ext = np.atleast_2d(np.c_[[[1, 0, 0], [0, 1, 0]]])
            b_ext = np.atleast_2d(np.c_[ns[0], ew[0]])
            w_ext = np.atleast_2d(1 / np.c_[ns[1], ew[1]]**2)
            n_geom = 1

            if ns[0]==0 and ew[0] != 0:
                # if only EW exist, invert for EW and up
                rm_index[0] = True

            elif ns[0]!=0 and ew[0] == 0:
                # if only NS exist, invert for NS and up
                rm_index[1] = True
                
            elif ns[0]==0 and ew[0] == 0:
                # NOTE: project only to vertical
                # if ther is horz. motion, vert. will be overestimated
                rm_index[:2] = [True, True]

                
        # Create Design matrix
        A = np.c_[np.atleast_2d(dn).T,
                  np.atleast_2d(de).T,
                  np.atleast_2d(dv).T]

        # Observation vector
        b = container[:,0]
        # Weight vector
        w = 1 / (container[:,1]**2)

        # Append external
        A = np.r_[A, A_ext]
        b = np.append(b, b_ext)
        w = np.append(w, w_ext)

        # Remove b rows with 0
        ix = np.where(b ==0)[0]
        if ix.size > 0:
            A = np.delete(A, ix, axis=0)
            b = np.delete(b, ix, axis=0)
            w = np.delete(w, ix, axis=0)

        # Estimate only selected components
        A = np.delete(A, rm_index, axis=1)

        #invert
        try:
            x0, stdx0, mse, Qxx = lscov(A, b, weights=w)[:4]
            mse = np.round(mse, 4)
        except:
            raise ValueError(f'Could not invert, check A:{A}, b:{b}, w:{w}')

        #in the case when we have only one ascending and one descending
        if mse == 0:
            stdx0 = np.sqrt(np.diag(Qxx) * A.shape[0]) # use arbitrary scale n_obs
            #maybe would be good to check this on the collocated gps stations
        
        rm_index = np.array(rm_index)
        x[~rm_index] = np.squeeze(x0)
        stdx[~rm_index] = np.squeeze(stdx0)
        
    return x, stdx, mse, n_geom

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
        
    # maybe need to normalize weight so that they do not blow up to large number 
    # when using meters for std instead of mm

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
    wres = Bw - Q.dot(z)
    if n_r > 0:
        mse = np.sum(wres * np.conj(wres)) / n_r
    else:
        mse = np.zeros(1, np.int32(b.shape[1]))
        
    #Compute the covariance matrix of the LS estimates
    Rinv = np.triu(np.linalg.lstsq(R, np.eye(np.linalg.matrix_rank(Aw)), rcond=None)[0])
    # S matrix
    Qxx = np.zeros((n_x, n_x))
    Qxx[np.ix_(perm, perm)] = np.dot(Rinv, Rinv.T) #* mse
    # std
    stdx = np.sqrt(mse * np.diag(Qxx))
    
    #residuals: scale backwards to get unweighted residuals
    res = wres / np.sqrt(weights[:, np.newaxis])
    
    # predicted obs
    obs_hat = Q.dot(z)
    obs_hat /= np.sqrt(weights[:, np.newaxis])
    
    ### Aposteriori cofactor matrix of estimates
    if n_r > 0:
        Q = A.dot(Qxx / mse).dot(A.T)
    else:
        Q = np.zeros(Qxx.shape)
    
    return x, stdx, mse, Qxx, res, wres, obs_hat, Q