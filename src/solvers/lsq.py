#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:56:38 2022

@author: govorcin
"""

import numpy as np
import scipy.linalg
import scipy.sparse as spu

def lscov(A: np.narray, b: np.narray, weights: np.narray = None):
	'''
    Ax = b, where Ax - b = 0
    A  : array / sparse.coo.coo_matrix  : Design matrix, dimensions [n,x] where n is number of obs, and x number of unknowns
    b  : array                          : Response vector, (obs), [n] where n is number of obs, only one axis, below b gets converted to vector
    weights : array                     : Response weights w = np.sqrt(1. / (obs_std ** 2)), where weights is vector [n] with 1 axis, 
                                          use np.squeeze if b or weights is in format [nx1]
    
    n_r = n - m  :: redundatant measurements, n - number of observations, m - number of unknowns
    e = Ax - b   ::  equation of correction, matrix dim, 
                     e[nx1] - errors
                     A[nxm] - design matrix
                     x[mx1] - unknowns coeff
                     b[nx1] - observations
    Nx - n  = o  :: normal equations, where:
                     N = A.T * W * A
                    -n = A.T * W * y
    x = N^-1 * n
    Control:
        Ne - n = A.T * W *s
        A.T * W * e = 0 - sum of errors need to be zero
    '''

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
        
    # maybe need to normalize weight so that thy do not blow up to large number 
    # when using meters for std instead of mm

    # Weights given, scale rows of design matrix and response.
    Aw = A * np.sqrt(weights[:, np.newaxis])
    Bw = b * np.sqrt(weights[:, np.newaxis])
    
    #Factor the design matrix, incorporate covariances or weights into the
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
    #Add a new line
    #Qxx = Qxx  / mse
    
    ### Aposteriori cofactor matrix of estimates
    if n_r > 0:
        Q = A.dot(Qxx / mse).dot(A.T)
    else:
        Q=np.zeros(Qxx.shape)
    
    return x, stdx, mse, Qxx, res, wres, obs_hat, Q

 def analitic_lsq(A: np.narray, b: np.narray, weights: np.narray = None):

 	if weights is None:
	    # construct weight matrix from observation standard deviation
	    # w = np.sqrt(1./(std**2))
	    weights = np.ones(b.shape)
    
    b = b[:, np.newaxis] #to turn it to vector [n_obs x 1]
    [n_obs, n_x] = A.shape
    n_r = n_obs - n_x # redundant obs
    
    # Check if design matrix is sparse
    sparse_flag = spu.issparse(A)

	W = np.diag(weights)
    if sparse_flag:
        #need to figure out how to deal with sparse matrixes in this workflow
        A = A.toarray() 
    
    #s = A.dot(np.ones(n_x)) - l
    #normal equations Nx - n = 0
    N = np.linalg.multi_dot([A.T, W, A])
    n = np.linalg.multi_dot([A.T, W, b])
    
    #cholesky
    # x = N^-1 * n = Qxx * n = (c.T * c)^-1 * n
    C = np.linalg.inv(np.linalg.cholesky(N))
    # Matrix covariance unknowns
    Qxx = np.dot(C.T, C) # or np.linalg.inv(N), cholesky tend to be faster
    
    #unkowns
    x = np.dot(Qxx, n)
    #x = np.linalg.lstsq(N, n, rcond=None)[0]
    
    #residuals
    e = b - A.dot(x)
    
    #control of errors A.T * W * e = 0
    ce = np.around(np.linalg.multi_dot([A.T, W, e]), decimals=tolerance_dim)
    
    #control of adjustment e.T * W * e = -l.T * W * e
    eTwe =  np.linalg.multi_dot([e.T, W, e])
    bTwe =  -np.linalg.multi_dot([b.T, W, e])
    ca = eTwe - bTwe
    
    #new obs l + e = l_hat
    b_hat = b + e
    
    #accuarcy: std  s0 = np.sqrt([e.T * W * e]  / n_r )
    s0 = np.sqrt(eTwe / n_r)
    
    # accuarcy of observations s0 * np.sqrt(np.diag(W))
    Q = 1. / np.diag(W)
    stdl = s0 * np.sqrt(Q)
    
    # accuarcy of unknows
    stdx =  s0 * np.sqrt(np.diag(Qxx))
    
    '''
    #control of unknown cofactor matrix Ne.T * Qx * e = m
    Ne = N.dot(np.ones(N.shape[1]))
    cqx = np.sum(np.dot(Ne.T,Qx).dot(np.ones(Qx.shape[1])))
    
    # Cofactor matrix of adjustement
    # Q = A * Qx * A.T
    Q = (np.dot(A,Qx)).dot(A.T)
    saa = s0 * np.sqrt(np.diag(Q))                                                
    #Control of cofactor matrix of adjustement tr(PQ) = n_x
    cq = np.trace(np.dot(W,Q))
    '''

    return x, stdx, s0, Qxx, e



