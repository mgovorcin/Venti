#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:56:38 2022

@author: govorcin

THIS MODULE NEEDS MORE CLEAN UP

Interesting ref:
https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals


how to use numpy

#matrix
N = np.linalg.multi_dot([A.T, W, A])
n = np.linalg.multi_dot([A.T, W, b])

# x = scipy.linalg.lstsq(N,n)[0]
x = np.linalg.lstsq(N, n, rcond=None)[0]

#Qxx = np.linalg.inv(N) # Option 1 to do matrix inverse
#Qxx = spu.linalg.spsolve(N, spu.eye(N.shape[0])) #other option to get sparse matrix inverse
Qxx = np.linalg.solve(N, np.eye(N.shape[0]))

e = (A.dot(x) - b)
eTwe =  np.linalg.multi_dot([e.T, W, e])
s0 = np.sqrt(eTwe / n_r)

stdx = s0 * np.sqrt(np.diag(Qxx))


"""

import numpy as np
import scipy.linalg
import scipy.sparse as spu

# LSQ output object
class lsq_model():
    def __init__(self):
        self.x = []
        self.stdx = []
        self.mse = []
        self.residuals = []
        self.Qxx = []
        self.Q = []
        self.observations_hat = []


def lscov(A: np.ndarray, b: np.ndarray, weights: np.ndarray = None) -> lsq_model:
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
    # Number of observations and unkowns
    [n_obs, n_x] = A.shape
    n_r = n_obs - n_x # redundant obs

    ##### Prepare input
    if spu.issparse(A):
        # Sparse matrixes not supported yet, missing QR implementation:
        # https://github.com/yig/PySPQR
        A = A.toarray()

    # if obs bector has ndim=1 turn it to vector with 2 dim [n_obs x 1]
    if b.ndim == 1:
        b = np.atleast_2d(b).T

    # maybe need to normalize weight so that thy do not blow up to large number
    # when using meters for std instead of mm
    if weights is None:
	    weights = np.ones(b.shape)

    # Weights needs to be a vector [n_obs x 1]
    elif weights.ndim == 1:
        weights = np.atleast_2d(weights).T

    ##### Weights given, scale rows of design matrix and response.
    Aw = A * np.sqrt(weights)
    Bw = b * np.sqrt(weights)

    # Factor the design matrix, incorporate covariances or weights into the
    # system of equations, and transform the response vector.
    [Q, R, perm] = scipy.linalg.qr(Aw, mode='economic', pivoting=True)

    z = np.dot(Q.T, Bw)

    # Use the rank-revealing QR to remove dependent columns of A.
    r_diag = np.diag(R)
    keepCols = abs(r_diag) > abs(r_diag[0]) * max(n_obs, n_x)*np.finfo(R.dtype).eps
    rankA = np.sum(keepCols)
    if rankA < n_x:
        print('Warning: A is rank deficient to within machine precision')
        '''
        When rank is deficient, lscov gives me different solution than matlab
        CHECK IT TO, solution is not stable
        https://www.heinrichhartmann.com/posts/2021-03-08-rank-decomposition/
        '''
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
    res = wres / np.sqrt(weights)

    # predicted obs
    obs_hat = Q.dot(z)
    obs_hat /= np.sqrt(weights)

    ### Aposteriori cofactor matrix of estimates
    if n_r > 0:
        Qhat = A.dot(Qxx / mse).dot(A.T)
    else:
        Qhat=np.zeros(Qxx.shape)

    # Inversion output
    out = lsq_model()
    out.x = x # vector of unknowns
    out.stdx = stdx # standad deviation of unknowns
    out.mse = mse # mean square error
    out.residuals = res # Resdiuals
    out.Qxx = Qxx # covariance matrix of unknowns
    out.Q = Qhat # Cofactor matrix of adjustement
    out.obs_hat = obs_hat # predicted values

    return out

def analitic(A: np.ndarray, b: np.ndarray, weights: np.ndarray = None):

    # Number of observations and unkowns
    [n_obs, n_x] = A.shape
    n_r = n_obs - n_x # redundant obs

    ##### Prepare input
    if spu.issparse(A):
        # Need to figure out how to deal with sparse matrixes in this workflow
        A = A.toarray()

    # if obs bector has ndim=1 turn it to vector with 2 dim [n_obs x 1]
    if b.ndim == 1:
        b = np.atleast_2d(b).T

    # maybe need to normalize weight so that thy do not blow up to large number
    # when using meters for std instead of mm
    if weights is None:
	    weights = np.ones(b.shape)

    # Weights needs to be a vector [n_obs]
    elif weights.ndim > 1:
        weights = np.squeeze(weights)

    W = np.diag(weights)

    #s = A.dot(np.ones(n_x)) - l
    #normal equations Nx - n = 0
    N = np.linalg.multi_dot([A.T, W, A])
    n = np.linalg.multi_dot([A.T, W, b])

    if np.linalg.matrix_rank(A) < A.shape[1]:
        print('Warning matrix A is singular. Inversion result can be biased!')
        '''
         See how to deal with singulat matrixes or matrix rank deficiency
         https://github.com/numpy/numpy/issues/2074

        '''

    try:
        #cholesky
        # x = N^-1 * n = Qxx * n = (c.T * c)^-1 * n
        C = np.linalg.inv(np.linalg.cholesky(N))

        # Matrix covariance unknowns
        Qxx = np.dot(C.T, C) # or np.linalg.inv(N), cholesky tend to be faster
        x = np.dot(Qxx, n)

    except:
        #u,s,v = np.linalg.svd(N)
        # Qxx = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        Qxx = np.linalg.inv(N)
        #x = np.linalg.lstsq(N, n, rcond=None)[0]
        x = np.dot(Qxx, n)

    #residuals
    e = b - A.dot(x)

    #control of errors A.T * W * e = 0
    ce = np.around(np.linalg.multi_dot([A.T, W, e]), decimals=5)

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



