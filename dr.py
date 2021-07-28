from numpy import *
from pylab import *
import util


def pca(X, K):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})

    should return a tuple (P, Z, evals)
    
    where P is the projected data (N*K) where
    the first dimension is the higest variance,
    the second dimension is the second higest variance, etc.

    Z is the projection matrix (D*K) that projects the data into
    the low dimensional space (i.e., P = X * Z).

    and evals, a K dimensional array of eigenvalues (sorted)
    '''

    N, D = X.shape

    """
    SHOULD BE 5 LINE FUNCTIONS IF WE DO IT RIGHT, USE NUMPY EIGENVALUE FUNCTIONS!
    
    Notes on correlation from STAT400:
    Correlation(X, Y) = E[(X - mu_x)(Y - mu_y)] = E(XY) - mu_x*mu_y
    
    Notes on Covariance from STAT400:
    Covariance(X, Y) = Correlation(X, Y) / (sigma_x * sigma_y), possible values range [-1, 1], sigma is standard deviation
    Covariance measures how linearly correlated two things are :
    If Cov(x, y) = -1 or 1, then the data is linearly correlated. That means Y is a linear combination of X, or Y=aX+b
    If Cov(x, y) = 0, then the X data is independent of the Y data completely. No linear correlation
    Else, the data is some amount linearly correlated
    It's almost like the dot product of two sets of data
    """

    # make sure we don't look for too many eigs!
    if K > N:
        K = N
    if K > D:
        K = D

    # first, we need to center the data
    ### TODO: YOUR CODE HERE
    # X = X - np.mean(X, axis=0)
    X = X.transpose()

    c = average(X, axis=1)
    Xc = (X.transpose() - c.transpose()).transpose()

    # next, compute eigenvalues of the data variance hint 1: look at 'help(matplotlib.pylab.eig)' hint 2: you'll want
    # to get rid of the imaginary portion of the eigenvalues; use: real(evals), real(evecs) hint 3: be sure to sort
    # the eigen(vectors,values) by the eigenvalues: see 'argsort', and be sure to sort in the right direction!
    #             
    ### TODO: YOUR CODE HERE
    M = dot(Xc, Xc.transpose())
    M = (1. / (N - 1)) * M

    evals, Z = linalg.eig(M)

    # real values only
    evals = real(evals)
    Z = real(Z)

    # sort
    idx = evals.argsort()[-K:][::-1]
    evals = evals[idx]
    Z = Z[:, idx]

    # projection
    P = dot(Xc.transpose(), Z).transpose()

    return P, Z, evals
