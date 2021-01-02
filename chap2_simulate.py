# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 20:37:59 2021

@author: Nam Kieu
Move all the functions from the Ipython notebook into this file
so that we can use Multi-processing to make the simulation process faster
"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from sklearn.neighbors import KernelDensity


def mpPDF(var, q, pts=100):
    # var: variance of the process
    # q = number of observations / number of factors
    # q = T/N according to Marco's notation
    # pts = number of points to create the pdf
    # output: pdf of the eigen values according to Marcenko-Pastur
    tmp = (1./q)**.5
    
    # eigen value between eMin and eMax are consistent with random behavior
    eMin, eMax = var *(1-tmp)**2, var*(1+tmp)**2
    eVal = np.linspace(eMin, eMax,pts)
    
    # this ithe probability density function of the eigen-values of the correlation matrix
    pdf = q/(2*np.pi*var*eVal) * ((eMax - eVal) *(eVal- eMin))**.5

    pdf = pd.Series(pdf, index=eVal)    

    return pdf

def getPCA(matrix):
    # eigen-value / eigen vectors
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1] # indexes of eigen values in descending order
    
    # slicing according to the size of eigen values
    eVal, eVec = eVal[indices], eVec[:, indices]
    
    # turn into a diagonal matrix
    eVal = np.diagflat(eVal)
    return eVal, eVec

def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    # Fit kernel to a series of observations
    # this is an empirical pdf
    # output the probability of obs
    # x is the array of values on which the fit KDE will be avaluated
    if len(obs.shape) == 1:
        obs=obs.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x=np.unique(obs).reshape(-1,1)
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    
    logProb=kde.score_samples(x) #log of density
    
    
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

from scipy.optimize import minimize

def errPDFs(var, eVal, q, bWidth, pts=1000):
    # fitting error given a variance
    pdf0= mpPDF(var[0], q, pts)
    pdf_kde = fitKDE(eVal, bWidth, x = pdf0.index.values)
    
    sse = np.sum((pdf_kde-pdf0)**2)
    return sse

def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x :errPDFs(*x), 
                   0.5, 
                   args=(eVal, q, bWidth),
                  bounds=((1e-5, 1-1e-5),))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1+(1./q) **0.5)**2
    return eMax, var    

def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    # random cov matrix from the data
    # not full rank because nCols > nFacts: we have more noises than signals
    cov = np.dot(w, w.T) 
    cov+= np.diag(np.random.uniform(size=nCols)) # full rank cov
    return cov
    
def cov2corr(cov):
    # Derive the correlation matrix from the covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr <-1], corr[corr > 1] = -1, 1 # fixing the numerical error
    return corr

def corr2cov(corr, std):
    cov = corr*np.outer(std, std)
    return cov

def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    # set the digonal to 1
    block[range(bSize), range(bSize)] = 1
    
    corr = block_diag(*([block] * nBlocks))
    return corr

def formTrueMatrix(nBlocks, bSize, bCorr):
    # Generate blocks of correlation matrix
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    
    # randomly shuffle the cols
    np.random.shuffle(cols)
    
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    
    # Noises. Note: corr0 is a square matrix with each of dimension of size (nBlocks * bSize)
    
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])
    
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0



nBlocks, bSize, bCorr = 10, 50, .5
np.random.seed(0)

mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)

from sklearn.covariance import LedoitWolf

def simCovMu(mu0, cov0, nObs, shrink=False):
    # nObs depend on how many observations we are simulating
    # ideally, nObs should be larger than the size of cov0
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    
    # take the empirical mean
    # and the empirical Covariance
    mu1 = x.mean(axis=0).reshape(-1, 1)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1= np.cov(x, rowvar=0)
    return mu1, cov1

def denoisedCorr(eVal, eVec, nFacts):
    # remove noise by fixing eigenvalues that are considered random
    # as opposed to simply getting rid of the eigen vectors like in PCA
    eVal_ = np.diag(eVal).copy()
    
    # the replacement value is such that the trace of the eigenvalues matrix is preserved
    replace_val = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_[nFacts:] = replace_val
    
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1

def deNoiseCov(cov0, q, bWidth):
    corr0= cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 =eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) **0.5)
    return cov1

def optPort(cov, mu=None):
    ## get the minimum variance portfolio
    ## of the risky asset portfolio
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    
    w = np.dot(inv, mu)
    w/= np.dot(ones.T, w)
    return w

nObs, nTrials, bWidth, shrink, minVarPortf = 1000, 1000, 0.01, False, True

def sample(i):
    np.random.seed(i)
    mul, cov1 = simCovMu(mu0, cov0, nObs, shrink=shrink)
    if minVarPortf: mu1=None
    cov1_d = deNoiseCov(cov1, nObs * 1./cov1.shape[1], bWidth)
    
    # Return item index
    return optPort(cov1, mu1).flatten(), optPort(cov1_d, mu1).flatten()
