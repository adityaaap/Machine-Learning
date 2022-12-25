"""
The gmm function takes in as input a data matrix X and a number of gaussians in
the mixture model

The implementation assumes that the covariance matrix is shared and is a
spherical diagonal covariance matrix

If you get this ImportError
    ImportError: cannot import name 'logsumexp' from 'scipy.special',
you may need to update your scipy install to >= 0.19.0
"""

from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
import numpy as np
import math

import matplotlib.pyplot as plt


def calc_logpdf(x, mean, cov):
    """Return log probability density."""
    x = multivariate_normal.logpdf(x, mean=mean, cov=cov)
    return x


def gmm(trainX, num_K, num_iter=10, plot=False):
    """Fit a gaussian mixture model on trainX data with num_K clusters.

    trainX is a NxD matrix containing N datapoints, each with D features
    num_K is the number of clusters or mixture components
    num_iter is the maximum number of EM iterations run over the dataset

    Description of other variables:
        - mu, which is KxD, the coordinates of the means
        - pk, which is Kx1 and represents the cluster proportions
        - zk, which is NxK, has at each z(n,k) the probability that the nth
          data point belongs to cluster k, specifying the cluster associated
          with each data point
        - si2 is the estimated (shared) variance of the data
        - BIC is the Bayesian Information Criterion (smaller BIC is better)
    """
    N = trainX.shape[0]
    D = trainX.shape[1]

    if num_K >= N:
        print("You are trying too many clusters")
        raise ValueError
    if plot and D != 2:
        print("Can only visualize if D = 2")
        raise ValueError

    si2 = 5  # Initialization of variance
    pk = np.ones((num_K, 1)) / num_K  # Uniformly initialize cluster proportions
    mu = np.random.randn(num_K, D)  # Random initialization of clusters
    zk = np.zeros(
        [N, num_K]
    )  # Matrix containing cluster membership probability for each point

    if plot:
        plt.ion()
        fig = plt.figure()
    for iter in range(0, num_iter):
        """Iterate through one loop of the EM algorithm."""
        if plot:
            plt.clf()
            xVals = trainX[:, 0]
            yVals = trainX[:, 1]
            x = np.linspace(np.min(xVals), np.max(xVals), 500)
            y = np.linspace(np.min(yVals), np.max(yVals), 500)
            X, Y = np.meshgrid(x, y)
            pos = np.array([X.flatten(), Y.flatten()]).T
            plt.scatter(xVals, yVals, color="black")
            pdfs = []
            for k in range(num_K):
                rv = multivariate_normal(mu[k], si2)
                pdfs.append(rv.pdf(pos).reshape(500, 500))
            pdfs = np.array(pdfs)
            plt.contourf(X, Y, np.max(pdfs, axis=0), alpha=0.8)
            plt.pause(0.01)

        """
        E-Step
        In the first step, we find the expected log-likelihood of the data
        which is equivalent to:
        Finding cluster assignments for each point probabilistically
        In this section, you will calculate the values of zk(n,k) for all n and
        k according to current values of si2, pk and mu
        """
        # Implement the E-step
        likelihood = []
        for k in range(num_K):
           likelihood.append(multivariate_normal(mean=mu[k], cov=si2))
        
        for n in range(N):
            val = []
            for k in range(num_K):
                val.append(pk[k] * likelihood[k].pdf(trainX[n]))
                sum = np.sum(val)
            for k in range(num_K):
                zk[n,k] = (pk[k] * likelihood[k].pdf(trainX[n])) / (sum)       
        """
        M-step
        Compute the GMM parameters from the expressions which you have in the spec
        """
        # Estimate new value of pk
        # 

        pk = np.mean(zk, axis=0)

        # Estimate new value for means
        

        mu = (1 / N) * (1 / pk).reshape(num_K, -1) * np.dot(zk.T, trainX)

        # Estimate new value for sigma^2
        
        
        for n in range(N):
            for k in range(num_K):
                a = trainX[n] - mu[k] 
                si2 = si2 + (zk[n,k]* np.dot(a.T,a))
        si2  = si2*(1/N)*(1/D)



    if plot:
        plt.ioff()
        plt.savefig('visualize_clusters.png')
    # Computing the expected log-likelihood of data for the optimal parameters computed
    
    # Compute the BIC for the current clustering
    LL_max = 0
    
    for n in range(N):
        LL = np.zeros([num_K])
        for k in range(num_K):
            LL[k] = LL[k] + np.log(pk[k]) + multivariate_normal.logpdf(trainX[n], mu[k], si2)
 
        LL_max = LL_max + logsumexp(LL)   
    

    
    BIC = (num_K*(D+1))*np.log(N) - 2*LL_max  # TODO: calculate BIC

    return mu, pk, zk, si2, BIC
    
