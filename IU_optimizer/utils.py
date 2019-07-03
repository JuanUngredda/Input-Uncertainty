import numpy as np
import scipy
from scipy.stats.norm import cdf as normcdf
from scipy.stats.norm import pdf as normpdf

# in this file put any simple little functions that are repeated and reused.

def Fit_Inputs(Y, MUSIG0, MU, SIG):
    # Takes data Y, and makes an unnormalized bar chart Dist.
    #
    # ARGS
    #  - Y : vector of samples
    #  - MUSIG0: matrix
    #  - MU: vector
    #  - SIG
    #
    # RETURNS
    #  - MU: vector
    #  - Dist: barchart

    Y = np.array(Y)
    Y = list(Y[~np.isnan(Y)])
    def Distr_Update():
        L = np.sum(np.array((np.matrix(MUSIG0[:,0]).T  - Y))**2.0,axis=1)
        L = np.exp(-(1.0/(2.0*MUSIG0[:,1])) * L)
        L = L*(1.0/np.sqrt(2*np.pi*MUSIG0[:,1]))**len(Y)
        L = np.array(L).reshape(len(MU),len(SIG))
        dmu = MU[1]-MU[0]
        dsig = SIG[1]-SIG[0]
        LN = np.sum(L*dmu*dsig)
        P = L/LN
        marg_mu = np.sum(P,axis=1)*dsig
        return marg_mu
    Dist = Distr_Update()
    return MU, Dist


def KG(mu, sig):
    # Takes a set of intercepts and gradients of linear functions and returns
    # the average hieght of the max of functions over Gaussain input.
    #
    # ARGS
    # - mu: length n vector, initercepts of linear functions
    # - sig: length n vector, gradients of linear functions
    #
    # IMPLICITLY ASSUMED ARGS
    # - None
    #
    # RETURNS
    # - out: scalar value is gaussain expectation of epigraph of lin. funs

    n = len(musig)
    O = sig.argsort()
    a = mu[O]
    b = sig[O]


    A=[0]
    C=[-float("inf")]
    while A[-1]<n-1:
        s = A[-1]
        si = range(s+1,n)
        Ci = -(a[s]-a[si])/(b[s]-b[si])
        bestsi=np.argmin(Ci)
        C.append(Ci[bestsi])
        A.append(si[bestsi])

    C.append(float("inf"))

    cdf_C = normcdf(C)
    diff_cdf_D = cdf_C[1:] - cdf_C[:-1]

    pdf_C = normpdf(C)
    diff_pdf_D = pdf_C[1:] - pdf_C[:-1]

    out = np.sum( a[A]*diff_cdf_D + b[A]*(diff_pdf_D) ) - np.max(mu)

    return out
