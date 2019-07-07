import GPy
import csv
import sys
import numpy as np
import scipy
from scipy.optimize import minimize
import time
# import pygmo as pg
from scipy.stats import uniform 
from pyDOE import *
from scipy import optimize
import pandas as pd

import scipy.integrate as integrate
import scipy.special as special

import time
from scipy.stats import truncnorm
from scipy.interpolate import interp1d

from utils import KG


def rep_concat(x, n):
    assert len(x.shape)==1; "cannot duplicate a matrix!"
    out = np.repeat(x, n)
    out = out.reshape(-1, n)
    out = out.T
    out = out.reshape(-1)
    return(out)


def COV(model, xa1, xa2, chol_K=None):
    # Takes a GP model, and 2 points, returns post cov.
    # ARGS
    #  model: a gpy object
    #  xa1: n1*d matrix
    #  xa2: n2*d matrix
    #  chol_K: cholesky decompositionb of kern(X,X)        
    #
    # RETURNS
    #  s2: posterior GP cov matrix

    assert len(xa1.shape)==2; "COV: xa1 must be rank 2"
    assert len(xa2.shape)==2; "COV: xa2 must be rank 2"
    assert  xa1.shape[1]==model.X.shape[1]; "COV: xa1 must have same dim as model"
    assert  xa2.shape[1]==model.X.shape[1]; "COV: xa2 must have same dim as model"

    if chol_K is None:
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))

    Lk1 = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))
    Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, xa2))

    K_ = model.kern.K(xa1, xa2)

    s2 = np.matrix(K_) - np.matmul(Lk1.T, Lk2)
    return s2


def VAR(model, xa1, chol_K=None):
    # Takes a GP model, and 1 point, returns post var.
    #
    # ARGS
    #  model: a gpy object
    #  xa1: n1*d matrix
    #  chol_K: cholesky decompositionb of kern(X,X)
    #
    # RETURNS
    #  s2: posterior GP cov matrix

    assert len(xa1.shape)==2; "VAR: xa1 must be rank 2"
    assert  xa1.shape[1]==model.X.shape[1]; "VAR: xa1 have same dim as model"

    if chol_K is None:
        K = model.kern.K(model.X,model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))

    Lk = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))
    K_ = model.kern.K(xa1, xa1)
    s2 = K_ - np.matmul(Lk.T, Lk)
    return s2


def test_func(xa, NoiseSD=np.sqrt(0.01), seed=11, gen=False):
    # A toy function for generator, GP
    #
    # ARGS
    #  xa: n*d matrix, points in space to eval testfun
    #  NoiseSD: additive gaussaint noise SD
    #  seed: int, RNG seed
    #  gen: boolean, create new test function?
    #
    # ASSUMED ARGS
    #  upper bounds: 100, 100, 100
    #  lower bounds: 0,0,0
    #  generative GP hypers: lx=10,10,10, var=3
    #
    # RETURNS
    #  output: vector of length nrow(xa)
    
    KERNEL = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=([10,10,10]), ARD = True)
    
    if gen == True or ~hasattr(testc_fun, "invCZ") or ~hasattr(test_func, "XtFi"):
        
        X0 = np.linspace(0, 100, 8)
        F1 = np.linspace(0, 100, 8)
        F2 = np.linspace(0, 100, 8)

        XF = np.array([[i,j,k] for i in X0 for j in F1 for k in F2])
        mu = np.zeros( XF.shape[0] )
        C  = KERNEL.K(XF, XF)

        Z = np.random.multivariate_normal(mu, C).reshape(-1,1)
        invC = np.linalg.inv(C + np.eye(C.shape[0])*1e-3)

        test_func.invCZ = np.dot(invC, Z)
        test_func.XF    = XF

    if xa.shape==(3,) or xa.shape==(3): xa = xa.reshape(1, 3)

    assert len(xa.shape)==2, "Test_func: xa must be a matrix"
    assert xa.shape[1]==3, "Test_func: xa must 3d"

    ks = KERNEL.K(xa, test_func.XF)
    out = np.dot(ks, test_func.invCZ)

    E = np.random.normal(0, NoiseSD, len(xa))

    return (out.reshape(-1,1) + E.reshape(-1, 1))


def Input_Source(n, s, mv_gen = False):
    # A Toy info source generator!
    # An mv norm sampler with randomly set params.
    #
    # ARGS
    #  n: numer of input sources
    #  s: size of data to generate
    #  mv_gen: boolean, regenerate mvnrom params
    #
    # ASSUMED ARGS
    #  Multi_Input_Uncert.var
    #  sampler hypers
    #
    # RETURNS
    #  rn: matrix of infor source samples.
    
    ub = 85
    lb = 15
    var = np.random.random(n)*(20-5)+5
    Mult_Input_Uncert.var = var
    if mv_gen == True or ~hasattr(Input_Source, "f_mean") or ~hasattr(Input_Source, "f_cov"):
        Input_Source.f_mean = np.random.random(n)*(ub-lb)+lb
        Input_Source.f_cov = np.multiply(np.identity(n), var)

    rn = np.random.multivariate_normal(Input_Source.f_mean, Input_Source.f_cov, s)
    return rn


def Fit_Inputs(Y, MUSIG0, MU, SIG):
        # Computes MU and margs_mu from observations
        # 
        # ARGS
        #  Y: observations
        #  MUSIG0:
        #  MU:
        #  SIG:
        #
        # RETURNS
        #  MU: posterior mean?
        #  marg_mu: marginal mean

        Y = np.array(Y)
        Y = list(Y[~np.isnan(Y)])

        L1 = (1.0/(2.0*MUSIG0[:,1]))
        L1 = L1*np.sum(np.array((np.matrix(MUSIG0[:,0]).T  - Y))**2.0,axis=1)
        L1 = np.exp(-L1)
        L2 = (1.0/np.sqrt(2*np.pi*MUSIG0[:,1]))**len(Y)

        L = L1*L2
        
        L = np.array(L).reshape(len(MU),len(SIG))

        dmu = MU[1]-MU[0]
        dsig = SIG[1]-SIG[0]
        LN = np.sum(L*dmu*dsig)
        P = L/LN
        marg_mu = np.sum(P, axis=1)*dsig

        return MU, marg_mu

def Gen_Sample(Dist, N=500):
    # Given a pmf generates samples assuming pmf is over equally
    # spaced points in 0,...,100
    #
    # ARGS
    #  Dist: vector of probalilities
    #  N: sample size
    #
    # RETURNS
    #  val: samples from set of qually spaced points in 0,..,100

    elements = np.linspace(0,100,len(Dist))
    probabilities = Dist/np.sum(Dist)
    val = np.random.choice(elements, N, p=probabilities)
    return val        

def Mult_Input_Uncert():
    global Nx, Nr, init_sample, x, MU, SIG, MUSIG0,MU_L, SIG_L, MUSIG0_L, Stop


    def sample_predict_dens(Data, N, MUSIG_L, MU_L, SIG_L):
        # Generate samples from the haluciated future pmf
        #
        # ARGS
        #  Data: matrix of observations of info sources
        #  N: sample size
        #
        # ASSUMED ARGS
        #  MUSIG0:
        #  MU_L:
        #  SIG_L:
        #
        # RETURNS
        #  zn: samples from updated distro

        # global MUSIG0, MU_L, SIG_L, MUSIG0_L
        
        Data = list(Data[~np.isnan(Data)])
        def Distr_Update(Data):
            Y = Data
            L = []
            fy = []
            y_n1 = np.linspace(0, 100, 200)
          
            for i in MUSIG0_L:
                fy.append(np.exp(-(1.0/(2.0*i[1]))*(i[0] - y_n1)**2.0))
                L.append(np.exp(-(1.0/(2.0*i[1]))*np.sum((i[0] - Y)**2.0))*(1.0/np.sqrt(2*np.pi*i[1]))**len(Y))
            dmu = MU_L[1]-MU_L[0]
            dsig = SIG_L[1]-SIG_L[0]
            dy_n1 = y_n1[1]-y_n1[0]

            L = np.matrix(L)
            fy = np.matrix(fy)
            D = np.array((np.matrix(L))*np.matrix(fy)*dmu*dsig)
            D = np.array((D/np.sum(D*dy_n1)))[0]
            return D
        
        pdf_zn = Distr_Update(Data)
        
        zn = Gen_Sample(pdf_zn, N)        
        return zn


    def Delta_Loss(Data, idx, model, Xr):
        # Compute the improvement due to queerying the info source
        #
        # ARGS
        #  Data: n*Ns matrix, info source observations
        #  idx: which info source to compute DL for.
        #
        # IMPLICIT ARGS
        #  Nr: 
        #  Nx:
        #
        # RETURNS
        #  DL: the delta loss for source idx!

        global Nr,Nx
        
        Nd = 100
        
        def W_aj(Y, a):
            # 
            MU = np.linspace(0,100, 60)
            SIG = np.linspace(0.25, 100, 60)
            d = np.vstack(np.hstack(Y))
            N = Y.shape[1]
            dimY = Y.shape[0]
            expo = np.exp(np.vstack(-(1.0/(2.0*SIG)))*np.hstack(np.sum(np.split((d-MU)**2,dimY,axis=0),axis=1)))

            consts =  np.vstack((1.0/np.sqrt(2*np.pi*SIG))**N)

            L = np.split(expo*consts, dimY, axis=1)
            marg_mu_dist = np.sum(L,axis=1)*(SIG[1]-SIG[0])
            C = np.sum(marg_mu_dist,axis=1)*(MU[1]-MU[0])
            marg_mu_dist = marg_mu_dist*(1/np.vstack(C))

            expo = np.exp(np.vstack(-(1.0/(2.0*SIG)))*np.hstack(np.sum(np.split((d-a)**2,dimY,axis=0),axis=1)))
            consts =  np.vstack((1.0/np.sqrt(2*np.pi*SIG))**N)
            L = np.split(expo*consts,dimY,axis=1)
            marg_mu_val = np.sum(L,axis=1)*(SIG[1]-SIG[0])*(1/np.vstack(C))
            return marg_mu_val
        
        Data_idx = np.array([list(Data[~np.isnan(Data[:,idx]),idx])])

        if len(Data_idx[0]) > 0:

            z1 = sample_predict_dens(Data_idx, N=Nd)
            Sample_I = Sample[:,idx+1]
            Sample_XA = Sample
            W_D = W_aj(Y=Data_idx, a=Sample_I)
            
            dj = np.c_[Data_idx*Nd, z1]
        else:
    
            z1 = np.random.random(Nd)*100
            Sample_I = Sample[:,idx+1]
            Sample_XA = Sample
            W_D = np.array([list(np.repeat([1.0/100],Nx*Nr))])
            dj = np.vstack(z1) 

        R_IU = []
        W_D1 = W_aj(Y=dj, a=Sample_I)

        Wi = W_D1/W_D
        Mult_Input_Uncert.Wi = Wi.reshape(Nd,Nx,Nr)
        
        Prd = model.predict(np.array(Sample_XA))[0].reshape(Nx, Nr)    
        Mult_Input_Uncert.Prd = Prd
        IU_D1 = np.mean(np.multiply(Mult_Input_Uncert.Wi,Mult_Input_Uncert.Prd),axis=2)
        
        max_IU_D1 = np.max(IU_D1,axis=1)

        Prd_D = model.predict(np.array(np.c_[np.repeat(Xr, Nr),Sample[:Nr,1:3]]))[0].T
        Mult_Input_Uncert.Prd_D = Prd_D
     
        IU_D = np.mean(np.mean(np.multiply(Mult_Input_Uncert.Wi,Mult_Input_Uncert.Prd_D),axis=2),axis=1)
        DL = np.mean(max_IU_D1 - IU_D)

        return DL
    

    def KG_Mc_Input(XA, model, A1_samples, A2_samples, Nx=15, Ns=20):
        # Takes a GPy model, constructs and optimzes KG and 
        # returns the best xa and a y value.
        #
        # ARGS
        #  XA: N*d matrix of observed locations, points
        #  model: gpy model
        #  Nx: int, numebr of KG discretization points.
        #  Ns: number of random starts for the optimizer
        #
        # IMPLICIT ARGS
        #  m: gp model
        #
        # RETURNS
        #  bestxa: the optimal xa for simulating
        #  Y: the output of the simulation

        assert len(A1_samples)==len(A2_samples); "code assumes same number of both A1 and A2 samples"

        # optimizer initial optim
        KG_Mc_Input.bestEVI = -10
        KG_Mc_Input.bestxa  = [-10,-10,-10]

        noiseVar = model.Gaussian_noise.variance[0] 

        # The past oberserved X values.
        Xd = np.array(XA[:,0]).reshape(-1,1)
        len_Xd = Xd.shape[0]

        # The given sampled A1 and A2 values
        Ad     = np.c_[A1_samples, A2_samples] #1000
        len_Ad = Ad.shape[0]

        # The matrix of discretized points.
        XdAd   = np.c_[np.repeat(Xd, len_Ad), 
                       rep_concat(A1_samples, len_Xd), 
                       rep_concat(A2_samples, len_Xd)]

        # Precompute the posterior mean added integrated over A1, A2 samples.
        MM0 = np.sum(model.predict(XdAd)[0].reshape(len_Xd,len_Ad), axis=1)
        MM0 = MM0.reshape(1,-1)

        # Precompute cholesky decomposition.
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(K.shape[0]))

        def KG_IU(xa):
            xa = np.array(xa).reshape((1, -1))

            if np.any(xa>100) or np.any(xa<0):# (np.abs(xa[0]-50)>50 or np.abs(xa[1]-50)>50 or np.abs(xa[2]-50)>50):
                return(1000000)
            else:
                # The current x with all the A samples
                newx = np.c_[xa[0,0]*np.ones((len_Ad, 1)), Ad]

                # The mean of new x integrated over A samples
                MMx  = np.sum(model.predict(newx)[0])

                # print(MMx); sys.exit()
                MM   = np.c_[MMx, MM0].reshape(-1)
                MM   = MM*(1/len_Ad)

                # The average warping integrated over A1, A2 for XdAd
                sigt_d = np.array(COV(model, xa, XdAd, chol_K))
                sigt_d = sigt_d.reshape(len_Xd, len_Ad)
                sigt_d = np.squeeze(np.sum(sigt_d, axis=1))
                
                # average warping for newx
                sigt_x = np.sum(COV(model, xa, newx, chol_K)).reshape((-1,))

                # variance of new observation
                var_xa = VAR(model, xa, chol_K) + noiseVar
                inv_SD_xa = 1/np.sqrt(var_xa).reshape(())

                # put the peices together!
                SIGT  = np.hstack([sigt_x, sigt_d])* inv_SD_xa
                SIGT  = SIGT * (1/len_Ad)

                # Finally compute KG!
                out  = KG(MM, SIGT)
                if out > KG_Mc_Input.bestEVI:
                    KG_Mc_Input.bestEVI = out
                    KG_Mc_Input.bestxa = xa
                return -out

        # XAs = np.array(np.c_[lhs(1, samples=Ns)*100,
        #                      lhs(1, samples=Ns)*100,
        #                      lhs(1, samples=Ns)*100])
        XAs  = lhs(3, Ns)*100
        A    = [minimize(KG_IU, i, method='nelder-mead', options={'maxiter': 80}) for i in XAs]
        
        return KG_Mc_Input.bestxa, KG_Mc_Input.bestEVI


    #=============================================================================================
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #==============================================================================================
    #Main Algorithm
    
    #initialise Parameters
    
        #Random variable generation for inputs
    
    init_sample = 10 # Initial Sample size
    Nx          = 101 # Granularity x value
    Nr          = 100
    EndN        = 100
    dim         = 2
    x           = np.linspace(0, 100, Nx) #vector of input variable

    # Make lattice over IU parameter space.
    precision = 101
    MU = np.linspace(0, 100, precision)
    SIG = np.linspace(0.025, 100, precision)
    rep_MU = np.repeat(MU, precision)
    rep_SIG = rep_concat(SIG, precision)
    MUSIG0 = np.c_[rep_MU, rep_SIG]
    
    MU_L  = np.linspace(0,100,101)
    SIG_L = np.linspace(0.025,100,101)
    X_L   = np.repeat(MU_L, 101)
    W_L   = rep_concat(SIG_L, 101)
    MUSIG0_L = np.c_[X_L, W_L]

    init_input_source = Input_Source(dim, 0)
    
    #Train GP
    XA = lhs(3, init_sample)*100
    Y  = test_func(xa=XA, gen=True)
    
    ker = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=([10,10,10]), ARD=True)

    # iLoss = []
    # OC = []
    #===================================================================================
    #True Func
    
    # XW0 = np.c_[ x, np.array([Input_Source.f_mean]*len(x))]
    # True_obj_v = test_func(xa=XW0,NoiseSD=0,gen=False)
    # obj     = lambda a: -1*np.mean(test_func(np.c_[[a],[Input_Source.f_mean]],NoiseSD=0,gen=False))
    # topX    = x[np.argmax(True_obj_v)]

    
    # if topX >= 100:    
    #     topX=99
    # elif topX <=0:
    #     topX=1
        
    # topX = optimize.fminbound(obj, topX-1, topX+1,xtol =1e-16)
    # best_ = -1*obj(topX)

    #=================================================================================================
    Data =  np.zeros((0, 2))
    Ndata = np.sum(~np.isnan(Data))

    while XA.shape[0] +  Ndata < EndN:
       
        Ndata = np.sum(~np.isnan(Data))
        print(XA.shape[0]+Ndata, ":")

        # Fit model to simulation data.
        GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1,1), ker, noise_var=0.01)

        # Fit model to IU data and generate samples.
        if Ndata > 0:
            _, A1_pdf = Fit_Inputs(Data[:,0], MUSIG0, MU, SIG)
            _, A2_pdf = Fit_Inputs(Data[:,1], MUSIG0, MU, SIG)
            A1_samples = Gen_Sample(A1_pdf, Nr)
            A2_samples = Gen_Sample(A2_pdf, Nr)
        else:
            A1_samples = np.random.random(Nr)*100
            A2_samples = np.random.random(Nr)*100

        Sample = np.c_[np.repeat(x, Nr), 
                       rep_concat(A1_samples, Nx),
                       rep_concat(A2_samples, Nx)]

        IU = np.mean(GPmodel.predict(Sample)[0].reshape(Nx, Nr),axis=1)
        Xr = x[np.argmax(IU)]

        # Get KG of both simulation and Input uncertainty.
        topxa, topKG = KG_Mc_Input(XA, GPmodel, A1_samples, A2_samples, Nx=Nx, Ns=20)
        DL = np.array([Delta_Loss(Data, i, GPmodel, Xr) for i in range(dim)])
        
        
        if topKG > np.max(DL):
            # if simulation is better
            print("Best is simulator: ", topxa, topKG)
            new_y = test_func(topxa)
            XA = np.vstack([XA, topxa])
            Y = np.concatenate([Y, new_y])
        
        else:
            # if info source is better
            QA = np.argmax(DL)
            print("Best is inf_src:", QA)
            new_d = np.array([[np.nan, np.nan]])
            new_d[0, QA] = Input_Source(2, 1)[0, QA]
            Data = np.vstack([Data, new_d])
            Ndata = np.sum(~np.isnan(Data))
        
        print(" ")

    return XA, Y, Data

OC= []
for i in range(1):
    identifier = 100 + np.random.random()
    OC ,N_I, var = Mult_Input_Uncert() 
    data = {'OC': OC,'len': [N_I]*len(OC),'var1':[var[0]]*len(OC),'var2':[var[1]]*len(OC)}
    print('data',data)
    gen_file = pd.DataFrame.from_dict(data)
    path ='/home/rawsys/matjiu/PythonCodes/PHD/Input_Uncertainty/With_Input_Selection/Data_MC100/OC_'+str(i)+'_' + str(identifier) + '.csv'
    gen_file.to_csv(path_or_buf=path)
