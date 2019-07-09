import GPy
import sys
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy import optimize
from scipy.stats import uniform, truncnorm, norm

from pyDOE import lhs

import time


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

    n = len(mu)
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

    cdf_C = norm.cdf(C)
    diff_CDF = cdf_C[1:] - cdf_C[:-1]

    pdf_C = norm.pdf(C)
    diff_PDF = pdf_C[1:] - pdf_C[:-1]

    out = np.sum( a[A]*diff_CDF + b[A]*diff_PDF ) - np.max(mu)

    return out


def rep_concat(x, n):
    """ x is a 1D array, repeat and concat array n times"""
    assert len(x.shape)==1; "cannot duplicate a matrix!"
    out = np.repeat(x, n)
    out = out.reshape(-1, n)
    out = out.T
    out = out.reshape(-1)
    return(out)


def COV(model, xa1, xa2, chol_K=None):
    """Takes a GP model, and 2 points, returns post cov.
    ARGS
     model: a gpy object
     xa1: n1*d matrix
     xa2: n2*d matrix
     chol_K: optional precomuted cholesky decompositionb of kern(X,X)
    
    RETURNS
     s2: posterior GP cov matrix
     """

    assert len(xa1.shape)==2; "COV: xa1 must be rank 2"
    assert len(xa2.shape)==2; "COV: xa2 must be rank 2"
    assert xa1.shape[1]==model.X.shape[1]; "COV: xa1 must have same dim as model"
    assert xa2.shape[1]==model.X.shape[1]; "COV: xa2 must have same dim as model"
    assert chol_K.shape[0]==model.X.shape[0]; "chol_K is not same dim as model.X"

    if chol_K is None:
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))

    Lk1 = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))
    Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, xa2))

    K_ = model.kern.K(xa1, xa2)

    s2 = np.matrix(K_) - np.matmul(Lk1.T, Lk2)
    return s2


def VAR(model, xa1, chol_K=None):
    """Takes a GP model, and 1 point, returns post var.
    
    ARGS
     model: a gpy object
     xa1: n1*d matrix
     chol_K: cholesky decompositionb of kern(X,X)
    
    RETURNS
     s2: posterior GP cov matrix
     """

    assert len(xa1.shape)==2; "VAR: xa1 must be rank 2"
    assert  xa1.shape[1]==model.X.shape[1]; "VAR: xa1 have same dim as model"

    if chol_K is None:
        K = model.kern.K(model.X,model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))

    Lk = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))
    K_ = model.kern.K(xa1, xa1)
    s2 = K_ - np.matmul(Lk.T, Lk)
    return s2


def toy_test_func(xa, NoiseSD=np.sqrt(0.01), gen_seed=11, gen=False):
    """A toy function (and generator if necessary upon first call), GP
    
    ARGS
     xa: n*d matrix, points in space to eval testfun
     NoiseSD: additive gaussaint noise SD
     seed: int, RNG seed
     gen: boolean, create new test function?
    
    RETURNS
     output: vector of length nrow(xa)
     """

    if len(xa.shape)==1 and xa.shape[0]==3: xa = xa.reshape((1,3))
    assert len(xa.shape)==2; "xa must be an N*3 matrix"
    assert xa.shape[1]==3; "xa must be 3d"
     
    
    KERNEL = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=([10,10,10]), ARD = True)
    
    if gen or not hasattr(toy_test_func, "invCZ"):
        # If the test function has not been generated then generate it.

        print("Generating test function")
        np.random.seed(gen_seed)
        X0 = np.linspace(0, 100, 8)
        F1 = np.linspace(0, 100, 8)
        F2 = np.linspace(0, 100, 8)

        XF = np.array([[i,j,k] for i in X0 for j in F1 for k in F2])
        mu = np.zeros( XF.shape[0] )
        C  = KERNEL.K(XF, XF)

        Z = np.random.multivariate_normal(mu, C).reshape(-1,1)
        invC = np.linalg.inv(C + np.eye(C.shape[0])*1e-3)

        toy_test_func.invCZ = np.dot(invC, Z)
        toy_test_func.XF    = XF
        toy_test_func.lb    = np.zeros((3,))
        toy_test_func.ub    = np.ones((3,))*100

    # if xa.shape==(3,) or xa.shape==(3): xa = xa.reshape(1, 3)

    # assert len(xa.shape)==2, "xa must be an N*3 matrix, each row a 3d point"
    # assert xa.shape[1]==3, "Test_func: xa must 3d"

    ks = KERNEL.K(xa, toy_test_func.XF)
    out = np.dot(ks, toy_test_func.invCZ)

    E = np.random.normal(0, NoiseSD, xa.shape[0])

    return (out.reshape(-1,1) + E.reshape(-1, 1))


def toy_inf_src(s, n=2, gen_seed=11, gen=False):
    """
    A Toy info source generator!
    An mv norm sampler with randomly set params.
    
    ARGS
     n: numer of input sources
     s: size of data to generate
     mv_gen: boolean, regenerate mvnrom params
    
    RETURNS
     rn: matrix of info source samples.
    """
    
    ub = 85
    lb = 15
    
    if gen or not hasattr(toy_inf_src, "f_mean"):
        print("Generating params for " + str(n) + " info sources")

        np.random.seed(gen_seed)
        var = np.random.random(n)*(20-5)+5
        toy_inf_src.f_mean = np.random.random(n)*(ub-lb)+lb
        toy_inf_src.f_cov = np.multiply(np.identity(n), var)
        toy_inf_src.n_sources = n

    rn = np.random.multivariate_normal(toy_inf_src.f_mean, toy_inf_src.f_cov, s)

    return rn[:,s],reshape(-1)


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


def sample_predict_dens(Data, N, MUSIG0_L, MU_L, SIG_L):
        # Generate samples from the haluciated future pmf
        #
        # ARGS
        #  Data: matrix of observations of info sources
        #  N: sample size
        #  MUSIG0_L:
        #  MU_L:
        #  SIG_L:
        #
        # RETURNS
        #  zn: samples from updated distro
        
        Data = list(Data[~np.isnan(Data)])
        # def Distr_Update(Data):
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
        # return D
        
        pdf_zn = D #istr_Update(Data)
        
        zn = Gen_Sample(pdf_zn, N)        
        return zn


def Mult_Input_Uncert(test_func, lb, ub, IU_dims, inf_src, 
                      n_fun_init=10, 
                      n_inf_init=0, 
                      Budget=100,
                      Nx=101, 
                      Nr=100):

    """
    Optimizes the test function integrated over IU_dims. The integral
    is also chaning ver time and learnt.

    ARGS
        test_func: callable function, returns scalar
        lb: lower bounds on input to test_fun
        ub: upper bounds on input to test_fun
        IU_dims: int. list of dims of test_func that are IU
        inf_src: callable function returning info source data
        n_fun_init: number of inital points for GP model
        n_inf_init: number of intial points for info source
        Budget: total budget of calls to test_fun and inf_src
        Nx: int, discretization size of X
        Nr: int, discretization size of X

    RETURNS
        X: observed test_func inputs
        Y: observed test_func outputs
        Data: list of array of inf_src observations
        rec_X: the recomended X values
    """

    IU_dims = np.array(IU_dims)

    assert len(lb.shape)==1; "lb must be 1d array"
    assert len(ub.shape)==1; "ub must be 1d array"
    assert ub.shape[0] == lb.shape[0]; "lb and ub must be the same shape!"
    assert np.all(IU_dims<ub.shape[0]); "IU_dims out of too high!"
    assert np.all(IU_dims>=0); "IU_dims too low!"

    x = np.linspace(0, 100, Nx) #vector of input variable
    dim = len(IU_dims)

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

    
    def Delta_Loss(Data, idx, model, Xr, Nr=102, Nx=101, Nd=100):
        """
        Compute the improvement due to queerying the info source
        
        ARGS
         Data: n*Ns matrix, info source observations
         idx: which info source to compute DL for.
         Xr: recomended X value with current data
         Nr: int
         Nx: int
         Nd: int
        
        RETURNS
         DL: the delta loss for source idx!

        """
        
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
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #=============================================================================================
    
    # Initilize GP model
    XA = lhs(3, n_fun_init)*100
    Y  = test_func(xa=XA)
    ker = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=([10,10,10]), ARD=True)

    # Initilize input uncertainty model
    Data =  np.zeros((0, 2))
    Ndata = np.sum(~np.isnan(Data))

    print("Initialization complete", XA.shape[0] +  Ndata)

    while XA.shape[0] +  Ndata < Budget:
       
        Ndata = np.sum(~np.isnan(Data))
        print("Iteration ", XA.shape[0]+Ndata+1, ":")

        # Fit model to simulation data.
        GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1,1), ker, noise_var=0.01)

        # Fit model to IU data and generate samples for KG MC integral.
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
        topis, topDL = np.argmax(DL), np.max(DL)
        
        if topKG > np.max(DL):
            # if simulation is better
            print("Best is simulator: ", topxa, topKG)
            new_y = test_func(topxa)

            XA = np.vstack([XA, topxa])
            Y = np.concatenate([Y, new_y])
        
        else:
            # if info source is better
            print("Best is info source: ", topis, topDL)
            new_d = np.array([ [np.nan]*dim ])
            new_d[0, is] = inf_src(topis)

            Data = np.vstack([Data, new_d])
        
        print(" ")

    return XA, Y, Data#, rec_X 



if __name__=="__main__":

    
    print("Calling Test Function")
    xa = np.random.uniform(size=(10, 3))*100

    print(xa)

    print(toy_test_func(xa[0,:].reshape(-1), NoiseSD=0.))

    print(toy_test_func(xa, NoiseSD=0.))

    print("Calling info source")

    print(toy_inf_src(0, 10))
    print(toy_inf_src(1, 10))
    
    print("\nCalling optimizer")

    X, Y, Data = Mult_Input_Uncert(test_func=toy_test_func,
                                   lb=toy_test_func.lb,
                                   ub=toy_test_func.lb,
                                   IU_dims=[1,2],
                                   inf_src=toy_inf_src)
    
    