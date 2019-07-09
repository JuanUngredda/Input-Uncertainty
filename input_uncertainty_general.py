import GPy
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from scipy.stats import uniform, truncnorm, norm


# Toy test function and info source.
def toy_func(xa, NoiseSD=np.sqrt(1), gen_seed=11, gen=False):
    """
    A toy function (and generator if necessary upon first call), GP
    
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
     
    
    KERNEL = GPy.kern.RBF(input_dim=3, variance=10000., lengthscale=([10,10,10]), ARD = True)
    
    if gen or not hasattr(toy_func, "invCZ"):
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

        toy_func.invCZ = np.dot(invC, Z)
        toy_func.XF    = XF

    toy_func.lb    = np.zeros((3,))
    toy_func.ub    = np.ones((3,))*100

    # if xa.shape==(3,) or xa.shape==(3): xa = xa.reshape(1, 3)

    # assert len(xa.shape)==2, "xa must be an N*3 matrix, each row a 3d point"
    # assert xa.shape[1]==3, "Test_func: xa must 3d"

    ks = KERNEL.K(xa, toy_func.XF)
    out = np.dot(ks, toy_func.invCZ)

    E = np.random.normal(0, NoiseSD, xa.shape[0])

    return (out.reshape(-1,1) + E.reshape(-1, 1))


def toy_infsrc(n, src, n_srcs=2, gen_seed=11, gen=False):
    """
    A Toy info source generator!
    An mv norm sampler with randomly set params.
    
    ARGS
        n: number of samples
        src: which source to use
        n_srcs: int, number of source to generate when initiliazing
        gen_seed: int, rng seed for generated initilization
        gen: boolean, regenerate mvnrom params
    
    RETURNS
        rn: vector of info source samples.
    """
    
    ub = 85
    lb = 15
    
    if gen or not hasattr(toy_inf_src, "f_mean"):
        print("Generating params for " + str(n_srcs) + " info sources")

        np.random.seed(gen_seed)
        var = np.random.random(n_srcs)*(20-5)+5
        toy_inf_src.f_mean = np.random.random(n_srcs)*(ub-lb)+lb
        toy_inf_src.f_cov = np.multiply(np.identity(n_srcs), var)
        toy_inf_src.n_srcs = n_srcs

    assert src < toy_inf_src.n_srcs; "info source out of range"
    # import pdb; pdb.set_trace()
    rn = np.random.normal(loc=toy_inf_src.f_mean[src], 
                          scale=toy_inf_src.f_cov[src, src], 
                          size=n)
    
    return rn.reshape(-1)


# Utilities, these functinos work as standalone functions.
def lhs_box(n, lb, ub):
    """
    Random samples uniform lhs in a box with lower and upper bounds.
    ARGS
        n: number of points
        lb: vector, lower bounds of each dim
        ub: vector, upper bounds of each dim
    
    RETURNS
        LL: an lhs in the box, lb < x < ub 
    """

    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    assert lb.shape[0]==ub.shape[0]; "bounds must be same shape"
    assert np.all(lb<=ub); "lower must be below upper!"

    LL = lb + lhs(len(lb), samples=n)*(ub-lb)

    return(LL)


def KG(mu, sig):
    """
    Takes a set of intercepts and gradients of linear functions and returns
    the average hieght of the max of functions over Gaussain input.
    
    ARGS
    - mu: length n vector, initercepts of linear functions
    - sig: length n vector, gradients of linear functions
    
    RETURNS
    - out: scalar value is gaussain expectation of epigraph of lin. funs
    """

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

    out = np.sum( a[A]*diff_CDF - b[A]*diff_PDF ) - np.max(mu)

    return out


def COV(model, xa1, xa2, chol_K=None, Lk2=None):
    """Takes a GP model, and 2 points, returns post cov.
    ARGS
        model: a gpy object
        xa1: n1*d matrix
        xa2: n2*d matrix
        chol_K: optional precomuted cholesky decompositionb of kern(X,X)
        Lk2: optional precomputed solve(chol_K, kernel(model.X, xa2)) (eg for XdAd)
    
    RETURNS
        s2: posterior GP cov matrix
     """

    assert len(xa1.shape)==2; "COV: xa1 must be rank 2"
    assert len(xa2.shape)==2; "COV: xa2 must be rank 2"
    assert xa1.shape[1]==model.X.shape[1]; "COV: xa1 must have same dim as model"
    assert xa2.shape[1]==model.X.shape[1]; "COV: xa2 must have same dim as model"

    if chol_K is None:
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))
    else:
        assert chol_K.shape[0]==model.X.shape[0]; "chol_K is not same dim as model.X"


    Lk1 = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))

    if Lk2 is None:
        Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, xa2))
    else:
        assert Lk2.shape[0]==model.X.shape[0]; "Lk2 is not same dim as model.X"
        assert Lk2.shape[1]==xa2.shape[0]; "Lk2 is not same dim as xa2"

    K_ = model.kern.K(xa1, xa2)

    s2 = np.matrix(K_) - np.matmul(Lk1.T, Lk2)

    s2 = np.array(s2)

    # make sure the output is correct!
    assert s2.shape[0] == xa1.shape[0]; "output dim is wrong!" 
    assert s2.shape[1] == xa2.shape[0]; "output dim is wrong!" 

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

    s2 = np.array(s2)

    return s2


def KG_Mc_Input(model, Xd, Ad, lb, ub, Ns=5000, Nc=2, maxiter=80):
        """Takes a GPy model, constructs and optimzes KG and 
        returns the best xa and best KG value.
        
        ARGS
            model: GPy model
            Xd: Nx*x_dim matrix discretizing X
            Ad: Na*a_dim matrix discretizing A
            lb: lower bounds on (x,a)
            ub: upper bounds on (x,a)
            Ns: number of START points, initial random search of KG
            Nc: number of CONTINUED points, top Nc points from the Na to perform a Nelder Mead run.
            maxiter: iterations of each Nelder Mead run.
        
        RETURNS
            bestxa: the optimal xa
            bestEVI: the largest KG value
         """

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)

        assert Ns > Nc; "more random points than optimzer points"
        assert len(Xd.shape)==2; "Xd must be a matrix"
        assert len(Ad.shape)==2; "Ad must be a matrix"
        assert Xd.shape[0]>1; "Xd must have more than one row"
        assert Ad.shape[0]>1; "Ad must have more than one row"
        assert Ad.shape[1]+Xd.shape[1] == model.X.shape[1]; "Xd, Ad, must have same dim as data"
        assert lb.shape[0]==ub.shape[0]; "bounds must have same dim"
        assert lb.shape[0]==model.X.shape[1]; "bounds must have same dim as data"
        assert np.all(lb<=ub); "lower must be below upper!"

        # optimizer initial optim
        KG_Mc_Input.bestEVI = -10
        KG_Mc_Input.bestxa  = [-10]*model.X.shape[1]

        noiseVar = model.Gaussian_noise.variance[0] 

        dim_X  = Xd.shape[1]
        len_Xd = Xd.shape[0]
        len_Ad = Ad.shape[0]

        XdAd   = np.hstack([np.repeat(Xd, len_Ad, axis=0), 
                            np.tile(Ad, (len_Xd, 1))])

        # Precompute the posterior mean at X_d integrated over A.
        M_Xd = model.predict(XdAd)[0].reshape(len_Xd, len_Ad)
        M_Xd = np.mean(M_Xd, axis=1).reshape(1,-1)

        # Precompute cholesky decomposition.
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(K.shape[0]))

        Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, XdAd))

        def KG_IU(xa):
            xa = np.array(xa).reshape((1, -1))

            if np.any(xa<lb) or np.any(xa>ub):
                return(1000000)
            else:
                # The current x with all the A samples
                tile_x = np.tile(xa[:,:dim_X], (len_Ad, 1))
                newx_Ad = np.hstack([tile_x, Ad])

                # The mean integrated over A.
                # M_Xd is precomputed
                M_x  = np.mean(model.predict(newx_Ad)[0])
                MM   = np.c_[M_x, M_Xd].reshape(-1)

                # The mean warping integrated over Ad
                S_Xd = COV(model, xa, XdAd, chol_K, Lk2)
                S_Xd = S_Xd.reshape(len_Xd, len_Ad)
                S_Xd = np.mean(S_Xd, axis=1).reshape(1,-1)
                S_x = np.mean(COV(model, xa, newx_Ad, chol_K))
                SS  = np.c_[S_x, S_Xd].reshape(-1)

                # variance of new observation
                var_xa = VAR(model, xa, chol_K) + noiseVar
                inv_sd = 1/np.sqrt(var_xa).reshape(())

                SS = SS*inv_sd

                # Finally compute KG!
                out  = KG(MM, SS)
                if out > KG_Mc_Input.bestEVI:
                    KG_Mc_Input.bestEVI = out
                    KG_Mc_Input.bestxa = xa

                # print(out)
                return -out

        # Optimize that badboy! First do Ns random points, then take best Nc results 
        # and do Nelder Mead starting from each of them.
        XA_Ns  = lhs_box(Ns, lb, ub)
        KG_Ns = np.array([KG_IU(XA_i) for XA_i in XA_Ns])
        XA_Nc = KG_Ns.argsort()[-Nc:]
        XA_Nc = XA_Ns[XA_Nc, :]

        _  = [minimize(KG_IU, XA_i, method='nelder-mead', options={'maxiter': maxiter}) for XA_i in XA_Nc]
        
        return KG_Mc_Input.bestxa, KG_Mc_Input.bestEVI


# These functions still assume 0...100 and MUSIG etc
def Gen_Sample(Dist, N=500):
    """Given a pmf generates samples assuming pmf is over equally
    spaced points in 0,...,100
    
    ARGS
     Dist: vector of probalilities
     N: sample size
    
    RETURNS
     val: samples from set of qually spaced points in 0,..,100
    """

    elements = np.linspace(0, 100, len(Dist))
    probabilities = Dist/np.sum(Dist)
    val = np.random.choice(elements, N, p=probabilities)
    return val        
    

def Fit_Inputs(Y, MUSIG0, MU, SIG):
    """
    Computes MU and margs_mu from observations
    
    ARGS
        Y: observations
        MUSIG0:
        MU:
        SIG:
    
    RETURNS
        MU: posterior mean?
        marg_mu: marginal mean
    """

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


def sample_predict_dens(Data, N, MUSIG0_L, MU_L, SIG_L):
        """Generate samples from the haluciated future pmf
        
        ARGS
         Data: matrix of observations of info sources
         N: sample size
         MUSIG0_L:
         MU_L:
         SIG_L:
        
        RETURNS
         zn: samples from updated distro
        """

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
                      Na=102, 
                      Nr=103):

    """
    Optimizes the test function integrated over IU_dims. The integral
    is also changing over time and learnt.

    ARGS
        test_func: callable function, returns scalar
        lb: lower bounds on (x,a) vector input to test_fun
        ub: upper bounds on (x,a) vector input to test_fun
        IU_dims: int, list of dims of (x,a) that are 'a' (eg 2nd and 3rd dim [1,2])
        inf_src: callable function returning info source data
        n_fun_init: number of inital points for GP model
        n_inf_init: number of intial points for info source
        Budget: total budget of calls to test_fun and inf_src
        Nx: int, discretization size of X
        Na: int, discretization size of A
        Nr: int, discretization size of ?

    RETURNS
        X: observed test_func inputs
        Y: observed test_func outputs
        Data: list of array of inf_src observations
        rec_X: the recomended X values
    """

    IU_dims = np.array(IU_dims)

    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    assert lb.shape[0]==ub.shape[0]; "bounds must be same shape"
    assert np.all(lb<=ub); "lower must be below upper!"

    assert ub.shape[0] == lb.shape[0]; "lb and ub must be the same shape!"
    assert np.all(IU_dims<ub.shape[0]); "IU_dims out of too high!"
    assert np.all(IU_dims>=0); "IU_dims too low!"

    # TODO: implement lb and ub, currently 0,100 is hardcoded.
    # TODO: implelemt arbitrary dimensions of x, a, currently dim_x=1, dim_a=2 is hardcoded.
    # TODO: rename application specific variables "MU_L", "SIG_L", "MUSIG0" etc to generalised "A".
    # TODO: use IU_dims! currently IU_dims=[1,2] is assumed.
    # TODO: future work, implement different priors/posteriors for IU? Beta?


    x = np.linspace(0, 100, Nx) #vector of input variable
    dim = len(IU_dims)

    # Make lattice over IU parameter space.
    MU = np.linspace(0, 100, Na)
    SIG = np.linspace(0.025, 100, Na)
    rep_MU = np.repeat(MU, Na)
    rep_SIG = np.tile(SIG, Na)
    MUSIG0 = np.c_[rep_MU, rep_SIG]
    
    MU_L  = np.linspace(0,100,101)
    SIG_L = np.linspace(0.025,100,101)

    X_L   = np.repeat(MU_L, 101)
    W_L   = np.tile(SIG_L, 101)
    MUSIG0_L = np.c_[X_L, W_L]

    
    def Delta_Loss(model, Data, src, Xr, XA_grid, Nr=102, Nx=101, Nd=100):
        return(-100000000)
        """
        Compute the improvement due to queerying the info source.
        
        ARGS
            model: GPy model
            Data: n*Ns matrix, info source observations
            src: which info source to compute DL for.
            Xr: recomended X value with current data
            Nr: int
            Nx: int
            Nd: int
        
        RETURNS
         DL: the delta loss for source idx!

        """
        
        Data_src = Data[~np.isnan(Data[:,src]), src]
        
        # import pdb; pdb.set_trace()
        def W_aj(Y, a):
            """
            ?
            ARGS
                Y: array
                a: unused?
            
            RETURNS
                marg_ma_val: float
            """
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
        
        

        if not Data_src.shape==(0,):

            z1 = sample_predict_dens(Data_src, N=Nd, MUSIG0_L=MUSIG0_L, MU_L=MU_L, SIG_L=SIG_L)
            Sample_I = XA_grid[:, src+1]
            Sample_XA = XA_grid
            W_D = W_aj(Y=Data_src, a=Sample_I)
            dj = np.c_[Data_src*Nd, z1]
        else:
    
            z1 = np.random.random(Nd)*100
            Sample_I = XA_grid[:, src+1]
            Sample_XA = XA_grid
            W_D = np.array([list(np.repeat([1.0/100], Nx*Nr))])
            dj = np.vstack(z1) 

        W_D1 = W_aj(Y=dj, a=Sample_I)

        Wi = W_D1/W_D
        Mult_Input_Uncert.Wi = Wi.reshape(Nd, Nx, Nr)
        
        Prd = model.predict(np.array(Sample_XA))[0].reshape(Nx, Nr)    
        Mult_Input_Uncert.Prd = Prd
        IU_D1 = np.mean(np.multiply(Mult_Input_Uncert.Wi,Mult_Input_Uncert.Prd),axis=2)
        
        max_IU_D1 = np.max(IU_D1,axis=1)

        Prd_D = model.predict(np.array(np.c_[np.repeat(Xr, Nr),Sample[:Nr,1:3]]))[0].T
        Mult_Input_Uncert.Prd_D = Prd_D
     
        IU_D = np.mean(np.mean(np.multiply(Mult_Input_Uncert.Wi,Mult_Input_Uncert.Prd_D),axis=2),axis=1)
        DL = np.mean(max_IU_D1 - IU_D)

        return DL


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

    print("Initialization complete, budget used: ", XA.shape[0] +  Ndata, "\n")

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

        # import pdb; pdb.set_trace()

        A1_samples = A1_samples.reshape(-1, 1)
        A2_samples = A2_samples.reshape(-1, 1)
        Xd  = x.reshape(-1,1)

        XA_grid = np.c_[np.repeat(Xd, Nr, axis=0), 
                        np.tile(A1_samples, (Nx,1)),
                        np.tile(A2_samples, (Nx,1))]

        A_grid = np.c_[A1_samples, A2_samples]
        X_grid = XA[:, 0].reshape(-1, 1)

        IU = np.mean(GPmodel.predict(XA_grid)[0].reshape(Nx, Nr), axis=1)
        Xr = x[np.argmax(IU)]


        # Get KG of both simulation and Input uncertainty.
        topxa, topKG = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub, Ns=20)
        
        DL = np.array([Delta_Loss(GPmodel, Data, src, Xr, XA_grid) for src in range(dim)])
        topsrc, topDL = np.argmax(DL), np.max(DL)
        
        if topKG > topDL:
            # if simulation is better
            print("Best is simulator: ", topxa, topKG)
            new_y = test_func(topxa)

            XA = np.vstack([XA, topxa])
            Y = np.concatenate([Y, new_y])
        
        else:
            # if info source is better
            print("Best is info source: ", topsrc, topDL)
            new_d = np.array([ [np.nan]*dim ])
            new_d[0, topsrc] = inf_src(s=1, src=topsrc)
            Data = np.vstack([Data, new_d])
        
        print(" ")

    return XA, Y, Data#, rec_X 



if __name__=="__main__":

    
    # print("Calling Test Function":wq
    xa = np.random.uniform(size=(10, 3))*100
    # print(xa)
    dead = toy_func(xa, NoiseSD=0.)


    # print("\nCalling info source")
    # print("source 0")
    # print(toy_inf_src(10, 0))
    # print("source 1")
    # print(toy_inf_src(10, 1))
    

    print("\nCalling optimizer")
    X, Y, Data = Mult_Input_Uncert(test_func=toy_func,
                                   lb=toy_func.lb,
                                   ub=toy_func.ub,
                                   IU_dims=[1,2],
                                   inf_src=toy_infsrc)
    
    
    
    