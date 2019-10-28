import GPy
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm, norm
import sys


# Toy test function
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
    assert len(xa.shape)==2; "xa must be an N*dimx matrix"
    # assert xa.shape[1]==3; "xa must be 3d"

    dimx = xa.shape[1]
     
    toy_func.lb    = np.zeros((dimx,))
    toy_func.ub    = np.ones((dimx,))*100
    
    KERNEL = GPy.kern.RBF(input_dim=dimx, variance=10000., lengthscale=(dimx*[10]), ARD = True)
    
    if gen or not hasattr(toy_func, "invCZ"):
        # If the test function has not been generated then generate it.

        print("Generating test function")
        np.random.seed(gen_seed)

        X0 = lhs_box(200*dimx, toy_func.lb, toy_func.ub)

        # XF = np.array([[i,j,k] for i in X0 for j in F1 for k in F2])
        mu = np.zeros( X0.shape[0] )
        C  = KERNEL.K(X0, X0)

        Z = np.random.multivariate_normal(mu, C).reshape(-1,1)
        invC = np.linalg.inv(C + np.eye(C.shape[0])*1e-3)

        toy_func.invCZ = np.dot(invC, Z)
        toy_func.XF    = X0


    # if xa.shape==(3,) or xa.shape==(3): xa = xa.reshape(1, 3)

    # assert len(xa.shape)==2, "xa must be an N*3 matrix, each row a 3d point"
    # assert xa.shape[1]==3, "Test_func: xa must 3d"

    ks = KERNEL.K(xa, toy_func.XF)
    out = np.dot(ks, toy_func.invCZ)

    E = np.random.normal(0, NoiseSD, xa.shape[0])

    return (out.reshape(-1,1) + E.reshape(-1, 1))


# Utilities, these work as standalone functions.
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


def KG(mu, sig):
    """
    Takes a set of intercepts and gradients of linear functions and returns
    the average hieght of the max of functions over Gaussain input.
    
    ARGS
        mu: length n vector, initercepts of linear functions
        sig: length n vector, gradients of linear functions
    
    RETURNS
        out: scalar value is gaussain expectation of epigraph of lin. funs
    """

    mu = np.squeeze(mu)
    sig = np.squeeze(sig)

    assert len(mu.shape)==1; "mu must be 1d vector"
    assert len(sig.shape)==1; "sig must be 1d vector"
    assert len(mu)==len(sig); "mu and sig must be same length"

    # TODO: make this not shit
    # temporary hack to get me started
    out = mu.reshape(-1,1) + sig.reshape(-1,1)*np.random.normal(size=(1,1000))
    out = np.max(out, axis=0)
    return np.mean(out) - np.max(mu)

    n = len(mu)
    O = sig.argsort()
    a = mu[O]
    b = sig[O]


    A=[0]
    C=[-np.inf]
    while A[-1]<n-1:
        s = A[-1]
        si = range(s+1,n)
        Ci = -(a[s]-a[si])/(b[s]-b[si])
        bestsi=np.argmin(Ci)
        C.append(Ci[bestsi])
        A.append(si[bestsi])

    C.append(np.inf)

    cdf_C = norm.cdf(C)
    diff_CDF = cdf_C[1:] - cdf_C[:-1]

    pdf_C = norm.pdf(C)
    diff_PDF = pdf_C[1:] - pdf_C[:-1]

    out = np.sum( a[A]*diff_CDF - b[A]*diff_PDF ) - np.max(mu)

    assert out>=0; "KG cannot be negative"

    return out


def get_best_KG(model, Xd, lb, ub, Ns=2000, Nc=5, maxiter=80):
    """
    Takes a GPy model, constructs and optimzes KG and 
    returns the best xa and best KG value.
    
    ARGS
        model: GPy model
        Xd: Nx*x_dim matrix discretizing X
        lb: lower bounds on (x,a)
        ub: upper bounds on (x,a)
        Ns: number of START points, initial  search of KG
        Nc: number of CONTINUED points, from the Ns the top Nc points to perform a Nelder Mead run.
        maxiter: iterations of each Nelder Mead run.
    
    RETURNS
        best_x: the optimal x
        best_KG: the largest KG value
    """

    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    assert Ns > Nc; "more random points than optimzer points"
    assert len(Xd.shape)==2; "Xd must be a matrix"
    assert Xd.shape[0]>1; "Xd must have more than one row"
    assert lb.shape[0]==ub.shape[0]; "bounds must have same dim"
    assert lb.shape[0]==model.X.shape[1]; "bounds must have same dim as data"
    assert np.all(lb<=ub); "lower must be below upper!"

    # optimizer initial optim, updated as evaluate_KG is called
    get_best_KG.best_KG = -10
    get_best_KG.best_x  = 0

    # get noise var from GPy model
    noiseVar = model.Gaussian_noise.variance[0] 
    
    # Precompute the posterior mean at X_d
    M_Xd = model.predict(Xd)[0]

    # Precompute cholesky decomposition.
    K = model.kern.K(model.X, model.X)
    chol_K = np.linalg.cholesky(K + (0.01**2.0)*np.eye(K.shape[0]))
    Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, Xd))

    get_best_KG.calls=0

    def evaluate_KG(xa):

        get_best_KG.calls +=1
        xa = np.array(xa).reshape((1, -1))

        if np.any(xa<lb) or np.any(xa>ub):
            return(1000000)
        else:

            # The change in post mean at Xd caused by xa
            SS = COV(model, xa, Xd, chol_K, Lk2).reshape((-1))

            
            # variance of new observation
            var_xa = VAR(model, xa, chol_K) + noiseVar
            inv_sd = 1./np.sqrt(var_xa).reshape(())

            SS = SS*inv_sd

            # M_xa = model.predict(xa)[0]

            # Finally compute KG!
            out  = KG(M_Xd, SS)
            
            if out > get_best_KG.best_KG:
                get_best_KG.best_KG = out
                get_best_KG.best_x = xa

            # the optmizer is doing minimisation
            return -out

    # Optimize that badboy! First do Ns random points, then take best Nc results 
    # and do Nelder Mead starting from each of them.
    X_Ns = lhs_box(Ns, lb, ub)
    KG_Ns = np.array([evaluate_KG(X_i) for X_i in X_Ns])
    X_ind = KG_Ns.argsort()[:Nc]
    X_Nc = X_Ns[X_ind, :]

    # Run the optimizer starting from the best Nc points, the eavluate_KG function
    # internally keeps track of the best point so far.
    _  = [minimize(evaluate_KG, X_i, method='nelder-mead', options={'maxiter': maxiter}) for X_i in X_Nc]

    return get_best_KG.best_x, get_best_KG.best_KG


def plot_GP(model, lb, ub, ax=None):
    """
    Plots the 1D GP 
    Arguments:
        model: GPy model
        lb: lower bound of X
        ub: upper bound of X
        ax: optional, matplotlib axes object

    Returns:
        ax: maplotlib axes object
    """

    if len(lb)>1:
        print("plotting only workds for 1D fuinctions. Skipping plotting")

    else:
        if ax is None:
            _,ax = plt.subplots(1,1,figsize=(12,6))
        else:
            ax.clear()
    
        X_test = np.linspace(lb, ub, num=100)
        mu_test, var_test = model.predict(X_test)

        se_test = 10*np.sqrt( var_test )
        
        X_test = X_test.reshape(-1)
        mu_test = mu_test.reshape(-1)
        se_test = se_test.reshape(-1)

        # plot the GP mean and uncertainty
        ax.plot(X_test, mu_test, color="orange", label="GP", zorder=1)
        ax.fill_between(X_test, mu_test+se_test, mu_test-se_test, color="orange", alpha=0.2)

        # plot the predicted peak
        i = mu_test.argmax()
        ax.scatter(X_test[i], mu_test[i], color="k", label="predicted peak", marker="*", zorder=10, s=300)
        
        # plot datapoints and newest point
        ax.scatter(model.X, model.Y, color="blue", label="Data", zorder=5)
        ax.scatter(model.X[-1], model.Y[-1], color="red", label="Newest data", zorder=11)

        # the legend is made automatiaclly from the labels given to plotting commands :)
        ax.legend()

        plt.pause(0.25)
        plt.show()


# The actual optimizer that puts all the peices together!
def KGBO(test_fun, 
         lb, 
         ub, 
         n_init=10, 
         n_final=100,
         Nx=101,
         plotting=False):

    """
    Optimizes the test function within the bounds lb < x < ub.

    ARGS
        test_fun: callable simulator function, input (x,a), returns scalar
        lb: lower bounds on (x,a) vector input to sim_fun
        ub: upper bounds on (x,a) vector input to sim_fun
        n_init: number of inital points for GP model
        n_final: total budget of calls to test_fun and inf_src
        Nx: int, discretization size of X
        plotting: bool, show pretty pictures?

    RETURNS
        X: observed test_func inputs
        Y: observed test_func outputs
        TODO: any extra tracking variables!
    """


    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    assert lb.shape[0]==ub.shape[0]; "bounds must be same shape"
    assert np.all(lb<=ub); "lower must be below upper!"


    # we will need this for making discretizations.
    X_sampler = lambda n: lhs_box(n, lb, ub)

    if plotting:
        _, ax = plt.subplots(1,1,figsize=(12,6))
        plt.ion()

   
    ####################################################################
    ############ OPTIMIZATION INITIALISE ###############################
    # Initilize the data and the GP model
    X   = lhs_box(n_init, lb, ub)
    Y   = test_fun(X)
    ker = GPy.kern.RBF(input_dim=lb.shape[0], variance=1., lengthscale=(ub-lb)*0.1, ARD=True)
    GPmodel = GPy.models.GPRegression(X, Y.reshape(-1,1), ker, noise_var=0.01)
    if plotting:
        _ = plot_GP(GPmodel, lb, ub, ax)
        plt.show()

    print("Initialization complete, budget used: ", n_init, "\n")


    ####################################################################
    ############ OPTIMIZATION ITERATE ##################################
    
    # TODO: add tracking such as timings, Xr, OC, hyperparameters, include them all in the returned outputs
    # Iterating through the budget, let's get the party started! 
    while X.shape[0] < n_final:

        print("Iteration ", X.shape[0] + 1, ":")

        # Discretize X by lhs
        X_grid = X_sampler(Nx)

        # Get x with the best KG 
        top_x, top_KG  = get_best_KG(GPmodel, X_grid, lb, ub)
        
        print("Best x and KG: ", top_x, top_KG)
        new_y = test_fun(top_x)

        X = np.vstack([X, top_x])
        Y = np.concatenate([Y, new_y])
        
        # Fit model to simulation data.
        GPmodel = GPy.models.GPRegression(X, Y.reshape(-1,1), ker, noise_var=0.01)
        if plotting:
            _ = plot_GP(GPmodel, lb, ub, ax)

    # TODO: return extra variables to be tracked, Opportunity cost, recomended X, KG/DL time series, hyper parameters
    return X, Y


if __name__=="__main__":

    
    x = np.random.uniform(size=(10, 1))*100

    # initialise the test function
    _ = toy_func(x, NoiseSD=0.)
    

    print("\nCalling optimizer")
    X, Y = KGBO(test_fun=toy_func,
                      lb=toy_func.lb,
                      ub=toy_func.ub,
                      n_init=5,
                      n_final=25,
                      plotting=True
                      )
