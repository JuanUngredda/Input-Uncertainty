import GPy
import numpy as np
import itertools
from pyDOE import lhs
from scipy.optimize import minimize
from scipy.stats import uniform, truncnorm, norm
import matplotlib.pyplot as plt
from matplotlib import cm

# Toy test function and info source.

class testfunction(object):
    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.dx = None

    def __call__(self, x):
        raise ValueError("Function not defined")

    def get_range(self):
        return np.array([self.xmin, self.xmax])

    def check_input(self, x):
        if not x.shape[1] == self.dxa or (x > self.xmax).any() or (x < self.xmin).any():
            raise ValueError("x is wrong dim or out of bounds")
        return x

    def RandomCall(self):
        x = np.random.uniform(self.xmin, self.xmax)
        f = self.__call__(x)
        # f = self.testmode(x)
        print("\n\nFunction: {}".format(type(self).__name__))
        print("Input = {}".format(x))
        print("Output = {}".format(f))

    def testmode(self, x):
        return self.__call__(x, noise_std=0)


class GP_test(testfunction):
    """
A toy function GP

ARGS
 min: scalar defining min range of inputs
 max: scalar defining max range of inputs
 seed: int, RNG seed
 x_dim: designs dimension
 a_dim: input dimensions
 xa: n*d matrix, points in space to eval testfun
 NoiseSD: additive gaussaint noise SD

RETURNS
 output: vector of length nrow(xa)
 """

    def __init__(self, xamin, xamax, seed=11, x_dim=2, a_dim=1):
        self.seed = seed
        self.dx = x_dim
        self.da = a_dim
        self.dxa = x_dim + a_dim
        self.xmin = np.array([xamin for i in range(self.dxa)])
        self.xmax = np.array([xamax for i in range(self.dxa)])
        self.KERNEL = GPy.kern.RBF(input_dim=self.dxa, variance=10000., lengthscale=([10] * self.dxa), ARD=True)
        self.generate_function()

    def __call__(self, xa, noise_std=1):
        assert len(xa.shape) == 2, "xa must be an N*d matrix, each row a d point"
        assert xa.shape[1] == self.dxa, "Test_func: wrong dimension inputed"

        xa = self.check_input(xa)

        ks = self.KERNEL.K(xa, self.XF)
        out = np.dot(ks, self.invCZ)

        E = np.random.normal(0, noise_std, xa.shape[0])

        return (out.reshape(-1, 1) + E.reshape(-1, 1))

    def generate_function(self):
        print("Generating test function")
        np.random.seed(self.seed)

        self.XF = np.random.uniform(size=(50, self.dxa)) * (self.xmax - self.xmin) + self.xmin

        print("elf.XF",self.XF)
        mu = np.zeros(self.XF.shape[0])

        C = self.KERNEL.K(self.XF, self.XF)

        Z = np.random.multivariate_normal(mu, C).reshape(-1, 1)
        invC = np.linalg.inv(C + np.eye(C.shape[0]) * 1e-3)

        self.invCZ = np.dot(invC, Z)

class toy_infsrc():

    def __init__(self,lb =25,ub=65,n_srcs =1 ,seed=11):
        self.lb = lb
        self.ub = ub
        self.n_srcs = n_srcs
        self.seed = seed
        self.input_tags = range(n_srcs)
        self.generate()

    def __call__(self, n, src):
        assert src in self.input_tags, "info source out of range"
        rn = np.random.normal(loc=self.f_mean[src],
                              scale=self.f_cov[src, src],
                              size=n)
        return rn.reshape(-1)

    def generate(self):
        print("Generating params for " + str(self.n_srcs) + " info sources")
        np.random.seed(self.seed)
        var = np.random.random(self.n_srcs)*(20-5) + 5
        self.f_mean = np.random.random(self.n_srcs)*(self.ub-self.lb) + self.lb
        self.f_cov = np.multiply(np.identity(self.n_srcs), var)

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

    assert out>=0; "KG cannot be negative"

    return out


def KG_Mc_Input(model, Xd, Ad, lb, ub, Ns=2000, Nc=5, maxiter=80):
        """Takes a GPy model, constructs and optimzes KG and 
        returns the best xa and best KG value.
        
        ARGS
            model: GPy model
            Xd: Nx*x_dim matrix discretizing X
            Ad: Na*a_dim matrix discretizing A !!!MUST BE SAMPLES FROM POSTERIOR OVER A!!!!!
            lb: lower bounds on (x,a)
            ub: upper bounds on (x,a)
            Ns: number of START points, initial random search of KG
            Nc: number of CONTINUED points, from the Ns the top Nc points to perform a Nelder Mead run.
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
        KG_Mc_Input.bestxa  = 0

        noiseVar = model.Gaussian_noise.variance[0] 

        dim_X  = Xd.shape[1]
        Nx = Xd.shape[0]
        Na = Ad.shape[0]

        XdAd   = np.hstack([np.repeat(Xd, Na, axis=0), 
                            np.tile(Ad, (Nx, 1))])

        # Precompute the posterior mean at X_d integrated over A.
        M_Xd = model.predict(XdAd)[0].reshape(Nx, Na)
        M_Xd = np.mean(M_Xd, axis=1).reshape(1,-1)

        # Precompute cholesky decomposition.
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (0.1**2.0)*np.eye(K.shape[0]))

        Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, XdAd))

        KG_Mc_Input.calls=0

        def KG_IU(xa):

            KG_Mc_Input.calls +=1
            xa = np.array(xa).reshape((1, -1))

            if np.any(xa<lb) or np.any(xa>ub):
                return(1000000)
            else:
                # The current x with all the A samples
                tile_x = np.tile(xa[:,:dim_X], (Na, 1))
                newx_Ad = np.hstack([tile_x, Ad])

                # The mean integrated over A.
                # M_Xd is precomputed
                M_x  = np.mean(model.predict(newx_Ad)[0])
                MM   = np.c_[M_x, M_Xd].reshape(-1)

                # The mean warping integrated over Ad
                S_Xd = COV(model, xa, XdAd, chol_K, Lk2)
                S_Xd = S_Xd.reshape(Nx, Na)
                S_Xd = np.mean(S_Xd, axis=1).reshape(1,-1)
                S_x  = np.mean(COV(model, xa, newx_Ad, chol_K))
                SS   = np.c_[S_x, S_Xd].reshape(-1)

                # variance of new observation
                var_xa = VAR(model, xa, chol_K) + noiseVar
                inv_sd = 1./np.sqrt(var_xa).reshape(())

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
        XA_Ns = lhs_box(Ns, lb, ub)
        KG_Ns = np.array([KG_IU(XA_i) for XA_i in XA_Ns])
        XA_Nc = KG_Ns.argsort()[-Nc:]
        XA_Nc = XA_Ns[XA_Nc, :]

        _  = [minimize(KG_IU, XA_i, method='nelder-mead', options={'maxiter': maxiter}) for XA_i in XA_Nc]

        return KG_Mc_Input.bestxa, KG_Mc_Input.bestEVI


# TODO: this function is untested code, it requires a post_maker before it can run
def DeltaLoss(model, Data, Xd, Ad, Wd, distribution, Nd=100):
    """Takes a GPy model, constructs and evaluates DeltaLoss and 
    returns the best info sourse and Delta Loss value.

    ARGS
        model: GPy model
        Data: list of arrays of inf_source observations
        Xd: Nx*x_dim matrix discretizing X
        Ad: Na*a_dim matrix discretizing A !!!MUST BE SAMPLES FROM POSTERIOR OVER A!!!!!
        Wd: vector of weights, the probability densities that produced Ad!
        Nd: sample size for Delta loss evaluation
        distribution: string, which IU distribution to use.

    RETURNS
        topsrc: the optimal info source
        topDL: the largest DL value
    """
    # TODO: copy assert stuff from KG_Mc_Input
    # TODO: implement this babdoy!
    if distribution is "trunc_norm":
        post_maker = trunc_norm_post
    elif distribution is "beta":
        post_maker = beta_post
    elif distribution is "MUSIG":
        post_maker = MUSIG_post
    else:
        raise NotImplementedError
    
    # make a full lattice over X x A and get the GP mean at each point.
    Nx = Xd.shape[0]
    Na = Ad.shape[0]
    XdAd = np.hstack([  np.repeat(Xd, Na), np.tile(Ad, (Nx, 1)) ])

    # matrix of GP mean, each row is x_i from Xd, each column is a_i from Ad
    M_XA = model.predict(XdAd)[0].reshape(Nx, Na)

    # precompute inverse weights of Ad points.
    invWd = np.sum(Wd)/Wd
    _, _, cur_Data_sampler = post_maker(Data)

    # get the index of the current top recomended x value.
    M_X = np.mean(M_XA, axis=1)
    cur_topX_index = np.argmax(M_X)

    # loop over IU parameters / A dims / inf sources.
    DL = []
    for src in range(len(Data)):

        # loop over individual DL samples
        DL_src=[]
        for i in range(Nd):
            # sample a new observation and add it to the original Data
            tmp_Data_i = cur_Data_sampler(n=1)[src]
            tmp_Data = Data
            tmp_Data[src] = np.concatenate([tmp_Data[src], tmp_Data_i])

            # get the importance weights of the Ad points from new posterior
            tmp_post_A_dens, _, _ = post_maker(tmp_Data)
            Wi = tmp_post_A_dens(Ad)
            Wi = Wi * invWd

            # now we have weights, get the peak of reweighted GP means
            M_X_i = np.sum(M_XA*Wi, axis=1)
            DL_i  = np.max(M_X_i) - M_X_i[cur_topX_index]

            # keep this single MC sample of DL improvement
            DL_src.append(DL_i)
        
        # get the average over DL samples for this source and save in the list.
        DL.append(np.mean(DL_src))

    # get the best source and its DL.
    topsrc = np.argmax(DL)
    topDL  = np.max(DL) #- np.max(M_X)

    return topsrc, topDL

    """
    I assume trunc_norm_post/beta_post/MUSIG_post can be easily used with Delta loss by:

    DELTA LOSS ALGORITHM (Distribution agonistic)
    1. with current Data,
        a. get current functions; cur_post_A_dens, cur_A_sampler, cur_Data_sampler = trunc_norm_post(Data)
        b. get samples over A and their post_densities: A_0=cur_A_sampler(n), W_0=cur_post_A_dens(A_0), W_0 = W_0/sum(W_0)
        c. precompute inverse weights invW_0 = 1/W_0 (multiplication is faster than division, so always precompute recycled denominators)

    2. set array of delta losses: DeltaLosses = []

    3. for i in range(DeltaLoss samples):
        a. sample new data:      Data_i = cur_Data_sampler(1)
        a. get new post_density: post_A_dens_i, _, _ = trunc_norm_post(Data with Data_i)
        b. compute new weights:  W_i = post_A_dens_i(A_0), W_i = W_i/sum(W_i)
        c. DeltaLoss_i =  max_x sum(GP.mu(x, A_0) * W_i * invW_0 )  -  sum(GP.mu(Xr, A_0) * W_i * invW_0)
        c. DeltaLosses.append(  DeltaLoss_i )

    4. return mean(DeltaLossses)

    Steps 2-4 can be repeated for each info source, just change Data_i to Data_i[src], only update 
    the Data observations with partial info.

    Thereby Delta loss does not care about what kind of prior or likelihood we use, all 
    delta loss needs is weights and samples, Delta loss itself is input uncertainty 
    distribution agnostic. To generalise if we can write a function
    from Data to ( post_A_dens, post_A_sampler, post_Data_sampler),
    then we can give this to delta loss and it can draw A samples+get current weights, 
    then do MC over delta losses by: generate new Data_i -> new posterior_i -> new weight_i -> DL_i.

    So as well as "trunc_norm_post", there could be "beta_post", "trunc_norm_wishart_post", any prior 
    and likelihood combination we can be bothered to code in, delta loss will just work with any of them.
    """


# Distributions for the Input uncertainty. These three post_makers functions
# go from observed Data -> A_density, A_sampler, Data_sampler
# TODO: make sure trunc_norm_post works!
class MUSIG_post():
    """
    Given i.i.d observations, builds a posterior density and a sampler
    (which can then be used with Delta Loss).
    Inference details:
    Normal Likelihood
    Uniform prior over the input "a" and uncertainty

    ARGS
        src_data: matrix of observations for a given source
        xmin = lower bound
        xmax = upper bound


    RETURNS
        post_dens: pdf over input domain A. Method: iu_pdf
        post_dens_samples: samples over post_dens. Method: iu_pdf_sampler
        post_predict_sampler: samples from posterior predictive density ynew. Method: data_predict_sampler

    """

    def __init__(self, xmin, xmax, src_data):

        self.xmin = xmin
        self.xmax = xmax
        self.h = 101
        self.sig_arr, self.dltsig = np.linspace(1e-6, 100, self.h, retstep=True)
        self.a_arr, self.dlta = np.linspace(1e-6, 100, self.h, retstep=True)
        self.y_n1, self.dlty_n1 = np.linspace(0, 100, 5000, retstep=True)

        self.Data = src_data[~np.isnan(src_data)]
        self.norm_const()

    def __call__(self, a=5, n=100, method='iu_pdf'):
        """

        :param a: specific value of input a
        :param n: number of samples from a distribution pdf
        :param method: type of method used.
            iu_pdf: estimates the input uncertainty pdf given a data set
            iu_pdf_sampler: given the iu pdf. generate samples
            data_predict_sampler: given the predictive dist pdf. generate samples
        :return:
        """
        if method == 'iu_pdf':
            return self.post_dens(a)
        elif method == 'iu_pdf_sampler':
            return self.post_A_sampler(n)
        elif method == 'data_predict_sampler':
            return self.post_Data_sampler(n)
        else:
            raise NotImplementedError

    def add(self, datum):
        """
        including a new data point to the self.Data attribute and
        update constant for pdf's.
        :param datum: new data point
        """
        print("\n data point included")
        self.Data = np.concatenate((self.Data, datum))
        self.norm_const()

    def log_prior_A_dens(self, a):
        """
        log prior using uniform distribution
        :param a: value of input narray x sigma narray
        :return:  value of logprior
        """
        assert len(a.shape) == 2;
        "a must be a matrix"
        assert a.shape[1] == 2;
        "a must have 2 columns"
        Lprior = np.zeros(len(a))
        max_ls = self.xmax;
        min_ls = self.xmin;

        prior = np.product(1.0 * ((a > min_ls) & (a < max_ls)), axis=1)
        # avoiding numerical errors with np.log(0.0)
        Lprior[prior != 0] = np.log(prior[prior != 0])
        Lprior[prior == 0] = -np.inf
        return Lprior

    def log_lhood_d_i(self, a, data_i):
        """
        log likelihood of normal distribution
        :param a: value of input narray x sigma narray
        :param data_i: data in narray
        :return: log likelihood narray
        """
        assert len(a.shape) == 2;
        "a must be a matrix"
        assert a.shape[1] == 2;
        "a must have 2 columns"
        mu = a[:, 0]
        var = a[:, 1]
        Llikelihood_i = (-1.0 / 2) * (1.0 / var) * (data_i - mu) ** 2

        return Llikelihood_i

    def norm_const(self):
        """
        calculates normalisation constant. Particularly useful to normalise
        individual values of a from post_dens_unnormalised
        :return: normalisation constant of posterior distribution
        """
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        full_post = self.post_dens_unnormalised(Dom_crssprd)
        self.nrm_cnst = np.sum(full_post) * self.dltsig * self.dlta

    def cross_prod(self, arr_1, arr_2):
        """
        cartesian product between arrays. aux function
        :param arr_1: array
        :param arr_2: array
        :return: cartesian product ndarray
        """
        arr_1 = np.array(arr_1).reshape(-1)
        Dom_sets = [arr_1, arr_2]
        Dom_crssprd = np.array([list(i) for i in itertools.product(*Dom_sets)])
        return Dom_crssprd

    def post_dens_unnormalised(self, a):
        """
        # This implementation style means that even if there is no data,
        # the second summation term will be 0 and only the prior will contribute
        # i.e. this style uses one method for both prior and posterior, prior is NOT a special case.
        :param a: value of a
        :return: joint pdf calculated in a
        """


        log_lhood = np.sum([self.log_lhood_d_i(a, d_i) for d_i in self.Data], axis=0)
        log_post = self.log_prior_A_dens(a) + log_lhood
        self.post = np.exp(log_post)
        return self.post

    def marg_post_dens(self, a):
        """
        marginilise posterior joint distribution of input A and variance sigma of normal
        distribution

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        joint_post = self.post_dens_unnormalised(a)
        joint_post = joint_post.reshape(self.Na, len(self.sig_arr))
        return np.sum(joint_post, axis=1) * self.dltsig

    def post_dens(self, a):
        """
        Posterior marginilised density estimation over A. First models posterior over
        the parameter A and uncertainty sigma from input source. Then marginilises out sigma
        and normalise the distribution over A

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        a = np.array(a).reshape(-1)
        self.Na = len(a)
        Dom_crssprd = self.cross_prod(a, self.sig_arr)
        pdf_post = self.marg_post_dens(Dom_crssprd)
        return pdf_post / self.nrm_cnst

    def sampler(self, n, dist, domain):
        """

        :param n: number of samples
        :param dist: pdf of distribution. normalised inside the function
        :param domain: discreatised domain
        :return: set of samples
        """
        assert not len(dist) == 1, "Trying to generate samples from scalar. Hint: Insert pdf"
        dist = dist / np.sum(dist)
        probabilities = dist * (1 / np.sum(dist))
        val = np.random.choice(domain, n, p=probabilities)
        return val

    def post_A_sampler(self, n):
        """
        Sampler for posterior marginilised density over input A
        :param n: number of samples for posterior density over A
        :return: samples over domain
        """
        DomA = np.linspace(self.xmin, self.xmax, 5000)
        Dist = self.post_dens(DomA)
        return self.sampler(n, dist=Dist, domain=DomA)

    def post_Data_sampler(self, n):
        """
        Sampler for posterior predictive density ynew
        :param n: number of samples for posterior predictive density
        :return: samples over domain
        """
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        pdf_musig = self.post_dens_unnormalised(Dom_crssprd)
        pdf_yn1_musig = np.exp(self.log_lhood_d_i(Dom_crssprd, self.y_n1[:, None]))
        pdf_yn1 = np.dot(pdf_yn1_musig, pdf_musig)
        return self.sampler(n, dist=pdf_yn1, domain=self.y_n1)


def trunc_norm_post(Data, noisevar=200, lb=np.zeros(2,), ub=100*np.ones(2,)):
    """
    Given i.i.d observations, builds a posterior density and a sampler 
    (which can then be used with Delta Loss)
    ARGS
        Data: matrix of observations
        noisevar: the noise of gaussian likelihood, same noise used for all sources
        lb: lower bound on a
        ub: upper bounds on a
    
    RETURNS
        log_post: function from A to scalars
    """
    # TODO: sanity check inputs with assert statements.
    # eg assert len lb = len ub = cols Data, Data is a matrix, noisevar is a scalar

    dim_a = Data.shape[1]
    uniform_dens = np.exp(-1*np.sum(np.log(ub-lb)))
    i_var = -0.5/noisevar
    log_norm_const = np.log( 1/(2*np.pi*np.sqrt(noisevar)) )

    # Convert Data to a list of arrays
    Data = [Data[~np.isnan(Data[:,src]), src] for src in range(Data.shape[1])]

    def log_uniform_prior(a):
        """
        Given a matrix of points in A, return vetor of densities form the uniform 
        distribution.
        ARGS
            a: n*dim_a matrix
        RETURNS

        """
        assert len(a.shape)==2; "a must be a matrix"
        assert a.shape[1]==dim_a; "a must have dim_a columns"
        # TODO: return 0 if any a<lb or ub<a, enforce bounds, row wise
        return(np.repeat(uniform_dens, a.shape[0]))
    
    def log_Gauss_lhood_1D(a_i, d_src):
        """
        1 dimensional Gauss lhood. Given a single scalar parameter value a_i
        and a vector of oberservations d_src, compute the scalar log likelihood.
        Even is there is no data, d_src=[], the output is 0 as desired.

        ARGS
            a_i: scalar, one param value for single source
            d_src: vector, observations from single source
        
        RETURNS
            output: scalar log likelihood
        """
        assert a_i.shape==(); "a_i must be a scalar"
        assert len(d_src.shape)<=1; "d_src must be empty or vector"

        sq = np.sum([(a_i-di)**2 for di in d_src])
        output = log_norm_const + sq * i_var
        return(output)
    
    def log_Gauss_lhood(a):
        """
        Given a matrix of points in A, compute the likelihood of each row 
        accounting for all of the obersations in Data.

        ARGS
            a: n*dim_a matrix

        RETURNS
            LH: length n vector of lieklihoods
        """

        assert len(a.shape)==2; "a must be a matrix"
        assert a.shape[1]==dim_a; "a have dim_a columns"
        a_dims = range(dim_a)
        a_rows = range(a.shape[0])
        LH = [[log_Gauss_lhood_1D(a[j,i], Data[i]) for i in a_dims] for j in a_rows]
        LH = np.sum(np.array(LH), axis=1)
        return(LH)

    def post_A_dens(a):
        """
        Given matrix a, and Data, compute posterior density of a.
        ARGS
            a: n*dim_a matrix
        RETURNS
            density: length n vector of posterior densities
        """

        assert len(a.shape)==2; "a must be a matrix"
        assert a.shape[1]==lb.shape[0]; "a must be same dim as lb"

        # TODO: matrix input -> vector output
        log_post = log_uniform_prior(a) + log_Gauss_lhood(a)
        density = np.exp(log_post)
        return(density)
    
    def post_A_sampler(n):
        """
        Get samples from posterior over A
        """
        # sample poitns from current posterior over a truncated normal
        # TODO: implelemnt!
        return(0)
        
    def post_Data_sampler(n):
        """
        This function samples new "Data" values given the current posterior distribtion
        over a. So first sample an 'a' value from the posterior, then add on 
        np.random.normal(scale=np.sqrt(noisevar), size=n) to get new fake "Data".
        """
        # TODO: get truncated normal samples from the posterior
        post_a_samples = post_A_sampler(n)
        Data_samples = post_a_samples + np.random.normal(size=n, scale=np.sqrt(noisevar))
        return(Data_samples)
    
    return post_A_dens, post_A_sampler, post_Data_sampler
    
# TODO: (not urgent) implement beta_post
def beta_post(Data):
    """
    Beta distribution for input unceraitnty
    """

    def post_A_dens():
        return 0
    
    def post_A_sampler(n):
        return(0)
    
    def post_Data_sampler(n):
        return(0)
    
    return post_A_dens, post_A_sampler, post_Data_sampler

# TODO: adapt code from Gen_Sample, Fit_Inputs, sample_predict_dens
def MUSIG_post(Data):

    """
    Gauss mean and Var distribution for input unceraitnty
    Fit_Inputs, Gen_Sample, sample_predict_dens can all be put in here.
    """

    def log_prior_A_dens(a):
        """
        log of uniform distribution defined over the dominion of the test function

        :param a: nd array of evaluations for prior
        :return: value for log prior distribution.
        """
        assert len(a.shape) == 2;
        "a must be a matrix"
        assert a.shape[1] == 2;
        "a must have 2 columns"

        max_ls = 100.0;
        min_ls = 0.0;
        Lprior = np.log(1.0 * ((a > min_ls) & (a < max_ls)))
        return Lprior
    
    def log_lhood(a, data_i):
        # TODO: implement the likelihood foa single point
        return(0)

    def post_A_dens(a):
        # This implementation style means that even if there is no data,
        # the second summation term will be 0 and only the prior will contribute
        # i.e. this style uses one method for both prior and posterior, prior is NOT a special case.
        log_post = log_prior_A_dens(a) + np.sum([log_lhood(a, d_i) for d_i in Data], axis=1)
        return(np.exp(log_post))
    
    def post_A_sampler(n):
        # TODO: implement
        return(0)
    
    def post_Data_sampler(n):
        # TODO imeplement
        return(0)
    
    return post_A_dens, post_A_sampler, post_Data_sampler


# The actual optimizer that puts all the peices together!


def Mult_Input_Uncert(sim_fun, lb, ub, dim_X, inf_src,
                      distribution = "MUSIG",
                      n_fun_init = 10,
                      n_inf_init = 0,
                      Budget = 100,
                      Nx = 101,
                      Na = 102,
                      Nd = 103):

    """
    Optimizes the test function integrated over IU_dims. The integral
    is also changing over time and learnt.

    ARGS
        sim_fun: callable simulator function, input (x,a), returns scalar
        lb: lower bounds on (x,a) vector input to sim_fun
        ub: upper bounds on (x,a) vector input to sim_fun
        dim_X: int, how many of input dims to sim_fun are X, rest are assumed A
        inf_src: callable function returning info source data
        distribution: which prior/posterior to use for the uncertain parameters
        n_fun_init: number of inital points for GP model
        n_inf_init: number of intial points for info source
        Budget: total budget of calls to test_fun and inf_src
        Nx: int, discretization size of X
        Na: int, sample size for MC over A
        Nd: int, sample size for Delta Loss

    RETURNS
        X: observed test_func inputs
        Y: observed test_func outputs
        Data: list of array of inf_src observations
        TODO: any extra tracking variables!
        """

    print("lb", lb)
    print("ub", ub)
    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    assert dim_X < lb.shape[0], "More X dims than possible"
    assert lb.shape[0]==ub.shape[0], "bounds must be same shape"
    assert np.all(lb<=ub), "lower must be below upper!"

    assert ub.shape[0] == lb.shape[0], "lb and ub must be the same shape!"
    # assert np.all(IU_dims<ub.shape[0]); "IU_dims out of too high!"
    # assert np.all(IU_dims>=0); "IU_dims too low!"


    # set the distribution to use for A dimensions.
    if distribution is "trunc_norm":
        post_maker = trunc_norm_post
    elif distribution is "beta":
        post_maker = beta_post
    elif distribution is "MUSIG":
        post_maker = MUSIG_post
    else:
        raise NotImplementedError


    # we will need this for making discretizations.
    X_sampler = lambda n: lhs_box(n, lb[:dim_X], ub[:dim_X])


    # Initilize GP model
    XA  = lhs_box(n_fun_init, lb, ub)
    Y   = sim_fun(xa=XA)
    ker = GPy.kern.RBF(input_dim=lb.shape[0], variance=1., lengthscale=(ub-lb)*0.1, ARD=True)

    # Initilize input uncertainty data via round robin allocation
    dim_A  = lb.shape[0] - dim_X
    alloc  = np.arange(n_inf_init)%dim_A
    alloc  = [np.sum(alloc=i) for i in range(dim_A)]
    Data   = [inf_src(alloc[i], i) for i in range(dim_A)]

    # this can be called at any time to get the number of Data collected
    Ndata = lambda: np.sum([d_src.shape[0] for d_src in Data])

    print("Initialization complete, budget used: ", n_fun_init + n_inf_init, "\n")

    # TODO: add tracking such as timings, Xr, OC, hyperparameters, include them all in the returned outputs

    # Let's get the party started!
    while XA.shape[0] +  Ndata() < Budget:

        print("Iteration ", XA.shape[0] + Ndata() + 1, ":")

        # Fit model to simulation data.
        GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1,1), ker, noise_var=0.01)

        # Discretize X by lhs and discretize A with posterior samples as required.
        X_grid = X_sampler(Nx)

        # KG+DL take a standard unweighted average over A_grid, i.e. A_grid must
        # be samples from posterior over A! Don't use linspace!
        A_density, A_sampler, _ = post_maker(Data)
        A_grid = A_sampler(Na)
        W_A    = A_density(A_grid)
        W_A    = W_A*(1 / np.sum(W_A))

        # Get KG of both simulation and Input uncertainty.
        topxa, topKG  = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub)
        topsrc, topDL = DeltaLoss(GPmodel, Data, X_grid, A_grid, W_A, distribution, Nd)

        if topKG > topDL:
            # if simulation is better
            print("Best is simulator: ", topxa, topKG)
            new_y = sim_fun(topxa)

            XA = np.vstack([XA, topxa])
            Y = np.concatenate([Y, new_y])

        else:
            # if info source is better
            print("Best is info source: ", topsrc, topDL)
            new_d = inf_src(s=1, src=topsrc)
            Data[topsrc] = np.concatenate([Data[topsrc], new_d])

        print(" ")

    # TODO: return extra variables to be tracked, Opportunity cost, recomended X, KG/DL time series, hyper parameters
    return XA, Y, Data


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# OLD CODE

# TODO: Rewrite as post_A_dens, post_A_sampler, post_Data_Sampler = MUSIG(Data)
def Gen_Sample(Dist, N=500):
    """
    Given a pmf generates samples assuming pmf is over equally
    spaced points in 0,...,100

    ARGS
     Dist: vector of probalilities
     N: sample size

    RETURNS
     val: samples from set of qually spaced points in 0,..,100
    """

    elements = np.linspace(0, 100, len(Dist))
    probabilities = Dist*(1/np.sum(Dist))
    val = np.random.choice(elements, N, p=probabilities)
    return val        
    

def Fit_Inputs(Y, MUSIG0, MU, SIG):
    """
    Computes MU and margs_mu from observations
    
    ARGS
        Y: info source observations
        MUSIG0: discretization over A1 x A2
        MU: discretiation over A1
        SIG: discretization over A2
    
    RETURNS
        MU: posterior mean?
        marg_mu: marginal mean
    """

    Y = np.array(Y)
    Y = list(Y[~np.isnan(Y)])
    # def Distr_Update():

    inv_var = 0.5/MUSIG0[:,1]
    sum_sq = np.sum( (np.matrix(MUSIG0[:,0]).T  - Y)**2.0, axis=1)
    norm_const = 1.0/np.sqrt(2*np.pi*MUSIG0[:,1])**len(Y)

    L = norm_const * np.exp( 1- sum_sq * inv_var )

    L = L.reshape(len(MU),len(SIG))
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

        Data = Data[~np.isnan(Data)]
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


def old_Mult_Input_Uncert(sim_fun, lb, ub, IU_dims, inf_src,
                      distribution="trunc_norm",
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
        sim_fun: callable simulator function, input (x,a), returns scalar
        lb: lower bounds on (x,a) vector input to test_fun
        ub: upper bounds on (x,a) vector input to test_fun
        IU_dims: int, list of dims of (x,a) that are 'a' (eg 2nd and 3rd dim [1,2])
        inf_src: callable function returning info source data
        distribution: which prior/posterior to use for the uncertain parameters
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
    # TODO: use IU_dims, The optimizer needs to know which dims of sim_fun are X and which are A
    # TODO: restructure 'Data' to be a list of arrays rather than a matrix of nan/floats?

    # TODO: generalize dimension and bounds
    x = np.linspace(0, 100, Nx) #vector of input variable
    dim = len(IU_dims)

    # Make lattice over IU parameter space.
    # TODO: generalise to arbitrary A dims/info sources
    # TODO: refresh the lattice every iteration?
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

    # TODO: make this function standalone, callable from outside of Multi_Inpur_Uncert, just like KG_Mc_Input
    # TODO: make this function IU distribution agnostic as described below trunc_norm_post function
    def Delta_Loss(model, Data, src, Xr, XdAd, Nr=102, Nx=101, Nd=100):
        # return(-100000000)
        """
        Compute the improvement due to querying an info source.

        ARGS
            model: GPy model
            Data: n*Ns matrix, info source observations
            src: which info source to compute DL for.
            Xr: recomended X value with current data
            XdAd: discretization over X x A
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

            # TODO: split up into multiple lines
            expo = np.exp(np.vstack(-(1.0/(2.0*SIG)))*np.hstack(np.sum(np.split((d-MU)**2,dimY,axis=0),axis=1)))
            consts =  np.vstack((1.0/np.sqrt(2*np.pi*SIG))**N)
            L = np.split(expo*consts, dimY, axis=1)
            marg_mu_dist = np.sum(L,axis=1)*(SIG[1]-SIG[0])

            C = np.sum(marg_mu_dist,axis=1)*(MU[1]-MU[0])
            marg_mu_dist = marg_mu_dist*(1/np.vstack(C))

            # TODO: split up into multiple lines
            expo = np.exp(np.vstack(-(1.0/(2.0*SIG)))*np.hstack(np.sum(np.split((d-a)**2,dimY,axis=0),axis=1)))
            consts =  np.vstack((1.0/np.sqrt(2*np.pi*SIG))**N)
            L = np.split(expo*consts, dimY, axis=1)
            marg_mu_val = np.sum(L,axis=1)*(SIG[1]-SIG[0])*(1/np.vstack(C))

            return marg_mu_val

        Xr = Xr.reshape((1,-1))
        dim_X = Xr.shape[1]

        # We need two functions, a posterior density and a posterior sampler.
        # TODO: merge both of these cases into one: sample_predict_dens also handles prior.
        if not Data_src.shape[0]==0:
            # If we have data, sample from posterior
            # TODO: replace MU, MUSIG etc with Ad as appropriate, generalise A dimension
            z1 = sample_predict_dens(Data_src, N=Nd, MUSIG0_L=MUSIG0_L, MU_L=MU_L, SIG_L=SIG_L)
            Ad = XdAd[:, src+dim_X]
            W_D = W_aj(Y=Data_src, a=Ad)
            dj = np.c_[Data_src*Nd, z1]

        else:
            # If we have no data, sample from prior
            z1 = np.random.random(Nd)*100
            Ad = XdAd[:, src+dim+X]
            # TODO: generalize bound
            W_D = np.array([list(np.repeat([1.0/100], Nx*Nr))])
            dj = np.vstack(z1)

        W_D1 = W_aj(Y=dj, a=Ad)

        Wi = W_D1/W_D
        Wi = Wi.reshape(Nd, Nx, Nr)

        M_XdAd = model.predict(XdAd)[0].reshape(Nx, Nr)

        IU_D1 = np.mean(np.multiply(Wi, M_XdAd), axis=2)
        max_IU_D1 = np.max(IU_D1, axis=1)

        # TODO: generalise dimension change 1:3 to dim_X:(dim_X+dim_A)
        M_Xr = model.predict(np.array(np.c_[np.repeat(Xr, Nr), XdAd[:Nr,1:3]]))[0].T


        IU_D = np.mean(np.mean(np.multiply(Wi, M_Xr), axis=2), axis=1)
        DL = np.mean(max_IU_D1 - IU_D)

        return DL


    #=============================================================================================
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #=============================================================================================

    # Initilize GP model
    XA = lhs(3, n_fun_init)*100
    Y  = sim_fun(xa=XA)
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
        # TODO: generalise to arbitrary IU variables
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
        topxa, topKG = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub)

        DL = np.array([Delta_Loss(GPmodel, Data, src, Xr, XA_grid) for src in range(inf_src.n_srcs)])
        topsrc, topDL = np.argmax(DL), np.max(DL)

        if topKG > topDL:
            # if simulation is better
            print("Best is simulator: ", topxa, topKG)
            new_y = sim_fun(topxa)

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
    verbose = True
    xdim = 1
    adim = 1
    dim = xdim + adim

    toy_func = GP_test(xamin=0, xamax=100, seed=11, x_dim=xdim, a_dim=adim)

    if verbose == True:
        h = 30
        somelists = [np.linspace(0, 100, h) for i in range(dim)]
        crssprd = np.array([list(i) for i in itertools.product(*somelists)])

        print("crssprd", crssprd)
        P = toy_func(crssprd,  noise_std=0)

        print("P", P)
        X, Y = np.meshgrid(somelists[0], somelists[0])

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=(15, 5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # plot a 3D surface like in the example mplot3d/surface3d_demo
        surf = ax.plot_surface(X, Y, np.array(P).reshape(h, h), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.clabel(surf, inline=1, fontsize=10)
        plt.xlabel('X')
        plt.ylabel('Theta')
        plt.show()

    print("\nCalling info source")
    input_source = toy_infsrc(n_srcs =2)

    n1 = input_source(n =500, src=0)
    n2 = input_source(n=500, src=1)

    if verbose == True:
        plt.hist(n1)
        plt.hist(n2)
        plt.show()

    print("\nCalling optimizer")
    X, Y, Data = Mult_Input_Uncert(sim_fun=toy_func,
                                   lb=toy_func.xmin,
                                   ub=toy_func.xmax,
                                   dim_X=toy_func.dx,
                                   inf_src=toy_infsrc,
                                   distribution="MUSIG")
    
    
# This stuff goes into a new file problem_runner.py or jupyter notebook
# from input_uncertainty_general import Multi_Input_Incertn
# from testproblems import ambulance, patient_source
# from testproblems import ATO, price_source
# from testproblems import toy_fun, toy_source
# import pickle

# id = np.random.uniform(10)
# output = Mult_Input_Uncert(test_func=myambfun,
#                            lb=myambfun.lb,
#                            ub=myambfun.ub,
#                            dim_X=6,
#                            inf_src=patient_distro)
    
# pickle.dump(output, open("mysavefile", ,".pkl", "wb"))
