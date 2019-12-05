import numpy as np
import GPy

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

    def __init__(self, xamin=[0,0], xamax=[100,100], seed=11, x_dim=1, a_dim=1):
        self.seed = seed
        self.dx = x_dim
        self.da = a_dim
        self.dxa = x_dim + a_dim

        self.xmin = np.zeros((self.dx,))
        self.xmax = np.ones((self.dx,))*xamax[:self.dx]

        # uncertainty parameter
        self.amin = np.zeros((self.da,))
        self.amax = np.ones((self.da,))*xamax[self.da:]

        self.xamin = np.concatenate((self.xmin,self.amin))
        self.xamax = np.concatenate((self.xmax,self.amax))

        vr = 1.
        ls = 10
        self.HP =  [vr,ls]
        self.KERNEL = GPy.kern.RBF(input_dim=self.dxa, variance=vr, lengthscale=([ls] * self.dxa), ARD=True)
        self.generate_function()

    def __call__(self, x, w, noise_std=1):
        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert len(w.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == self.dx, "Test_func: wrong dimension inputed"
        assert w.shape[1] == self.da, "Test_func: wrong dimension inputed"

        xw = np.c_[x,w]
        ks = self.KERNEL.K(xw, self.XF)
        out = np.dot(ks, self.invCZ)

        #print("noise_std",noise_std)
        E = np.random.normal(0, noise_std, xw.shape[0])
        #rint("E",E)
        return (out.reshape(-1, 1) + E.reshape(-1, 1))

    def generate_function(self):
        print("Generating test function")
        np.random.seed(self.seed)

        self.XF = np.random.uniform(size=(50, self.dxa)) * (self.xmax - self.xmin) + self.xmin


        mu = np.zeros(self.XF.shape[0])

        C = self.KERNEL.K(self.XF, self.XF)

        Z = np.random.multivariate_normal(mu, C).reshape(-1, 1)
        invC = np.linalg.inv(C + np.eye(C.shape[0]) * 1e-3)

        self.invCZ = np.dot(invC, Z)

class toyfun(testfunction):


    def __init__(self):

        self.dx = 1
        self.da = 1
        self.dxa = self.dx + self.da

        self.xmin = np.zeros((self.dx,))
        self.xmax = np.ones((self.dx,))

        # uncertainty parameter
        self.amin = np.zeros((self.da,))
        self.amax = np.ones((self.da,))

        self.xamin = np.concatenate((self.xmin,self.amin))
        self.xamax = np.concatenate((self.xmax,self.amax))

        self.noise_std = 0


    def __call__(self, x, w, *args, **kwargs):
        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert len(w.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == self.dx, "Test_func: wrong dimension inputed"
        assert w.shape[1] == self.da, "Test_func: wrong dimension inputed"

        #
        # ARGS
        #  x: scalar decision variable
        #  w: scalaer input parameter
        #
        # RETURNS
        #  y: scalar output function
        x = x.reshape(-1)
        w = w.reshape(-1)

        r = np.sqrt(((x - 0.2) ** 2 + (w - 0.2) ** 2))

        y = np.cos(r * 2. * np.pi) / (r + 2)

        return (y)

class toysource():

    def __init__(self,lb =-5,ub=5,d=1):
        self.lb = lb
        self.ub = ub
        self.f_mean = 40*np.ones(d)#np.random.random(d)
        self.f_cov = np.repeat(np.array([10]), d)
        self.n_srcs = d

    def __call__(self, n, src, *args,**kwargs):
        return self.f_mean[src] + np.random.normal(size=(n))*np.sqrt(self.f_cov[src])





