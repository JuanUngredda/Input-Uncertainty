import numpy as np
from scipy.stats import norm
from numba import jit
import time
mathpi = 3.141592653589793


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
        x = x.reshape((-1))
        if not x.shape[0] == self.dx or any(x > self.xmax) or any(x < self.xmin):
            raise ValueError("x is wrong dim or out of bounds")
        return x

    def RandomCall(self):
        x = np.random.uniform(self.xmin, self.xmax)
        f = self.__call__(x)
        # f = self.testmode(x)
        print("\n\nFunction: {}".format(type(self).__name__))
        print("Input = {}".format(x))
        print("Output = {}".format(f))




class newsvendor_deterministic(testfunction):
    def __init__(self, dim=2):
        self.dx = dim
        self.xamin = 0 * np.ones(dim)
        self.xamax = 100 * np.ones(dim)
        self.dx = 1
        self.da = 1
        self.xmin = self.xamin[:self.dx]
        self.xmax = self.xamax[:self.dx]
        self.amin = self.xamin[self.dx:]
        self.amax = self.xamax[self.dx:]
        self.p = 5
        self.l = 3

    def __call__(self, x, u, sig=np.sqrt(10.0), *args, **kwargs):

        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert len(u.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
        assert u.shape[1] == self.da, "Test_func: wrong u input dimension"

        #x = self.check_input(x)
        def E_min(x, u, sig=1.0):
            r = (x - u) / sig
            return x + (u - x) * norm.cdf(r) - sig * norm.pdf(r)

        out = self.p * E_min(x, u, sig) - self.l * x
        return out.reshape(-1,1)

class newsvendor_noisy(testfunction):
    def __init__(self, dim=2):
        self.dx = dim
        self.xamin = 0 * np.ones(dim)
        self.xamax = 100 * np.ones(dim)
        self.dx = 1
        self.da = 1
        self.xmin = self.xamin[:self.dx]
        self.xmax = self.xamax[:self.dx]
        self.amin = self.xamin[self.dx:]
        self.amax = self.xamax[self.dx:]
        self.p = 5
        self.l = 3

    def __call__(self, x, u, sig=np.sqrt(10.0), noise_std=None, *args, **kwargs):

        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert len(u.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
        assert u.shape[1] == self.da, "Test_func: wrong u input dimension"

        if noise_std != None:

            def E_min(x, u, sig=1.0):
                r = (x - u) / sig
                return x + (u - x) * norm.cdf(r) - sig * norm.pdf(r)

            out = self.p * E_min(x, u, sig) - self.l * x
            return out.reshape(-1, 1)

        else:
            c = np.random.normal(u.reshape(-1),np.ones(1)*sig,(1,x.shape[0])).reshape(-1)
            xc = np.concatenate((x.reshape(-1,1), c.reshape(-1,1)),axis=1)
            out = self.p * np.min(xc,axis=1).reshape(-1) - self.l * x.reshape(-1)

            return out.reshape(-1,1)


class newsvendor_noisy_2(testfunction):
    def __init__(self,True_Demand=None, Assumed_Demand=None, dimx=1, dima =2):
        dim = dimx + dima
        self.xamin = np.array([0,0,1e-21])
        self.xamax = np.array([100,100,40])
        # print("self.xamin",self.xamin, "self.xamax",self.xamax)
        self.dx = dimx
        self.da = dima
        self.dr = 1
        self.xmin = self.xamin[:self.dx]
        self.xmax = self.xamax[:self.dx]
        self.amin = self.xamin[self.dx:]
        self.amax = self.xamax[self.dx:]
        # print("self.xmin ",self.xmin ,"self.xmax",self.xmax,"self.amin", self.amin, "self.amax", self.amax)
        self.True_Demand = True_Demand
        MC_samples = 10000000
        print("self.True_Demand",self.True_Demand)
        self.True_Demand_Samples =  self.True_Demand[0].rvs(MC_samples).reshape(-1)

        self.Assumed_Demand = Assumed_Demand
        self.p = 5
        self.l = 3

        # import matplotlib.pyplot as plt
        # x=np.linspace(0,100,100)
        # vals = self.true_performance(np.atleast_2d(x).T)
        # plt.plot(x, vals)
        # plt.show()


    @staticmethod
    @jit(nopython=True)
    def true_performance(x, dx, dr,True_Demand_Samples, p, l ):
        x = np.atleast_2d(x)

        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == dx, "Test_func: wrong x input dimension"

        XDemand = np.zeros((x.shape[0], x.shape[1]+dr))
        Benefit = np.zeros((len(True_Demand_Samples), XDemand.shape[0]))
        for d in range(len(True_Demand_Samples)):
            Demand = True_Demand_Samples[d] #self.True_Demand_Samples
            XDemand[:,:dx] = x #np.concatenate((x.reshape(-1, 1), Demand.reshape(-1, 1)), axis=1)
            XDemand[:, dx:] = Demand
            for xd in range(XDemand.shape[0]):
                benefit = p * np.min(XDemand[xd]) - l * x[xd]
                Benefit[d,xd] = benefit[0]

        return Benefit

    def __call__(self, x, u, true_performance_flag=False, *args, **kwargs):

        if true_performance_flag:
            assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
            Benefit_Simulations = self.true_performance(x, dx=self.dx,dr=self.dr, True_Demand_Samples=self.True_Demand_Samples, p=self.p, l=self.l )
            # print("results",results, "shape", results.shape)
            Expected_Benefit = np.mean(Benefit_Simulations,axis=0)
            Expected_Benefit = (Expected_Benefit - 15.292935763364486)/94.28618452053284
            return Expected_Benefit.reshape(-1, 1)

        else:
            print("x",x,"u",u)
            assert x.shape[0] == u.shape[0], "wrong x or u dimensions"
            assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert len(u.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
            assert u.shape[1] == self.da, "Test_func: wrong u input dimension"

            mean = u[:,0]
            sig = np.sqrt(u[:,1])
            reps = 100

            self.Assumed_Demand_Samples = self.Assumed_Demand[0](mean, sig, (reps, x.shape[0]))
            Benefit_Simulations = self.core_simulator(X=x,mean=mean,sig=sig,reps=reps, dx=self.dx, dr=self.dr,Assumed_Demand=self.Assumed_Demand_Samples, p=self.p, l=self.l)
            Expected_Benefit = np.mean(Benefit_Simulations,axis=0)
            Expected_Benefit = (Expected_Benefit - 15.292935763364486) / 94.28618452053284
            # print("mean", np.mean(Benefit_Simulations,axis=0),"len",Benefit_Simulations.shape[0],"MSE", np.std(Benefit_Simulations,axis=0)/np.sqrt(Benefit_Simulations.shape[0]))
            return Expected_Benefit.reshape(-1,1)

    @staticmethod
    @jit(nopython=True)
    def core_simulator(X, mean, sig, reps, dx,dr, Assumed_Demand, p, l):
        X = np.atleast_2d(X)

        assert len(X.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert X.shape[1] == dx, "Test_func: wrong x input dimension"

        Benefit = np.zeros((reps, X.shape[0]))
        for d in range(reps):
            XDemand = np.zeros((X.shape[1] + dr))
            for xd in range(X.shape[0]):
                Demand = Assumed_Demand[d, xd]
                XDemand[:dx] = X[xd]
                XDemand[dx:] = Demand
                benefit = p * np.min(XDemand) - l *X[xd]
                Benefit[d, xd] = benefit[0]
        return Benefit


# x = np.array([[50],
#               [60]])
# a = np.array([[40,10],
#               [40,10]])
# True_Input_distributions = [norm(loc=40, scale=np.sqrt(10))]  # [gamma(a=k,loc=0,scale=theta)]#
# Assumed_Input_Distributions = [np.random.normal]
# #
# # # plt.hist(True_Input_distributions[0].rvs(1000), bins=200, density=True)
# # # plt.hist(np.random.normal(mu, np.sqrt(var), (1, 1000)).reshape(-1), bins=200, density=True)
# # # plt.show()
# #
# Simulator = newsvendor_noisy_2(True_Demand=True_Input_distributions, Assumed_Demand=Assumed_Input_Distributions)
# N=10000
# x = np.random.random((N,1))*100
# a = np.random.random((N,2))*[100,40]
#
#
# out =Simulator(x,a, true_performance_flag=False)
# print("out", np.mean(out), np.std(out))

# # # stop = time.time()
# # #
# # # start = time.time()
# # # Simulator(x,a, true_performance_flag=True)
# # # stop = time.time()
# # # print("time", stop-start)
# #
# import matplotlib.pyplot as plt
# X = np.linspace(0,100,60).reshape((-1,1))
# a = np.array([[40,10]])
# x= np.array([[99.67831948]])
# u= np.array([[41.52744512 ,18.08146504]])
# a = np.repeat(a,60,axis=0)
# topxa =np.array([[99.67831948, 41.52744512, 18.08146504]])
# dim_X = 1
# out = Simulator(x,u, true_performance_flag=False)
# # print("a.shape", a.shape)
# plt.scatter(X, out)
# plt.show()

