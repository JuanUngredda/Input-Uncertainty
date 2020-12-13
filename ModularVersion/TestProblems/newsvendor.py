import numpy as np
from scipy.stats import norm

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
        # raise


    def true_performance(self, x):
        x = np.atleast_2d(x)

        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"

        mean_repetitions = []

        Partition = 100
        Expected_Benefit_d = []
        for d in np.split(self.True_Demand_Samples, Partition):
            Demand = d #self.True_Demand_Samples

            XDemand = np.array(np.meshgrid(x, Demand))
            Benefit = self.p * np.min(XDemand, axis=0) - self.l * x.reshape(-1)
            Expected_Benefit_d.append(np.mean(Benefit, axis=0))
        Expected_Benefit = np.mean(Expected_Benefit_d,axis=0)
        # print("x",x)
        # print("Expected_Benefit",Expected_Benefit)
        # raise
        return Expected_Benefit

    def __call__(self, x, u, true_performance_flag=False, *args, **kwargs):

        if true_performance_flag:
            assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
            return self.true_performance(x).reshape(-1, 1)

        else:
            assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert len(u.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
            assert u.shape[1] == self.da, "Test_func: wrong u input dimension"


            mean = u[:,0]
            sig = np.sqrt(u[:,1])
            reps = 320
            rev = []
            print("")
            for i in range(reps):
                c = self.Assumed_Demand[0](mean.reshape(-1),np.zeros(1)*sig,(1,x.shape[0])).reshape(-1)
                xc = np.concatenate((x.reshape(-1,1), c.reshape(-1,1)),axis=1)
                #print("x",xc)
                #print("np.min(xc,axis=1)",np.min(xc,axis=1))
                out = self.p * np.min(xc,axis=1).reshape(-1) - self.l * x.reshape(-1)
                rev.append(out)
            # print("mean", np.mean(rev,axis=0), "std", np.std(rev,axis=0), "MSE",np.std(rev,axis=0)/reps )
            # print("max", np.max(np.std(rev,axis=0)/reps ),"min", np.min(np.std(rev,axis=0)/reps ))
            results = np.mean(rev, axis=0)
            # raise
            return results.reshape(-1,1)


def main():

    if __name__ == "__main__":
        main()
