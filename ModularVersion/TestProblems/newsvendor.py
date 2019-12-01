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


def main():

    if __name__ == "__main__":
        main()
