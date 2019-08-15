import numpy as np

class Ambulance_Delays:
    def __init__(self, num_ambulances=3):
        
        # decision variables
        self.dx = num_ambulances * 2
        self.xmin = np.zeros((self.dx,))
        self.xmax = np.ones((self.dx,))

        # uncertainty parameter
        self.pmin = np.zeros((2,))
        self.pmax = np.ones((2,))
        
        # simulator stuff
        self.nC = 10 * num_ambulances

        self.m = 45 / 60.0
        self.s = 15 / 60.0
        self.rate = 1 / 60.0
        self.vf = 60 / 30.0 * np.sqrt(3 / num_ambulances)
        self.vs = 40 / 30.0 * np.sqrt(3 / num_ambulances)

    def Gen_CallLocs(self, n=30, peak=None):

        if peak is None:
            peak = 0.8 * np.ones((2,))

        CallLocs = np.zeros((0, 2))

        while CallLocs.shape[0] < n:
            nGen = (n - CallLocs.shape[0]) * 2
            u = np.random.uniform(size=(nGen, 3))
            accept = 1.6 * u[:, 2] <= 1.6 - np.abs(u[:, 0] - peak[0]) - np.abs(u[:, 1] - peak[1])
            CallLocs = np.vstack([CallLocs, u[accept, 0:2]])

        CallLocs = CallLocs[0:n, :]

        return CallLocs


    def Gen_CallTimes(self, n, rate=None):
        if rate is None:
            rate = 1.0 / 60
        CallTimes = np.random.exponential(size=n, scale=1 / rate)
        CallTimes = np.cumsum(CallTimes)
        return CallTimes


    def Gen_ServTimes(self, n, m=None, s=None):
        if m is None:
            m = 45 / 60.0

        if s is None:
            s = 15 / 60.0

        shp = (m / s) ** 2
        sca = m / shp

        ServTimes = np.random.gamma(size=(n,), shape=shp, scale=sca)

        return ServTimes


    def Dist_2_call(self, lc, b, nc, d):

        if d > np.sum(np.abs(lc - b)):
            pres_loc = b

        elif d > np.abs(lc[1] - b[1]):
            d -= np.abs(lc[1] - b[1])
            if lc[0] < b[0]:
                pres_loc = np.asarray([lc[0] + d, b[1]])
            else:
                pres_loc = np.asarray([lc[0] - d, b[1]])

        else:
            if lc[1] < b[1]:
                pres_loc = np.asarray([lc[0], lc[1] + d])
            else:
                pres_loc = np.asarray([lc[0], lc[1] - d])

        return np.sum(np.abs(pres_loc - nc))


    def Simulate(self, x, nC=30, peak=None, rate=None, m=None, s=None, vf=None, vs=None, seed=None):

        if seed is not None:
            np.random.seed(seed)

        if seed is not None:
            np.random.seed(seed)

        nA = x.shape[0]

        if vf is None:
            vf = 60.0 / 30
        if vs is None:
            vs = 40.0 / 30

        Ambulances = np.zeros((nA, 5))
        Ambulances[:, 1:3] = x
        Ambulances[:, 3:5] = x
        ExitTimes = np.zeros((nC,))
        AmbArrTimes = np.zeros((nC,))

        CallLocs = self.Gen_CallLocs(nC, peak)
        CallTimes = self.Gen_CallTimes(nC, rate)
        ServTimes = self.Gen_ServTimes(nC, m, s)

        ArrivalTimes = np.zeros((nC,))
        ExitTimes = np.zeros((nC,))

        for i in range(nC):

            D2call = 100000 * np.ones((nA,))

            for j in range(nA):
                driven = vs * (CallTimes[i] - Ambulances[j, 0])
                if driven > 0:
                    D2call[j] = self.Dist_2_call(Ambulances[j, 1:3], Ambulances[j, 3:5], CallLocs[i, :], driven)

            if np.any(D2call < 10000):
                closestA = np.argmin(D2call)
                depart = CallTimes[i]
                minD2call = np.min(D2call)

            else:
                closestA = np.argmin(Ambulances[:, 0])
                depart = Ambulances[closestA, 0]
                minD2call = np.sum(np.abs(Ambulances[closestA, 1:3] - CallLocs[i, :]))

            ArrivalTimes[i] = depart + minD2call / vf
            ExitTimes[i] = ArrivalTimes[i] + ServTimes[i]
            Ambulances[closestA, 0] = ExitTimes[i]
            Ambulances[closestA, 1:3] = CallLocs[i, :]

        Delay = np.mean(ArrivalTimes - CallTimes)

        return Delay


    def get_range(self):
        # just returns bounds of x
        return np.vstack([self.xmin, self.xmax])

    def __check_x(self, x):
        # sanity checking input, dimensions, bounds

        if not len(x) == self.dx:
            raise ValueError("x is wrong dimension")

        if not np.all(x > 0):
            raise ValueError("x is out of bounds")

        if not np.all(x < 1):
            raise ValueError("x is out of bounds")

        return x
    
    def __check_peak(self, peak):
        # sanity checking uncertainty parameter, bounds etc
        if not len(peak) == 2:
            raise ValueError("peak is wrong dimension")

        if not np.all(peak > self.pmin):
            raise ValueError("peak is out of bounds")

        if not np.all(peak < self.pmax):
            raise ValueError("peak is out of bounds")

        return peak

    def __call__(self, x=None, peak=None, seed=None,*args,**kwargs):

        assert len(x.shape) == 2 , "insert ndarray"

        out = np.zeros(x.shape[0])

        for idx, x_val in enumerate(x):

            if seed is not None:
                np.random.seed(seed)

            assert x[idx] is not None, "give an x vector!"
            assert peak[idx] is not None, "give a p vector!"

            x_vect = self.__check_x(x[idx])
            x_vect = x_vect.reshape((-1, 2))

            peak_vect = self.__check_peak(peak[idx])

            output = self.Simulate(x_vect, peak=peak_vect, rate=self.rate, m=self.m, s=self.s, vf=self.vf, vs=self.vs)

            out[idx] = output

        print("out",out)
        return out

    def testmode(self, x, p, num_seeds=100):
        testseeds = np.linspace(1000000, 1000000 + num_seeds, num_seeds).astype(int)
        test_f = np.array([self.__call__(x, p, s) for s in testseeds])

        return test_f.mean()


if __name__ == "__main__":

    # decision variable is 3 (x,y) points in unit square (reshaped to vector)
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    
    # input unceratinty parameter is the peak of the patient density in the unit square
    p = np.array([0.8, 0.8])

    
    # instantiate the objective function, a simulator for 3 ambulance bases (hence 6dims, can increase/decrese)
    # this simulator computes the journay time of amnbulances to reach patients, the objective must be minimized!
    f = Ambulance_Delays(num_ambulances=3)
    
    # now call the function with a x and p
    print("decision varaible, ambulance base locations: ", x)
    print("uncertain parameter peak of distro of patients: ", p)
    print("call simulator once, ambulance journary time is...", f(x, p, seed=1))
    print("call simulator 1000 times with different seeds and average (for evaluation)", f.testmode(x, p, num_seeds=1000))
    

