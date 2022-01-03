import numpy as np


def Gen_CallLocs(n=30, peak=None):

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


def Gen_CallTimes(n, rate=None):
    if rate is None:
        rate = 1.0 / 60
    CallTimes = np.random.exponential(size=n, scale=1 / rate)
    CallTimes = np.cumsum(CallTimes)
    return CallTimes


def Gen_ServTimes(n, m=None, s=None):
    if m is None:
        m = 45 / 60.0

    if s is None:
        s = 15 / 60.0

    shp = (m / s) ** 2
    sca = m / shp

    ServTimes = np.random.gamma(size=(n,), shape=shp, scale=sca)

    return ServTimes


def Dist_2_call(lc, b, nc, d):

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


def Simulate(x, nC=30, peak=None, rate=None, m=None, s=None, vf=None, vs=None, seed=None):

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

    CallLocs = Gen_CallLocs(nC, peak)
    CallTimes = Gen_CallTimes(nC, rate)
    ServTimes = Gen_ServTimes(nC, m, s)

    ArrivalTimes = np.zeros((nC,))
    ExitTimes = np.zeros((nC,))

    for i in range(nC):

        D2call = 100000 * np.ones((nA,))

        for j in range(nA):
            driven = vs * (CallTimes[i] - Ambulances[j, 0])
            if driven > 0:
                D2call[j] = Dist_2_call(Ambulances[j, 1:3], Ambulances[j, 3:5], CallLocs[i, :], driven)

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


class Ambulance_Delays:
    def __init__(self, num_ambulances=3):
        self.dx = num_ambulances * 2
        self.xmin = np.zeros((self.dx,))
        self.xmax = np.ones((self.dx,))

        self.nC = 10 * num_ambulances
        self.peak = 0.8 * np.ones((2,))
        self.m = 45 / 60.0
        self.s = 15 / 60.0
        self.rate = 1 / 60.0
        self.vf = 60 / 30.0 * np.sqrt(3 / num_ambulances)
        self.vs = 40 / 30.0 * np.sqrt(3 / num_ambulances)

    def get_range(self):
        return np.vstack([self.xmin, self.xmax])

    def __check_x(self, x):
        if not len(x) == self.dx:
            raise ValueError("x is wrong dimension")

        if not np.all(x > 0):
            raise ValueError("x is out of bounds")

        if not np.all(x < 1):
            raise ValueError("x is out of bounds")

        return x

    def __call__(self, x, seed=None):

        if seed is not None:
            np.random.seed(seed)

        x = self.__check_x(x)
        x = x.reshape((-1, 2))

        output = Simulate(x, peak=self.peak, rate=self.rate, m=self.m, s=self.s, vf=self.vf, vs=self.vs)
        return output

    def testmode(self, x, num_seeds=100):
        testseeds = np.linspace(1000000, 1000000 + num_seeds, num_seeds).astype(int)
        test_f = np.array([self.__call__(x, s) for s in testseeds])

        return test_f.mean()


if __name__ == "__main__":

    import sys

    args = sys.argv

    if len(sys.argv) == 1:
        f = Ambulance_Delays(30)

        x = np.random.uniform(f.xmin, f.xmax)

        print("Input = {}".format(x))
        print("Output = {}".format(f.testmode(x)))

    else:
        x = np.array(sys.argv[1:]).astype(np.float)
        if len(x) % 2 == 0:
            f = Ambulance_Delays(int(0.5 * len(x)))
            print(f(x))
