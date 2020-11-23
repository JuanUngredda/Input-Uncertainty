import numpy as np
from scipy.stats import expon, norm
from numba import njit


@njit(debug=False)
def Simulate_Xie_ATO(Multiplier, x, Arrival, ProdTimeKey, ProdTimeNonKey, Items0, Products,
                     seed, self_seed):
    """
    ARGS:
        x: (8,) int, vector of restocking levels
        Arrival: (8, 1000?) RNG time series of inter-arrival times
        ProdTimeKey: (8, 1000?) RNGtime series of Key item
        ProdTimeNonKey: (8, 1000?) RNGtime series of NonKey item
        Items: (nItems) parameter matrix
        Products: (nItems, nProducts) parameter matrix

    RETURNS:
        Profit: scalar float
    """
    if seed is None:
        seed = self_seed

    nProds = 5
    nItems = 8
    nKey = 6
    Profit = 0.0

    Orders = np.zeros(nProds)  # vector of next event times
    # A = np.ones(nProds, dtype=np.int)  # vector of indices iterating through Arrivals
    # A = nProds * [1]
    A = [1 for _ in range(nProds)]

    Inventory = x.copy()  # starting stock level is target level
    Items = Items0.copy()
    Items[:, 2] *= Multiplier  # add multiplier

    ItemAvail = -np.ones((nItems, 100))  # time series of ordered items on their way

    PrevTime = 0.0
    Time = 0.0
    warmup = 20.0
    Tmax = 70.0
    Prod = 0
    ptime_i = 0
    pntime_i = 0
    p = 1
    q = 1
    r = 0
    keyavail = True
    usedI = 0.0
    BaseClock = 0.0
    Orders = Arrival[:, 0]  # vector of next events for each product

    while min(Orders) < Tmax:

        ##########################################
        # PART 1 / 4
        # the prodct that arrived and the current time
        Prod = np.argmin(Orders)
        Time = np.min(Orders)

        # update the time to next event in RNG series
        Orders[Prod] += Arrival[Prod, A[Prod]];
        A[Prod] += 1;

        # if too late, stop simulating
        if Time > Tmax: break

        ##########################################
        # PART 2 / 4
        # Update inventory arrived since last time and items
        #  requested but not arrived.
        for q in range(nItems):

            # count arrived items
            r = 0
            while ItemAvail[q, r] > 0.0 and ItemAvail[q, r] < Time:
                r += 1
            Inventory[q] += r

            # overwrite arrived items from ItemAvail time series
            p = 0
            while ItemAvail[q, p] > 0.0:
                ItemAvail[q, p] = ItemAvail[q, p + r]
                p += 1

        ###########################################
        # PART 3 / 4
        # Does the inventory have enough stock? Then make the sale!
        keyavail = np.all(Inventory >= Products[Prod, 1:])

        if keyavail:
            # Decrease key items inventory and place replenishment
            # orders for the amount of key items used.
            for i in range(nKey):
                used_inv = int(Products[Prod, i + 1])
                if used_inv > 0:

                    # key item profit
                    if Time > warmup: Profit += Items[i, 0] * used_inv

                    # key item inventory level
                    Inventory[i] -= used_inv

                    # compute time at last item availabilty
                    ia = 0
                    BaseClock = -10.0
                    while ItemAvail[i, ia] > 0.0:
                        BaseClock = ItemAvail[i, ia]
                        ia += 1

                    if BaseClock < Time:
                        BaseClock = Time

                    # update the time series of items in transit
                    for j in range(used_inv):
                        ItemAvail[i, ia + j] = BaseClock + Items[i, 2] + Items[i, 3] * ProdTimeKey[ptime_i]
                        ptime_i += 1
                        BaseClock = ItemAvail[i, ia + j]

            # Decrease nonkey items inventory and place replenishment
            # orders for the amount of key items used
            for i in range(nKey, nItems):
                used_inv = int(Products[Prod, i + 1])

                if used_inv > 0 and used_inv <= Inventory[i]:

                    # profit
                    if Time > warmup: Profit += Items[i, 0] * used_inv

                    # invenroty level
                    Inventory[i] -= used_inv

                    # compute time at last item availabilty
                    ia = 0
                    BaseClock = -10.0
                    while ItemAvail[i, ia] > 0.0:
                        BaseClock = ItemAvail[i, ia]
                        ia += 1

                    if BaseClock < Time:
                        BaseClock = Time

                    # update the time series of items in transit
                    for j in range(used_inv):
                        ItemAvail[i, ia + j] = BaseClock + Items[i, 2] + Items[i, 3] * ProdTimeNonKey[pntime_i]
                        pntime_i += 1
                        BaseClock = ItemAvail[i, ia + j]

                        ##########################################
        # PART 4 / 4
        # subtract holding costs from profit
        if Time > warmup:
            Profit -= np.sum(Inventory * Items[:, 1]) * (Time - PrevTime)
        PrevTime = Time

    return Profit / (Tmax - warmup);


# @njit(debug=False)
class ATO_Xie:
    def __init__(self, seed=1, simreps=5):
        self.seed = seed * 1000

        self.items = [1, 2, .15, .0225, 20,
                      2, 2, .40, .06, 20,
                      3, 2, .25, .0375, 20,
                      4, 2, .15, .0225, 20,
                      5, 2, .25, .0375, 20,
                      6, 2, .08, .012, 20,
                      7, 2, .13, .0195, 20,
                      8, 2, .40, .06, 20]

        self.items = np.array(self.items).reshape(8, 5)

        self.xmin = np.zeros([8])
        self.xmax = 20 * np.ones([8])
        self.action_bounds = np.vstack((self.xmin, self.xmax))
        self.state_bounds = np.vstack((0.5, 1.5))

        self.products = [3.6, 1, 0, 0, 1, 0, 1, 1, 0,
                         3, 1, 0, 0, 0, 1, 1, 1, 0,
                         2.4, 0, 1, 0, 1, 0, 1, 0, 0,
                         1.8, 0, 0, 1, 1, 0, 1, 0, 1,
                         1.2, 0, 0, 1, 0, 1, 1, 1, 0]

        self.products = np.array(self.products).reshape(5, 9)

        # Simulation params
        self.nItems = self.items.shape[0]
        self.nProducts = 5
        self.numberkey = 6
        self.numbernk = self.nItems - self.numberkey
        self.Tmax = 70

        # Generate RNG streams
        # np.random.seed(seed)
        # fnSum = 0.0
        self.simreps = simreps
        
    def simulate_vec(self, Multipliers, BaseStockLevels, seed=None):
        results = []
        Multipliers = np.atleast_2d(Multipliers)
        BaseStockLevels = np.atleast_2d(BaseStockLevels)
        for s, x in zip(Multipliers, BaseStockLevels):
            result = self.Simulate(s, x, seed)
            results.append(result)
        return np.array(results)

    def Simulate(self, Multiplier=np.ones([1]), BaseStockLevel=np.ones([8]), seed=None):
        fnSum = 0.0
        if seed is None:
            np.random.seed(self.seed)
            self.seed += 1
        else:
            np.random.seed(seed)

        for _ in range(self.simreps):
            nGen = int(10 * self.Tmax * np.round(np.sum(self.products[:, 0])))
            Arrival_RNG = expon.rvs(size=(5, nGen)) * (1 / self.products[:, 0, None])
            ProdTimeKey_RNG = norm.rvs(size=(nGen))
            ProdTimeNonKey_RNG = norm.rvs(size=(nGen))

            Profit = Simulate_Xie_ATO(Multiplier, BaseStockLevel, Arrival_RNG, ProdTimeKey_RNG, ProdTimeNonKey_RNG,
                                      self.items,
                                      self.products,
                                      seed, self.seed)
            fnSum += Profit

        # end of all simulation reps
        fnAvg = fnSum / self.simreps

        return fnAvg  # maximize profit!

    def __call__(self, s, x, seed=None):
        assert np.all(0.5 <= s) and np.all(s <= 1.5), "ATO input params must be in [0.5, 1.5]"
        assert np.all(0 <= x) and np.all(x <= 20), "ATO decision variables must be in [0, 20]^8"
        return self.simulate_vec(s, x, seed)

    def test_mode(self, s, x, num_seeds=100):
        rng_state = np.random.get_state()
        testseeds = np.linspace(1000000, 1000000 + num_seeds, num_seeds).astype(int)
        test_f = np.array([self(s, x, seed=seed) for seed in testseeds])
        np.random.set_state(rng_state)
        return test_f.mean(axis=0)



fun = ATO_Xie()

def ATO(s, x, seed=None, test=False):
    if test:
        return fun.test_mode(s, x)
    else:
        return fun(s, x, seed)


if __name__ == "__main__":

    """
    From other files, you can type

    from ato_xie import ATO

    # during BO
    y = ATO(s, x) 

    # at the end to measure true theta(s, x) quality
    true_y = ATO(s, x, test=True) 
    
    """


    for seed in range(10):
        # pick random sate between 0.5, 1.5, x in [0, 20]^8
        s = 0.5 + np.random.uniform()
        x = np.asarray(20 * np.random.uniform(size=8)).astype(int)
        print("\nCall the function once,    s:", str(s)[:4], "\t x:", x, ",\t   y =", ATO(s, x))
        print("Average 100 seeds \"truth\", s:", str(s)[:4], "\t x:", x, ",\t E[y]=", ATO(s, x, test=True))

    
