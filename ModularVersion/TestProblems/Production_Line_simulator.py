"""
Carwash example.

Covers:

- Waiting for other processes
- Resources: Resource

Scenario:
  A carwash has a limited number of washing machines and defines
  a washing processes that takes some (random) time.

  Car processes arrive at the carwash at a random time. If one washing
  machine is available, they start the washing process and wait for it
  to finish. If not, they wait until they an use one.

"""
import random
import simpy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from scipy.stats import expon
# import logging

SIM_TIME = 2000     # Simulation time in minutes
WARM_UP_PERIOD = 1000

class Carwash(object):
    """A carwash has a limited number of machines (``NUM_MACHINES``) to
    clean cars in parallel.

    Cars have to request one of the machines. When they got one, they
    can start the washing processes and wait for it to finish (which
    takes ``washtime`` minutes).

    """
    def __init__(self, env):
        self.env = env

        self.queue1 = simpy.Resource(env, capacity=10)
        self.machine1 = simpy.Resource(env, capacity=1)
        self.queue2 = simpy.Resource(env, capacity=10)
        self.machine2 = simpy.Resource(env, capacity=1)
        self.queue3 = simpy.Resource(env, capacity=10)
        self.machine3 = simpy.Resource(env, capacity=1)
        self.queue4 = simpy.Resource(env, capacity=10)
        self.machine4 = simpy.Resource(env, capacity=1)
        self.dispatch = simpy.Container(env, capacity=9e99, init=0)


    def wash1(self, car, mu):
        """The washing processes. It takes a ``car`` processes and tries
        to clean it."""
        mu = np.reciprocal(mu)
        time = np.random.exponential(scale=mu)
        yield self.env.timeout(time)

    def wash2(self, car, mu):
        """The washing processes. It takes a ``car`` processes and tries
        to clean it."""
        mu = np.reciprocal(mu)
        time = np.random.exponential(scale=mu)
        yield self.env.timeout(time)

    def wash3(self, car, mu):
        """The washing processes. It takes a ``car`` processes and tries
        to clean it."""
        mu = np.reciprocal(mu)
        time = np.random.exponential(scale=mu)
        yield self.env.timeout(time)

    def wash4(self, car, mu):
        """The washing processes. It takes a ``car`` processes and tries
        to clean it."""
        mu = np.reciprocal(mu)
        time = np.random.exponential(scale=mu)
        yield self.env.timeout(time)

def car(env, name, cw, MU):
    """The car process (each car has a ``name``) arrives at the carwash
    (``cw``) and requests a cleaning machine.

    It then starts the washing process, waits for it to finish and
    leaves to never come back ...

    """

    with cw.machine1.request() as request:
        yield env.process(cw.wash1(name, mu=MU[:, 0]))
        yield request
        # print('%s enters the machine 1 at %.2f.' % (name, env.now))

        # print('%s leaves the machine 1 at %.2f.' % (name, env.now))

    req2 = cw.queue2.request()
    yield req2
    # print('%s enters the queue 2 at %.2f.' % (name, env.now), " level ", cw.queue2.count)
    with cw.machine2.request() as request:
        yield request
        cw.queue2.release(req2)
        # print('%s gets out the queue 2 at %.2f.' % (name, env.now), " level ", cw.queue2.count)
        # print('%s enters the machine 2 at %.2f.' % (name, env.now))
        yield env.process(cw.wash2(name, mu=MU[:,1]))
        # print('%s leaves the machine 2 at %.2f.' % (name, env.now))

    req3 = cw.queue3.request()
    yield req3
    # print('%s enters the queue 3 at %.2f.' % (name, env.now))
    with cw.machine3.request() as request:
        yield request
        cw.queue3.release(req3)
        # print('%s enters the machine 3 at %.2f.' % (name, env.now))
        yield env.process(cw.wash3(name, mu=MU[:,2]))
        # print('%s leaves the machine 3 at %.2f.' % (name, env.now))


    # print('%s leaves at the carwash at %.2f.' % (name, env.now))
    if env.now> WARM_UP_PERIOD:
        cw.dispatch.put(1)

def setup(env, carwash,  MU, Parameter):
    """Create a carwash, a number of initial cars and keep creating cars
    approx. every ``t_inter`` minutes."""
    # Create the carwash
    i = 0
    while True:

        Parameter = np.reciprocal(Parameter) #Change lambda to Beta as numpy parameter is Beta
        yield env.timeout(expon.rvs(scale=Parameter))
        env.process(car(env, 'Car %d' % i, carwash, MU=MU))
        i += 1


class Production_Line():
    def __init__(self,True_rate=None, dimx=3, dima =1):

        self.xamin = np.array([0, 0, 0, 0])
        self.xamax = np.array([2, 2, 2, 7])
        # print("self.xamin",self.xamin, "self.xamax",self.xamax)
        self.dx = dimx
        self.da = dima
        self.xmin = self.xamin[:self.dx]
        self.xmax = self.xamax[:self.dx]
        self.amin = self.xamin[self.dx:]
        self.amax = self.xamax[self.dx:]
        # print("self.xmin ",self.xmin ,"self.xmax",self.xmax,"self.amin", self.amin, "self.amax", self.amax)
        self.True_rate = True_rate
        self.MC_samples = 20
        # print("self.True_Demand", self.True_Demand)

    def __call__(self, X, U, true_performance_flag=False, *args, **kwargs):

        if true_performance_flag:
            assert len(X.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert X.shape[1] == self.dx, "Test_func: wrong x input dimension"
            out = np.zeros((X.shape[0],1))

            for i in range(X.shape[0]):
                out[i,:] = self.simulator(x = X[i], u = self.True_rate, MC_samples = 100)
            return out
        else:
            assert len(X.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert len(U.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert X.shape[1] == self.dx, "Test_func: wrong x input dimension"
            assert U.shape[1] == self.da, "Test_func: wrong u input dimension"


            out = np.zeros((X.shape[0],1))
            for i in range(X.shape[0]):
                out[i,:] = self.simulator(x = X[i], u = U[i], MC_samples = 1)
            return out

    def simulator(self, x, u, MC_samples,true_performance_flag=False, *args, **kwargs):
        x = np.atleast_2d(x)
        u = np.atleast_2d(u)

        assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert len(u.shape) == 2, "x must be an N*d matrix, each row a d point"
        assert x.shape[1] == self.dx, "Test_func: wrong x input dimension"
        assert u.shape[1] == self.da, "Test_func: wrong u input dimension"

        # Setup and start the simulation

        # Create an environment and start the setup process

        self.x = x
        self.u = u
        self.SIM_TIME = SIM_TIME
        self.WARM_UP = WARM_UP_PERIOD

        # pool = multiprocessing.Pool()

        with multiprocessing.Pool() as pool:
            Revenue= pool.map(self.parallelised_sim, range(MC_samples))
        # Revenue= pool.map(self.parallelised_sim, range(MC_samples))
        # pool.close()
        mean_Revenue = np.mean(Revenue)
        MSE_Revenue = np.sqrt(np.var(Revenue) / len(Revenue))
        # if MSE_Revenue>5:
        # plt.hist(Revenue)
        # plt.show()
        # print("x", self.x,"u", self.u)
        print("mean_rev", mean_Revenue, "var",np.var(Revenue),"len(Revenue)",len(Revenue),"MSE rev", MSE_Revenue, "min", np.min(Revenue), "max", np.max(Revenue))
        return mean_Revenue.reshape(-1)

    def parallelised_sim(self, identifier):
        r = 10000
        c0 = 1
        c1 = 400
        c = np.array([[1, 5, 9]])

        env = simpy.Environment()
        carwash = Carwash(env)
        env.process(setup(env, carwash, MU=self.x, Parameter=self.u))
        # Execute!
        env.run(until=SIM_TIME)
        Throughtput = carwash.dispatch.level / ((self.SIM_TIME - self.WARM_UP) * 1.0)

        Revenue = ((r * Throughtput) / (c0 + np.sum(c * self.x))) - c1

        Revenue = (Revenue-(-102.839362))/168.76971540118384

        return Revenue


# import time
# Simulator = Production_Line(True_rate=0.5)
#
#
# D=4
# N=1000
# ub = np.array([[2,2,2,2]])
# lb = np.array([[1e-99,1e-99,1e-99,1e-99]])
# X =np.array([[0.1, 7.29685763e-01, 6.65953669e-01]]) # np.random.random((N,D))*(ub-lb) + lb
# X[:,-1] = 0.5
#
# start = time.time()
# out = Simulator(X[:,:3], X[:,3:], true_performance_flag=True)
# out = out.reshape(-1)
# x_r = X[np.argmax(out)]
# print("x_r", x_r, "val", np.max(out))
# print("sorted X", X[np.argsort(out)])
# print("sorted vals", np.sort(out))
# # print("out", out, "mean", np.mean(out), "std", np.std(out))
# stop = time.time()
# print("time", stop-start)
#


# rev = []
# time_step = np.linspace(1000,1500,50)
# for sim in time_step:
#     out = Simulator(X, rate, true_performance_flag=False)
#     rev.append(out)

# plt.plot(time_step, np.array(rev).reshape(-1))
# plt.show()
# N=1
# D=4
# ub = np.array([[2,2,2,2]])
# lb = np.array([[1e-99,1e-99,1e-99,1e-99]])
# np.random.seed(1)
# X = np.random.random((N,D))*(ub-lb) + lb
#
#
# N=1000
# D=4
# test_X = np.linspace(1e-9,9,N)
# import time
# Simulator = Production_Line(True_rate=0.5)
# k=-1
#
# test_X=np.random.random((N,D-1))*2
#
#
# print("start simulations")
#
# out = Simulator(X=np.array([[1.1622165,0.791415 ,0.7978742]]), U=np.array([[0.5]]), true_performance_flag=True)
# print("max", np.max(out), "np.argmax", test_X[np.argmax(out)])
# stop = time.time()

# print("Test_matrix",Test_matrix)
# plt.scatter(test_X, out)
# plt.show()

# print("X",X[:,3:])
# print("Simulator output", Simulator(X, rate,true_performance_flag=False))
# stop = time.time()
# print(stop-start)