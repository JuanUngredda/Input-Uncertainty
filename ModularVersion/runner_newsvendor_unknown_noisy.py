# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)

import matplotlib.pyplot as plt
from IU_optimizer import *
from TestProblems import Information_Source
from TestProblems.newsvendor import newsvendor_noisy, newsvendor_noisy_2
import subprocess as sp
from scipy.stats import norm

print("\nCalling optimizer")
myoptimizer = Mult_Input_Uncert()

"""
Choose optimsiation method between:
	- BICO: Use Knowledge gradient and Value of Information for external data sources for sampling.
	- Benchmark: use fixed quantity of data source points initially and optimise by KG.

Choose distribution method between:
-trunc_norm: Normal Likelihood and Uniform prior for input. Assumes known variance in the data.
-MUSIG : Normal Likelihood and Uniform prior for input. Assumes unknown variance in the data.

"""
def function_caller(rep):
    np.random.seed(rep+200)
    True_Input_distributions = [norm(loc=40, scale=np.sqrt(10))]
    Assumed_Input_Distributions = [np.random.normal]

    # plt.hist(True_Input_distributions[0].rvs(1000), bins=200, density=True)
    # plt.hist(np.random.normal(mu, np.sqrt(var), (1, 1000)).reshape(-1), bins=200, density=True)
    # plt.show()

    Simulator = newsvendor_noisy_2(True_Demand=True_Input_distributions, Assumed_Demand=Assumed_Input_Distributions)
    Information_Source_Generator = Information_Source(Distribution=True_Input_distributions, lb=Simulator.amin,
                                                      ub=Simulator.amax, d=1)


    [XA], [Y], [Data] = myoptimizer(sim_fun=Simulator, inf_src=Information_Source_Generator,
                                    lb_x=Simulator.xmin, ub_x=Simulator.xmax,
                                    lb_a=Simulator.amin, ub_a=Simulator.amax,
                                    distribution="MUSIG",
                                    n_fun_init=10,
                                    n_inf_init=20,
                                    Budget=100,
                                    Nx=100,
                                    Na=101,
                                    Nd=101,
                                    GP_train=True,
                                    GP_train_relearning=True,
                                    var_data=None,
                                    opt_method="BICO",
                                    rep=str(rep),
                                    save_only_last_stats=False,
                                    calculate_true_optimum=False,
                                    results_name="Newsvendor_BICO_RESULTS")

# for r in range(5,30,1):
# function_caller(rep=1)