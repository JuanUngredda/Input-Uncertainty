# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)

import matplotlib.pyplot as plt
from IU_optimizer import *
from TestProblems import Information_Source
# from TestProblems.newsvendor import newsvendor_noisy, newsvendor_noisy_2
from TestProblems.Production_Line_simulator import Production_Line
import subprocess as sp
from scipy.stats import norm, expon

def function_caller(rep):
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
	np.random.seed(rep+400)

	True_rate = 0.5
	True_Input_distributions = [expon(scale=np.reciprocal(True_rate))]  # [gamma(a=k,loc=0,scale=theta)]#
	Assumed_Input_Distributions = [np.random.normal]


	Simulator = Production_Line(True_rate=True_rate)
	Information_Source_Generator = Information_Source(Distribution=True_Input_distributions, lb=Simulator.amin,
													  ub=Simulator.amax, d=1)

	[XA], [Y], [Data] = myoptimizer( sim_fun = Simulator, inf_src= Information_Source_Generator,
					  lb_x = Simulator.xmin, ub_x = Simulator.xmax,
					  lb_a = Simulator.amin, ub_a = Simulator.amax,
					  distribution = "Exponential",
					  n_fun_init = 50,
					  n_inf_init = 5,
					  Budget = 100,
					  Nx = 100,
					  Na = 101,
					  Nd = 100,
					  GP_train = True,
					  GP_train_relearning = True,
					  var_data= None,
					  opt_method="BICO",
					  rep = str(rep+400),
					  save_only_last_stats=True,
					  calculate_true_optimum=False,
					  results_name="Production_line_BICO_RESULTS")

function_caller(rep=6)