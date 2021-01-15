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

	# f = newsvendor()
	# x = np.linspace(f.xmin,f.xmax,100)
	# y = np.linspace(f.amin,f.amax,100)
	# X,Y = np.meshgrid(x,y)
	# plt.contourf(X,Y,f(x,y).reshape(len(x),len(y)))
	# plt.plot()
	# # now run the optimizer 100 times and save all outputs
	"""
	Choose optimsiation method between:
	- KG_DL: Use Knowledge gradient and Delta Loss for sampling.
	- KG_fixed_iu: use fixed quantity of data source points initially and optimise by KG.

	Choose distribution method between:
	-trunc_norm: Normal Likelihood and Uniform prior for input. Assumes known variance in the data.
	-MUSIG : Normal Likelihood and Uniform prior for input. Assumes unknown variance in the data.

	"""
	np.random.seed(rep+50)

	True_rate = 0.5
	True_Input_distributions = [expon(scale=np.reciprocal(True_rate))]  # [gamma(a=k,loc=0,scale=theta)]#
	Assumed_Input_Distributions = [np.random.normal]

	# plt.hist(True_Input_distributions[0].rvs(1000), bins=200, density=True)
	# plt.hist(np.random.normal(mu, np.sqrt(var), (1, 1000)).reshape(-1), bins=200, density=True)
	# plt.show()

	Simulator = Production_Line(True_rate=True_rate)
	Information_Source_Generator = Information_Source(Distribution=True_Input_distributions, lb=Simulator.amin,
													  ub=Simulator.amax, d=1)

	proportions = [25.0, 30.0, 35.0,40.0]#np.linspace(5,40,8)
	for i in proportions:
		[XA], [Y], [Data] = myoptimizer( sim_fun = Simulator, inf_src= Information_Source_Generator,
						  lb_x = Simulator.xmin, ub_x = Simulator.xmax,
						  lb_a = Simulator.amin, ub_a = Simulator.amax,
						  distribution = "Exponential",
						  n_fun_init = 20,
						  n_inf_init = i,
						  Budget = 100,
						  Nx = 100,
						  Na = 100,
						  Nd = 100,
						  GP_train = True,
						  GP_train_relearning = True,
						  var_data= None,
						  opt_method="KG_fixed_iu",
						  rep = str(rep+50),
						  save_only_last_stats=True,
					      calculate_true_optimum=False,
						  results_name="Production_line_Fixed_Proportion_")

# function_caller(rep=1)