# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)

import matplotlib.pyplot as plt
from IU_optimizer import *
from TestProblems import toyfun, toysource
from TestProblems.newsvendor import newsvendor_noisy

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
N = 10
tag = np.random.random(N)*100
for rp in range(N):
    [XA], [Y], [Data] = myoptimizer( sim_fun = newsvendor_noisy(), inf_src= toysource(lb =newsvendor_noisy().amin,ub=newsvendor_noisy().amax,d=1),
                          lb_x = newsvendor_noisy().xmin, ub_x = newsvendor_noisy().xmax,
                          lb_a = newsvendor_noisy().amin, ub_a = newsvendor_noisy().amax,
                          distribution = "MUSIG",
                          n_fun_init = 10,
                          n_inf_init = 0,
                          Budget = 12,
                          Nx = 100,
                          Na = 100,
                          Nd = 100,
                          GP_train = True,
                          GP_train_relearning = True,
                          var_data= 10,
                          opt_method="KG_DL",
                          rep = tag[rp] )