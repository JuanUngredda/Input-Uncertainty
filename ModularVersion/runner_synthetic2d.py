# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer import *
from TestProblems import toyfun, toysource
from TestProblems.experiments2d import rosenbrock

print("\nCalling optimizer")
myoptimizer = Mult_Input_Uncert()
rosenbrock().plot()
# # now run the optimizer 100 times and save all outputs
"""
Choose optimsiation method between:
- KG_DL: Use Knowledge gradient and Delta Loss for sampling.
- KG_fixed_iu: use fixed quantity of data source points initially and optimise by KG.

Choose distribution method between:
-trunc_norm: Normal Likelihood and Uniform prior for input. Assumes known variance in the data.
-MUSIG : Normal Likelihood and Uniform prior for input. Assumes unknown variance in the data.

"""
for rp in range(1):
    [XA], [Y], [Data] = myoptimizer(sim_fun = rosenbrock(), inf_src= toysource(lb =rosenbrock().amin,ub=rosenbrock().amax,d=1),
                          lb_x = rosenbrock().xmin, ub_x = rosenbrock().xmax,
                          lb_a = rosenbrock().amin, ub_a = rosenbrock().amax,
                          distribution = "trunc_norm",
                          n_fun_init = 10,
                          n_inf_init = 0,
                          Budget = 100,
                          Nx = 101,
                          Na = 100,
                          Nd = 100,
                          GP_train = True,
                          var_data= 1,
                          opt_method="KG_fixed_iu",
                          rep = rp)
