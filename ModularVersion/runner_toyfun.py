# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer import *
from TestProblems import toyfun, toysource


print("\nCalling optimizer")
myoptimizer = Mult_Input_Uncert()


# # now run the optimizer 100 times and save all outputs
"""
Choose optimsiation method between:
- KG_DL: Use Knowledge gradient and Delta Loss for sampling.
- KG_fixed_iu: use fixed quantity of data source points initially and optimise by KG.

Choose distribution method between:
-trunc_norm: Normal Likelihood and Uniform prior for input. Assumes known variance in the data.
-MUSIG : Normal Likelihood and Uniform prior for input. Assumes unknown variance in the data.

"""
for rp in range(100):
    [XA], [Y], [Data] = myoptimizer(sim_fun = toyfun(), inf_src= toysource(d=1),
                          lb_x = toyfun().xmin, ub_x = toyfun().xmax,
                          lb_a = toyfun().amin, ub_a = toyfun().amax,
                          distribution = "trunc_norm",
                          n_fun_init = 10,
                          n_inf_init = 0,
                          Budget = 100,
                          Nx = 101,
                          Na = 100,
                          Nd = 100,
                          GP_train = True,
                          var_data= np.array([1]),
                          opt_method="KG_DL",
                          rep = rp)
