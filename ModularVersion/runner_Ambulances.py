
# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)

from IU_optimizer import *
from TestProblems import toysource as ambsource
from TestProblems.Ambulance_simulator import Ambulance_Delays as ambfun

print("\nCalling optimizer")
myoptimizer = Mult_Input_Uncert()

# initilize the optimizer
for rp in range(1000):
    [XA], [Y], [Data] = myoptimizer(sim_fun = ambfun(), inf_src= ambsource(d=2),
                                     lb_x = ambfun().xmin, ub_x = ambfun().xmax,
                                     lb_a = ambfun().pmin, ub_a = ambfun().pmax,
                                     distribution = "MUSIG",
                                     n_fun_init = 40,
                                     n_inf_init = 0,
                                     Budget = 100,
                                     Nx = 101,
                                     Na = 100,
                                     Nd = 100,
                                     GP_train = True, rep = rp)
