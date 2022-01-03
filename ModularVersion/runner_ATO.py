# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer import *
from IU_optimizer.utils import toysource as atosource
from TestProblems.ATO import assemble_to_order as atofun



# initilize the optimizer
myoptimizer = Mult_Input_Uncert(sim_fun = atofun(), inf_src= atosource(d=2),
                                 lb_x = atofun().xmin, ub_x = ambfun().xmax,
                                 lb_a = atofun().pmin, ub_a = ambfun().pmax,
                                 distribution = "MUSIG",
                                 n_fun_init = 40,
                                 n_inf_init = 0,
                                 Budget = 100,
                                 Nx = 101,
                                 Na = 1000,
                                 Nd = 1000,
                                 GP_train = True)

