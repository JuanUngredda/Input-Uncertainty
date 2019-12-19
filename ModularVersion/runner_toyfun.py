# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer import *
from TestProblems import GP_test, toysource





# # now run the optimizer 100 times and save all outputs
"""
Choose optimsiation method between:
- KG_DL: Use Knowledge gradient and Delta Loss for sampling.
- KG_fixed_iu: use fixed quantity of data source points initially and optimise by KG.

Choose distribution method between:
-trunc_norm: Normal Likelihood and Uniform prior for input. Assumes known variance in the data.
-MUSIG : Normal Likelihood and Uniform prior for input. Assumes unknown variance in the data.

"""
def function_caller(rep):
    print("\nCalling optimizer")
    myoptimizer = Mult_Input_Uncert()
    np.random.seed(rep)

    # for i in range(0, 90, 5):
    #     print("i",i)
    i=5
    [XA], [Y], [Data] = myoptimizer(sim_fun = GP_test(x_dim=1, a_dim=2), inf_src= toysource(d=2),
                          lb_x = GP_test().xmin, ub_x = GP_test().xmax,
                          lb_a = GP_test().amin, ub_a = GP_test().amax,
                          distribution = "trunc_norm",
                          n_fun_init = 10,
                          n_inf_init = i,
                          Budget = 50,
                          Nx = 101,
                          Na = 100,
                          Nd = 100,
                          GP_train = False,
                          var_data= 10,
                          Gpy_Kernel = GP_test().KERNEL ,
                          opt_method="KG_fixed_iu",
                          rep = str(i) +"_"+str(rep))

function_caller(rep=5)