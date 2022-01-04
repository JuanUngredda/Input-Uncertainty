# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer import *
from TestProblems.Toy import GP_test
from TestProblems import Information_Source

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

    rep = rep
    var_mix = [[5,10]]
    for v_mx in var_mix:
        mu0 = 40
        mu1 = 40
        var0 = 5
        var1 = 10

        True_Input_distributions = [norm(loc=mu0, scale=np.sqrt(var0)), norm(loc=mu1, scale=np.sqrt(var1)),]  # [gamma(a=k,loc=0,scale=theta)]#

        Information_Source_Generator = Information_Source(Distribution=True_Input_distributions, lb=np.zeros(2),
                                                          ub=np.ones(2)*100, d=2)
        Simulator = GP_test(xamin=[0,0,0], xamax=[100,100,100], seed=11, x_dim=1, a_dim=2, true_params=[mu0,mu1])


        for m0 in range(2,50,5):
            for m1 in range(2,50,5):
                np.random.seed(rep)
                [XA], [Y], [Data] = myoptimizer(sim_fun = Simulator, inf_src= Information_Source_Generator,
                                    lb_x=Simulator.xmin, ub_x=Simulator.xmax,
                                    lb_a=Simulator.amin, ub_a=Simulator.amax,
                                    distribution = "trunc_norm",
                                    n_fun_init=10,
                                    n_inf_init=np.array([m0,m1]),
                                    Budget=100,
                                    Nx=100,
                                    Na=150,
                                    Nd=200,
                                    GP_train=False,
                                    GP_train_relearning=False,
                                    var_data=np.array([var0,var1]),
                                    opt_method="Benchmark",
                                    Gpy_Kernel=Simulator.KERNEL,
                                    rep=str(rep),
                                    save_only_last_stats=False,
                                    calculate_true_optimum=False,
                                    results_name="synthetic_different_vars_fp_"+str(m0)+"_"+str(m1)+"_RESULTS")

function_caller(rep=9)