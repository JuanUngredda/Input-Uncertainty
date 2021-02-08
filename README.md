# Bayesian Optimisation vs. Input Uncertainty Reduction

Simulators often require calibration inputs estimated from real world data and the quality of the estimate can significantly
affect simulation output. Particularly when performing simulation optimisation to find an optimal solution, the uncertainty in
the inputs significantly affects the quality of the found solution. One remedy is to search for the solution that has the best
performance on average over the uncertain range of inputs yielding an optimal compromise solution. We consider the more
general setting where a user may choose between either running simulations or instead collecting real world data. A user
may choose an input and a solution and observe the simulation output, or instead query an external data source improving
the input estimate enabling the search for a more focused, less compromised solution. We explicitly examine the trade-off
between simulation and real data collection in order to find the optimal solution of the simulator with the true inputs. Using a
value of information procedure, we propose a novel unified simulation optimisation procedure called Bayesian Information
Collection and Optimisation (BICO) that, in each iteration, automatically determines which of the two actions (running
simulations or data collection) is more beneficial.

Code for paper "Bayesian Optimisation vs. Input Uncertainty Reduction" https://arxiv.org/abs/2006.00643

Author Details:
e-mail: 

## Requirements
We strongly recommend using the anaconda python distribution and python version 3.8.6. With anaconda you can install all packages as following 
by the following:

```
pip install -r requirements.txt
```

```
gpy version 1.9.9
matplotlib version 3.3.3
numba version 0.52.0
numpy version 1.19.4
pandas version 1.2.0
pydoe version 0.3.8
scipy version 1.6.0
```

# Tutorial

1-Import all functions related to the optimisation.

```
from IU_optimizer import *  #Imports the main function that performs the optimisation loop and stores all statistics.
from TestProblems.Toy import GP_test #Imports the "simualator". In this case is a synthetic test function
from TestProblems import Information_Source #Imports the external and uncontrollable synthetic information source.
```

2-Initialise the simulator class. It must accept a numpy nd.array "x" for design variables and numpy nd.array "w" as
input variables. 

```
#xamin: concatenated min range of design variables and input variables. The order is first all design variables and then
all input variables 
#xamax: concatenated max range of design variables and input variables. The order is first all design variables and then
all input variables 
#seed: random seed.
#x_dim: number of dimensions of design space
#a_dim: number of dimensions of input space
#true_params: True underlying parameters for "a"


Simulator = GP_test(xamin=[0,0,0], xamax=[100,100,100], seed=11, x_dim=1, w_dim=2, true_params=[mu0,mu1])
```

An example of a simulator class structure can be seen below. Simulator_function() should perform all the operations
done by a simulator. Further examples in ./ModularVersion/TestProblems

```
class Simulator_Class(testfunction):
    """
    A toy function    
    ARGS
     min: scalar defining min range of inputs
     max: scalar defining max range of inputs
     seed: int, RNG seed
     x_dim: designs dimension
     a_dim: input dimensions
     x: n*x_dim matrix, points in space to eval testfun
     w: n*w_dim matrix, points in space to eval testfun
     NoiseSD: additive gaussaint noise SD
    
    RETURNS
     output: vector of length nrow(xa)
     """
    
        def __init__(self, xamin=[0,0], xamax=[100,100], seed=11, x_dim=1, w_dim=1, true_params=[40]):
            
            #Initialise class variables
            self.seed = seed
            self.dx = x_dim
            self.da = a_dim
            self.dxa = x_dim + a_dim
    
            self.xmin = np.zeros((self.dx,))
            self.xmax = np.ones((self.dx,))*xamax[:self.dx]
    
            # uncertainty parameter
            self.amin = np.zeros((self.da,))
            self.amax = np.ones((self.da,))*xamax[self.da:]
    
            self.xamin = np.concatenate((self.xmin,self.amin))
            self.xamax = np.concatenate((self.xmax,self.amax))
            self.true_params = true_params
    
        def __call__(self, x, w=None, noise_std=0.01, true_performance_flag=True):
    
            assert x.shape[0] == w.shape[0], "wrong x or u dimensions"
            assert len(x.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert len(w.shape) == 2, "x must be an N*d matrix, each row a d point"
            assert x.shape[1] == self.dx, "Test_func: wrong dimension inputed"
            assert w.shape[1] == self.da, "Test_func: wrong dimension inputed"
    
            out = simulator_function(x,w,  true_performance_flag=True)
            
            return out
```

3-Initialise the External Data source Class. This is a Class that produces different values
for a random variable when is queried. It needs as an input a distribution. For simplicity,
we use the parametric family of functions in scipy.

```
True_Input_distributions = [norm(loc=mu0, scale=np.sqrt(var0)), norm(loc=mu1, scale=np.sqrt(var1)),]
Information_Source_Generator = Information_Source(Distribution=True_Input_distributions, lb=np.zeros(2),
                                                  ub=np.ones(2)*100, d=2)
```

4- Run the optimiser with the BICO parameters, Simulator and External data source as an input. For every run of the 
algorithm, solutions to replicate the results are shown in /RESULTS folder with the name given in results_name="Results_name".

```
 Optimizes the test function integrated over IU_dims. The integral
        is also changing over time and learnt.

        sim_fun: callable simulator function, input (x,a), returns scalar
        inf_src: callable data source function, returns scalar
        lb_x: lower bounds on (x) vector design to sim_fun
        ub_x: upper bounds on (x) vector design to sim_fun
        lb_a: lower bounds on (a) vector input to sim_fun
        ub_a: upper bounds on (a) vector input to sim_fun
        distribution: which prior/posterior to use for the uncertain parameters
        n_fun_init: number of inital points for GP model
        n_inf_init: number of intial points for info source
        Budget: total budget of calls to test_fun and inf_src
        Nx: int, discretization size of X
        Na: int, sample size for MC over A
        Nd: int, sample size for Delta Loss
        Gpy_Kernel: GPy object, include GPy kernel with learnt hyperparameters.
        GP_train: Bool. True, Hyperparameters are trained in every iteration. False, uses pre-set parameters
        opt_method: "BICO" proposed approach in the paper. "Benchmark" two-stage approach described in the paper.
        save_only_last_stats: Bool. True, only compute True performance in the end. False, compute at each
        iteration. True setting is recommended for expensive experiments.
        calculate_true_optimum. True. Produces noisy performance instead of real expected performance.
        opt_method: Method for IU-optimisation:
               -"KG_DL": Compares Knowledge Gradient and Delta Loss for every iteration of the algorithm.
               -"KG_fixed_iu": Updates the Data/Input posterior initially with n_inf_init and only
               uses Knowledge gradient.

        :param rep: int, number that identifies one specific run of the experiment (whole budget)


myoptimiser = Mult_Input_Uncert()
myoptimiser(sim_fun = Simulator, inf_src= Information_Source_Generator,
                            lb_x=Simulator.xmin, ub_x=Simulator.xmax,
                            lb_a=Simulator.amin, ub_a=Simulator.amax,
                            distribution = "trunc_norm",
                            n_fun_init=10,
                            n_inf_init=4,
                            Budget=100,
                            Nx=100,
                            Na=150,
                            Nd=200,
                            GP_train=False,
                            GP_train_relearning=False,
                            opt_method="BICO",
                            Gpy_Kernel=Simulator.KERNEL,
                            rep=str(rep),
                            save_only_last_stats=False,
                            calculate_true_optimum=False,
                            results_name="Results_name")

```

