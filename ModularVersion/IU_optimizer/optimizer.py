from .utils import *
import GPy
import numpy as np


class Mult_Input_Uncert():

    def __init__(self):
        pass

    def __call__(self, sim_fun, inf_src ,
                      lb_x, ub_x,
                      lb_a, ub_a,
                      distribution="MUSIG",
                      n_fun_init = 10,
                      n_inf_init = 0,
                      Budget = 9,
                      Nx = 101,
                      Na = 102,
                      Nd = 103,
                      var_data = None,
                      GP_train = True,
                      opt_method = "KG_DL", rep = None):

        """
        Optimizes the test function integrated over IU_dims. The integral
        is also changing over time and learnt.

        :param sim_fun: callable simulator function, input (x,a), returns scalar
        :param inf_src: callable data source function, returns scalar
        :param lb_x: lower bounds on (x) vector design to sim_fun
        :param ub_x: upper bounds on (x) vector design to sim_fun
        :param lb_a: lower bounds on (a) vector input to sim_fun
        :param ub_a: upper bounds on (a) vector input to sim_fun
        :param distribution: which prior/posterior to use for the uncertain parameters
        :param n_fun_init: number of inital points for GP model
        :param n_inf_init: number of intial points for info source
        :param Budget: total budget of calls to test_fun and inf_src
        :param Nx: int, discretization size of X
        :param Na: int, sample size for MC over A
        :param Nd: int, sample size for Delta Loss
        :param var_data: int, True variance of Data for inference in distribution "KG_fixed_iu".
        :param GP_train: Bool. True, Hyperparameters are trained in every iteration. False, uses pre-set parameters
        :param opt_method: Method for IU-optimisation:
               -"KG_DL": Compares Knowledge Gradient and Delta Loss for every iteration of the algorithm.
               -"KG_fixed_iu": Updates the Data/Input posterior initially with n_inf_init and only
               uses Knowledge gradient.

        :param rep: int, number that identifies one specific run of the experiment (whole budget)

        RETURNS
            X: observed test_func inputs
            Y: observed test_func outputs
            Data: list of array of inf_src observations

            """

        lb_x = lb_x.reshape(-1)
        ub_x = ub_x.reshape(-1)

        lb_a = lb_a.reshape(-1)
        ub_a = ub_a.reshape(-1)

        lb = np.concatenate((lb_x,lb_a))
        ub = np.concatenate((ub_x,ub_a))

        dim_X = sim_fun.dx

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)

        assert dim_X < lb.shape[0], "More X dims than possible"
        assert lb.shape[0] == ub.shape[0], "bounds must be same shape"
        assert np.all(lb <= ub), "lower must be below upper!"

        assert ub.shape[0] == lb.shape[0], "lb and ub must be the same shape!"
        # assert np.all(IU_dims<ub.shape[0]); "IU_dims out of too high!"
        # assert np.all(IU_dims>=0); "IU_dims too low!"

        # set the distribution to use for A dimensions.
        if distribution is "trunc_norm":
            """
            trunc_norm_post: by specifying the variance of the data var_data, calculates data posterior
            and input posterior using uniform prior and gaussian likelihood.
            """
            if var_data is None:
                var_data = np.repeat(1,len(lb_a))

            post_maker = trunc_norm_post(amin=inf_src.lb, amax=inf_src.ub, var= var_data )
        elif distribution is "MUSIG":
            """ 
            trunc_norm_post: by specifying the variance of the data var_data, calculates data posterior
            and input posterior using uniform prior and gaussian likelihood.
            """
            post_maker = MUSIG_post(amin=inf_src.lb, amax=inf_src.ub)
        else:
            raise NotImplementedError

        # we will need this for making discretizations.
        X_sampler = lambda n: lhs_box(n, lb[:dim_X], ub[:dim_X])

        # Initilize GP model
        print("\ninitial design")
        XA = lhs_box(n_fun_init, lb, ub)
        Y = sim_fun(XA[:,0:dim_X], XA[:,dim_X:XA.shape[1]])
        ker = GPy.kern.RBF(input_dim=lb.shape[0], variance=10000., lengthscale=(ub - lb) * 0.1, ARD=True)

        # Initilize input uncertainty data via round robin allocation
        dim_A = lb.shape[0] - dim_X
        alloc = np.arange(n_inf_init) % dim_A
        alloc = [np.sum(alloc == i) for i in range(dim_A)]
        Data = [inf_src(n=alloc[i], src=i) for i in range(dim_A)]

        # this can be called at any time to get the number of Data collected
        Ndata = lambda: np.sum([d_src.shape[0] for d_src in Data])

        print("Initialization complete, budget used: ", n_fun_init + n_inf_init, "\n")

        print("\nStoring Data...")
        # Calculates statistics of the simulation run. It's decorated to save stats in a csv file.


        stats = store_stats(sim_fun, inf_src, dim_X, lb.shape[0], lb, ub, rep = rep)

        # Let's get the party started!

        while XA.shape[0] + Ndata() < Budget:
            print("Iteration ", XA.shape[0] + Ndata() + 1, ":")

            #GP Maximum likelihood training
            Gaussian_noise = 0.01
            if GP_train == True:
                GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var=0.01)
                GPmodel.optimize_restarts(10, robust=True, verbose=True)
                Gaussian_noise = GPmodel.Gaussian_noise.variance
                if Gaussian_noise < 1e-9:
                    Gaussian_noise = 1e-3

            # Fit model to simulation data.
            GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var= Gaussian_noise)

            # Discretize X by lhs and discretize A with posterior samples as required.
            X_grid = X_sampler(Nx)

            # KG+DL take a standard unweighted average over A_grid, i.e. A_grid must
            # be samples from posterior over A! Don't use linspace!

            A_density, A_sampler, _ = post_maker(Data)
            A_grid = [A_sampler(n=Na, src_idx=i) for i in range(inf_src.n_srcs)]
            W_A = [A_density(A_grid[i], src_idx=i) for i in range(inf_src.n_srcs)]

            # Get KG of both simulation and Input uncertainty.
            if opt_method is "KG_DL":
                XA,Y,Data = self.KG_DL_alg(sim_fun, inf_src,GPmodel, XA, Y, Data, X_grid,
                                           A_grid, W_A, post_maker,
                                           Nd, lb, ub,stats,dim_X)
            elif opt_method is "KG_fixed_iu":
                XA,Y,Data = self.KG_alg(sim_fun, inf_src,GPmodel, XA, Y, Data, X_grid,
                                           A_grid, W_A, post_maker,
                                           Nd, lb, ub,stats,dim_X)
            else:
                raise NotImplementedError

        return [XA], [Y], [Data]

    def KG_DL_alg(self , sim_fun,inf_src,GPmodel, XA,
                  Y, Data, X_grid, A_grid, W_A,
                  post_maker, Nd, lb, ub,stats,dim_X):
        """

        :param sim_fun: callable simulator function, input (x,a), returns scalar
        :param inf_src: callable data source function, returns scalar
        :param GPmodel: Trained Gaussian Process model
        :param XA: ndarray, observed test_func inputs
        :param Y: ndarray, output of test_func inputs
        :param Data: list of arrays, observed output data from input source n_source
        :param X_grid: X grid set by latin hypercube
        :param A_grid: A grid specified by sampling from input posterior
        :param W_A: density evaluated at specified grid.
        :param post_maker: object, object that collects posterior input density, posterior density sampler,
        posterior data sampler.
        :param Nd: int, sample size for Delta Loss
        :param lb: concatenated lower bounds (x,a)
        :param ub: concatenated upper bounds (x,a)
        :param stats: object, storing stats functions
        :param dim_X: dimensionality of input design for simulator function

        RETURNS
            X: observed test_func inputs
            Y: observed test_func outputs
            Data: list of array of inf_src observations
        """

        topxa, topKG = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub)
        print("Best is simulator: ", topxa, topKG)

        topsrc, topDL = DeltaLoss(GPmodel, Data, X_grid, A_grid, W_A, post_maker, Nd)
        print("Best is info source: ", topsrc, topDL)

        stats(model = GPmodel,
              Data = Data,
              XA = XA,
              Y = Y,
              A_sample = A_grid,
              KG = [topxa, topKG],
              DL = [topsrc, topDL],
              HP_names=GPmodel.parameter_names(),
              HP_values=GPmodel.param_array)

        if topKG > topDL:
            # if simulation is better
            print("topxa", topxa)

            print("topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]]",topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]])
            new_y = func_caller(func = sim_fun, x = topxa[:, 0:dim_X], a = topxa[:, dim_X:XA.shape[1]])

            XA = np.vstack([XA, topxa])
            Y = np.concatenate([Y, new_y])

        else:
            # if info source is better
            new_d = inf_src(n=1, src=topsrc)
            print("new_d", new_d, "Data[topsrc]", Data[topsrc])
            Data[topsrc] = np.concatenate([Data[topsrc], new_d.reshape(-1)])
        return XA,Y,Data

    def KG_alg(self , sim_fun,inf_src,GPmodel, XA,
                  Y, Data, X_grid, A_grid, W_A,
                  post_maker, Nd, lb, ub,stats,dim_X):

        """

        :param sim_fun: callable simulator function, input (x,a), returns scalar
        :param inf_src: callable data source function, returns scalar
        :param GPmodel: Trained Gaussian Process model
        :param XA: ndarray, observed test_func inputs
        :param Y: ndarray, output of test_func inputs
        :param Data: list of arrays, observed output data from input source n_source
        :param X_grid:
        :param A_grid:
        :param W_A:
        :param post_maker:
        :param Nd: int, sample size for Delta Loss
        :param lb: concatenated lower bounds (x,a)
        :param ub: concatenated upper bounds (x,a)
        :param stats: object, storing stats functions
        :param dim_X: dimensionality of input design for simulator function

        RETURNS
            X: observed test_func inputs
            Y: observed test_func outputs
            Data: list of array of inf_src observations
        """

        topxa, topKG = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub)
        print("Best is simulator: ", topxa, topKG)

        topsrc, topDL = [np.nan,np.nan]

        stats(GPmodel, Data, XA, Y, A_grid, [topxa, topKG], [topsrc, topDL], HP_names=GPmodel.parameter_names(),
              HP_values=GPmodel.param_array)

        print("topxa", topxa)

        new_y = func_caller( sim_fun, topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]])

        XA = np.vstack([XA, topxa])
        Y = np.concatenate([Y, new_y])
        return XA,Y,Data





