from .utils import *
import GPy
import numpy as np


class Mult_Input_Uncert():

    def __init__(self):
        pass

    def __call__(self, sim_fun, inf_src ,
                      lb_x, ub_x,
                      lb_a, ub_a,
                      distribution="MU_t_S",
                      n_fun_init = 10,
                      n_inf_init = 0,
                      Budget = 9,
                      Nx = 101,
                      Na = 102,
                      Nd = 103,
                      var_data = None,
                      GP_train = True,
                      GP_train_relearning=False,
                      Gpy_Kernel = None,
                      opt_method = "BICO", rep = None,
                       save_only_last_stats=False,
                      calculate_true_optimum=True,
                    results_name="SIM_RESULTS"):

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
        :param Gpy_Kernel: GPy object, include GPy kernel with learnt hyperparameters.
        :param GP_train: Bool. True, Hyperparameters are trained in every iteration. False, uses pre-set parameters
        :param save_only_last_stats: Bool. True, only compute True performance in the end. False, compute at each
        iteration. True setting is recommended for expensive experiments.
        :param: calculate_true_optimum. True. Produces noisy performance instead of real expected performance.
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

        self.calculate_true_optimum = calculate_true_optimum
        self.kill_signal = False
        lb_x = lb_x.reshape(-1)
        ub_x = ub_x.reshape(-1)

        lb_a = lb_a.reshape(-1)
        ub_a = ub_a.reshape(-1)

        lb = np.concatenate((lb_x,lb_a))
        ub = np.concatenate((ub_x,ub_a))

        dim_X = sim_fun.dx

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        print("dimX", dim_X, "lb", lb.shape[0])
        assert dim_X < lb.shape[0], "More X dims than possible"
        assert lb.shape[0] == ub.shape[0], "bounds must be same shape"
        assert np.all(lb <= ub), "lower must be below upper!"

        assert ub.shape[0] == lb.shape[0], "lb and ub must be the same shape!"

        # set the distribution to use for A dimensions.
        if distribution == "trunc_norm":
            """
            trunc_norm_post: by specifying the variance of the data var_data, calculates data posterior
            and input posterior using uniform prior and gaussian likelihood.
            """
            if var_data is None:
                var_data = np.repeat(1,len(lb_a))

            post_maker = trunc_norm_post(amin=inf_src.lb, amax=inf_src.ub, var= var_data )
        elif distribution == "MU_t_S":
            """ 
            MUSIG_post: marginalises uncertainty over the variance
            """
            post_maker = MU_s_T_post(amin=inf_src.lb, amax=inf_src.ub)

        elif distribution == "MUSIG":
            """ 
            MUSIG_post: joint distribution mu and sigma given data. Same MUSIG_post but without marganilising variance
            """

            post_maker = Gaussian_musigma_inference(amin=inf_src.lb, amax=inf_src.ub, prior_n_pts=n_inf_init,
                                                    lb=lb_a,ub=ub_a, lbx=lb_x, ubx=ub_x)

        elif distribution == "Exponential":
            post_maker = Exponential_inference(amin=inf_src.lb, amax=inf_src.ub, prior_n_pts=n_inf_init,
                                                    lb=lb_a,ub=ub_a, lbx=lb_x, ubx=ub_x)

        else:
            raise NotImplementedError

        # we will need this for making discretizations.
        X_sampler = lambda n: lhs_box(n, lb[:dim_X], ub[:dim_X])

        # Initilize GP model
        print("\ninitial design")

        XA = lhs_box(n_fun_init, lb, ub)
        print("XA",XA)
        Y = sim_fun(XA[:,0:dim_X], XA[:,dim_X:XA.shape[1]])
        ker = GPy.kern.RBF(input_dim=lb.shape[0], variance=10000., lengthscale=(ub - lb) * 0.1, ARD=True)

        if Gpy_Kernel != None:
            ker = Gpy_Kernel



        # Initilize input uncertainty data via round robin allocation
        dim_A =  inf_src.n_srcs

        if type(n_inf_init) is int:
            alloc = np.arange(n_inf_init) % dim_A
            alloc = [np.sum(alloc == i) for i in range(dim_A)]
            Data = [inf_src(n=alloc[i], src=i) for i in range(dim_A)]
        elif (type(n_inf_init) is np.ndarray) | (type(n_inf_init) is list):
            n_inf_init = np.array(n_inf_init)
            n_inf_init = np.round(n_inf_init)

            assert np.any(n_inf_init <0) == False, "Please insert an allocation greater than 0"
            assert len(n_inf_init)  == dim_A, "Dimension of allocation for data sources != DimA"

            Data = [inf_src(n=n_inf_init[i], src=i) for i in range(dim_A)]
        else:
            print("please introduce np.array,or list to specify number of samples. Insert int for uniform allocation")
            raise

        # this can be called at any time to get the number of Data collected
        Ndata = lambda: np.sum([d_src.shape[0] for d_src in Data])


        print("Initialization complete, budget used: ", n_fun_init + np.sum(n_inf_init), "\n")

        print("\nStoring Data...")
        # Calculates statistics of the simulation run. It's decorated to save stats in a csv file.


        stats = store_stats(sim_fun, inf_src, dim_X, lb.shape[0], lb, ub, fp=str(np.sum(n_inf_init)), B=int(Budget-np.sum(n_inf_init)) , rep=rep,results_name=results_name,
                            save_only_last_stats=save_only_last_stats,calculate_true_optimum=self.calculate_true_optimum)

        # Let's get the party started!

        # GP Maximum likelihood training
        Gaussian_noise = 0.01
        if GP_train == True:
            # print("XA", XA)
            GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var=0.01)
            GPmodel.optimize_restarts(10, robust=True, verbose=True)
            GPmodel.Gaussian_noise.variance.constrain_bounded(1e-2, 2)
            Gaussian_noise = GPmodel.Gaussian_noise.variance
            if Gaussian_noise < 1e-9:
                Gaussian_noise = 1e-3

        else:
            GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), Gpy_Kernel, noise_var=0.01)


        while XA.shape[0] + Ndata() < Budget:
            start = time.time()
            print("Iteration ", XA.shape[0] + Ndata() + 1, ":")

            if GP_train_relearning == True:

                Gaussian_noise = 0.01

                GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var=0.01)
                GPmodel.Gaussian_noise.variance.constrain_bounded(1e-4, 0.2)
                GPmodel.optimize_restarts(10, robust=True, verbose=True)

                Gaussian_noise = GPmodel.Gaussian_noise.variance
                if Gaussian_noise < 1e-9:
                    Gaussian_noise = 1e-3
            else:
                GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var=0.01)
                    # GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var= Gaussian_noise)
            # Fit model to simulation data.

            # Discretize X by lhs and discretize A with posterior samples as required.
            X_grid = X_sampler(Nx)

            # be samples from posterior over A!
            A_density, A_sampler, _ = post_maker(Data)

            A_grid = [A_sampler(n=Na, src_idx=i) for i in range(inf_src.n_srcs)]
            W_A = [A_density(A_grid[i], src_idx=i) for i in range(inf_src.n_srcs)]


            # Get KG of both simulation and Input uncertainty.

            if XA.shape[0] + Ndata() == Budget-1:
                self.kill_signal = True
            else:
                self.kill_signal = False


            if opt_method == "BICO":

                XA,Y,Data = self.BICO_alg(sim_fun, inf_src,GPmodel, XA, Y, Data, X_grid,
                                           A_grid, W_A, post_maker,
                                           Nd, lb, ub,stats,dim_X)

            elif opt_method == "Benchmark":
                XA,Y,Data = self.KG_alg(sim_fun, inf_src,GPmodel, XA, Y, Data, X_grid,
                                           A_grid, W_A, post_maker,
                                           Nd, lb, ub,stats,dim_X)
            else:
                raise NotImplementedError

        return [XA], [Y], [Data]

    def BICO_alg(self , sim_fun,inf_src,GPmodel, XA,
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

        #Combutes Value of information for simualtion data
        topxa, topKG = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub)
        print("Best is simulator: ", topxa, topKG)

        #Computes Value of Information for external data
        topsrc, topDL = DeltaLoss(GPmodel, Data, X_grid, A_grid, W_A, post_maker,  lb, ub, Nd)
        print("Best is info source: ", topsrc, topDL)

        #Compute statistics and produce file
        stats(model = GPmodel,
              Data = Data,
              XA = XA,
              Y = Y,
              Decision = topDL>topKG,
              A_sample = A_grid,
              KG = [topxa, topKG],
              DL = [topsrc, topDL],
              kill_signal=self.kill_signal,
              HP_names=GPmodel.parameter_names(),
              HP_values=GPmodel.param_array)

        #Update Data set variables
        if topKG > topDL:
            # if simulation is better
            print("topxa", topxa)

            # print("topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]]",topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]])
            new_y = func_caller(func = sim_fun, x = topxa[:, 0:dim_X], a = topxa[:, dim_X:XA.shape[1]])

            XA = np.vstack([XA, topxa])
            Y = np.concatenate([Y, new_y])

        else:
            # if info source is better
            new_d = inf_src(n=1, src=topsrc)
            print("new_d", new_d, "Data[topsrc]", Data[topsrc])
            Data[topsrc] = np.concatenate([Data[topsrc], new_d.reshape(-1)])

        print("XA", XA)
        print("Y", Y)
        print("Data", Data)
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

        #Combutes Value of information for simualtion data
        topxa, topKG = KG_Mc_Input(GPmodel, X_grid, A_grid, lb, ub)
        print("Best is simulator: ", topxa, topKG)

        topsrc, topDL = [np.nan,np.nan]

        stats(GPmodel, Data, XA, Y, A_grid, [topxa, topKG], [topsrc, topDL], kill_signal=self.kill_signal,HP_names=GPmodel.parameter_names(),
              HP_values=GPmodel.param_array)

        print("topxa", topxa)

        new_y = func_caller( sim_fun, topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]])

        XA = np.vstack([XA, topxa])
        Y = np.concatenate([Y, new_y])
        return XA,Y,Data





