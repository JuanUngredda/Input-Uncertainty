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
                      opt_method = "KG_DL", rep = None,
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
        # print("ub_x,ub_a",ub_x,ub_a)
        lb = np.concatenate((lb_x,lb_a))
        ub = np.concatenate((ub_x,ub_a))
        # print("lb", lb, "ub", ub)
        dim_X = sim_fun.dx

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        print("dimX", dim_X, "lb", lb.shape[0])
        assert dim_X < lb.shape[0], "More X dims than possible"
        assert lb.shape[0] == ub.shape[0], "bounds must be same shape"
        assert np.all(lb <= ub), "lower must be below upper!"

        assert ub.shape[0] == lb.shape[0], "lb and ub must be the same shape!"
        # assert np.all(IU_dims<ub.shape[0]); "IU_dims out of too high!"
        # assert np.all(IU_dims>=0); "IU_dims too low!"

        # set the distribution to use for A dimensions.
        if distribution == "trunc_norm":
            """
            trunc_norm_post: by specifying the variance of the data var_data, calculates data posterior
            and input posterior using uniform prior and gaussian likelihood.
            """
            if var_data == None:
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
        # XA = np.array([[1.58629827e+00, 1.12826632e+00, 1.83781194e+00, 2.61517108e+00],
        #  [1.03466221e+00, 1.36917957e+00, 1.97372671e+00, 2.07025041e+00],
        #  [1.36024569e+00, 6.44166460e-01, 3.48116867e-01, 3.95743697e+00],
        #  [1.12443977e+00, 3.31711047e-01, 4.74528235e-01, 4.09505285e+00],
        #  [1.41562871e+00, 1.73214191e+00, 1.74370563e+00, 3.31912687e+00],
        #  [1.74792517e+00, 5.72845649e-01, 1.03539184e+00, 2.96859293e+00],
        #  [2.91744785e-01, 1.97030227e+00, 5.42175804e-01, 1.64590151e+00],
        #  [9.58222465e-01, 9.87260106e-01, 9.78933922e-01, 1.78155599e+00],
        #  [1.93642212e+00, 4.60958207e-03, 8.99947173e-01, 3.22521449e+00],
        #  [3.18628227e-01, 8.64292729e-01, 2.33225985e-01, 1.08673714e-02],
        #  [1.60270970e+00, 1.85782629e+00, 1.25251691e+00, 3.67033037e+00],
        #  [6.23222257e-01, 1.22800718e+00, 1.62663268e+00, 4.38956729e+00],
        #  [4.70498257e-01, 4.31467693e-01, 7.08319793e-01, 4.17128929e-01],
        #  [5.60822646e-01, 1.46637884e+00, 1.49428085e+00, 2.30452192e+00],
        #  [1.11642370e-01, 1.59115066e+00, 1.19093066e-01, 4.92064663e+00],
        #  [1.21427690e+00, 1.60386569e-01, 1.32825949e+00, 1.34847705e+00],
        #  [2.50923624e-02, 7.74135188e-01, 6.76816241e-02, 4.55092417e+00],
        #  [7.10003138e-01, 2.41878009e-01, 1.58153405e+00, 7.67380117e-01],
        #  [1.83936505e+00, 1.07215006e+00, 1.18262815e+00, 1.09955320e+00],
        #  [8.32289153e-01, 1.62409477e+00, 6.37302139e-01, 5.70758409e-01],
        #  [1.63815176e+00, 1.12101479e+00, 1.75206389e+00, 2.18173388e-01],
        #  [1.26471082e+00, 1.11451234e+00, 9.33643496e-01, 3.29236516e-01],
        #  [1.19310374e+00, 6.51616257e-01, 9.08950886e-01, 2.06495773e-01],
        #  [1.10540769e+00, 1.88056275e+00, 8.94925329e-01, 3.67143310e-01],
        #  [1.20448954e-01, 6.69445940e-01, 9.49352597e-01, 2.34854109e-01],
        #  [1.72618629e-04, 1.85343277e-01, 7.53126999e-01, 3.13257423e-01],
        #  [2.34834138e-01, 5.14390013e-01, 1.16910970e+00, 3.45051097e-01],
        #  [1.99833583e+00, 1.04161644e+00, 6.94053120e-01, 4.66191807e-01],
        #  [1.99951736e+00, 9.88672762e-01, 8.78755979e-01, 3.81259307e-01],
        #  [2.12786182e-01, 5.34723743e-01, 3.91692211e-01, 4.07637789e-01],
        #  [2.90507450e-04, 4.75542663e-01, 3.47846155e-02, 5.24134750e-01],
        #  [1.83868828e+00, 1.07659589e+00, 1.18990709e+00, 3.20475118e-01],
        #  [4.55828773e-01, 1.63549230e-03, 1.05942336e+00, 4.03091365e-01],
        #  [1.99286203e+00, 1.64558880e+00, 9.85976809e-01, 4.59362461e-01],
        #  [2.15321066e-01, 1.68928890e+00, 1.45496329e+00, 5.42903936e-01],
        #  [8.27088110e-01, 1.58677030e+00, 1.01210762e+00, 3.64956238e-01],
        #  [1.13153645e-01, 1.81816277e+00, 1.32270977e+00, 3.34935576e-01],
        #  [2.31452169e-01, 1.40361775e+00, 1.87945860e+00, 4.43185511e-01],
        #  [1.27224717e+00, 1.02572320e+00, 1.94677700e-01, 3.47050522e-01],
        #  [1.10953710e-01, 1.31699566e-01, 1.10407147e+00, 4.33359350e-01],
        #  [3.82197923e-01, 1.06618214e+00, 1.32591972e+00, 3.66344345e-01],
        #  [1.45922283e+00, 1.78571005e+00, 1.39653719e+00, 3.42294579e-01],
        #  [2.15757083e-01, 8.75000888e-01, 1.52816048e+00, 3.52775247e-01],
        #  [3.89882565e-01, 1.46353442e+00, 4.16761256e-01, 3.35508530e-01],
        #  [4.69672580e-01, 1.16214953e-01, 1.67491407e+00, 3.90846875e-01],
        #  [6.29830464e-02, 7.08129118e-01, 4.28931123e-01, 3.13029302e-01],
        #  [7.00932538e-01, 1.79655960e+00, 5.49410332e-01, 2.13723888e-01],
        #  [1.96421667e+00, 1.49084099e+00, 5.84478707e-01, 2.52741607e-01],
        #  [2.50525919e-01, 7.72892711e-01, 5.05885114e-01, 2.25691056e-01],
        #  [1.99999506e+00, 6.37101214e-01, 8.56043949e-01, 3.29540086e-01],
        #  [2.91259025e-01, 6.63407632e-01, 1.24135873e-01, 2.95107440e-01],
        #  [7.91622054e-01, 4.34609916e-01, 3.17057897e-01, 3.93523474e-01],
        #  [1.25998895e+00, 3.67648647e-01, 4.51302523e-01, 4.17388390e-01],
        #  [7.48407922e-02, 9.42574782e-01, 6.16939465e-01, 3.66173790e-01],
        #  [3.77544470e-01, 1.10004230e+00, 4.97470940e-01, 5.45145434e-01],
        #  [9.22215928e-01, 7.34079279e-01, 5.68838512e-01, 3.96105428e-01],
        #  [1.78338999e+00, 7.55123529e-01, 5.64083717e-01, 3.92343706e-01],
        #  [5.01113415e-01, 8.44659197e-01, 6.74819304e-01, 4.22160523e-01],
        #  [5.16428156e-01, 7.76743114e-01, 4.44515624e-01, 3.25276508e-01],
        #  [1.83121732e-01, 6.07249772e-01, 6.55141047e-01, 5.75589855e-01],
        #  [4.37020600e-01, 1.99288117e+00, 3.05021071e-03, 4.12429182e-01],
        #  [3.98153067e-02, 9.47132696e-01, 7.59311172e-01, 5.01216774e-01],
        #  [8.61432725e-03, 5.71159851e-01, 4.60458278e-01, 2.49375089e-01],
        #  [1.10140610e-02, 7.49240864e-01, 7.53210881e-01, 4.37796742e-01],
        #  [1.98438536e+00, 3.30036203e-04, 9.84214368e-02, 4.13858899e-01],
        #  [8.20517572e-02, 1.02266017e+00, 8.24831973e-01, 3.70597302e-01],
        #  [1.86387737e+00, 5.61994234e-01, 7.95587131e-01, 3.99951121e-01],
        #  [1.99962474e+00, 1.99997882e+00, 4.49316461e-01, 3.33258937e-01],
        #  [1.31773511e+00, 5.92591293e-01, 5.89073216e-01, 3.09074718e-01],
        #  [5.44665992e-01, 1.99887787e+00, 1.88471468e+00, 4.11389071e-01],
        #  [3.61804877e-01, 6.13203961e-01, 5.37150765e-01, 4.17432108e-01],
        #  [1.99993933e+00, 3.87904032e-01, 1.67266348e+00, 4.12152490e-01],
        #  [1.82238287e+00, 7.94392101e-01, 8.95277578e-01, 3.33852092e-01]])
        XA = lhs_box(n_fun_init, lb, ub)
        print("XA",XA)
        Y = sim_fun(XA[:,0:dim_X], XA[:,dim_X:XA.shape[1]])
        ker = GPy.kern.RBF(input_dim=lb.shape[0], variance=10000., lengthscale=(ub - lb) * 0.1, ARD=True)

        # ker.lengthscale.constrain_bounded(0, np.max(ub - lb) * 0.40)

        if Gpy_Kernel != None:
            print("true jeje")
            ker = Gpy_Kernel



        # Initilize input uncertainty data via round robin allocation
        dim_A =  inf_src.n_srcs #lb.shape[0] - dim_X
        # print("lb.shape[0], dim_X", lb.shape[0] ,dim_X)
        # raise
        alloc = np.arange(n_inf_init) % dim_A
        alloc = [np.sum(alloc == i) for i in range(dim_A)]
        Data = [inf_src(n=alloc[i], src=i) for i in range(dim_A)]
        print("DATA", Data)
        # this can be called at any time to get the number of Data collected
        Ndata = lambda: np.sum([d_src.shape[0] for d_src in Data])

        print("Initialization complete, budget used: ", n_fun_init + n_inf_init, "\n")

        print("\nStoring Data...")
        # Calculates statistics of the simulation run. It's decorated to save stats in a csv file.

        if save_only_last_stats:
            stats = store_stats(sim_fun, inf_src, dim_X, lb.shape[0], lb, ub, fp =str(n_inf_init), B=int(Budget-n_inf_init) ,rep = rep, results_name=results_name,calculate_true_optimum =self.calculate_true_optimum)
        else:
            stats = store_stats(sim_fun, inf_src, dim_X, lb.shape[0], lb, ub, results_name,fp=str(n_inf_init), rep=rep,results_name=results_name,
                                calculate_true_optimum=self.calculate_true_optimum)

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

        while XA.shape[0] + Ndata() < Budget:
            print("Iteration ", XA.shape[0] + Ndata() + 1, ":")

            if GP_train_relearning == True:

                Gaussian_noise = 0.01
                if GP_train == True:
                    # print("XA", XA)
                    # print("Y", Y)
                    GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var=0.01)
                    GPmodel.Gaussian_noise.variance.constrain_bounded(1e-4, 0.2)
                    GPmodel.optimize_restarts(10, robust=True, verbose=True)


                    Gaussian_noise = GPmodel.Gaussian_noise.variance
                    if Gaussian_noise < 1e-9:
                        Gaussian_noise = 1e-3
                    # GPmodel = GPy.models.GPRegression(XA, Y.reshape(-1, 1), ker, noise_var= Gaussian_noise)
            # Fit model to simulation data.

            # Discretize X by lhs and discretize A with posterior samples as required.
            X_grid = X_sampler(Nx)

            # KG+DL take a standard unweighted average over A_grid, i.e. A_grid must
            # be samples from posterior over A! Don't use linspace!

            A_density, A_sampler, _ = post_maker(Data)
            # print("inf_src.n_srcs",inf_src.n_srcs)


            A_grid = [A_sampler(n=Na, src_idx=i) for i in range(inf_src.n_srcs)]
            W_A = [A_density(A_grid[i], src_idx=i) for i in range(inf_src.n_srcs)]

            # print("A_grid[i]",A_grid[0][np.argmin(W_A)])
            # print("W_A", np.min(W_A))

            # Get KG of both simulation and Input uncertainty.

            if XA.shape[0] + Ndata() == Budget-1:
                self.kill_signal = True
            else:
                self.kill_signal = False


            if opt_method == "KG_DL":

                XA,Y,Data = self.KG_DL_alg(sim_fun, inf_src,GPmodel, XA, Y, Data, X_grid,
                                           A_grid, W_A, post_maker,
                                           Nd, lb, ub,stats,dim_X)
            elif opt_method == "KG_fixed_iu":
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

        topsrc, topDL = DeltaLoss(GPmodel, Data, X_grid, A_grid, W_A, post_maker,  lb, ub, Nd)
        print("Best is info source: ", topsrc, topDL)

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

        stats(GPmodel, Data, XA, Y, A_grid, [topxa, topKG], [topsrc, topDL], kill_signal=self.kill_signal,HP_names=GPmodel.parameter_names(),
              HP_values=GPmodel.param_array)

        print("topxa", topxa)

        new_y = func_caller( sim_fun, topxa[:, 0:dim_X], topxa[:, dim_X:XA.shape[1]])

        XA = np.vstack([XA, topxa])
        Y = np.concatenate([Y, new_y])
        return XA,Y,Data





