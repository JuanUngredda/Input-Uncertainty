import numpy as np
import itertools
import pandas as pd
from pyDOE import lhs
from scipy import optimize
from scipy.optimize import minimize
from scipy.stats import norm
import os
from functools import wraps
import logging
from matplotlib import pyplot as plt
import sys
import __main__ as main
import pandas as pd
import time

from scipy.stats import gamma
from scipy.stats import invgamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import t


# In this file put any simple little functions that can be abstracted out of the
# main code, eg samplers, plotting, saving, more complex mathematical operations
# tat don't have any persistent state etc.



logger = logging.getLogger(__name__)


class writeCSV_time_stats():
    """
    This decorator saves the running times and then converts it into a csv file
    """
    def __init__(self, function):
        self.function = function
        self.folder = os.path.basename(sys.argv[0])[0:-3] + "_stats/" + "time"
        if os.path.isdir(self.folder) == False:
            os.makedirs(self.folder)
        self.time_data = {}

    def __call__(self, *args, **kwargs):

        start = time.time()
        result = self.function(*args, **kwargs)
        end = time.time()

        if self.function.__name__ not in self.time_data.keys():
            self.time_data[self.function.__name__] = [end - start]
        else:
            self.time_data[self.function.__name__].append(end - start)

        data = self.time_data
        gen_file = pd.DataFrame.from_dict(data)
        script_dir = os.path.dirname(__file__)
        project_path = script_dir[:-27]
        if not os.path.exists(project_path +"RESULTS" +  "/" + self.folder):
            os.makedirs(project_path +"RESULTS" +  "/" + self.folder)
        path = project_path +"RESULTS" +  "/" + self.folder + '/time_'+ self.function.__name__ + '.csv'
        gen_file.to_csv(path_or_buf=path)
        return result

def writeCSV_run_stats():
    """
    :return: generates file with statistics from store_stats()
    """
    folder = os.path.basename(sys.argv[0])[0:-3] + "_stats/" + "run_stats"
    if os.path.isdir(folder) == False:
        os.makedirs(folder)

    def decorator(func):
        def wrapcsv(*args, **kwargs):
            data = func(*args, **kwargs)
            gen_file = pd.DataFrame.from_dict(data)
            script_dir = os.path.dirname(__file__)
            project_path = script_dir[:-27]

            if not os.path.exists(project_path + "RESULTS" + "/" + folder):
                os.makedirs(project_path + "RESULTS" + "/" + folder)
            path = project_path +"RESULTS" + "/" + folder + '/stats_'+ str(data['file_number'])+'.csv'
            print("path", path)

            gen_file.to_csv(path_or_buf=path)
            return data
        return wrapcsv
    return decorator

@writeCSV_time_stats
def func_caller(func, x, a):
    """calls the simulation function and return its value."""
    return func(x, a)

class store_stats():
    """
    Calculates the important statistics that will be recorded during the run of the algorithm. Specifically,
    X_r: Recommended design using the mean of the Gaussian process.
    OC: Opportunity cost. difference between true function evaluated at best design with true inputs and
      function evaluated at recommended design with true inputs. This is done by discretising the space using
      latin hypercube to find the true and recommended value. The discretisation is the same in both cases so the max
      values between the two can be compared.
    KG: Knowledge Gradient recommended design and value for that design.
    DL: Delta Loss recommended source and value of Delta Loss
    HP: Hyperparameters of gaussian process name/values.


    """
    def __init__(self,test_func,test_infr, dimX , dimXA, lb,ub,  rep, max_prob = True):

        self.rep = rep
        self.lb = lb
        self.ub = ub
        self.dimX = dimX
        self.dimXA = dimXA
        self.dimA = dimXA - dimX
        self.test_func = test_func
        self.test_infr = test_infr
        self.tp = (1)
        self.dimX = dimX
        self.lb_X = lb[:dimX]
        self.ub_X = ub[:dimX]
        self.Xd = lhs_box(50000, self.lb_X, self.ub_X)
        self.top_XA, self.best_quality = self.find_top_X()
        self.X_r = []
        self.OC = []
        self.KG = []
        self.DL = []
        self.HP_names = []
        self.HP_values = []
        self.Best_Quality = []
        self.Best_Recommended_Quality = []
        self.mean_input= []
        self.P5_input = []
        self.P95_input = []
        self.Decision = []

    def __call__(self,model, Data, XA, Y, A_sample, KG = np.nan, DL = np.nan ,Decision =None,HP_names = None,HP_values=None):

        # print("ENTEREEEEEEEEEEEEEEEEEEEED")

        mean_input = np.mean(A_sample[0],axis=0)
        P5 = np.percentile(A_sample[0],5,axis=0)
        P95 = np.percentile(A_sample[0],95,axis=0)
        X_r = self.recommended_X( model,A_sample)
        OC, val_recom, val_opt = self.Opportunity_cost(X_r)
        self.Decision.append(Decision)
        self.X_r.append(X_r)
        self.OC.append(OC)
        self.KG.append(KG)
        self.DL.append(DL)
        self.HP_names.append(HP_names)
        self.HP_values.append(HP_values)
        self.Best_Recommended_Quality.append(val_recom)
        self.Best_Quality.append(val_opt)

        self.mean_input.append(mean_input)
        self.P5_input.append(P5)
        self.P95_input.append(P95)

        ##Adjusting variables to proper data types:
        OC = np.array(self.OC).reshape(-1)
        best_r_quality = np.array(self.Best_Recommended_Quality).reshape(-1)
        Best_Quality = np.array(self.Best_Quality).reshape(-1)
        Decision = np.array(self.Decision).reshape(-1)


        #file accesed by the decorator to be printed in a csv file.
        registered_vars = self.log_file(X_r = self.X_r,
                                        Decision = Decision,
                                        best_r_quality = best_r_quality,
                                        top_quality = Best_Quality,
                                        OC=OC,
                                        mean_input = self.mean_input,
                                        P5_input = self.P5_input,
                                        P95_input = self.P95_input,
                                        KG = self.KG,
                                        DL = self.DL,
                                        HP_names = self.HP_names,
                                        HP_vars = self.HP_values,
                                        file_number = self.rep)

        return registered_vars

    @writeCSV_run_stats()
    def log_file(self, **kwargs):
        '''
        dummy function. receives and outputs the same arguments. but allows writeCSV to pick up the names of the
        variables to be printed in the csv file
        '''
        return kwargs

    def find_top_X(self):
        """
        Finds best value of true function given discretisation lhs_box.
        :return: narray. design from max test function
        """
        Xd = lhs_box(n=20, lb=self.lb_X, ub = self.ub_X)

        def best_value_function(X):
            X = np.atleast_2d(X)
            Nx = X.shape[0]
            if np.any(X < self.lb_X) or np.any(X > self.ub_X):
                return (1000000)
            else:
                M_Xd = self.test_func(X, None, true_performance_flag=True)
                return -M_Xd



        M_Xd = best_value_function(Xd)
        # a = [[40,10]]
        # reps = len(Xd)
        # noisy_function = self.test_func(Xd, np.array(a*reps), true_performance_flag=False)
        # plt.scatter(Xd, -M_Xd)
        # plt.scatter(Xd, noisy_function)
        # plt.show()
        self.M_true = M_Xd

        anchor_point = Xd[np.argmin(M_Xd)]
        topX = [minimize(best_value_function, x, method='nelder-mead').x for x in anchor_point]

        self.topX = np.array(topX)
        best_quality = self.test_func(self.topX,None, true_performance_flag=True)
        # print("self.topX",self.topX)
        print("self.topX, best_quality",self.topX, best_quality)
        return self.topX, best_quality

    def Opportunity_cost(self, X_r):
        """

        :param X_r: Recommended design given lhs discretisation.
        :return: OC: int, Opportunity cost. difference between true function evaluated at best design with true inputs and
      function evaluated at recommended design with true inputs. This is done by discretising the space using
      latin hypercube to find the true and recommended value. The discretisation is the same in both cases so the max
      values between the two can be compared.
        """

        # XA_r = np.c_[X_r, [self.test_infr.f_mean]]
        X_r = np.atleast_2d(X_r) #np.vstack(np.array([X_r]))
        self.topX = np.atleast_2d(self.topX)

        val_recom = self.test_func(X_r, None, true_performance_flag=True)
        val_opt = self.test_func(self.topX, None, true_performance_flag=True)

        OC = val_opt - val_recom

        return OC, val_recom, val_opt

    def recommended_X(self, model, A_sample):
        """

        :param model: trained GP model
        :param A_sample: Sample from posterior input distribution to marginilise input.
        :return: recommended design, X_r. using lhs discretisation
        """
        print("recommended X to choose...")

        Xd = self.Xd

        Ad = np.concatenate(A_sample)
        Nx = Xd.shape[0]
        Na = Ad.shape[0]
        # print("Ad", Ad)
        lb = self.lb.reshape(-1)
        ub = self.ub.reshape(-1)

        def marginal_current(X, var_flag=False):

            if np.any(X < lb[:Xd.shape[1]]) or np.any(X > ub[:Xd.shape[1]]):
                return (1000000)
            else:
                X = np.atleast_2d(X)
                Nx_internal = X.shape[0]
                XdAd = np.hstack([np.repeat(X, Na, axis=0),
                                  np.tile(Ad, (Nx_internal, 1))])


                M_XA = model.predict(XdAd)[0].reshape(Nx_internal, Na)
                # now we have weights, get the peak of reweighted GP means
                M_X_i = np.mean(M_XA, axis=1)
                if var_flag:
                    var_XA = model.predict(XdAd)[1].reshape(Nx_internal, Na)
                    # now we have weights, get the peak of reweighted GP means
                    varX = np.mean(var_XA, axis=1)
                    return -M_X_i, varX
                return -M_X_i

        M_X, varX = marginal_current(Xd, var_flag=True)
        current_top_X = Xd[np.argmin(M_X)]

        X_r = np.array(
            [minimize(marginal_current, x_discrete, method='Nelder-Mead').x for x_discrete in current_top_X]).reshape(
            -1)

        print("recommended design", X_r)
        # plt.scatter(Xd, -M_X )
        # plt.scatter(Xd, -M_X + 1.96*np.sqrt(varX ))
        # plt.scatter(Xd, -M_X - 1.96*np.sqrt(varX ))
        # plt.show()
        return X_r

class Gaussian_musigma_inference():

    """ Given i.i.d observations, builds a posterior density and a sampler
    (which can then be used with Delta Loss).
    Inference details:
    Normal Likelihood
    Prior over mu and variance

    ARGS
        src_data: matrix of observations for a given source
        xmin = lower bound
        xmax = upper bound


    RETURNS
        post_dens: pdf over input domain A. Method: iu_pdf
        post_dens_samples: samples over post_dens. Method: iu_pdf_sampler
        post_predict_sampler: samples from posterior predictive density ynew. Method: data_predict_sampler

    """
    def __init__(self, amin=0, amax=100, prior_n_pts=5, lb=None, ub=None, lbx=None, ubx=None):
        self.xmin = amin
        self.xmax = amax
        self.h = 101
        self.prior_n_pts = prior_n_pts
        self.sig_arr, self.dltsig = np.linspace(1e-3, 20, self.h, retstep=True)
        self.a_arr, self.dlta = np.linspace(1e-3, self.xmax, self.h, retstep=True)
        self.y_n1, self.dlty_n1 = np.linspace(0, self.xmax, 5000, retstep=True)
        self.lb = lb
        self.ub = ub
        self.lbx = lbx
        self.ubx = ubx
        print("Unknown mu and sigma posterior distribution")

    def __call__(self, src_data):
        """

        :param src_data: list of narrays. include whole data.
        :return: functions post_dens: posterior density distribution given source. post_A_sampler: samples from
        posterior density distribution.
        """
        self.Data_post = src_data
        return self.post_dens, self.post_A_sampler, self.post_Data_sampler

    def post_dens(self, a, src_idx):
        """
        Posterior marginilised density estimation over A. First models posterior over
        the parameter A and uncertainty sigma from input source. Then marginilises out sigma
        and normalise the distribution over A

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        # print("DENSITY")
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"
        assert a.shape[1]==2, "more than 2 inputs"
        assert self.prior_n_pts >3, "include at least 3 prior data points from external source"
        # print("post dens")
        # print("a",a)

        # print("self.Data_post", self.Data_post)
        self.Data_i = self.Data_post[src_idx]
        Y = self.Data_i

        # Parameter updating
        n = len(Y) - self.prior_n_pts
        Y_data = Y[self.prior_n_pts:]
        Y_prior = Y[:self.prior_n_pts]
        # print("Y",Y,"Y_data", Y_data, "Y_prior",Y_prior)
        if n > 0:
            Ymean = np.mean(Y_data)
            s = np.var(Y_data)
            m0 = np.mean(Y_prior)
            s0 = np.var(Y_prior)
            n0 = len(Y_prior)
            v0 = n0 - 1
        else:
            Ymean = 0
            s = 0
            m0 = np.mean(Y_prior)
            s0 = np.var(Y_prior)
            n0 = len(Y_prior)
            v0 = n0 - 1

        # print("Ymean", Ymean, "s",s,"m0", m0, "s0", s0, "n0", n0, "v0", v0)
        mn = (n * Ymean + n0 * m0) / (n + n0)

        vn = v0 + n
        nn = n0 + n
        varn = (1.0 / vn) * (s * (n - 1) + s0 * v0 + (n0 * n / nn) * (Ymean - m0) ** 2.0)
        # print("mn", mn, "varn", varn)
        mu_grid = a[:, 0]
        var_grid = a[:, 1]
        alpha = vn / 2.0
        beta = (varn * vn) / 2.0
        var_pdf = invgamma.pdf(var_grid, a=alpha, loc=0, scale=beta)
        mu_pdf = norm.pdf(mu_grid, loc=mn, scale=var_grid / nn)
        # print("mu_pdf", mu_pdf.shape, "phi_pdf", var_pdf.shape)
        pdf_post = mu_pdf * var_pdf
        # print("pdf_post",pdf_post,"self.renormalisation_constant",self.renormalisation_constant)
        return pdf_post/self.renormalisation_constant


    def post_A_sampler(self, n, src_idx):
        """
        Sampler for posterior marginilised density over input A
        :param n: number of samples for posterior density over A
        :return: samples over domain
        """

        # print("SAMPLER")
        feasable_counter = 0
        feasable_final_samples = []
        self.renormalisation_constant = []
        while feasable_counter< n:
            # print("self.Data_post",self.Data_post)
            self.Data_i = self.Data_post[src_idx]
            Y = self.Data_i
            MC_samples = n
            n_data = len(Y) - self.prior_n_pts
            Y_data = Y[self.prior_n_pts:]
            Y_prior = Y[:self.prior_n_pts]
            # print("Y", Y, "Y_data", Y_data, "Y_prior", Y_prior)
            if n_data > 0:
                Ymean = np.mean(Y_data)
                s = np.var(Y_data)
                m0 = np.mean(Y_prior)
                s0 = np.var(Y_prior)
                n0 = len(Y_prior)
                v0 = n0 - 1
            else:
                Ymean = 0
                s = 0
                m0 = np.mean(Y_prior)
                s0 = np.var(Y_prior)
                n0 = len(Y_prior)
                v0 = n0 - 1
            mn = (n_data * Ymean + n0 * m0) / (n_data + n0)
            vn = v0 + n_data
            nn = n0 + n_data
            varn = (1.0 / vn) * (s * (n_data - 1) + s0 * v0 + (n0 * n_data / nn) * (Ymean - m0) ** 2.0)

            # print("mn", mn, "varn", varn)
            alpha = vn / 2.0
            beta = (varn * vn) / 2.0
            var_samples = invgamma.rvs(size=MC_samples, a=alpha, loc=0, scale=beta)
            mu_samples = multivariate_normal.rvs(size=1, mean=np.repeat(mn, MC_samples),
                                                 cov=np.identity(MC_samples) * var_samples / nn)

            joint_samples = np.stack([mu_samples,var_samples],axis=1)

            bounds_violation = np.any(np.logical_and(joint_samples> self.lb, joint_samples < self.ub), axis=1)
            feasable_counter += np.sum(bounds_violation)
            # print("feasable_counter",feasable_counter)
            feasable_samples = joint_samples[bounds_violation]
            feasable_final_samples.append(feasable_samples)
            self.renormalisation_constant.append(feasable_samples.shape[0]*1.0 /MC_samples)
        # print("self.lb",self.lb, "self.ub", self.ub)
        # print("np.concatenate(feasable_final_samples)[:n]",np.concatenate(feasable_final_samples)[:n])
        # plt.hist(mu_samples, bins=20)
        # plt.show()
        # plt.hist(var_samples, bins=20)
        # plt.show()
        self.renormalisation_constant= np.mean(self.renormalisation_constant)
        # raise
        samples = np.concatenate(feasable_final_samples)[:n]
        # mean_input = np.mean(samples, axis=0)
        # P5 = np.percentile(samples, 5, axis=0)
        # P95 = np.percentile(samples, 95, axis=0)
        # print("mean_input ",mean_input ,"P5", P5, "P95", P95)
        return samples


    def post_Data_sampler(self, n, src_idx):
        """
        Sampler for posterior predictive density ynew
        :param n: number of samples for posterior predictive density
        :return: samples over domain
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        feasable_counter = 0
        feasable_final_samples = []
        while feasable_counter< n:
            self.Data_i = self.Data_post[src_idx]
            Y = self.Data_i
            n_data = len(Y)
            Ymean = np.mean(Y)
            s = np.var(Y)
            m0 = np.mean(Y[:self.prior_n_pts])
            s0 = np.var(Y[:self.prior_n_pts])

            n0 = 2
            v0 = n0 - 1
            mn = (n * Ymean + n0 * m0) / (n + n0)
            vn = v0 + n_data
            nn = n0 + n_data
            varn = (1.0 / vn) * (s * (n_data - 1) + s0 * v0 + (n0 * n_data / nn) * (Ymean - m0) ** 2.0)

            MC_samples = n
            predictive_samples = t.rvs(size=MC_samples, df=vn, loc=mn, scale=np.sqrt(varn*(1+1.0/nn)))
            bounds_violation = np.logical_and(predictive_samples > self.lbx, predictive_samples < self.ubx)
            feasable_counter += np.sum(bounds_violation)
            feasable_predictive_samples = predictive_samples[bounds_violation]
            feasable_final_samples.append(feasable_predictive_samples)
        return np.concatenate(feasable_final_samples)[:n]

class MU_s_T_post():
    """
    Given i.i.d observations, builds a posterior density and a sampler
    (which can then be used with Delta Loss).
    Inference details:
    Normal Likelihood
    Uniform prior over the input "a" and uncertainty

    ARGS
        src_data: matrix of observations for a given source
        xmin = lower bound
        xmax = upper bound


    RETURNS
        post_dens: pdf over input domain A. Method: iu_pdf
        post_dens_samples: samples over post_dens. Method: iu_pdf_sampler
        post_predict_sampler: samples from posterior predictive density ynew. Method: data_predict_sampler

    """

    def __init__(self, amin=0, amax=100):
        self.xmin = amin
        self.xmax = amax
        self.h = 101
        self.sig_arr, self.dltsig = np.linspace(1e-3, 20, self.h, retstep=True)
        self.a_arr, self.dlta = np.linspace(1e-3, self.xmax, self.h, retstep=True)
        self.y_n1, self.dlty_n1 = np.linspace(0, self.xmax, 5000, retstep=True)

    def __call__(self, src_data):
        """

        :param src_data: list of narrays. include whole data.
        :return: functions post_dens: posterior density distribution given source. post_A_sampler: samples from
        posterior density distribution.
        """
        self.Data_post = src_data
        return self.post_dens, self.post_A_sampler, self.post_Data_sampler

    def log_prior_A_dens(self, a):
        """
        log prior using uniform distribution
        :param a: value of input narray x sigma narray
        :return:  value of logprior
        """
        assert len(a.shape) == 2;
        "a must be a matrix"
        assert a.shape[1] == 2;
        "a must have 2 columns"
        Lprior = np.zeros(len(a))
        max_ls = self.xmax;
        min_ls = self.xmin;
        prior = np.product(1.0 * ((a >= min_ls) & (a <= max_ls)), axis=1)
        Lprior[prior != 0] = np.log(prior[prior != 0])
        Lprior[prior == 0] = -np.inf
        return Lprior

    def log_lhood_d_i(self, a, data_i):
        """
        log likelihood of normal distribution
        :param a: value of input narray x sigma narray
        :param data_i: data in narray
        :return: log likelihood narray
        """
        assert len(a.shape) == 2, "a must be a matrix"
        assert a.shape[1] == 2, "a must have 2 columns"
        mu = a[:, 0]
        var = a[:, 1]
        Llikelihood_i = (-1.0 / 2) * (1.0 / var) * ((data_i - mu) ** 2) - np.log(np.sqrt(2 * np.pi * var))
        return Llikelihood_i

    def norm_const(self):
        """
        calculates normalisation constant. Particularly useful to normalise
        individual values of a from post_dens_unnormalised
        :return: normalisation constant of posterior distribution
        """
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        full_post = self.post_dens_unnormalised(Dom_crssprd)
        self.nrm_cnst = np.sum(full_post) * self.dltsig * self.dlta

    def cross_prod(self, arr_1, arr_2):
        """
        cartesian product between arrays. aux function
        :param arr_1: array
        :param arr_2: array
        :return: cartesian product ndarray
        """
        arr_1 = np.array(arr_1).reshape(-1)
        Dom_sets = [arr_1, arr_2]
        Dom_crssprd = np.array([list(i) for i in itertools.product(*Dom_sets)])
        return Dom_crssprd

    def post_dens_unnormalised(self, a):
        """
        # This implementation style means that even if there is no data,
        # the second summation term will be 0 and only the prior will contribute
        # i.e. this style uses one method for both prior and posterior, prior is NOT a special case.
        :param a: value of a
        :return: joint pdf calculated in a
        """
        log_lhood = np.sum([self.log_lhood_d_i(a, d_i) for d_i in self.Data_i], axis=0)
        log_post = self.log_prior_A_dens(a) + log_lhood
        self.post = np.exp(log_post)
        return self.post

    def marg_post_dens(self, a):
        """
        marginilise posterior joint distribution of input A and variance sigma of normal
        distribution

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        joint_post = self.post_dens_unnormalised(a)
        joint_post = joint_post.reshape(self.Na, len(self.sig_arr))
        return np.sum(joint_post, axis=1) * self.dltsig

    def post_dens(self, a, src_idx):
        """
        Posterior marginilised density estimation over A. First models posterior over
        the parameter A and uncertainty sigma from input source. Then marginilises out sigma
        and normalise the distribution over A

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        self.Data_i = self.Data_post[src_idx]
        self.norm_const()
        a = np.array(a).reshape(-1)
        self.Na = len(a)
        Dom_crssprd = self.cross_prod(a, self.sig_arr)
        pdf_post = self.marg_post_dens(Dom_crssprd)
        return pdf_post / self.nrm_cnst

    def sampler(self, n, dist, domain):
        """

        :param n: number of samples
        :param dist: pdf of distribution. normalised inside the function
        :param domain: discreatised domain
        :return: set of samples
        """
        assert not len(dist) == 1, "Trying to generate samples from scalar. Hint: Insert pdf"
        domain = domain.reshape(-1)
        dist = dist.reshape(-1)

        dist = dist / np.sum(dist)
        probabilities = dist * (1.0 / np.sum(dist))

        val = np.random.choice(domain, n, p=probabilities)
        return val

    def post_A_sampler(self, n, src_idx):
        """
        Sampler for posterior marginilised density over input A
        :param n: number of samples for posterior density over A
        :return: samples over domain
        """
        DomA = np.linspace(self.xmin, self.xmax, 5000)
        Dist = self.post_dens(DomA, src_idx)
        return self.sampler(n, dist=Dist, domain=DomA)

    def post_Data_sampler(self, n, src_idx):
        """
        Sampler for posterior predictive density ynew
        :param n: number of samples for posterior predictive density
        :return: samples over domain
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        self.Data_i = self.Data_post[src_idx]
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        pdf_musig = self.post_dens_unnormalised(Dom_crssprd)
        pdf_yn1_musig = np.exp(self.log_lhood_d_i(Dom_crssprd, self.y_n1[:, None]))
        pdf_yn1 = np.dot(pdf_yn1_musig, pdf_musig)
        return self.sampler(n, dist=pdf_yn1, domain=self.y_n1)

class MUSIG_post():
    """
    Given i.i.d observations, builds a posterior density and a sampler
    (which can then be used with Delta Loss).
    Inference details:
    Normal Likelihood
    Uniform prior over the input "a" and uncertainty

    ARGS
        src_data: matrix of observations for a given source
        xmin = lower bound
        xmax = upper bound


    RETURNS
        post_dens: pdf over input domain A. Method: iu_pdf
        post_dens_samples: samples over post_dens. Method: iu_pdf_sampler
        post_predict_sampler: samples from posterior predictive density ynew. Method: data_predict_sampler

    """

    def __init__(self, amin=0, amax=100):
        self.xmin = amin
        self.xmax = amax
        self.h = 101
        self.sig_arr, self.dltsig = np.linspace(1e-3, 20, self.h, retstep=True)
        self.a_arr, self.dlta = np.linspace(1e-3, self.xmax, self.h, retstep=True)
        self.y_n1, self.dlty_n1 = np.linspace(0, self.xmax, 5000, retstep=True)

    def __call__(self, src_data):
        """

        :param src_data: list of narrays. include whole data.
        :return: functions post_dens: posterior density distribution given source. post_A_sampler: samples from
        posterior density distribution.
        """
        self.Data_post = src_data
        return self.post_dens, self.post_A_sampler, self.post_Data_sampler

    def log_prior_A_dens(self, a):
        """
        log prior using uniform distribution
        :param a: value of input narray x sigma narray
        :return:  value of logprior
        """
        assert len(a.shape) == 2;
        "a must be a matrix"
        assert a.shape[1] == 2;
        "a must have 2 columns"
        Lprior = np.zeros(len(a))
        max_ls = self.xmax;
        min_ls = self.xmin;
        prior = np.product(1.0 * ((a >= min_ls) & (a <= max_ls)), axis=1)
        Lprior[prior != 0] = np.log(prior[prior != 0])
        Lprior[prior == 0] = -np.inf
        return Lprior

    def log_lhood_d_i(self, a, data_i):
        """
        log likelihood of normal distribution
        :param a: value of input narray x sigma narray
        :param data_i: data in narray
        :return: log likelihood narray
        """
        assert len(a.shape) == 2, "a must be a matrix"
        assert a.shape[1] == 2, "a must have 2 columns"
        mu = a[:, 0]
        var = a[:, 1]
        Llikelihood_i = (-1.0 / 2) * (1.0 / var) * ((data_i - mu) ** 2) - np.log(np.sqrt(2 * np.pi * var))
        return Llikelihood_i

    def norm_const(self):
        """
        calculates normalisation constant. Particularly useful to normalise
        individual values of a from post_dens_unnormalised
        :return: normalisation constant of posterior distribution
        """
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        full_post = self.post_dens_unnormalised(Dom_crssprd)
        self.nrm_cnst = np.sum(full_post) * self.dltsig * self.dlta

    def cross_prod(self, arr_1, arr_2):
        """
        cartesian product between arrays. aux function
        :param arr_1: array
        :param arr_2: array
        :return: cartesian product ndarray
        """
        arr_1 = np.array(arr_1).reshape(-1)
        Dom_sets = [arr_1, arr_2]
        Dom_crssprd = np.array([list(i) for i in itertools.product(*Dom_sets)])
        return Dom_crssprd

    def post_dens_unnormalised(self, a):
        """
        # This implementation style means that even if there is no data,
        # the second summation term will be 0 and only the prior will contribute
        # i.e. this style uses one method for both prior and posterior, prior is NOT a special case.
        :param a: value of a
        :return: joint pdf calculated in a
        """
        log_lhood = np.sum([self.log_lhood_d_i(a, d_i) for d_i in self.Data_i], axis=0)
        log_post = self.log_prior_A_dens(a) + log_lhood
        self.post = np.exp(log_post)
        return self.post

    def marg_post_dens(self, a):
        """
        marginilise posterior joint distribution of input A and variance sigma of normal
        distribution

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        joint_post = self.post_dens_unnormalised(a)
        joint_post = joint_post

        # print("joint_post",joint_post.shape)
        a = np.linspace(self.xmin, self.xmax, 5000)
        X, Y =np.meshgrid(a, self.sig_arr)
        plt.contourf(X,Y,joint_post.reshape(len(self.sig_arr), self.Na))
        plt.show()
        return joint_post

    def post_dens(self, a, src_idx):
        """
        Posterior marginilised density estimation over A. First models posterior over
        the parameter A and uncertainty sigma from input source. Then marginilises out sigma
        and normalise the distribution over A

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        self.Data_i = self.Data_post[src_idx]
        self.norm_const()
        a = np.array(a).reshape(-1)
        self.Na = len(a)
        Dom_crssprd = self.cross_prod(a, self.sig_arr)
        pdf_post = self.marg_post_dens(Dom_crssprd)
        return pdf_post / self.nrm_cnst

    def sampler(self, n, dist, domain):
        """

        :param n: number of samples
        :param dist: pdf of distribution. normalised inside the function
        :param domain: discreatised domain
        :return: set of samples
        """
        assert not len(dist) == 1, "Trying to generate samples from scalar. Hint: Insert pdf"
        domain = domain.reshape(-1)
        dist = dist.reshape(-1)

        dist = dist / np.sum(dist)
        probabilities = dist * (1.0 / np.sum(dist))

        val = np.random.choice(domain, n, p=probabilities)
        return val

    def post_A_sampler(self, n, src_idx):
        """
        Sampler for posterior marginilised density over input A
        :param n: number of samples for posterior density over A
        :return: samples over domain
        """
        DomA = np.linspace(self.xmin, self.xmax, 5000)
        Dist = self.post_dens(DomA, src_idx)
        return self.sampler(n, dist=Dist, domain=DomA)

    def post_Data_sampler(self, n, src_idx):
        """
        Sampler for posterior predictive density ynew
        :param n: number of samples for posterior predictive density
        :return: samples over domain
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        self.Data_i = self.Data_post[src_idx]
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        pdf_musig = self.post_dens_unnormalised(Dom_crssprd)
        pdf_yn1_musig = np.exp(self.log_lhood_d_i(Dom_crssprd, self.y_n1[:, None]))
        pdf_yn1 = np.dot(pdf_yn1_musig, pdf_musig)
        return self.sampler(n, dist=pdf_yn1, domain=self.y_n1)

class trunc_norm_post():
    """
    Given i.i.d observations, builds a posterior density and a sampler
    (which can then be used with Delta Loss).
    Inference details:
    Normal Likelihood
    Uniform prior over the input "a" and uncertainty

    ARGS
        src_data: matrix of observations for a given source
        xmin = lower bound
        xmax = upper bound


    RETURNS
        post_dens: pdf over input domain A. Method: iu_pdf
        post_dens_samples: samples over post_dens. Method: iu_pdf_sampler
        post_predict_sampler: samples from posterior predictive density ynew. Method: data_predict_sampler

    """

    def __init__(self, amin=0, amax=100, var=None):
        self.xmin = amin
        self.xmax = amax
        self.h = 101
        self.var_arr = np.array([var]).reshape(-1)
        self.sig_arr = np.array([var]).reshape(-1)
        self.a_arr, self.dlta = np.linspace(1e-3, self.xmax, self.h, retstep=True)
        self.y_n1, self.dlty_n1 = np.linspace(0, self.xmax, 5000, retstep=True)

    def __call__(self, src_data):
        """
        :param src_data: list of narrays. include whole data.
        :return: functions post_dens: posterior density distribution given source. post_A_sampler: samples from
        posterior density distribution.
        """
        assert len(src_data) == len(self.var_arr), "different number of variance for number of data sources"
        self.Data_post = src_data

        return self.post_dens, self.post_A_sampler, self.post_Data_sampler

    def log_prior_A_dens(self, a):
        """
        log prior using uniform distribution
        :param a: value of input narray x sigma narray
        :return:  value of logprior
        """
        assert len(a.shape) == 2;
        "a must be a matrix"
        assert a.shape[1] == 2;
        "a must have 2 columns"
        Lprior = np.zeros(len(a))
        max_ls = self.xmax;
        min_ls = self.xmin;
        min_condition = np.vstack((a[:, 0] >= min_ls))
        max_condition = np.vstack((a[:, 0] <= max_ls))
        prior = np.product(1.0 * (min_condition & max_condition), axis=1)
        Lprior[prior != 0] = np.log(prior[prior != 0])
        Lprior[prior == 0] = -np.inf
        return Lprior

    def log_lhood_d_i(self, a, data_i):
        """
        log likelihood of normal distribution
        :param a: value of input narray x sigma narray
        :param data_i: data in narray
        :return: log likelihood narray
        """
        assert len(a.shape) == 2, "a must be a matrix"
        assert a.shape[1] == 2, "a must have 2 columns"
        mu = a[:, 0]
        var = a[:, 1]
        Llikelihood_i = (-1.0 / 2) * (1.0 / var) * ((data_i - mu) ** 2) - np.log(np.sqrt(2 * np.pi * var))
        return Llikelihood_i

    def norm_const(self):
        """
        calculates normalisation constant. Particularly useful to normalise
        individual values of a from post_dens_unnormalised
        :return: normalisation constant of posterior distribution
        """
        Dom_crssprd = self.cross_prod(self.a_arr, self.var)
        full_post = self.post_dens_unnormalised(Dom_crssprd)
        self.nrm_cnst = np.sum(full_post) * self.dlta

    def cross_prod(self, arr_1, arr_2):
        """
        cartesian product between arrays. aux function
        :param arr_1: array
        :param arr_2: array
        :return: cartesian product ndarray
        """
        arr_1 = np.array(arr_1).reshape(-1)
        Dom_sets = [arr_1, arr_2]
        Dom_crssprd = np.array([list(i) for i in itertools.product(*Dom_sets)])
        return Dom_crssprd

    def post_dens_unnormalised(self, a):
        """
        # This implementation style means that even if there is no data,
        # the second summation term will be 0 and only the prior will contribute
        # i.e. this style uses one method for both prior and posterior, prior is NOT a special case.
        :param a: value of a
        :return: joint pdf calculated in a
        """
        log_lhood = np.sum([self.log_lhood_d_i(a, d_i) for d_i in self.Data_i], axis=0)
        log_post = self.log_prior_A_dens(a) + log_lhood
        self.post = np.exp(log_post)
        return self.post

    def marg_post_dens(self, a):
        """
        marginilise posterior joint distribution of input A and variance sigma of normal
        distribution

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        joint_post = self.post_dens_unnormalised(a)
        joint_post = joint_post.reshape(self.Na, len(self.sig_arr))
        return np.sum(joint_post, axis=1)

    def post_dens(self, a, src_idx):
        """
        Posterior marginilised density estimation over A. First models posterior over
        the parameter A and uncertainty sigma from input source. Then marginilises out sigma
        and normalise the distribution over A

        :param a: values over domain of A to calculate pdf.
        :return: pdf calculated in a
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        self.var = np.array(self.var_arr[src_idx]).reshape(-1)
        self.Data_i = self.Data_post[src_idx]
        self.norm_const()
        a = np.array(a).reshape(-1)
        self.Na = len(a)
        Dom_crssprd = self.cross_prod(a, self.sig_arr)
        pdf_post = self.marg_post_dens(Dom_crssprd)
        return pdf_post / self.nrm_cnst

    def sampler(self, n, dist, domain):
        """

        :param n: number of samples
        :param dist: pdf of distribution. normalised inside the function
        :param domain: discreatised domain
        :return: set of samples
        """
        assert not len(dist) == 1, "Trying to generate samples from scalar. Hint: Insert pdf"
        dist = dist / np.sum(dist)
        probabilities = dist * (1 / np.sum(dist))
        domain = domain.reshape(-1)
        probabilities = probabilities.reshape(-1)
        # print("probabilities", probabilities)
        # print("n",n)
        # print("domain", domain)
        val = np.random.choice(domain, n, p=probabilities)
        return val

    def post_A_sampler(self, n, src_idx):
        """
        Sampler for posterior marginilised density over input A
        :param n: number of samples for posterior density over A
        :return: samples over domain
        """
        DomA = np.linspace(self.xmin, self.xmax, 5000)
        Dist = self.post_dens(DomA, src_idx)
        return self.sampler(n, dist=Dist, domain=DomA)

    def post_Data_sampler(self, n, src_idx):
        """
        Sampler for posterior predictive density ynew
        :param n: number of samples for posterior predictive density
        :return: samples over domain
        """
        assert src_idx + 1 <= len(self.Data_post) and src_idx + 1 >= 1, "source index is out of bounds"

        self.var = np.array(self.var_arr[src_idx]).reshape(-1)

        self.Data_i = self.Data_post[src_idx]
        Dom_crssprd = self.cross_prod(self.a_arr, self.sig_arr)
        pdf_musig = self.post_dens_unnormalised(Dom_crssprd)
        pdf_yn1_musig = np.exp(self.log_lhood_d_i(Dom_crssprd, self.y_n1[:, None]))
        pdf_yn1 = np.dot(pdf_yn1_musig, pdf_musig)
        return self.sampler(n, dist=pdf_yn1, domain=self.y_n1)

def KG(mu, sig):
    """
    Takes a set of intercepts and gradients of linear functions and returns
    the average hieght of the max of functions over Gaussain input.

    ARGS
        mu: length n vector, initercepts of linear functions
        sig: length n vector, gradients of linear functions

    RETURNS
        out: scalar value is gaussain expectation of epigraph of lin. funs
    """

    n = len(mu)
    O = sig.argsort()
    a = mu[O]
    b = sig[O]
    # print("a",a)
    # print("b",b)
    A = [0]
    C = [-float("inf")]
    while A[-1] < n - 1:
        s = A[-1]
        si = range(s + 1, n)
        Ci = -(a[s] - a[si]) / (b[s] - b[si])
        bestsi = np.argmin(Ci)
        C.append(Ci[bestsi])
        A.append(si[bestsi])

    C.append(float("inf"))
    cdf_C = norm.cdf(C)
    diff_CDF = cdf_C[1:] - cdf_C[:-1]
    pdf_C = norm.pdf(C)
    diff_PDF = pdf_C[1:] - pdf_C[:-1]
    # print(" a[A]*diff_CDF - b[A]*diff_PDF", a[A]*diff_CDF - b[A]*diff_PDF)
    # print("np.max(mu)", np.max(mu))
    out = np.sum(a[A] * diff_CDF - b[A] * diff_PDF) - np.max(mu)

    # print("out",out)

    # assert out >= -1e-10;
    # "KG cannot be negative"

    return out

def lhs_box(n, lb, ub):
    """
    Random samples uniform lhs in a box with lower and upper bounds.
    ARGS
        n: number of points
        lb: vector, lower bounds of each dim
        ub: vector, upper bounds of each dim

    RETURNS
        LL: an lhs in the box, lb < x < ub
    """

    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    assert lb.shape[0]==ub.shape[0]; "bounds must be same shape"
    assert np.all(lb<=ub); "lower must be below upper!"

    LL = lb + lhs(len(lb), samples=n)*(ub-lb)

    return(LL)


def COV(model, xa1, xa2, chol_K=None, Lk2=None):
    """Takes a GP model, and 2 points, returns post cov.
    ARGS
        model: a gpy object
        xa1: n1*d matrix
        xa2: n2*d matrix
        chol_K: optional precomuted cholesky decompositionb of kern(X,X)
        Lk2: optional precomputed solve(chol_K, kernel(model.X, xa2)) (eg for XdAd)

    RETURNS
        s2: posterior GP cov matrix
     """
    # print("xa1",xa1)
    assert len(xa1.shape) == 2;
    "COV: xa1 must be rank 2"
    assert len(xa2.shape) == 2;
    "COV: xa2 must be rank 2"
    assert xa1.shape[1] == model.X.shape[1];
    "COV: xa1 must have same dim as model"
    assert xa2.shape[1] == model.X.shape[1];
    "COV: xa2 must have same dim as model"

    if chol_K is None:
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (1e-4 ** 2.0) * np.eye(len(K)))
    else:
        assert chol_K.shape[0] == model.X.shape[0];
        "chol_K is not same dim as model.X"

    Lk1 = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))

    if Lk2 is None:
        Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, xa2))
    else:
        assert Lk2.shape[0] == model.X.shape[0];
        "Lk2 is not same dim as model.X"
        assert Lk2.shape[1] == xa2.shape[0];
        "Lk2 is not same dim as xa2"

    K_ = model.kern.K(xa1, xa2)
    # print("K_", K_)
    # print("Lk1", Lk1)
    # print("Lk2", Lk2)
    # print("np.matmul(Lk1.T, Lk2)",np.matmul(Lk1.T, Lk2))
    s2 = np.matrix(K_) - np.matmul(Lk1.T, Lk2)
    # print("s2", s2)


    # print("model.posterior_covariance_between_points(xa1, xa2)",model.posterior_covariance_between_points(xa1, xa2))
    s2 = np.array(model.posterior_covariance_between_points(xa1, xa2))#np.array(s2)

    # make sure the output is correct!
    assert s2.shape[0] == xa1.shape[0];
    "output dim is wrong!"
    assert s2.shape[1] == xa2.shape[0];
    "output dim is wrong!"

    return s2

def VAR(model, xa1, chol_K=None):
    """Takes a GP model, and 1 point, returns post var.

    ARGS
     model: a gpy object
     xa1: n1*d matrix
     chol_K: cholesky decompositionb of kern(X,X)

    RETURNS
     s2: posterior GP cov matrix
     """

    assert len(xa1.shape) == 2;
    "VAR: xa1 must be rank 2"
    assert xa1.shape[1] == model.X.shape[1];
    "VAR: xa1 have same dim as model"

    if chol_K is None:
        K = model.kern.K(model.X, model.X)
        chol_K = np.linalg.cholesky(K + (1e-6 ** 2.0) * np.eye(len(K)))

    Lk = np.linalg.solve(chol_K, model.kern.K(model.X, xa1))
    K_ = model.kern.K(xa1, xa1)
    s2 = K_ - np.matmul(Lk.T, Lk)

    #s2 = np.array(s2)
    s2 = np.array(model.posterior_covariance_between_points(xa1, xa1))
    return s2

@writeCSV_time_stats
def DeltaLoss(model, Data, Xd, Ad, Wd, pst_mkr, lb, ub, Nd=101):
    # make a full lattice over X x A and get the GP mean at each point.
    """

    :param model: Gaussian Process trained model
    :param Data:list of arrays, observed source Data
    :param Xd:
    :param Ad:
    :param Wd:
    :param pst_mkr:
    :param Nd:
    :return:
    """
    Ad_list = Ad
    Ad = np.hstack(Ad)
    lb = lb.reshape(-1)
    ub = ub.reshape(-1)

    Nx = Xd.shape[0]
    Na = Ad.shape[0]

    # precompute inverse weights of Ad points.

    invWd =np.reciprocal(Wd)  # np.sum(Wd)/Wd

    _, _, cur_Data_sampler = pst_mkr(Data)
    # get the index of the current top recomended x value.
    def marginal_current(X):

        if np.any(X < lb[:Xd.shape[1]]) or np.any(X > ub[:Xd.shape[1]]):
            return (1000000)
        else:
            X = np.atleast_2d(X)
            Nx_internal = X.shape[0]
            XdAd = np.hstack([np.repeat(X, Na, axis=0),
                              np.tile(Ad, (Nx_internal, 1))])

            M_XA = model.predict(XdAd)[0].reshape(Nx_internal, Na)
            # now we have weights, get the peak of reweighted GP means
            M_X_i = np.mean(M_XA, axis=1)

            return -M_X_i

    M_X = marginal_current(Xd)
    current_top_X = Xd[np.argmin(M_X)]

    cur_topX_index = np.array([minimize(marginal_current, x_discrete , method = 'Nelder-Mead').x for x_discrete in current_top_X]).reshape(-1)

    # loop over IU parameters / A dims / inf sources.
    DL = []
    y_Data = [cur_Data_sampler(n=Nd, src_idx=src) for src in range(len(Data))]
    # print("y_Data", y_Data)
    for src in range(len(Data)):

        # loop over individual DL samples
        DL_src = []
        for i in range(Nd):
            # sample a new observation and add it to the original Data
            tmp_Data_i = np.array([y_Data[src][i]])

            tmp_Data = Data[:]
            tmp_Data[src] = np.concatenate([tmp_Data[src], tmp_Data_i])

            # get the importance weights of the Ad points from new posterior
            tmp_post_A_dens, _, _ = pst_mkr(tmp_Data)

            def marginal_one_step_ahead(X):
                if np.any(X < lb[:Xd.shape[1]]) or np.any(X > ub[:Xd.shape[1]]):
                    return (1000000)
                else:
                    X = np.atleast_2d(X)
                    Nx_internal = X.shape[0]
                    XdAd = np.hstack([np.repeat(X, Na, axis=0),
                                      np.tile(Ad, (Nx_internal, 1))])

                    M_XA = model.predict(XdAd)[0].reshape(Nx_internal, Na)

                    Wi = tmp_post_A_dens(Ad_list[src], src_idx=src)

                    # print("beg Wi", Wi)
                    # print("invWd[src]",invWd[src])
                    Wi = Wi * invWd[src]
                    # print("np.sum(Wi)",np.sum(Wi))
                    Wi = Wi / np.sum(Wi)
                    # print("end Wi", Wi)
                    # now we have weights, get the peak of reweighted GP means
                    M_X_i = np.sum(M_XA * Wi, axis=1)
                    return -M_X_i

            M_X_i = marginal_one_step_ahead(Xd)

            estimated_top_X = Xd[np.argmin(M_X_i)]
            # print("estimated_top_X",estimated_top_X)
            # print("M_X_i",M_X_i)
            top_val = [minimize(marginal_one_step_ahead, x_discrete , method = 'Nelder-Mead').fun for x_discrete in estimated_top_X ]
            top_val = -np.array(top_val).reshape(-1)
            # plt.scatter(Xd, -M_X, label="M_X", color="blue")
            # plt.scatter(Xd, -M_X_i, label="new M_X", color="magenta")
            # plt.show()
            DL_i = top_val - (-marginal_one_step_ahead(cur_topX_index))
            # print("DL_i",DL_i)
            #assert DL_i >= 0, "Delta Loss can't be negative"
            # keep this single MC sample of DL improvement
            DL_src.append(DL_i)

        # get the average over DL samples for this source and save in the list.
        DL.append(np.mean(DL_src))
 #       print("np.mean(DL_src)",np.mean(DL_src), "np.MSE(DL_src)",np.std(DL_src)/np.sqrt(len(DL_src)))
#        raise
    # get the best source and its DL.

    topsrc = np.argmax(DL)
    topDL = np.max(DL)  # - np.max(M_X)
    return topsrc, topDL

@writeCSV_time_stats
def KG_Mc_Input(model, Xd, Ad, lb, ub, Ns=2000, Nc=5, maxiter=80):
    """Takes a GPy model, constructs and optimzes KG and
    returns the best xa and best KG value.

    ARGS
        model: GPy model
        Xd: Nx*x_dim matrix discretizing X
        Ad: Na*a_dim matrix discretizing A !!!MUST BE SAMPLES FROM POSTERIOR OVER A!!!!!
        lb: lower bounds on (x,a)
        ub: upper bounds on (x,a)
        Ns: number of START points, initial random search of KG
        Nc: number of CONTINUED points, from the Ns the top Nc points to perform a Nelder Mead run.
        maxiter: iterations of each Nelder Mead run.

    RETURNS
        bestxa: the optimal xa
        bestEVI: the largest KG value
     """

    lb = lb.reshape(-1)
    ub = ub.reshape(-1)
    Ad = np.hstack(Ad)

    assert Ns > Nc;
    "more random points than optimzer points"
    assert len(Xd.shape) == 2;
    "Xd must be a matrix"
    assert len(Ad.shape) == 2;
    "Ad must be a matrix"
    assert Xd.shape[0] > 1;
    "Xd must have more than one row"
    assert Ad.shape[0] > 1;
    "Ad must have more than one row"
    assert Ad.shape[1] + Xd.shape[1] == model.X.shape[1];
    "Xd, Ad, must have same dim as data"
    assert lb.shape[0] == ub.shape[0];
    "bounds must have same dim"
    assert lb.shape[0] == model.X.shape[1];
    "bounds must have same dim as data"
    assert np.all(lb <= ub);
    "lower must be below upper!"

    # optimizer initial optim
    KG_Mc_Input.bestEVI = -9e10
    KG_Mc_Input.bestxa = 0

    noiseVar = model.Gaussian_noise.variance[0]

    dim_X = Xd.shape[1]
    Nx = Xd.shape[0]
    Na = Ad.shape[0]

    XdAd = np.hstack([np.repeat(Xd, Na, axis=0),
                      np.tile(Ad, (Nx, 1))])
    # Precompute the posterior mean at X_d integrated over A.
    M_Xd = model.predict(XdAd)[0].reshape(Nx, Na)
    # S_Xd = model.predict(XdAd, include_likelihood=False)[1].reshape(Nx, Na)


    M_Xd = np.mean(M_Xd, axis=1).reshape(1, -1)
    # S_Xd = np.mean(S_Xd,axis=1).reshape(1,-1)
    #
    # plt.scatter(Xd,M_Xd)
    # plt.scatter(Xd, M_Xd + 1.95*np.sqrt(S_Xd))
    # plt.scatter(Xd, M_Xd - 1.95* np.sqrt(S_Xd))
    # plt.show()
    #
    # plt.scatter(model.X[:,0],model.X[:,1])
    # plt.scatter(model.X[-1, 0], model.X[-1, 1],color="red")
    # plt.show()

    # Precompute cholesky decomposition.
    K = model.kern.K(model.X, model.X)
    chol_K = np.linalg.cholesky(K + (1e-4 ** 2.0) * np.eye(K.shape[0]))

    Lk2 = np.linalg.solve(chol_K, model.kern.K(model.X, XdAd))

    KG_Mc_Input.calls = 0


    def KG_IU(xa):

        KG_Mc_Input.calls += 1
        xa = np.array(xa).reshape((1, -1))

        if np.any(xa < lb) or np.any(xa > ub):
            return (1000000)
        else:

            # The current x with all the A samples
            tile_x = np.tile(xa[:, :dim_X], (Na, 1))
            newx_Ad = np.hstack([tile_x, Ad])

            # The mean integrated over A.
            # M_Xd is precomputed
            M_x = np.mean(model.predict(newx_Ad)[0])

            MM = np.c_[M_x, M_Xd].reshape(-1) # OK

            # The mean warping integrated over Ad

            S_Xd = COV(model, xa, XdAd, chol_K, Lk2)
            S_Xd = S_Xd.reshape(Nx, Na)
            S_Xd = np.mean(S_Xd, axis=1).reshape(1, -1)
            # print("S_Xd",S_Xd)
            S_x = np.mean(COV(model, xa, newx_Ad, chol_K))
            SS = np.c_[S_x, S_Xd].reshape(-1)

            # variance of new observation
            var_xa = VAR(model, xa, chol_K) + noiseVar
            inv_sd = 1. / np.asarray(var_xa**0.5).reshape(())

            SS = SS * inv_sd
            # Finally compute KG!
            out = KG(MM, SS)

            if out > KG_Mc_Input.bestEVI:
                # print("xa", xa)
                # print("XdAd", XdAd, "S_Xd", S_Xd)
                # print("XdAd", np.concatenate((XdAd,S_Xd)))

                KG_Mc_Input.bestEVI = out
                KG_Mc_Input.bestxa = xa
            return -out


    #dKG = grad(KG_IU)
    #print("dKG", dKG([0.5,0.5])) #Problem with var
    #import pdb; pdb.set_trace()

    # Optimize that badboy! First do Ns random points, then take best Nc results
    # and do Nelder Mead starting from each of them.
    lb_X = np.array(lb).reshape(-1)[:dim_X]
    ub_X = np.array(ub).reshape(-1)[:dim_X]
    X = lhs_box(Ns, lb=lb_X, ub = ub_X ) #lb, ub)
    XA_Ns = np.hstack([np.repeat(X, Na, axis=0),
                      np.tile(Ad, (Ns, 1))])

    proposed_idx = np.random.choice(len(XA_Ns), Ns)
    XA_Ns = XA_Ns[proposed_idx]
    KG_Ns = np.array([KG_IU(XA_i) for XA_i in XA_Ns])
    XA_Nc = KG_Ns.argsort()[-Nc:]
    XA_Nc = XA_Ns[XA_Nc, :]

    _ = [minimize(KG_IU, XA_i, method='nelder-mead', options={'maxiter': maxiter}) for XA_i in XA_Nc]

    return KG_Mc_Input.bestxa, KG_Mc_Input.bestEVI

