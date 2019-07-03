import GPy
import csv
import numpy as np
import scipy
from scipy.optimize import minimize
import time
import pygmo as pg
from scipy.stats import uniform
from pyDOE import lhs
from scipy import optimize
import pandas as pd

import scipy.integrate as integrate
import scipy.special as special

import time
from scipy.stats import truncnorm
from scipy.interpolate import interp1d

class Mult_Input_Uncert():

    def __init__(self, test_func, input_source, xran, wran):
        # This code here is executed when we make a new optimizer myoptimizer = Mult_Input_Uncert()
        #
        # ARGS:
        #  test_func: a callable function y = f(x, w) that returns simulations outputs
        #  input_source: a callable source of data that returns "external" observations
        #  xran: the upper and lower limit of x in f(x,w)
        #  wran: the upper and lower limit of w in f(x,w)
        #
        # RETURNS
        #  self: a Mult_Input_Uncert instance, __init__ is an initilizer/constructor!!

        self.test_func = test_func
        self.input_source = input_source
        self.xran = xran
        self.wran = wran

    def __call__(self, init_sample=10, Nx = 101, Nr=100, EndN=100, precision=100,
                 seed=1):
        # THIS IS WHERE THE MAIN CODE FOR OPTIMIZATION GOES :)
        # This code is run when myoptimizer(sim_init=...) is called as if
        # it were a function. We could also just do myoptimizer.__call__(....)
        # or we would make def optimize(self, sim_init=...) and call myoptimizer.optimize(...)
        # this __call__ is a python keyword that is executed when the object is treated as a function
        #
        # ARGS:
        #  init_sample: how many points to use to start the Gaussain process
        #  Nx: discretization size over x space
        #  Nr: discretization size over w space
        #  EndN: sampling budget for optimization
        #  precision:
        #
        # RETURNS
        #  OC: opportunity cost
        #  xw: GP data
        #  z: iu data
        #  otherstuff: I forgot....

        dim = self.xran.shape[1] + self.wran.shape[1]
        x= np.linspace(0, 100, Nx) #vector of input variable
        MU = np.linspace(0,100,precision)
        SIG = np.linspace(0.025,100,precision)
        X = np.repeat(list(MU),len(SIG))
        W=list(SIG)*len(MU)
        MUSIG0 = np.c_[X,W]

        MU_L = np.linspace(0,100,101)
        SIG_L = np.linspace(0.025,100,101)
        X_L = np.repeat(list(MU_L),len(SIG_L))
        W_L=list(SIG_L)*len(MU_L)
        MUSIG0_L = np.c_[X_L,W_L]
        init_input_source = Input_Source(dim,0)

        # First get some simulation ( (X, W), Y) pairs for training the GP.
        XA = np.c_[lhs(1, samples=init_sample)*100, lhs(1, samples=init_sample)*100, lhs(1, samples=init_sample)*100]
        Y = self.test_func(xa=XA)
        ker = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=([10,10,10]), ARD = True)

        iLoss = []
        OC = []
        #===================================================================================
        #True Func

        XW0 = np.c_[ x, np.array([Input_Source.f_mean]*len(x))]
        True_obj_v = test_func(xa=XW0,NoiseSD=0,gen=False)
        obj     = lambda a: -1*np.mean(test_func(np.c_[[a],[Input_Source.f_mean]],NoiseSD=0,gen=False))
        topX    = x[np.argmax(True_obj_v)]


        if topX >= 100:
            topX=99
        elif topX <=0:
            topX=1

        topX = optimize.fminbound(obj, topX-1, topX+1,xtol =1e-16)
        best_ = -1*obj(topX)

        #=================================================================================================
        Data =  np.array([[np.nan,np.nan]]) #init_input_source #
        Ndata = 0

        while len(XA) +  Ndata < EndN:

            Ndata = np.sum([len(Data[:,i][~np.isnan(Data[:,i])]) for i in range(dim)])
            self.model = GPy.models.GPRegression(np.array(XA) , np.array(Y).reshape(len(Y),1) , ker,noise_var=0.01)
            Mult_Input_Uncert.m = m

            Mult_Input_Uncert.KG=[]
            Mult_Input_Uncert.DL1 = []
            Mult_Input_Uncert.DL2 = []

        #ERROR CURVE
            if Ndata > 0:
    #             print('Nr',Nr)
                x_pdf1, pdf1 = Fit_Inputs(Data[:,0])
                x_pdf2, pdf2 = Fit_Inputs(Data[:,1])

                #Generation of RV
                pdf1_gen = Gen_Sample(pdf1,N=Nr)
                pdf2_gen = Gen_Sample(pdf2,N=Nr)
                # Sample variable x inputs
                Sample = np.c_[np.repeat(x,Nr) , list(pdf1_gen)*Nx, list(pdf2_gen)*Nx ]
            else:
                pdf1_gen = np.random.random(Nr)*100
                pdf2_gen = np.random.random(Nr)*100

                Sample = np.c_[np.repeat(x,Nr) , list(pdf1_gen)*Nx,list(pdf2_gen)*Nx]

            IU = np.mean(self.model.predict(np.array(Sample))[0].reshape(Nx,Nr),axis=1)

            Xr = x[np.argmax(IU)]


            topobj = -1*obj(Xr)
            DIF = best_ - topobj

            OC.append(DIF)

            #===============================================================================================

            xa, p = KG_Mc_Input(XA,Y,self.model,Nx=Nx,Ns=20)

            Comparison = np.concatenate(([KG_Mc_Input.bestEVI],[Delta_Loss(Data,i) for i in range(dim)]))
            chs = np.argmax(Comparison)

            d_point = np.array([[np.nan, np.nan]])
            if chs == 0:
                XA = np.concatenate((XA,xa)) ; Y = np.concatenate([Y,p])
            else:

                QA = chs - 1
                IS = Input_Source(2,1,mv_gen = False)[0][QA]
                d_point[:,QA] = IS
                Data = np.concatenate((Data, d_point))
                Ndata = np.sum([len(Data[:,i][~np.isnan(Data[:,i])]) for i in range(dim)])


            N_I = [len(Data[:,i][~np.isnan(Data[:,i])]) for i in range(dim)]
            return OC, N_I, Mult_Input_Uncert.var

    def COV(self, xa1, xa2):
        #K = self.model.kern.K(model.X,model.X)
        L = self.chol_K #np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))
        Lk1 = np.linalg.solve(L, self.model.kern.K(self.model.X, xa1))
        Lk2 = np.linalg.solve(L, self.model.kern.K(self.model.X, xa2))
        K_ = self.model.kern.K(xa1, xa2)
        s2 = np.matrix(K_) - np.matrix(np.dot(Lk2.T,Lk1))
        return s2

    def VAR(self, xa1):
        # K = model.kern.K(model.X, model.X)
        L = self.chol_K #np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))
        Lk = np.linalg.solve(L, model.kern.K(self.model.X, xa1))
        K_ = self.model.kern.K(xa1, xa1)
        s2 = K_ - np.sum(Lk**2, axis=0)
        return s2

    def Gen_Sample(Dist, N=500):
        # generates N samples between 0 and 100 with given 'Dist' weights
        elements = np.linspace(0, 100, len(Dist))
        probabilities = Dist/np.sum(Dist)
        val = np.random.choice(elements, N, p=probabilities)
        return val

    def sample_predict_dens(Data, N):
        global MUSIG0, MU_L, SIG_L, MUSIG0_L

        Data = list(Data[~np.isnan(Data)])
        def Distr_Update(Data):
            Y =Data
            L = []
            fy = []
            y_n1 = np.linspace(0,100,200)

            for i in MUSIG0_L:
                fy.append(np.exp(-(1.0/(2.0*i[1]))*(i[0] - y_n1)**2.0))
                L.append(np.exp(-(1.0/(2.0*i[1]))*np.sum((i[0] - Y)**2.0))*(1.0/np.sqrt(2*np.pi*i[1]))**len(Y))
            dmu = MU_L[1]-MU_L[0]
            dsig = SIG_L[1]-SIG_L[0]
            dy_n1 = y_n1[1]-y_n1[0]

            L = np.matrix(L)
            fy = np.matrix(fy)
            D = np.array((np.matrix(L))*np.matrix(fy)*dmu*dsig)
            D = np.array((D/np.sum(D*dy_n1)))[0]
            return D

        pdf_zn = Distr_Update(Data)

        zn = Gen_Sample(pdf_zn, N)
        return zn

    def Delta_Loss(Data, idx):
        global Nr,Nx

        Nd = 100
        #INPU: Matrix with N dimensions and S realisations
        def W_aj(Y=np.array([]),a=np.array([])):

            MU = np.linspace(0,100,60)
            SIG = np.linspace(0.25,100,60)
            d = np.vstack(np.hstack(Y))
            N = Y.shape[1]
            dimY = Y.shape[0]
            expo = np.exp(np.vstack(-(1.0/(2.0*SIG)))*np.hstack(np.sum(np.split((d-MU)**2,dimY,axis=0),axis=1)))

            consts =  np.vstack((1.0/np.sqrt(2*np.pi*SIG))**N)

            L = np.split(expo*consts, dimY, axis=1)
            marg_mu_dist = np.sum(L,axis=1)*(SIG[1]-SIG[0])
            C = np.sum(marg_mu_dist,axis=1)*(MU[1]-MU[0])
            marg_mu_dist = marg_mu_dist*(1/np.vstack(C))

            expo = np.exp(np.vstack(-(1.0/(2.0*SIG)))*np.hstack(np.sum(np.split((d-a)**2,dimY,axis=0),axis=1)))
            consts =  np.vstack((1.0/np.sqrt(2*np.pi*SIG))**N)
            L = np.split(expo*consts,dimY,axis=1)
            marg_mu_val = np.sum(L,axis=1)*(SIG[1]-SIG[0])*(1/np.vstack(C))
            return marg_mu_val

        if len(np.array([list(Data[~np.isnan(Data[:,idx]),idx])])[0]) > 0:

            z1 = sample_predict_dens(np.array([list(Data[~np.isnan(Data[:,idx]),idx])]),N=Nd)
            Sample_I = Sample[:,idx+1]
            Sample_XA = Sample
            W_D = W_aj(Y=np.array([list(Data[~np.isnan(Data[:,idx]),idx])]), a=Sample_I)


            dj = np.c_[np.array([list(Data[~np.isnan(Data[:,idx]),idx])]*Nd),z1]
        else:

            z1 = np.random.random(Nd)*100
            Sample_I = Sample[:,idx+1]
            Sample_XA = Sample
            W_D = np.array([list(np.repeat([1.0/100],Nx*Nr))])
            dj = np.vstack(z1)

        R_IU = []
        W_D1 = W_aj(Y=dj, a=Sample_I)

        Wi = W_D1/W_D
        Mult_Input_Uncert.Wi = Wi.reshape(Nd,Nx,Nr)

        Prd = Mult_Input_Uncert.m.predict(np.array(Sample_XA))[0].reshape(Nx, Nr)
        Mult_Input_Uncert.Prd = Prd
        IU_D1 = np.mean(np.multiply(Mult_Input_Uncert.Wi,Mult_Input_Uncert.Prd),axis=2)

        max_IU_D1 = np.max(IU_D1,axis=1)

        Prd_D = m.predict(np.array(np.c_[np.repeat(Xr,Nr),Sample[:Nr,1:3]]))[0].T
        Mult_Input_Uncert.Prd_D = Prd_D

        IU_D = np.mean(np.mean(np.multiply(Mult_Input_Uncert.Wi,Mult_Input_Uncert.Prd_D),axis=2),axis=1)
        DL = np.mean(max_IU_D1-IU_D)

        return DL


    def KG_Mc_Input(self, XA, P, model, Nx=15,Ns=20):
        global best_obj_v
        #np.random.seed(np.int(time.clock()*10000))
        KG_Mc_Input.Ns = Ns
        KG_Mc_Input.bestEVI = -10
        KG_Mc_Input.bestxa  = [-10,-10,-10]
        KG_Mc_Input.noiseVar = model.Gaussian_noise.variance[0]
        KG_Mc_Input.Xi = np.array(XA[:,0])
        KG_Mc_Input.Xi.sort()

        KG_Mc_Input.Ad = np.c_[pdf1_gen, pdf2_gen] #1000
        KG_Mc_Input.XiAd  = np.c_[np.repeat(KG_Mc_Input.Xi,len(KG_Mc_Input.Ad) ), list(pdf1_gen)*len(KG_Mc_Input.Xi), list(pdf2_gen)*len(KG_Mc_Input.Xi)]
        #print('KG_Mc_Input.Ad',KG_Mc_Input.Ad)

        obj_v = np.sum(m.predict(KG_Mc_Input.XiAd)[0].reshape(len(KG_Mc_Input.Xi),len(KG_Mc_Input.Ad)),axis=1)
        obj_v = np.array(obj_v).reshape(len(obj_v),1)


        def KG_IU(xa):

            if(np.abs(xa[0]-50)>50 or np.abs(xa[1]-50)>50 or np.abs(xa[2]-50)>50):
                return(1000000)
            else:

                newx = np.array(np.c_[np.repeat(xa[0],len(KG_Mc_Input.Ad)),KG_Mc_Input.Ad])

                MMx = np.sum(model.predict(newx)[0])

                MM     =  np.c_[MMx,obj_v.T].reshape(len(np.c_[MMx,obj_v.T].T),1)

                MM = MM/(len(KG_Mc_Input.Ad ))

                sigt2 = np.diag(COV(m,np.array([xa]),KG_Mc_Input.XiAd)).reshape(len(KG_Mc_Input.Xi),len(KG_Mc_Input.Ad))
                sigt2 = np.array(np.sum(sigt2,axis=1)).reshape(len(sigt2),1)

                sigt2 = np.c_[np.sum(np.diag(COV(m,np.array([xa]),newx))),sigt2.T].T
                sigt1   = self.VAR(np.array([xa]))
                sigt    = ((sigt2) / np.sqrt(sigt1+KG_Mc_Input.noiseVar))
                sigt = sigt/(len(KG_Mc_Input.Ad ))

                musig = np.c_[MM,sigt]
                out  = KG(musig)
                if out > KG_Mc_Input.bestEVI:
                    KG_Mc_Input.bestEVI = out
                    KG_Mc_Input.bestxa = xa
                return -out

        # random restarts for optimization
        XAs = np.array(np.c_[lhs(1, samples=KG_Mc_Input.Ns)*100,lhs(1, samples=KG_Mc_Input.Ns)*100,lhs(1, samples=KG_Mc_Input.Ns)*100])

        # run nelder mead for each random restart
        A    = [minimize(KG_IU, i, method='nelder-mead', options={'maxiter': 80}).x for i in XAs]
        Y = test_func(np.array([list(KG_Mc_Input.bestxa)]),gen=False)

        return np.array([list(KG_Mc_Input.bestxa)]) , np.array(Y)
