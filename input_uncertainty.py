import GPy
import csv
import numpy as np
import scipy
from scipy.optimize import minimize
import time
# import pygmo as pg
from scipy.stats import uniform 
from pyDOE import *
from scipy import optimize
import pandas as pd

import scipy.integrate as integrate
import scipy.special as special

import time
from scipy.stats import truncnorm
from scipy.interpolate import interp1d

def Mult_Input_Uncert():
    global Nx, Nr, init_sample, x, MU, SIG, MUSIG0,MU_L, SIG_L, MUSIG0_L,Stop

    def COV(model,xa1,xa2):
        K = model.kern.K(model.X,model.X)
        L = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))
        Lk1 = np.linalg.solve(L, model.kern.K(model.X, xa1))
        Lk2 = np.linalg.solve(L, model.kern.K(model.X, xa2))
        K_ = model.kern.K(xa1, xa2)
        s2 = np.matrix(K_) - np.matrix(np.dot(Lk2.T,Lk1))
        return s2

    def COV1(model,xa1):
        K = model.kern.K(model.X,model.X)
        L = np.linalg.cholesky(K + (0.1**2.0)*np.eye(len(K)))
        Lk = np.linalg.solve(L, model.kern.K(model.X, xa1))
        K_ = model.kern.K(xa1, xa1)
        s2 = K_ - np.sum(Lk**2, axis=0)
        return s2

    def true_Q(a):
        X = lhs(1, samples=1000)*100
        xa = [[i,a] for val in X for i in val]
        F_a = np.sum(test_func(xa, NoiseSD=0,gen=False))
        return F_a

    def predic_Q(a,model,Nx):
        X = lhs(1, samples=1000)*100
        xa = [[i,a] for val in X for i in val]
        F_x = np.mean(model.predict(np.array(xa))[0])
        return F_x

    def SUM_MU(X,a,model,Nx):
        xa = [[i,a] for val in X for i in val]
        F_a = np.sum(model.predict(np.array(xa))[0])
        return F_a

    def SUM_COV(a,xan,model,Nx):
        X = lhs(1, samples=Nx)*100
        xa = [[i,a] for val in X for i in val]
        COV = [model.kern.K(np.array([i]),np.array([xan])) for i in xa]
        SUM_COV = np.sum(COV)
        return SUM_COV

        
    def test_func(xa, NoiseSD=np.sqrt(0.01), seed=11,gen=True):
        #np.random.seed(np.int(time.clock()*10000))
        KERNEL = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=([10,10,10]), ARD = True)
        
        if gen == True:
            #np.random.seed(seed)
            Xt0=np.linspace(0,100,8) ; nX = len(Xt0)
            Ft1=np.linspace(0,100,8);  nF1 = len(Ft1)
            Ft2=np.linspace(0,100,8);  nF2 = len(Ft2)
            #Ft3=np.linspace(0,100,8);  nF3 = len(Ft3)

            test_func.XtFi = np.array([np.array([i,j,k]) for i in Xt0 for j in Ft1 for k in Ft2])
            mu = np.zeros(len(test_func.XtFi))
            C = KERNEL.K(np.array(test_func.XtFi), np.array(test_func.XtFi))

            Z = np.random.multivariate_normal(np.zeros(len(C)), C)
            invC = np.linalg.inv(C+np.eye(len(C))*1e-3)

            test_func.invCZ = np.dot(invC,np.matrix(Z).reshape(len(Z),1))

        ks = KERNEL.K(np.array(xa),np.array(test_func.XtFi))
        out = np.dot(ks,test_func.invCZ)

        E = np.random.normal(0,NoiseSD,len(xa))

        return (out.reshape(len(out),1) + E.reshape(len(E),1))
    
    def Input_Source(n,s,mv_gen = True):
        
        global f_mean, f_cov
        #np.random.seed(np.int(time.clock()*10000))
        ub = 85
        lb = 15
        var = np.random.random(n)*(20-5)+5
        Mult_Input_Uncert.var = var
        def init_params():  
            mean = np.random.random(n)*(ub-lb)+lb
            cov = np.multiply(np.identity(n),var)
            return mean, cov
        if mv_gen == True:
            f_mean,f_cov = init_params()
            Input_Source.f_mean = f_mean
#         print('mean',f_mean,'f_cov',f_cov)
        rn = np.random.multivariate_normal(f_mean, f_cov, s)
        return rn
    
    def Fit_Inputs(Y =[]):
        Y = np.array(Y)
        Y = list(Y[~np.isnan(Y)])
        def Distr_Update():

            L = (np.exp(-(1.0/(2.0*MUSIG0[:,1]))*np.sum(np.array((np.matrix(MUSIG0[:,0]).T  - Y))**2.0,axis=1))*(1.0/np.sqrt(2*np.pi*MUSIG0[:,1]))**len(Y))
            L = np.array(L).reshape(len(MU),len(SIG))
            dmu = MU[1]-MU[0]
            dsig = SIG[1]-SIG[0]
            LN = np.sum(L*dmu*dsig)
            P = L/LN
            marg_mu = np.sum(P,axis=1)*dsig
            return marg_mu
        Dist = Distr_Update()
        return MU, Dist
    



    def Gen_Sample(Dist, N=500):
        elements = np.linspace(0,100,len(Dist))
        probabilities = Dist/np.sum(Dist)
        val = np.random.choice(elements, N, p=probabilities)
        return val
    
    def sample_predict_dens(Data,N):
        global MUSIG0,MU_L, SIG_L, MUSIG0_L
        
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

    def Delta_Loss(Data,idx):
        global Nr, Nx
        
        Nd = 100
        #INPU: Matrix with N dimensions and S realisations
        def W_aj(Y=np.array([]), a=np.array([])):
           
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
    

    def KG_Mc_Input(XA,P,model, Nx=15,Ns=20):
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

        def KG(musig):
            n = len(musig)
            O = musig[:,1].argsort()
            b = musig[O][:,1]
            a = musig[O][:,0]

            A=[0]
            C=[-float("inf")]
            while A[-1]<n-1:
                s = A[-1]
                si = range(s+1,n)
                Ci = -(a[s]-a[si])/(b[s]-b[si])
                bestsi=np.argmin(Ci)
                C.append(Ci[bestsi])
                A.append(si[bestsi])

            C.append(float("inf"))
            
            out = np.sum(a[A]*(scipy.stats.norm.cdf(C[1:len(C)])-scipy.stats.norm.cdf(C[0:-1])) + b[A]*(scipy.stats.norm.pdf(C[0:-1])-scipy.stats.norm.pdf(C[1:len(C)]))) - np.max(musig[:,0])
            return out

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
                sigt1   = COV1(m,np.array([xa]))
                sigt    = ((sigt2) / np.sqrt(sigt1+KG_Mc_Input.noiseVar))
                sigt = sigt/(len(KG_Mc_Input.Ad ))
                
                musig = np.c_[MM,sigt]
                out  = KG(musig)
                if out > KG_Mc_Input.bestEVI:
                    KG_Mc_Input.bestEVI = out
                    KG_Mc_Input.bestxa = xa
                return -out
        XAs = np.array(np.c_[lhs(1, samples=KG_Mc_Input.Ns)*100,lhs(1, samples=KG_Mc_Input.Ns)*100,lhs(1, samples=KG_Mc_Input.Ns)*100])
        

        
        A    = [minimize(KG_IU, i, method='nelder-mead', options={'maxiter': 80}).x for i in XAs]
        Y = test_func(np.array([list(KG_Mc_Input.bestxa)]),gen=False)
        
        return np.array([list(KG_Mc_Input.bestxa)]) , np.array(Y)

    #=============================================================================================
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #==============================================================================================
    #Main Algorithm
    
    #initialise Parameters
    
        #Random variable generation for inputs
    
    init_sample = 10 # Initial Sample size
    Nx = 101 # Granularity x value
    Nr =  100
    EndN = 100
    dim = 2
    x = np.linspace(0,100, Nx) #vector of input variable

    precision = 101
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
    
    #Train GP
    XA = np.c_[lhs(1, samples=init_sample)*100,lhs(1, samples=init_sample)*100,lhs(1, samples=init_sample)*100]
    Y = test_func(xa=XA,gen=True)
    
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
        
        print(len(XA) +  Ndata)
        Ndata = np.sum([len(Data[:,i][~np.isnan(Data[:,i])]) for i in range(dim)])
        m = GPy.models.GPRegression(np.array(XA) , np.array(Y).reshape(len(Y),1) , ker,noise_var=0.01)
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
         
            Sample = np.c_[np.repeat(x,Nr) , list(pdf1_gen)*Nx, list(pdf2_gen)*Nx ]

        IU = np.mean(m.predict(np.array(Sample))[0].reshape(Nx,Nr),axis=1)

        Xr = x[np.argmax(IU)]


        topobj = -1*obj(Xr)
        DIF = best_ - topobj

        OC.append(DIF)

        #===============================================================================================
        
        xa, p = KG_Mc_Input(XA,Y,m,Nx=Nx,Ns=20)

        Comparison = np.concatenate(([KG_Mc_Input.bestEVI],[Delta_Loss(Data,i) for i in range(dim)]))
        chs = np.argmax(Comparison)

        d_point = np.array([[np.nan,np.nan]])
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


OC= []
for i in range(100):
    identifier = 100 + np.random.random()
    OC ,N_I, var = Mult_Input_Uncert() 
    data = {'OC': OC,'len': [N_I]*len(OC),'var1':[var[0]]*len(OC),'var2':[var[1]]*len(OC)}
    print('data',data)
    gen_file = pd.DataFrame.from_dict(data)
    path ='/home/rawsys/matjiu/PythonCodes/PHD/Input_Uncertainty/With_Input_Selection/Data_MC100/OC_'+str(i)+'_' + str(identifier) + '.csv'
    gen_file.to_csv(path_or_buf=path)
