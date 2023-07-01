import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mv
#import pymc3 as pm
from scipy.integrate import nquad
from numpy.linalg import inv
import estimators as est
np.random.seed(1234)

# this file produces the benchmark results for the skew Gaussian distribution with d=2. Results are saved as 'MI_skew.csv' and plotted in a seperate notebook



# the correlation matrix from which the 2d gaussian data are samples, is based on CyTOF data. The gaussian data is used to derive skew gaussian data, as in Azzalini et al.

# in the git version, another cell lines is applied due to previous usage of 
# that cell line in the examples_notebook file
X = pd.read_csv('data/MCF12A.csv')
X.loc[(X.treatment=='EGF')&(X.time==0.0),:]

effect = ['cleavedCas', 'p.AMPK']

Cor = X.loc[:,effect].corr()
d = 2

#%%
n = 1000
MI = []
SD = []

Omega = np.array(Cor)[:d,:d]
vec = np.zeros(d)
for dlt in 0.1*np.arange(1,9):
    delta = np.array([0.5,dlt])
    Delta = np.diag((1-delta**2)**(0.5))
    lmbd= delta/((1-delta**2)**(0.5))
    a = (np.transpose(lmbd.reshape(-1,1))@inv(Omega)@inv(Delta))/((1+np.transpose(lmbd.reshape(-1,1))@inv(Omega)@lmbd.reshape(-1-1))**(0.5))
    Phi = Delta@(Omega+lmbd.reshape(-1,1)@np.transpose(lmbd.reshape(-1,1)))@Delta
    rv =mv(mean=vec, cov=Phi, allow_singular=False)
    
    
    #independent
    Omega_ = Omega.copy()
    Omega_[-1,:-1]=0
    Omega_[:-1,-1]=0
    delta_ = delta.copy()
    delta_[-1]=0
    
    Delta_ = np.diag((1-delta_**2)**(0.5))
    lmbd_= delta_/((1-delta_**2)**(0.5))
    a_ = (np.transpose(lmbd_.reshape(-1,1))@inv(Omega_)@inv(Delta_))/((1+np.transpose(lmbd_.reshape(-1,1))@inv(Omega_)@lmbd_.reshape(-1-1))**(0.5))
    Phi_ = Delta_@(Omega_+lmbd_.reshape(-1,1)@np.transpose(lmbd_.reshape(-1,1)))@Delta_
    rv_ =mv(mean=vec, cov=Phi_, allow_singular=False)
    
    
    def to_integrate(x1,x2):
        x = np.array([x1,x2])
        return np.log(rv.cdf(a*x)/rv_.cdf(a_*x))*2*rv.pdf(x)*rv.cdf(a*x)
        
    
  
    supp = [-10,10]
    
    S = []
    for i in range(d):
        S.append(supp)
    
    
    options={'limit':500}
    r = nquad(to_integrate,S, opts=[options, options])
    numeric = r[0]
   
    
    MIs = []
    
    for j in range(25):
    
        
        Y_0 = mv(0,1).rvs(n)
        Y = mv(vec, Omega).rvs(n)
        
        
        
        # sample: create samples via normals
        Z = []
        for i in range(d):
            Z.append(delta[i]*np.abs(Y_0) + (1-delta[i]**2)**(0.5)*Y[:,i])
        z = np.array(Z)
        
        #%%
        
        b = est.BannMI((z[0,:],z[1,:]),measure='MI', k = 10, unit='nats', prior='empiric')
        bx = est.BannMI((z[0,:],z[1,:]),measure='MI', k = 10, unit='nats', prior=0.1)
        ksg = est.KSG((z[0,:],z[1,:]), k = 1)
        nmi = est.NMI((z[0,:],z[1,:]), measure='MI', k = 10, unit='nats', eps=True)
        w = est.WMI((z[0,:],z[1,:]), measure='MI', k = 2, unit='nats', eps=True)
        MIs.append([b,bx,ksg,nmi,w])
    x = np.append(numeric, np.array(MIs).mean(axis = 0))
    print(x)
    MI.append(x)
    SD.append(np.array(MIs).std(axis = 0))    
#%%    
 
MI = pd.DataFrame(MI, columns = ['num', 'BannMI','uBannMI','KSG', 'NMI', 'WMI'])  
SD = pd.DataFrame(SD, columns = ['BannMI','uBannMI', 'KSG', 'NMI', 'WMI'])
MI.to_csv('MI_skew.csv')
SD.to_csv('SD_skew.csv')
