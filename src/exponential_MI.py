import pymc3 as pm
import numpy as np
import pandas as pd
import estimators as est
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.integrate import nquad
import random

#this file produces benchmark results for bivariate exponential. Results are saved and plottet in seperate jupyter notebook



random.seed(1234)
print(f"Running on PyMC3 v{pm.__version__}")

D = np.arange(0.1,1,step=0.1)
Deltas = []

#formula for bivariate distribution, Gumble et al.
for delta in D:
    def logp(x,y,delta = delta):
        return np.log(1+(x+y-1)*delta + x*y*delta**2)-(x+y+x*y*delta)
         
    
    
    
    with pm.Model() as model:
        x = pm.Exponential('x', lam=1)
        y = pm.Exponential('y', lam=1)
        #delta = pm.Beta('delta', mu = 0.5, sigma=0.2)
        #mv_exp = pm.DensityDist('mv_exp', logp, observed = dict(x=x, y=y, delta=delta))
        mv_exp = pm.DensityDist('mv_exp', logp, observed = dict(x=x, y=y))
        trace = pm.sample(tune=5000, draws=1000, chains=25, target_accept=0.9,
                          return_inferencedata=True, idata_kwargs={"density_dist_obs": False})
        
    
    for i in range(1):
        plt.scatter(trace['posterior']['x'][i,:],trace['posterior']['y'][i,:])
        
    plt.show()
        
    
    x = expon.rvs(size=1000)
    y = expon.rvs(size=1000)
    
    plt.scatter(x,y)
    plt.show()
    
    mis = []
    for i in range(25):
        y = (np.array(trace['posterior']['x'][i,:]).reshape(-1,1),
             np.array(trace['posterior']['y'][i,:]).reshape(-1,1))
        mis.append(est.KSG(y, k=2))
        mis.append(est.BannMI(y, k=10, prior='empiric', unit='nats', measure='MI'))
        mis.append(est.BannMI(y, k=10, prior=0.1, unit='nats', measure='MI'))
        mis.append(est.NMI(y, k=10, unit='nats', eps=True, measure='MI'))
        mis.append(est.WMI(y, k=2, unit='nats', eps=1e-6, measure='MI'))
    mi = np.array(mis)
    mi = mi.reshape(-1,5)
    MI = pd.DataFrame(mi, columns=['KSG', 'BannMI','uBannMI', 'NMI', 'WMI'])
    Deltas.append(MI)
    
MiMu = [i.mean(axis=0) for i in Deltas]

MiMu = pd.DataFrame(MiMu, index = D)

#%%

# numeric integration of KLD / MI formula 
truth = []
for delta in D:
    def fxy(x,y):
        return (1+(x+y-1)*delta + x*y*delta**2)*np.exp(-x-y-x*y*delta)
    
    def fxfy(x,y):
        return np.exp(-x-y)
    
    def KLD(x,y):
        return (np.log(1 + (x+y-1)*delta + x*y*delta**2) - delta*x*y)*fxy(x,y)
    
    
    m = 100
    result = nquad(KLD, [[0,m],[0,m]]) 
    truth.append(result[0])
    
MiMu['truth'] = truth
MiMu.to_csv('MV_expo.csv')


MiSd = [i.std(axis=0) for i in Deltas]

MiSd = pd.DataFrame(MiSd, index = D)
MiSd.to_csv('MV_expo_sd.csv')
