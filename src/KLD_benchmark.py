import numpy as np
import pandas as pd
from itertools import combinations_with_replacement as cwr
from functools import partial
import scipy.integrate as integrate
import estimators as est
from multiprocessing import Pool

# this file is used to produce KLD estimation results as in the supplementary data
# KLD benchmark is performed on random variables with unbounded support.
# outcommented are testing procedure for random varaibles with compact 
# or non-negative support



# x > 0
#from scipy.stats import expon, lognorm, rayleigh, gamma, chi

# x in R
from scipy.stats import norm, skewnorm, logistic, cauchy, t

# 0 < x < 1
#from scipy.stats import beta, uniform, powerlaw

# random varaibles as their names
#positive = [expon(0), expon(0,2), lognorm(1), rayleigh(0), gamma(2) , gamma(3), chi(3), chi(2)]
#positive_n = ['expon1', 'expon2', 'lognorm', 'rayleigh', 'gamma2', 'gamma3',  'chi3', 'chi2']

unbounded = [norm(0,1), norm(0,2), norm(2,1), skewnorm(2) ,logistic(1), cauchy(1), t(5)]
unbounded_n = ['std_norm', 'norm_var2', 'norm_shift2', 'skewnorm2', 'logistic', 'cauchy1', 't']

#compact = [beta(2,2), beta(0.5,0.5), beta(0.5,3), uniform(0,1), powerlaw(2)]
#compact_n = ['beta22', 'beta0505' , 'beta053', 'uniform', 'powerlaw2']



#%%

# form tuples of random variables
P = list(cwr(unbounded, 2))
p_names = list(cwr(unbounded_n, 2))

col = pd.MultiIndex.from_tuples(p_names, names=('p', 'q'))

# numerical integration
kld = []
for i in P:
    def f(x):
        return np.log(i[0].pdf(x)/i[1].pdf(x))*i[0].pdf(x)
    kld.append(integrate.quad(f, -15,15))

Numericals = pd.DataFrame(kld, columns = ['result', 'error'], index = col)

#%%
# next, results for every estimator are derived seperatly due to different parameters

print('BannMI')


dim = [2,5]
size = [200, 500, 1000, 5000]
neighbors = [5,10,15,20,50]
prior = [0.1, 0.2, 0.5, 1.0, 'empiric']


M = []
STD = []
INDEX = []
for p in prior:
    for n in size:
        for k in neighbors:
            for d in dim:
                mean = []
                std = []
                for i in P:
                    tupel = []
                    for j in range(25):
                        tupel.append((i[0].rvs(d*n).reshape(n,d), i[1].rvs(d*n).reshape(n,d)))
                    if __name__=='__main__':
                        with Pool(12) as pl:
                            res = pl.map(partial(est.BannMI, measure='KLD', k=k,
                                                prior=p, unit='nats'), tupel)                        
                        
                    mean.append(np.mean(res))
                    std.append(np.std(res))          
                M.append(mean)
                STD.append(std)
                INDEX.append((d,n,k,p))
                print((d,n,k,p))
ind = pd.MultiIndex.from_tuples(INDEX, names=('dimension', 'samplesize', 'neighbors', 'prior'))
BannMI_mean=pd.DataFrame(M, index=ind, columns=col)

n = Numericals.iloc[:,0]
n.name=(0,0,0,0)
BannMI = pd.concat((n.to_frame().T, BannMI_mean))
BannMI_std=pd.DataFrame(STD, index=ind, columns=col)   
BannMI.to_csv('BannMI_unb.csv')
BannMI_std.to_csv('BannMI_std_unb.csv')

#%%

print('Noshad')


neighbors = [10,20,50,150, 'adapt']
eps = [True, 'adapt', 1e-6, 0.01]



#Samples = []
M = []
STD = []
INDEX = []
for e in eps:
    for n in size:
        for k in neighbors:
            for d in dim:
                mean = []
                std = []
                for i in P:
                    tupel = []
                    for j in range(25):
                        tupel.append((i[0].rvs(d*n).reshape(n,d), i[1].rvs(d*n).reshape(n,d)))
                    if __name__=='__main__':
                        with Pool(12) as pl:
                            res = pl.map(partial(est.NMI, measure='KLD', k=k,
                                                eps=e, unit='nats'), tupel)                        
                        
                    mean.append(np.mean(res))
                    std.append(np.std(res))   
          
                M.append(mean)
                STD.append(std)
                INDEX.append((d,n,k,e))
                print((d,n,k,e))
ind = pd.MultiIndex.from_tuples(INDEX, names=('dimension', 'samplesize', 'neighbors', 'epsilon'))
NMI_mean=pd.DataFrame(M, index=ind, columns=col)
n = Numericals.iloc[:,0]
n.name=(0,0,0,0)
NMInn = pd.concat((n.to_frame().T, NMI_mean))
NMI_std=pd.DataFrame(STD, index=ind, columns=col)   
NMInn.to_csv('NMInn_unb.csv')
NMI_std.to_csv('NMInn_std_unb.csv')


#%%

print('WANG')


neighbors = [2,4,6,8]
eps = [True, 1e-6, 0.01]




#Samples = []
M = []
STD = []
INDEX = []
for e in eps:
    for n in size:
        for k in neighbors:
            for d in dim:
                mean = []
                std = []
                for i in P:
                    tupel = []
                    for j in range(25):
                        tupel.append((i[0].rvs(d*n).reshape(n,d), i[1].rvs(d*n).reshape(n,d)))
                    if __name__=='__main__':
                        with Pool(12) as pl:
                            res = pl.map(partial(est.WMI, measure='KLD', k=k,
                                                eps=e, unit='nats'), tupel)                        
                        
                    mean.append(np.mean(res))
                    std.append(np.std(res))  
      
                M.append(mean)
                STD.append(std)
                INDEX.append((d,n,k,e))
                print((d,n,k,e))
ind = pd.MultiIndex.from_tuples(INDEX, names=('dimension', 'samplesize', 'neighbors', 'epsilon'))
WMI_mean=pd.DataFrame(M, index=ind, columns=col)
n = Numericals.iloc[:,0]
n.name=(0,0,0,0)

WMInn = pd.concat((n.to_frame().T, WMI_mean))
WMI_std=pd.DataFrame(STD, index=ind, columns=col)
WMInn.to_csv('WMInn_unb.csv')   
WMI_std.to_csv('WMInn_std_unb.csv') 
