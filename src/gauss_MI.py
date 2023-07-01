import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool
from functools import partial
import estimators as est
from scipy.stats import multivariate_normal as mv
np.random.seed(1234)

# benchmark computations for Gaussian variables with covariance matrices derived from cell lines. covariance matrices were computed before in python file. here, parallel computing is used and dimension of variables is only max 10

# names of cell lines
lines = ['184A1', 'CAL148', 'HBL100', 'HCC1599', 'HCC3153', 'MACLS2', 'MDAMB231',  'MPE600',
         'ZR751', '184B5', 'CAL51', 'HCC1806', 'HCC38', 'MCF10A', 'MDAMB361', 'MX1', 'ZR7530',
         'AU565', 'CAL851', 'HCC1187', 'HCC1937', 'HCC70', 'MCF10F', 'MDAMB415', 'OCUBM',
         'BT20', 'HCC1395', 'HCC1954', 'HDQP1', 'MCF12A', 'MDAMB436', 'SKBR3', 'BT474', 'DU4475',
         'HCC1419', 'HCC202', 'Hs578T', 'MCF7', 'MDAMB453', 'T47D', 'BT483', 'EFM19', 'HCC1428',
         'HCC2157', 'JIMT1', 'MDAMB134VI', 'MDAMB468', 'UACC3199', 'BT549', 'EFM192A', 'HCC1500',
         'HCC2185', 'MDAMB157', 'MDAkb2', 'UACC812', 'EVSAT', 'HCC1569', 'HCC2218', 'LY2',
         'MDAMB175VII', 'MFM223', 'UACC893']


# funiction to compute analytical Kullback-Leibler divergence (MI) value 
def kl_theo(x_0, x_1, cov0=False, cov1=False, samples=True):
    
    """
    compute theoretical Kullback Leibler Divergence on Gaussians
    usable with input parameters or samples
    fix: investigate eps scenario and cases
    """
    
    if samples:
        k = x_0.shape[1]
        mu0 = np.mean(x_0, axis=0)
        mu1 = np.mean(x_1, axis=0)
        cov0 = np.cov(x_0, rowvar=False)
        cov1 = np.cov(x_1, rowvar=False)
    else:
        if np.isscalar(x_0):
            x_0 = np.array([x_0])
        if np.isscalar(x_1):
            x_1 = np.array([x_1])
        k = len(x_0)
        
        
        mu0 = x_0
        mu1 = x_1
      
    if k==1:
        return 0.5*((cov0/cov1)**2+(mu1-mu0)**2/cov1**2-1+2*np.log(cov1/cov0))
    
    else:
        part1 = np.trace(np.linalg.inv(cov1) @ cov0)
        
        part2 = (mu1 - mu0).reshape((-1, k)) @ np.linalg.inv(cov1) @ (mu1 - mu0) - k
        
        if np.isnan(np.linalg.det(cov1)/np.linalg.det(cov0)):
            return np.nan
            
        else: 
            part3 = np.log(np.linalg.det(cov1)/np.linalg.det(cov0))
       
        return max((0.5*(part1 + part2 + part3)[0]), 0)
    

#%%

# parameters for the testing
n = 1000
rep = 25
dim = [2,5,10]
neighbors = [1,2]



mutual_info = []
MSE = []
index=[]


# loop over all parameters to be tested an all cell lines

for d in dim:
    vec = np.zeros(d)
    for j in neighbors:
 
        TRUTH = []
        tpl = []
        for i in lines:
            try:
                Cov = pd.read_hdf('data/Cov_matrix.h5', key = i)
            except FileNotFoundError:
                continue
            Cov = np.array(Cov)[:d,:d]
            
            Cov2=np.copy(Cov)
            Cov2[:-1,-1]=0
            Cov2[-1,:-1]=0
            
            # function for analytical value
            TRUTH.append(kl_theo(vec, vec, Cov, Cov2, samples=False))
            


            for l in range(rep):
                X = mv(vec,Cov).rvs(size = n)
                tpl.append((X[:,:-1],X[:,-1]))
            
        
        if __name__=='__main__':
            with Pool(12) as p:
                mi_Be = p.map(partial(est.BannMI, measure = 'MI', k = 10*j, 
                                      prior = 'empiric', unit='nats'), tpl)
                mi_B = p.map(partial(est.BannMI, measure = 'MI', k = 10*j, 
                                     prior = 0.1, unit='nats'), tpl)
                mi_N = p.map(partial(est.NMI, measure = 'MI', k = 10*j, 
                                     unit='nats', eps=True), tpl)
                mi_KSG = p.map(partial(est.KSG, k = j), tpl)
                mi_W = p.map(partial(est.WMI, measure = 'MI', k = (j+1), 
                                     eps=1e-06, unit='nats'), tpl)
                p.close()
                p.join()
            


        x = [mi_Be, mi_B, mi_N, mi_KSG, mi_W]
        x = [np.array(s).reshape(-1,rep) for s in x]
        
        # means and standard deviations are derived with respect to the 'rep' computations.
        # Hence, a mean value and a std exists for each cell line seperately
        means = [np.nanmean(s, axis = 1) for s in x] 
        sd = [np.nanstd(s, axis = 1) for s in x]         
        means = [TRUTH] + means
        sd = [TRUTH] + sd    
        
        cols = ['truth', 'BannMIe', 'BannMI','NMI', 'KSG', 'WMI']
        MI = pd.DataFrame(means,index=cols, columns = lines).transpose()
        MI.to_hdf('MI_gauss.h5', key = 'dim_' + str(d) + '_k_' + str(j), mode = 'a')
        
        SD = pd.DataFrame(sd,index=cols, columns = lines).transpose()
        SD.to_hdf('MI_gauss_sd.h5', key = 'dim_' + str(d) + '_k_' + str(j), mode = 'a')
    
    # just check how far we have come so far. higher data dimension implies longer 
    # computationsal time    
    print(d)
        


