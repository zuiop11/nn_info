from scipy.stats import multivariate_normal as mv
from scipy.special import digamma, gamma
import estimators as est
import numpy as np



class info:
    '''
    general class for computation of information theoretical measures. 
    Choose measure ('MI', 'KLD' or 'entropy') via measure
    Choose approach ('nn_ratio', 'nn_distance') to select algorithm. 
    '''
    def __init__(self, measure, approach = 'nn_ratio', k = 3, 
                 unit = 'bits', prior = 'empiric', epsilon = False):
        
        measure_value = {'MI', 'KLD', 'entropy'}
        approach_value = {'nn_ratio', 'nn_distance'}
        unit_value = {'bits', 'nats'}

        if measure not in measure_value:
            raise ValueError("info: measure must be one of %r." % measure_value)
            
        if approach not in approach_value:
            raise ValueError("info: approach must be one of %r." % approach_value)

        if unit not in unit_value:
            raise ValueError("info: unit must be one of %r." % unit_value)
        
        self.measure = measure
        self.approach = approach
        self.k = k
        self.unit = unit
        self.prior = prior
        self.epsilon = epsilon
    
            
    #        
    def estimator(self, tpl):
        
        if self.approach == 'nn_ratio':
        
            return est.BannMI(tpl, 
                          measure=self.measure, 
                          k = 5*self.k, 
                          unit=self.unit, 
                          prior = self.prior)
        
        elif self.approach == 'nn_distance':
            
            if self.measure == 'MI':
                
                if self.unit == 'bits':
                    print('please select nats as unit')
                
                elif self.unit == 'nats':
                    return est.KSG(tpl, 
                               k = self.k)
            
            elif self.measure == 'KLD':
                
                return est.WMI(tpl, 
                           measure=self.measure, 
                           k = self.k + 1, 
                           unit = self.unit, 
                           epsilon = self.epsilon)
            elif self.measure == 'entropy':
                
                return est.entropy(tpl, 
                                   k = self.k,
                                   unit = self.unit)
            
        else:
            print('please select either nn_ratio or nn_distance as approach')
  
        
#%%
# just other stuff to ease analytical computation of IT measures


def KLD_gaussian(tpl=False, mu0=False, mu1=False, 
                 cov0=False, cov1=False, samples=True, density_estimate=False):
    
    """
    compute Kullback Leibler Divergence on Gaussians
    either estimate Gaussian parameters based on samples 
    or input parameters
    in the latter case, result is analytical
    """
    
    if samples:
        X = tpl[0]
        Y = tpl[1]
        if X.ndim == 1:
            k=1
        else:
            k = X.shape[1]
        mu0 = np.mean(X, axis=0)
        mu1 = np.mean(Y, axis=0)
        cov0 = np.cov(X, rowvar=False)
        cov1 = np.cov(Y, rowvar=False)


    if density_estimate:
        px = mv.pdf(X, mean=mu0, cov=cov0)
        py = mv.pdf(X, mean=mu1, cov=cov1)
        
        return np.nanmean(np.log(px/py))

    
    if np.isscalar(mu0) or len(mu0)==1:
        x = 0.5*((cov0/cov1)**2+(mu1-mu0)**2/cov1**2-1+2*np.log(cov1/cov0))
        if not np.isscalar(x):
            x = x[0]
        return x

    else:
        k = len(mu0)
        part1 = np.trace(np.linalg.inv(cov1) @ cov0)
        
        part2 = (mu1 - mu0).reshape((-1, k)) @ np.linalg.inv(cov1) @ (mu1 - mu0) - k
        
        if np.isnan(np.linalg.det(cov1)/np.linalg.det(cov0)):
            return np.nan
            
        else: 
            part3 = np.log(np.linalg.det(cov1)/np.linalg.det(cov0))
       
        return max((0.5*(part1 + part2 + part3)[0]), 0)
    

def KLD_gamma(tpl=False, a0=False, a1=False, b0=False, b1=False, samples=True):
    
    """
    compute Kullback Leibler Divergence on Gamma distributions.
    parameters are a=alpha (shape), b=beta (rate)
    either estimate parameters based on samples 
    or input parameters
    in the latter case, result is analytical
    """
    
    if samples:
        X = tpl[0]
        Y = tpl[1]
        
        mu = np.mean(X), np.mean(Y)
        var = np.var(X), np.var(Y)
        
        a0 = mu[0]**2/var[0]
        a1 = mu[1]**2/var[1]
        b0 = mu[0]/var[0]
        b1 = mu[1]/var[1]
    
    part1 = (a0-a1)*digamma(a0) - np.log(gamma(a0)) + np.log(gamma(a1))
    part2 = a1*(np.log(b0)-np.log(b1)) + a0*((b1-b0)/b0)
    res = part1 + part2
    return max(res,0)
    

  