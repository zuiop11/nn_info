import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma, gamma
from numpy.random import choice
import sympy



#%%

def BannMI(tpl, measure = 'MI', k = 15, unit='bits', prior='empiric', estimate='mean'):
    
    """
    Bayesian nearest neighbor ratio estimation for mutual information
    and Kullback-Leibler divergences, as proposed in
    Schmidt et al., 2023, soon to come...
    In particular, we recommend BannMI for dependency analysis 
    of phosphoproteomic data (multivariate protein-protein-interactions)
    The algorithm was inspired by NMI (see function)
    """
    
    X = np.array(tpl[0])
    Y = np.array(tpl[1])
    
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)

    if measure=='KLD':
        p = X
        q = Y
    
    elif measure=='MI':
        #shuffle data such that dependencies disappear
        Y_per = Y.copy()
        np.random.shuffle(Y_per)
        p = np.hstack((X,Y))
        q = np.hstack((X,Y_per))
    
    n = p.shape[0]
    n2 = 2*n


    both = np.concatenate((p, q))
    # shuffle data such that neighbors are mixed in case of identical values
    permut = choice(n2,n2,replace = False)
    both[permut]

    
    label = np.concatenate((np.zeros(n), np.ones(n)))
    label[permut]
    nbrs = NearestNeighbors(n_neighbors=k).fit(both)
    indices = nbrs.kneighbors(p, return_distance=False)

    q_neighbors = np.sum(label[indices], axis=1)
    c = k - q_neighbors

    # case: distributions are identical (t_var==0)
    t_var = np.nanvar((c/k))
    if not t_var:
        return 0.
    
    if prior=='empiric':
        t_mu = np.nanmean((c/k))
        # hyperparameters
        b = np.array((t_mu-t_mu**2-t_var)*(1-t_mu)/t_var)
        a = np.array(b*t_mu/(1-t_mu))
        
        # case for < 1 prior beta is convex
        b[b<1]=1
        a[a<1]=1        
        
    else:
        b = prior
        a = prior

    a_new = a + c
    b_new = b + k - c

    if estimate == 'mean':
        theta = (a_new)/(a_new + b_new)
    elif estimate == 'mode':
        theta = (a_new + 1)/(a_new + b_new + 2)
    ratio = theta/(1-theta)
    
    if unit == 'bits':
        res = np.mean(np.log2(ratio))
    elif unit == 'nats':
        res = np.mean(np.log(ratio))
    else:
        print('please select nats or bits as unit')
    
    return max(res, 0.)


#%%
    
def WMI(tpl, measure = 'KLD', k = 5, unit='nats', eps=False):
    
    """
    Erstimation of (multivariate) differential Kullback-Leibler Divergence
    via nearest neighbor distances as proposed in 
    Wang, Q. et al. (2009). Divergence Estimation for Multidimensional
    Densities Via k-Nearest-Neighbor Distances. IEEE Transactions of
    Information Theory, 55(5), 2392–2405.
    here, epsilon parameter is introduced to adjust for distance 0.
    if measure is not 'divergence', WMI computes mutual information between 
    tpl[0] and tpl[1]
    
    """
    
    X = np.array(tpl[0])
    Y = np.array(tpl[1])
    
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)

    if measure == 'KLD':
        p = X
        q = Y
    
    elif measure == 'MI': 
        Y_per = Y.copy()
        np.random.shuffle(Y_per)
        p = np.hstack((X,Y))
        q = np.hstack((X,Y_per))
        
    n = p.shape[0]
    d = p.shape[1] 

        
    Xnbrs = NearestNeighbors(n_neighbors= k + 1).fit(p)
    Ynbrs = NearestNeighbors(n_neighbors= k).fit(q)
    Xdistances, indices = Xnbrs.kneighbors(p, return_distance=True)
    Ydistances, indices = Ynbrs.kneighbors(p, return_distance=True)
    xdist = Xdistances[:,k]
    ydist = Ydistances[:,k-1]
    
    if eps:
        xdist[xdist==0]=np.finfo(float).eps
        ydist[ydist==0]=np.finfo(float).eps
            
    px = k/(xdist**d*(n-1))
    py = k/(ydist**d*n)
    
    if unit == 'bits':
        res = np.mean(np.log2(px/py))
    elif unit == 'nats': 
        res = np.mean(np.log(px/py))
    else:
        print('please select nats or bits as input')
           
    return max(res, 0.)

#%%

def KSG(tpl, k = 2):

    """
    function sklearn.feature_selection.mutual_info_regression 
    https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/feature_selection.rst
    adjusted for multivariate variables. Algorithm was proposed in
    Kraskov, A. et al. (2004). Estimating mutual information. Phys. Rev. E,
69, 066138.
    """

    x = tpl[0]
    y = tpl[1]
    n_samples = x.shape[0]

    if x.ndim == 1:
        x = x.reshape((-1, 1))
    if y.ndim == 1: 
        y = y.reshape((-1, 1))
    xy = np.hstack((x, y))
    permut = choice(n_samples,n_samples,replace = False)
    xy[permut]

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=k)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (digamma(n_samples) + digamma(k)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)



#%%

def entropy(X, k=1, unit = 'nats'): 
    
    """
    nearest neighbor estimation of differential entropy as proposed in
    Kozachenko, L. and Leonenko, N. (1987). Sample Estimate of the Entropy
    of a Random Vector. Problems of Information Transmission, 23(2),
    95–101.
    """
    
    X = np.array(X)
    gam = sympy.EulerGamma.evalf()

    
    if X.ndim == 1:
        X = X.reshape(-1,1)
        
     
    n = X.shape[0]
    d = X.shape[1] 

        
    Xnbrs = NearestNeighbors(n_neighbors= k + 1).fit(X)
    Xdistances, indices = Xnbrs.kneighbors(X, return_distance=True)
    xdist = Xdistances[:,k]
    v = np.exp(np.array([gam], dtype = float))[0]
    px = (k*gamma(d/2+1))/(xdist**d*(n-1)*v*np.pi**(d/2))
    
    if unit == 'bits':
        res = np.mean(np.log2(px))
    elif unit == 'nats':
        res = np.mean(np.log(px))
    else:
        print('please select nats or bits as unit')
           
    return max(-res, 0.)



#%%

def NMI(tpl, k=50, eps=True, measure = 'KLD', algo='ball_tree', leaf=30, unit='nats'):
    
    """
    Estimation of (multivariate) Kullback-Leibler divergence 
    estimation via nearest neighbor ratios. The algorithm was proposen 
    in Noshad, M. et al. (2017). Direct estimation of information divergence
    using nearest neighbor ratios. In 2017 IEEE International Symposium
    on Information Theory (ISIT 2017)", pages 903 – 907, Aachen, Germany.
    Institute of Electrical and Electronics Engineers ( IEEE ).
    The proposed f-divergence estimator is adjusted for Kullback-
    Leibler divergence estimation (with functional '-log' as plug-in). 
    As the nearest neighbor of a datapoint is the datapoint itself, 
    there is no need for '+1' adjustment. Furthermore, we case discriminate
    the ratio directly via application of max function instead of -log(ratio)
    as in the latter case, eps would be a large number
    """
        
    X = np.array(tpl[0])
    Y = np.array(tpl[1])
    
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)

    if measure == 'KLD':
        p = X
        q = Y
    
    elif measure == 'MI': 
        Y_per = Y.copy()
        np.random.shuffle(Y_per)
        p = np.hstack((X,Y))
        q = np.hstack((X,Y_per))
        
    n = p.shape[0]
    d = p.shape[1] 
    
    
    if eps:
        eps = np.finfo(float).eps
    elif eps == 'adapt':
        eps = 1/(n*d)


    if k == 'adapt':
        k = 3*int(np.sqrt(n))


    both = np.concatenate((p, q))
    label = np.concatenate((np.zeros(n), np.ones(n)))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algo, 
                            leaf_size=leaf).fit(both)
    indices = nbrs.kneighbors(p, return_distance=False)

    # sum over q-label neighbors
    q_neighbors = np.sum(label[indices], axis=1)
    p_neighbors = k - q_neighbors
    ratio = q_neighbors/p_neighbors
    ratio[ratio < eps] = eps
    # p dist on top
    #ratio = 1/ratio
        
    if unit == 'nats':
        res = -np.mean(np.log(ratio))
    elif unit=='bits':
        res = -np.mean(np.log2(ratio))
    else:
        print('please select nats or bits as unit')
        
    return max(res,0.)