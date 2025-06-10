import numpy as np

def multivariate_normal(dim = 3, cov = None, mean = None, npoints = 1000, seed = None):
    if mean is None : mean = np.zeros(dim)
    if cov is None : cov = np.eye(dim)*0.1
    values = np.random.default_rng(seed).multivariate_normal(mean, cov, npoints)
    mask = None
    for i in range(dim):
        mk = np.abs(values[:,i])<1.0
        if mask is None:
            mask = mk
        else:
            mask = mask & mk
    values = values[mask]
    return values
