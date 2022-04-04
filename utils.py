import numpy as np
import pandas as pd
from numpy import pi
from pyDOE import *   


    
#latin hypercube sampling
def lhc_samples(n, dim,ranges): 
    samples=lhs(dim, samples=n, criterion='center')
    for i in range(dim): 
       samples[:,i]=samples[:,i]*(ranges[(2*i+1)]-ranges[2*i]) + ranges[2*i]
    return samples

# random integer sampling
def random_integer_sampling(n,dim,ranges):
    for i in range(dim): 
       one_dim_samples=np.random.random_integers(ranges[2*i],ranges[(2*i+1)], size=(n,1))
       if i==0:
         samples= one_dim_samples
       else: 
         samples= np.concatenate((samples,one_dim_samples),axis=1) 
    return samples



