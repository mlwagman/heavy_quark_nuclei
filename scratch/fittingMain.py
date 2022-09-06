from julia import Main
Main.include("shrinkage.jl")
import numpy as np
import random
from analysis import *
import h5py


x = np.random.normal(loc= 300.0, size=1000)
bootx
#print(x)
print(bootstrap(x,Nboot=200,f=mean))
print(covar_from_boots(bootstrap(x,Nboot=200,f=mean)))
print(covar_from_boots(x))
#print(bin_data(x,2,silent_trunc=True))
#print(Main.optimalShrinkage(x))
