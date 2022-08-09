from julia import Main
Main.include("shrinkage.jl")
import numpy as np
import random
from analysis import *
import h5py


#wv = np.array([[0.1,0.02], [0.3,0.05], [0.1,0.02], [0.1,0.02]])
x = np.random.normal(loc= 300.0, size=1000)
#print(x)

print(bootstrap(x,Nboot=200,f=mean))
print(covar_from_boots(bootstrap(x,Nboot=200,f=mean)))
#print(bin_data(x,2,silent_trunc=True))
#print(Main.optimalShrinkage(x))
