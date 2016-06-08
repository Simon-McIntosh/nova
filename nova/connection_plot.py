from png_tools import data_mine
import pylab as pl
import shelve
import pickle as pk
import numpy as np

path = '../Figs/datamine/'
file = 'Louter_vs_rus_2015'
file = 'CREATE'

#data_ID = 'tmp'
#data_mine(path, file, data_ID, x_fig=[0,0.005], y_fig=[10,10000])


ID = 'tmp'
with open(path+file+'_'+ID+'.dat', 'rb') as f:  # load segment geometory
    seg = pk.load(f)
r,z = seg[:,0],seg[:,1]

b = np.log10(10e3/10)/(10e3-10)
a = 10e3/10**(b*10e3)

y = a*10**(b*z)
print(y)

