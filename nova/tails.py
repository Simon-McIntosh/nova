import numpy as np
import pylab as pl

def tails(x,dx=0.1):
    if x > (1-dx):
        y = (1-dx) + dx*(1-np.exp(-(x-(1-dx))))
    elif x < dx:
        y = dx - dx*(1-np.exp(-(dx-x)))
    else:
        y = x
    return y
        
X = np.linspace(-10,20,1000)
Y = np.zeros(len(X))
for i,x in enumerate(X): 
    Y[i] = tails(x,dx=0.1)
pl.plot(X,Y)
            
        