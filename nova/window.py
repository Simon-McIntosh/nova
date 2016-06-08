import numpy as np
import pylab as pl

start = 0.1
end = 0.7
trans = 0.2
N = 601

def window(N,start,end,trans):
    L = np.linspace(0,1,N)
    w = np.ones(N)
    w[(L<start) | (L>end)] = 0
    index = (L>=start) & (L<start+trans)
    w[index] = 0.5+np.sin(np.pi*(L[index]-start)/trans-np.pi/2)/2
    index = (L<=end) & (L>end-trans)
    w[index] = 0.5+np.sin(np.pi*(end-L[index])/trans-np.pi/2)/2
    return w

window(N,start,end,trans)
pl.plot(w)
