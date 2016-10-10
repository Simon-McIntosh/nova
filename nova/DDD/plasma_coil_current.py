import numpy as np
import pylab as pl
from scipy.interpolate import interp1d as interp1

I = 16e6  # MA

theta = np.linspace(-np.pi,np.pi,100)
zo = 0
ro = 9.08
A = 3.89
k = 1.69
delta = 0.44

GSdel = A*(np.sqrt(1+A**-2)-1)

def elipsoid(theta,ro,zo,A,k,delta):
    R = ro+ro/A*np.cos(theta+delta*np.sin(theta))
    Z = zo+ro/A*k*np.sin(theta)
    return (R,Z)
    
pl.figure(figsize=(10,14))
N = 8

dcoil = 1.5*ro/(N*A)
rshift = np.linspace(0.4,0,N)
rmax = ro/A
rspace = np.linspace(0,rmax,N)

for shift,r in zip(rshift,rspace):
    R,Z = elipsoid(theta,ro+shift,zo,ro/r,k,delta)
    pl.plot(R,Z,'k-')

Rc,Zc,nc = np.array([]),np.array([]),np.array([])
for shift,r in zip(rshift[:-1],rspace[:-1]):
    if r  == 0.0:
        R,Z = ro+shift,zo
        Rc,Zc = np.append(Rc,R),np.append(Rc,Z)
        nc = np.append(nc,1)
    else:
        R,Z = elipsoid(theta,ro+shift,zo,ro/r,k,delta)
        L = np.cumsum(np.sqrt(np.gradient(R)**2+np.gradient(Z)**2))
        Linterp = np.linspace(L[0],L[-1],np.round(L[-1]/dcoil))
        Rc = np.append(Rc,interp1(L,R)(Linterp))
        Zc = np.append(Zc,interp1(L,Z)(Linterp))
        nc = np.append(nc,len(interp1(L,R)(Linterp)))

Ci = (1-np.arange(N-1)/(N-1))*nc  # I = Io*sum(Ci) = Io*sum(nci(1-xi/(N-1)))
Io = I/np.sum(Ci)
Ic = np.zeros(np.shape(Rc))
ncOld = 0
for i in range(len(nc)):
    ncSum = sum(nc[0:i+1])
    Ic[ncOld:ncSum] = Io*(1-i/(N-1))
    ncOld = ncSum

print(sum(Ic))

pl.plot(Rc,Zc,'ro')
    
