import numpy as np
import pylab as pl
from cross_coil import Bcoil

coil = {'a':{'I':8e6, 'r':5, 'z':5, 'rc':0.5},
        'b':{'I':8e6, 'r':5, 'z':0, 'rc':0.5}}


# contor grid
res = 0.4
xlim, zlim = ([-10,10], [-5,10])
dx, dz = (xlim[1]-xlim[0], zlim[1]-zlim[0])
nx, nz = (int(dx/res), int(dz/res))

x = np.linspace(xlim[0], xlim[1], nx)
z = np.linspace(zlim[0], zlim[1], nz)
xm, zm = np.meshgrid(x, z)
feild = np.zeros((nz,nx))
Bx = np.zeros((nz,nx))
Bz = np.zeros((nz,nx))
for i in range(nz):
    for j in range(nx):
        B = Bcoil(coil, [xm[i,j],zm[i,j]])
        Bx[i,j] = B[0]
        Bz[i,j] = B[2]
        #feild[i,j] = B[2]#sum(B*B)**0.5
        feild[i,j] = sum(B*B)**0.5

        
pl.figure(figsize=(12,12))
V = np.linspace(0,2,30)
#pl.contour(xm, zm, feild,V)
pl.plot(xm,zm,'k.')
pl.quiver(xm, zm, Bx, Bz)
pl.streamplot(xm, zm, Bx, Bz)
pl.axis('equal')
pl.xlim(xlim)
pl.ylim(zlim)