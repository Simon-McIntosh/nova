import numpy as np
import pylab as pl
from cross_coil import loop

path = './coil_data/'
file = 'Double_Decker_Coils'
plasma = loop(path,file,'plasma')

I = 20e6  # plasma current [MA]

pl.figure(figsize=(9, 12))

# contor grid
delta = 0.5
nx, nz = (int(plasma.dx/delta), int(plasma.dz/delta))
x,dx = np.linspace(plasma.xlim[0], plasma.xlim[1], nx, retstep='true')
z,dz = np.linspace(plasma.zlim[0], plasma.zlim[1], nz, retstep='true')
xm, zm = np.meshgrid(x, z)

xp,zp = [],[]

for i in range(nz):
    for j in range(nx):
        if plasma.check([xm[i,j],zm[i,j]]):
            xp.append(xm[i,j])
            zp.append(zm[i,j])
            mark = 'rx'
        else:
            mark = 'bx'

        pl.plot(xm[i,j],zm[i,j], mark)
        
J_plasma = I/(dx*dz*len(xp))

A = len(xp)*dx*dz
print(J_plasma)
pl.plot(plasma.xp,plasma.zp, 'k')
pl.axis('equal')