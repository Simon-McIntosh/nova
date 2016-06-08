import pylab as pl
from cross_coil import Bcoil, mu_o, loop, plasma_coils
from eqlib import eq
from Ptools import topo
import numpy as np
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD'  # 'DD' || 'Conv'
swing = 1
eqlib = eq(config)
eqlib.swing(swing)
coil = eqlib.coil

path = './coil_data/'
file = 'Double_Decker_Coils'

I = 16e6  # plasma current [MA]
delta = 0.2  # node spaceing [m]
path = './coil_data/'
file = 'Double_Decker_Coils'
plasma = loop(path,file,'plasma')
plasma_coils(plasma, I, delta, coil)


r = [8]
z = [3.535]  
scale = -0.2

for i in range(100):    
    B = Bcoil(mu_o, coil, [r[-1],0,z[-1]])
    B_mag = sum(B*B)**0.5
    
    r_step = r[-1]+scale/2*B[0]/B_mag
    z_step = z[-1]+scale/2*B[-1]/B_mag
    
    B = Bcoil(mu_o, coil, [r_step,0,z_step])
    B_mag = sum(B*B)**0.5
    
    r.append(r[-1]+scale*B[0]/B_mag)
    z.append(z[-1]+scale*B[-1]/B_mag)
    
fig = pl.figure(figsize=(9, 12))
ax = fig.add_subplot(111)
plot = topo(config)
#plot.sol()  # scrape-off layer
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.plasma()
plot.plasma_coils(coil)
plot.coil_loc()

pl.plot(r,z,'b-x')
pl.axis('equal')


# contor grid
res = 0.4
xlim, zlim = ([3,14], [-11,5])
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
        B = Bcoil(mu_o, coil, [xm[i,j],0,zm[i,j]])
        Bx[i,j] = B[0]
        Bz[i,j] = B[2]
        feild[i,j] = sum(B*B)**0.5

V = np.linspace(0,10,30)
pl.streamplot(xm, zm, Bx, Bz, density = [2*nx/25,2*nz/25])
pl.axis('equal')
pl.xlim(xlim)
pl.ylim(zlim)
