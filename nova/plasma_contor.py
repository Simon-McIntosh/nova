import numpy as np
import pylab as pl
from cross_coil import mu_o, loop, plasma_coils, Pcoil
from Ptools import topo
from eqlib import eq
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD'  # 'DD' || 'Conv'
swing = 1
eqlib = eq(config)
eqlib.swing(swing)
coil = eqlib.coil

I = 16e6  # plasma current [MA]
delta = 0.2  # node spaceing [m]
path = './coil_data/'
file = 'Double_Decker_Coils'
plasma_shape = loop(path,file,'plasma')
plasma_coils(plasma_shape, I, delta, coil)


fig = pl.figure(figsize=(9, 12))
ax = fig.add_subplot(111)
plot = topo(config)
#plot.sol()  # scrape-off layer
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.plasma()
plot.coil_loc()

# contor grid
res = 0.5
xlim, zlim = ([3,14], [-11,5])
dx, dz = (xlim[1]-xlim[0], zlim[1]-zlim[0])
nx, nz = (int(dx/res), int(dz/res))

x = np.linspace(xlim[0], xlim[1], nx)
z = np.linspace(zlim[0], zlim[1], nz)
xm, zm = np.meshgrid(x, z)
phi = np.zeros((nz,nx))
for i in range(nz):
    for j in range(nx):
        phi[i,j] = Pcoil(mu_o, coil, [xm[i,j],zm[i,j]])
        
        
phi_ref = Pcoil(mu_o, coil, [8,3.535]) 

C = pl.contour(xm, zm, phi, levels=np.linspace(phi_ref-0.5, phi_ref+0.6, 10))

zc = C.collections[0]
pl.setp(zc, linewidth=[1, 2, 12])


pl.axis('equal')
pl.xlim(xlim)
pl.ylim(zlim)