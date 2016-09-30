import numpy as np
import pylab as pl
from cross_coil import Bcoil, mu_o, loop, plasma_coils
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
plasma = loop(path,file,'plasma')
plasma_coils(plasma, I, delta, coil)


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
rlim, zlim = ([3,14], [-11,5])
dr, dz = (rlim[1]-rlim[0], zlim[1]-zlim[0])
nr, nz = (int(dr/res), int(dz/res))

r = np.linspace(rlim[0], rlim[1], nr)
z = np.linspace(zlim[0], zlim[1], nz)
rm, zm = np.meshgrid(r, z)
Br = np.zeros((nz,nr))
Bz = np.zeros((nz,nr))
Bmag = np.zeros((nz,nr))
for i in range(nz):
    for j in range(nr):
        B = Bcoil(mu_o, coil, [rm[i,j],zm[i,j]])
        Br[i,j] = B[0]
        Bz[i,j] = B[1]
        Bmag[i,j] = sum(B*B)**0.5

V = np.linspace(0,10,30)
pl.streamplot(rm, zm, Br, Bz, density = [2*nr/25,2*nz/25])
pl.contour(rm, zm, Bmag, levels=[5*np.min(Bmag)])
pl.axis('equal')
pl.xlim(rlim)
pl.ylim(zlim)

if I > 0:
    plasma_dir = 'pos'
elif I == 0:
    plasma_dir = 'off' 
else:
    plasma_dir = 'neg'
#pl.savefig(fig_path+config+'_flux_conv_'+plasma_dir+'.png', dpi=300)


fig = pl.figure(figsize=(9, 12))
ax = fig.add_subplot(111)
pl.pcolor(rm, zm, Bmag, cmap='RdBu', vmin=0, vmax=2)
plot = topo(config)
#plot.sol()  # scrape-off layer
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.plasma()
plot.coil_loc()
pl.axis('equal')
pl.xlim(rlim)
pl.ylim(zlim)





