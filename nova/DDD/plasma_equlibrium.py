import numpy as np
from cross_coil import Bcoil, mu_o, loop, plasma_coils
import pylab as pl
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

fig = pl.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
plot = topo(config)
plot.sol()  # scrape-off layer
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.plasma()
plot.coil_loc()

config = 'DD'
swing = 1

path = './coil_data/'
file = 'Double_Decker_Coils'

eqlib = eq(config)
eqlib.swing(swing)
coil = eqlib.coil

I = 16e6  # plasma current [A]
delta = 0.2  # node spaceing [m]
plasma = loop(path,file,'plasma')
plasma_coils(plasma, I, delta, coil)
   
name = 'plasma'
dt = 1e-10
Pr,Pz,Fr,Fz = [],[],[],[]     
for i in range(50):
    B = Bcoil(mu_o, coil, [coil[name]['r'],0,coil[name]['z']])
    F = 2*np.pi*coil[name]['r']*np.cross(B,[0,coil[name]['I'],0])
    #Fmag = sum(coil[name]['Fv']**2)**0.5
    coil[name]['r'] += dt*F[0]
    coil[name]['z'] += dt*F[2]
    print(coil[name]['r'], coil[name]['z'], F)
    




pl.axis('equal')
#pl.ylim([-10, -3])
pl.grid()
pl.tight_layout()
pl.savefig(fig_path+'force_vector_swing_'+str(swing)+'.png', dpi=300)