import pylab as pl
import numpy as np
from config import layout
from extract import extract_geom
from streamfunction import SF
from cross_coil import Fcoil
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 0.4  # node spaceing [m]
geom.plasma_coils(I, delta)

Cmove = 'P6'
Pz = geom.coil[Cmove]['z']
Xmove = [1,10,100]
Nc = 5
cm = pl.get_cmap('cool', Nc)
plot = layout(geom)
sf = SF(geom.coil, res=0.2, rlim=[5,14], zlim=[-10,1])  # res=0.2
    
for dx in Xmove:
    
    fig = pl.figure(figsize=(4,4))
    #plot.sol()  # scrape-off layer
    plot.plates()  # divertor plates
    plot.TF()  # TF coil
    plot.FW()
    #plot.plasma()
    plot.coil_fill()
    plot.coil_label(['P6','P5'])
    plot.coil_sheild()
    sf.sol(color='k')
    
    Pz_delta = np.linspace(-dx*1e-3, dx*1e-3, Nc)
    for i,dz in enumerate(Pz_delta):
        print(i)
        geom.coil[Cmove]['z'] = Pz+dz
        sf.potential()
        sf.sol(color=[cm(i)])
    
    #sf.contour()
    pl.axis('equal')
    pl.xlim([6,13])
    pl.ylim([-10,-5])
    pl.xlabel('radial coordinate, r [m]')
    pl.ylabel('vertical coordinate, z [m]')
    pl.tight_layout()
    pl.savefig(fig_path+Cmove+'move_'+str(dx)+'mm.png', dpi=300)