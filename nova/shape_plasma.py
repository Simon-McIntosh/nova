import pylab as pl
import numpy as np
from config import layout
from extract import extract_geom
from streamfunction import SF
from cross_coil import Fcoil
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 0.4  # node spaceing [m]
geom.plasma_coils(I, delta)


fig = pl.figure(figsize=(12, 6))
plot = layout(geom)
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.FW()
plot.plasma()
plot.coil_fill()
plot.coil_sheild()

sf = SF(geom.coil, res=0.2, rlim=[4,14], zlim=[-10,4.5])
sf.sol()
sf.contour()
pl.axis('equal')
pl.ylim([-8, -5])
pl.tight_layout()