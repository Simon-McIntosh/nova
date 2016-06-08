import pylab as pl
from config import layout
from extract import extract_geom
from streamfunction import SF
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 0.4  # node spaceing [m]
geom.plasma_coils(I, delta)

#sf = SF(geom.coil, res=0.2, rlim=[0.5,18], zlim=[-10,4.5])
#sf.grid(res=0.3, rlim=[2,16], zlim=[-13,10])

pl.figure(figsize=(8, 14))
plot = layout(geom)
#plot.sol()  # scrape-off layer
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.FW()

plot.plasma()
plot.plasma_coils()
plot.coil_fill()
plot.coil_label()
plot.coil_sheild()

#sf.sol()
#sf.contour()
#plot.P6_support(coil)    
       
pl.axis('equal')
pl.ylim([-13,11])
pl.tight_layout()
pl.axis('off')
#pl.savefig(fig_path+'DD3_coil_sizing_plasma.png', dpi=400)
