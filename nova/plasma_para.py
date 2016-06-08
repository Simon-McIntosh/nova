import pylab as pl
from config import layout
from extract import extract_geom

config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 0.4  # node spaceing [m]
geom.plasma_coils(I, delta)


fig = pl.figure(figsize=(9, 12))
ax = fig.add_subplot(111)
plot = layout(geom)
plot.plasma()
plot.plasma_coils()
plot.sol()  # scrape-off layer 
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.coil_fill()  
plot.coil_sheild() 
plot.coil_label() 
pl.axis('equal')
