import pylab as pl
import pickle
from config import layout
from extract import extract_geom
from streamfunction import SF


config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 20e6  # plasma current [A]

fig = pl.figure(figsize=(10,14))
geom.plasma_coils(I,9)

pl.axis('equal')
#pl.xlim([5,10])
#pl.ylim([-5,0])
plot = layout(geom)

'''
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.FW()
plot.plasma()
plot.coil_fill()
#plot.coil_current([0,20],[-11,-2])
plot.coil_sheild()
plot.plasma_coils()
'''
sf = SF(geom.coil, res=0.2, rlim=[4,14], zlim=[-10,4.5])
sf.sol()
sf.contour()

pl.plot(geom.plasma['LCFS']['r'],geom.plasma['LCFS']['z'],'r')
#plot.sol()  # scrape-off layer

'''
with open('./plot_data/'+config+'_sol_LT.pkl', 'wb') as output:
    pickle.dump(sf,output,-1)
    pickle.dump(plot,output,-1)
    pickle.dump(geom,output,-1)
   
'''