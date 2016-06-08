import pickle
import pylab as pl
import numpy as np
from feild_calc import solCalc


with open('./plot_data/DD3_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)
    
sol = solCalc(sf,geom)

fig = pl.figure(figsize=(14,12))
pl.axis('equal')
pl.grid()
pl.xlim([4.5,13.5])
pl.ylim([-10,-5])
#plot.sol()  # scrape-off layer
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.FW()
plot.plasma()
plot.coil_fill()
#plot.coil_current([0,20],[-11,-2])
plot.coil_sheild()
#sf.sol()

p3arc = [[10,-9.53],[12,-7.8],[13,-5.8]]  # three-point strike arc
sol.strike_arc(p3arc,0.25,plot=True)    
    
for leg in ['inner','outer']:
    for layer_index in range(10):
        sol.legs(leg,layer_index,plot=True) 






  
'''

pl.plot(Rsol,Zsol,'c-x')




fig = pl.figure(figsize=(14,10))
pl.plot(L2D,Xi)

graze = 1.5*np.pi/180


'''




