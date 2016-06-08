import iapws
from iapws import IAPWS97
import numpy as np
import pylab as pl

#water properties
#sat_f = IAPWS97(P=self.P/10, x=0)

Pc = iapws.Pc

Pm = np.linspace(0.01,Pc,100)
T = np.array([])

for P in Pm:
    sat = IAPWS97(P=P, x=0)
    T = np.append(T,sat.T)

    

pl.figure(figsize=(9, 6))
pl.plot(T-273.15, Pm)
pl.grid()
pl.xlabel('Temparture, T [C]')
pl.ylabel('Pressure, P [MPa]')
pl.savefig('../Figs/saturation_curve.png', dpi=400)

'''
pl.figure(figsize=(12, 8))
pl.plot(hf, Pm, hg, Pm)
 
Pm = np.linspace(0.1,2*Pc,100)   
Tm = np.linspace(300,1000,50)
for T in Tm:
    hstate = []
    for P in Pm:
        state = IAPWS97(T=T, P=P)
        hstate.append(state.h)
    pl.plot(hstate, Pm)
'''    
