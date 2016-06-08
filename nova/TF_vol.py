import pylab as pl
import numpy as np
from radial_build import RB
from streamfunction import SF

filename = './eqlib_data/Super-X_equilibrium_2MC43F_v1_0.eqdsk'

def vol_calc(R,Z):
    dR = np.diff(R)
    dZ = np.diff(Z)
    V = 0
    for r,dr,dz in zip(R[:-1],dR,dZ):
        V += np.abs((r+dr/2)**2*dz)
    V *= np.pi
    return V
    
    
sf = SF(filename)
rb = RB(sf,config='SX',dr=0.15,Np=200)

xCoilo = [3.5,0.6,1.2,3.5,-0.5]

R,Z,L = rb.drawTF(xCoilo, Nspace=30) 

def loop_vol(R,Z):
    imin,imax = np.argmin(Z),np.argmax(Z)
    Rin = np.append(R[::-1][:imin+1][::-1],R[:imin+1])
    Zin = np.append(Z[::-1][:imin+1][::-1],Z[:imin+1])
    Rout = R[imin:imax+1]
    Zout = Z[imin:imax+1]
    return vol_calc(Rout,Zout)-vol_calc(Rin,Zin)

print(loop_vol(R,Z))