import numpy as np
import pylab as pl
import pickle
from radial_build import RB
from feild_calc import solCalc

TFopp = False # optimise TF coil placment
config = 'DD1'  # DD1/DD3

def vol_calc(R,Z):
    dR = np.diff(R)
    dZ = np.diff(Z)
    V = 0
    for r,dr,dz in zip(R[:-1],dR,dZ):
        V += (r+dr/2)**2*dz
    V *= np.pi
    return V

with open('./plot_data/'+config+'_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)

sol = solCalc(sf,geom,config)
rb = RB(geom,sf,config,Np=400)
#rb.internal_coils(['P6','P5','PS2','P6B','P5B','PS2','PS5'])

rb.TFopp(TFopp)
Rtf,Ztf,L = rb.drawTF(rb.xCoil,Nspace=300)
Rtf,Ztf = Rtf[::-1],Ztf[::-1]

imax = np.argmax(Ztf)
imin = np.argmin(Ztf)
Htf = np.max(Ztf)-np.min(Ztf)

pl.figure(figsize=(14,10))

R,Z = Rtf[:imax+1],Ztf[:imax+1]
Vin = 2*vol_calc(R,Z)
pl.plot(R,Z,'k-')

R,Z = Rtf[imax:imin+1][::-1],Ztf[imax:imin+1][::-1]
Vout = vol_calc(R,Z)
pl.plot(R,Z,'b--')

Vtf = Vout-Vin

Rp,Zp = sf.Rsol[0][:],sf.Zsol[0][:]
imax = np.argmax(Zp)
imin = np.argmin(Zp)

R,Z = Rp[imax:][::-1],Zp[imax:][::-1]
print(Z[0])
Vin = vol_calc(R,Z)
pl.plot(R,Z,'m-')

R,Z = Rp[:imax+1],Zp[:imax+1]
print(Z[0])
Vout = vol_calc(R,Z)
Vp = Vout-Vin
pl.plot(R,Z,'r--')

print(Vtf,Vp,Vp/Vtf,Htf)

