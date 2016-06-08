import numpy as np
from eqConfig import Config
from radial_build import RB
from scipy.interpolate import interp1d as interp1

class GEOM(object):
    
    def __init__(self,sf,inv,config='SN',Ncoil=5):
        self.inv = inv
        
        conf = Config(config)
        conf.TF(sf)
        rb = RB(conf,sf,Np=150)
        rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
        rb.TFfill()
        
        R,Z = rb.offset(rb.R,rb.Z,1)
        L = rb.length(R,Z)
        Ltrim = 0.14
        Lt = np.linspace(Ltrim,1-Ltrim,int((1-2*Ltrim)*len(L)))
        
        R,Z = interp1(L,R)(Lt),interp1(L,Z)(Lt)
        L = np.linspace(0,1,len(R))
        
        dL = 1/Ncoil
        self.Lo = np.linspace(dL/2,1-dL/2,Ncoil)
        self.Ritp,self.Zitp = interp1(L,R),interp1(L,Z)
        Ro,Zo = self.Ritp(self.Lo),self.Zitp(self.Lo)

        for r,z in zip(Ro,Zo):
            inv.add_coil(r,z,1,1)
            
    def move_coil(self,Lc):
        for name,L in zip(self.inv.coil['active'].keys(),Lc):
            r,z = self.Ritp(L),self.Zitp(L)
            self.inv.move_coil(int(name.split('Coil')[-1]),point=(r,z))
    