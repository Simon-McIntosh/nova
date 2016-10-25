import numpy as np
from DEMOxlsx import DEMO
from nova.coils import TF
from amigo import geom
import pylab as pl
import seaborn as sns

tf = TF('DEMO_SN',coil_type='S',nTF=16,objective='L',npoints=5000)
x = tf.coil.draw()
r,z = x['r'],x['z']
l = geom.length(r,z)
dl = np.mean(np.diff(l))

dr = np.gradient(r,dl)
dz = np.gradient(z,dl)
ddr = np.gradient(dr,dl)
ddz = np.gradient(dz,dl)

Rv,Zv = tf.coil.verticies()[:2]

index = np.zeros(len(Rv),dtype=int)
for i,(rv,zv) in enumerate(zip(Rv,Zv)):
    index[i] = np.argmin((r-rv)**2+(z-zv)**2)
    

tf.coil.plot()
    
pl.figure(figsize=(10,7))
k = (dr*ddz-dz*ddr)/(dr**2+dz**2)**(3/2)
pl.plot(l,k)
pl.plot(l[index],k[index],'s')
pl.xlabel('normalised length (anit-clockwise from inboard midplane)')
pl.ylabel(r'coil curvature, $\kappa$')
sns.despine()

