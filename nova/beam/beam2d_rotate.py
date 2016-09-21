import numpy as np
import pylab as pl
from finite_element import FE
from nova import coils
from nova.beam import Dcoil
from amigo.addtext import linelabel
from time import time

from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF
from nova.inverse import INV
from nova.coils import TF
from nova.coil_cage import coil_cage
import nova.cross_coil as cc
from amigo import geom

import seaborn as sns
rc = {'figure.figsize':[8*12/16,8],'savefig.dpi':120, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

config = 'DEMO_SN'

setup = Setup(config,eqdir='../../eqdsk/')
sf = SF(setup.filename)
tf = TF(config,coil_type='D')
tf.load(nTF=18,objective='L')

rb = RB(setup,sf)
pf = PF(sf.eqdsk)
eq = EQ(sf,pf,boundary=tf.get_loop(expand=0.5),n=5e3)  
eq.get_plasma_coil()
#inv = INV(sf,pf,eq)

pf.plot(coils=eq.coil,label=False,plasma=True,current=False) 
#inv.plot_coils()
sf.contour()

to = time()


 
tf.split_loop()
#tf.fill()
    

fe = FE(frame='3D')
fe.add_mat(0,E=1e2,I=1e2,A=1,G=5,J=5,rho=5e-2)

nodes = {}
for part in ['loop','nose']:  # ,'nose'
    x,y = tf.x[part]['r'],tf.x[part]['z']
    if part == 'nose':
        x = np.min(x)*np.ones(len(x))
    X = np.zeros((len(x),3))
    X[:,0],X[:,1] = x,y
    fe.add_nodes(X)
    nodes[part] = np.arange(fe.nndo,fe.nnd)
    
n = np.append(np.append(nodes['nose'][-1],nodes['loop']),nodes['nose'][0])
fe.add_elements(n=n,part_name='loop')  
fe.add_elements(n=nodes['nose'],part_name='nose')  
 
#fe.add_nodes([13,-12,-2])
fe.add_nodes([13,-12,0])
fe.add_elements(n=[fe.part['loop']['el'][20],fe.nndo],
                part_name='support')
     

fe.freeze()

fe.addBC(['u','w','rx','ry','rz'],'all',part='nose')
fe.addBC(['fix'],[-1],part='support') 


#fe.add_weight()  # add weight to all elements

cage = coil_cage(nTF=18,rc=tf.rc,plasma={'config':config},
                 coil={'cl':tf.x['cl']})


Rp = geom.rotate(np.pi/2,'x')
Rm = geom.rotate(-np.pi/2,'x')

i = np.argmax(tf.x['cl']['r'])
ro,zo = tf.x['cl']['r'][i],tf.x['cl']['z'][i]
bm = -ro*cage.point((ro,0,zo),variable='feild')[1]  # TF feild

for part in ['loop','nose']:
    for el in fe.part[part]['el']:
        n = fe.el['n'][el]  # node index pair
        point = np.zeros(3)
        for i in range(3):  # calculate load at element mid-point
            point[i] = np.mean(fe.X[n,i])   
        b = np.zeros(3)
        b[2] = bm/point[0]  # TF feild (fast version)
        #b = np.dot(Rm,cage.point(np.dot(Rp,point),variable='feild'))  # TF
        b[:2] += sf.Bpoint((point[:2]))  # PF feild (sf, fast version)
        w = np.cross(fe.el['dx'][el],b)
        fe.add_load(el=el,W=w)  # bursting/toppling load
        
print(w[2])

fe.plot_nodes()


fe.solve()
print('time {:1.3f}'.format(time()-to))


fe.plot_F(scale=5e-1)

fe.plot_displacment()
pl.axis('off')

fe.plot_twin()

fe.plot_curvature()

