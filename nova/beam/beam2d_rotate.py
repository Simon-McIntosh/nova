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

import seaborn as sns
rc = {'figure.figsize':[8*12/16,8],'savefig.dpi':120, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

setup = Setup('DEMO_SN',eqdir='../../eqdsk/')
sf = SF(setup.filename)

rb = RB(setup,sf)
pf = PF(sf.eqdsk)
eq = EQ(sf,pf,limit=[5,14,-7,7],n=5e3)  

eq.get_plasma_coil()
inv = INV(sf,pf,eq)

pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
#inv.plot_coils()
sf.contour()

to = time()

config = 'DEMO_SN'
tf = TF(config,coil_type='S',npoints=12)


tf.load(nTF=18,objective='L')

    


r,z = tf.x['cl']['r'][:-1],tf.x['cl']['z'][:-1]


X = np.zeros((len(r),3))
X[:,0] = r
X[:,1] = z

fe = FE(frame='3D')
fe.add_mat(0,E=1,I=1,A=1,G=1,J=1,rho=1e-4)

fe.add_nodes(X)
fe.add_elements(part_name='outerD')  # outer d coil


fe.add_elements(n=[fe.part['outerD']['el'][-2],fe.part['outerD']['el'][2]],
                part_name='innerD')  # straight segment



fe.add_nodes([13,-12,0])
fe.add_elements(n=[fe.part['outerD']['el'][5],fe.nndo],part_name='support')


fe.freeze()

fe.addBC(['fix'],[0,-1],part='outerD')  

#fe.addBC(['fix'],[-1],part='support') 


#fe.add_weight()  # add weight to all elements

bm = sf.bcentr*sf.rcentr
for el in fe.part['outerD']['el']:
    nm = fe.el['mat'][el]  # material index
    n = fe.el['n'][el]
    r = np.mean(fe.X[n,0])
    b = np.array([0,0,bm/r])
    w = -2e-3*np.cross(fe.el['dx'][el],b)
    fe.add_load(el=el,W=w)  # self weight


fe.plot_nodes()

fe.solve()
t1 = time()

print('time {:1.3f}'.format(t1-to))


#fe.plot_F(scale=1e2)


color = sns.color_palette('Set2',6)
for i,part in enumerate(fe.part):
    pl.plot(fe.part[part]['U'][:,0],fe.part[part]['U'][:,1],color=color[i+1])
pl.axis('equal')


pl.figure(figsize=([4,3*12/16]))
text = linelabel(value='',postfix='',Ndiv=5) 
color = sns.color_palette('Set2',6)
for i,part in enumerate(fe.part):
    pl.plot(fe.part[part]['l'],fe.part[part]['d2u'][:,1],color=color[i+1])
    text.add(part)
text.plot()
sns.despine()
pl.xlabel('part length')
pl.ylabel('part curvature')
