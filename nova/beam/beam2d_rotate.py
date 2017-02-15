import numpy as np
import pylab as pl
from nova.finite_element import FE
from nova import coils
from nova.beam import Dcoil
from amigo.addtext import linelabel
from time import time
from nova.config import Setup,select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF
from nova.inverse import INV
from nova.coils import TF
import nova.cross_coil as cc
from amigo import geom
from nova.loops import Profile
from nova.force import force_feild
import seaborn as sns
rc = {'figure.figsize':[8*12/16,8],'savefig.dpi':120, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

nTF,nPF,nCS = 18,4,3
config = {'TF':'dtt','eq':'SN'}
config,setup = select(config,nTF=nTF,nPF=nPF,nCS=nCS,update=False)
profile = Profile(config['TF'],family='S',part='TF',
                  nTF=nTF,obj='L',load=True,npoints=50)
sf = SF(setup.filename)
tf = TF(profile,sf=sf)
tf.fill()

#rb = RB(setup,sf)
pf = PF(sf.eqdsk)
eq = EQ(sf,pf,boundary=tf.get_loop(expand=0.5),n=1e3,sigma=0)  
eq.get_plasma_coil()
ff = force_feild(pf.index,pf.coil,eq.coil,eq.plasma_coil)
        
#eq.run()
#eq.plotj()
#pf.coil['Coil6']['r'] -= 1.5
#eq.coils()
#eq.gen_opp(sf.Mpoint[1])
#eq.resample()
#eq.plotb(alpha=1)

#inv = INV(sf,pf,eq)

pf.plot(coils=eq.coil,label=False,plasma=True,current=False,alpha=0.5) 
#inv.plot_coils()
sf.contour()


to = time()
tf.split_loop()
    
fe = FE(frame='3D')
fe.add_mat(0,E=5e1,I=8e1,A=0.5,G=5,J=5,rho=5e-2)
fe.add_mat(1,E=1e2,I=8e1,A=0.5,G=5,J=5,rho=5e-2)
fe.add_mat(2,E=1e3,I=8e3,A=0.5,G=50,J=50,rho=5e-2)

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
fe.add_elements(n=n,part_name='loop',nmat=0)
fe.add_elements(n=nodes['nose'],part_name='nose',nmat=0) 
fe.add_bc('nv','all',part='nose') 


nd_GS = fe.el['n'][fe.part['loop']['el'][10]][0]  # gravity support connect
 
fe.add_nodes([13,-12,-1])
fe.add_nodes([13,-12,1])
fe.add_nodes(fe.X[nd_GS])

fe.add_elements(n=[fe.nndo-2,fe.nndo,fe.nndo-1],part_name='support',nmat=1)
fe.add_bc('nz',[0],part='support',ends=0) 
fe.add_bc('nz',[-1],part='support',ends=1) 

fe.add_cp([fe.nndo,nd_GS],dof='fix')  #'nrx'

nd_OISo = fe.el['n'][fe.part['loop']['el'][15]][0]  # OIS
nd_OIS1 = fe.el['n'][fe.part['loop']['el'][22]][0]

'''
fe.add_nodes(np.dot(fe.X[nd_OISo],geom.rotate(np.pi/config['nTF'],axis='y')))
fe.add_nodes(np.dot(fe.X[nd_OISo],geom.rotate(-np.pi/config['nTF'],axis='y')))
fe.add_elements(n=[fe.nndo-1,nd_OISo,fe.nndo],part_name='OISo',nmat=2)
fe.add_cp([fe.nndo,fe.nndo-1],dof='fix',rotate=True,axis='y')
#fe.add_cp([nd_OISo,nd_OISo-1],dof='rx')
'''
fe.add_weight()  # add weight to all elements
fe.add_tf_load(config['eq'],ff,tf,sf.Bpoint,method='function')  # burst and topple

fe.solve()


fe.deform(scale=0.3)

print('time {:1.3f}'.format(time()-to))

fe.plot_nodes()
fe.plot_F(scale=0.6)

fe.plot_displacment()
pl.axis('off')
pl.tight_layout(rect=[-0.3,0.1,0.9,0.9])

fe.deform(scale=1.2)
fe.plot_3D(pattern=config['nTF'])
#fe.plot_twin()
#fe.plot_curvature()

