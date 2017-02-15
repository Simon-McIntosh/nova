import numpy as np
import pylab as pl
from nova.finite_element import FE
from time import time
import seaborn as sns

rc = {'figure.figsize':[4*16/12,4],'savefig.dpi':120, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

to = time() 
fe = FE(frame='2D')
fe.add_mat(0,E=1e0,I=1e0,A=1,G=5,J=1,rho=5e-2)

fe.add_nodes([0,0,0])

R = 2
nTF = 16

for i,theta in enumerate(np.linspace(0,2*np.pi,nTF,endpoint=False)):
    fe.add_nodes([R*np.cos(theta),R*np.sin(theta),0])
    fe.add_elements(n=[0,i+1],part_name='s{:d}'.format(i))
    fe.add_bc(['fix'],[0],part='s{:d}'.format(i),ends=0) 
    if i>0:
        fe.add_cp([1,i+1],dof='fix',rotate=True)



#fe.d(['u'],[1]) 

#fe.add_elements(n=[4,5],part_name='s3')
#fe.d(['fix'],[0],part='s3',ends=1) 


fe.add_nodal_load(1,'fy',0.5)
fe.add_nodal_load(1,'fx',2)

#fe.add_weight()  # add weight to all elements
#fe.add_tf_load(config,tf,sf.Bpoint,method='function')  # burst and topple

#fe.cp.add([1,2],dof=['u','v'])  # couple nodes

          
#fe.add_cp([1,2],dof='fix',rotate=True)
#fe.add_cp([1,3],dof='fix',rotate=True)

fe.solve()



print('time {:1.3f}'.format(time()-to))

fe.plot_nodes()
fe.plot_F(scale=5e-1)

fe.plot_displacment()
pl.axis('off')


#fe.plot_twin()
#fe.plot_curvature()

