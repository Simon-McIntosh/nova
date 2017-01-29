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
fe = FE(frame='3D')
fe.add_mat(0,E=1e0,I=1e0,A=1,G=5,J=1,rho=5e-2)

fe.add_nodes([0,0,0])
fe.add_nodes([1,0,0])
fe.add_nodes([0,-1,0])

 
fe.add_elements(n=[0,1],part_name='s1')
fe.d(['fix'],[0],part='s1',ends=0) 

fe.add_elements(n=[0,2],part_name='s2')
fe.d(['fix'],[0],part='s2',ends=0) 

fe.d(['u'],[0]) 

#fe.add_elements(n=[4,5],part_name='s3')
#fe.d(['fix'],[0],part='s3',ends=1) 


fe.add_nodal_load(1,'fy',1)
fe.add_nodal_load(2,'fz',-2)
#fe.add_weight()  # add weight to all elements
#fe.add_tf_load(config,tf,sf.Bpoint,method='function')  # burst and topple

#fe.cp.add([1,2],dof=['u','v'])  # couple nodes

          
fe.cp.add([1,2],dof='fix',theta=-np.pi/2)


fe.solve()



print('time {:1.3f}'.format(time()-to))

fe.plot_nodes()
fe.plot_F(scale=5e-1)

fe.plot_displacment()
pl.axis('off')


#fe.plot_twin()
fe.plot_curvature()

