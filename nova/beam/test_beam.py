import numpy as np
import pylab as pl
from nova.finite_element import FE
from time import time


to = time() 
fe = FE(frame='1D')
fe.add_mat(0,E=1e0,I=1e0,A=1,G=5,J=1,rho=5e-2)

fe.add_nodes([0,0,0])
fe.add_nodes([2.5,0,0])
fe.add_nodes([3.5,0,0])
fe.add_nodes([5,0,0])
 
fe.add_elements(n=[0,1],part_name='s1')
fe.addBC(['fix'],[0],part='s1',ends=0) 

fe.add_elements(n=[2,3],part_name='s2')
fe.addBC(['fix'],[0],part='s2',ends=1) 

#fe.addBC(['fix'],[-1],part='support') 

fe.add_weight()  # add weight to all elements
#fe.add_tf_load(config,tf,sf.Bpoint,method='function')  # burst and topple

fe.solve()



print('time {:1.3f}'.format(time()-to))

fe.plot_nodes()
fe.plot_F(scale=5e-1)

fe.plot_displacment()
pl.axis('off')


#fe.plot_twin()
fe.plot_curvature()

