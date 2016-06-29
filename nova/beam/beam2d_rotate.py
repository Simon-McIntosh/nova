import numpy as np
import pylab as pl
from finite_element import FE
from nova import coils

tf = coils.TF()

r,z = tf.drawTF([6.42769145,1.26028077,4.51906176,4.53712734,-2.22330594])[:2]

'''
L = 1
Nel = 20
X = np.zeros((Nel+1,3))
X[:,0] = np.linspace(0,L,Nel+1)
X[:,1] = np.linspace(0,L/3,Nel+1)
'''
X = np.zeros((len(r),3))
X[:,0] = r
X[:,1] = z

fe = FE(X,frame='3D')
fe.addBC('fix',[0,fe.nel])  # fix left hand node
fe.add_nodal_load('v',fe.nel/2,0.1)
fe.add_nodal_load('u',fe.nel/2,0.1)
fe.add_nodal_load('w',fe.nel/2,0.1)
fe.solve()
fe.plot()


#print('2D',fe.L[-2]*L**3/(3*fe.E*fe.Iz),fe.D['y'][-1])

