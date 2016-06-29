import numpy as np
import pylab as pl
from finite_element import FE
from nova import coils

tf = coils.TF()

xCoil = [6.42769145,1.26028077,4.51906176,4.53712734,-2.22330594]
r,z = tf.drawTF(xCoil,Nspace=100)[:2]

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

fe = FE(frame='3D')
fe.grid(X)
fe.initalise()

fe.addBC('fix',[0,fe.nel])  # fix left hand node

fe.add_force(0.5,[0,-1e-2,0]) 
fe.add_force(0.2,[-1e-2,-1e-2,1e-2])  # l, [Fx,Fy,Fz], global
fe.add_force(0.9,[3e-2,1e-2,-1e-2])  # l, [Fx,Fy,Fz], global
fe.add_force(0.1,[0,1e-2,1e-2])  # l, [Fx,Fy,Fz], global

fe.solve()
fe.interpolate()

fe.plot_nodes()
pl.plot(fe.shape['U'][:,0],fe.shape['U'][:,1])

pl.axis('equal')

pl.figure()
pl.plot(fe.shape['x'],fe.shape['d2u'][:,1],'-')
pl.plot(fe.shape['x'],fe.shape['d2u'][:,2],'-')


