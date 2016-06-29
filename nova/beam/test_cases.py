import numpy as np
import pylab as pl
from finite_element import FE

L,F,alpha = 1,100,0*np.pi/180
Nel = 10
X = np.zeros((Nel+1,3))
X[:,0] = np.linspace(0,L,Nel+1)
R = np.array([[np.cos(alpha),-np.sin(alpha),0],
               [np.sin(alpha),np.cos(alpha), 0],
               [0,            0,             1]])
X = np.dot(X,R)
fe = FE(frame='3D')

fe.grid(X)
fe.initalise()

fe.addBC('fix',[0,fe.nel])  # fix left hand node



#fe.add_force(0.5,[F*np.sin(alpha),F*np.cos(alpha),0])  # l, [Fx,Fy,Fz], global

fe.add_force(0.5,[0.5,50,-1])  # l, [Fx,Fy,Fz], global

fe.solve()
fe.plot_nodes()

fe.interpolate()


pl.plot(fe.shape['U'][:,0],fe.shape['U'][:,1])

pl.axis('equal')



pl.figure()
pl.plot(fe.shape['x'],fe.shape['u'][:,1])
pl.plot(fe.shape['x'],fe.shape['u'][:,2])

pl.figure()
pl.plot(fe.shape['x'],fe.shape['d2u'][:,1])
pl.plot(fe.shape['x'],fe.shape['d2u'][:,2])

print(F*fe.L[-1]**3/192,fe.shape['u'][:,1].max())

