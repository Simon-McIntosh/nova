import numpy as np
import pylab as pl
from finite_element import FE
import seaborn as sns
rc = {'figure.figsize':[6,6*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
        

L,F,alpha = 1,40,0*np.pi/180
Nel = 30
X = np.zeros((Nel+1,3))
X[:,0] = np.linspace(0,L,Nel+1)
R = np.array([[np.cos(alpha),-np.sin(alpha),0],
               [np.sin(alpha),np.cos(alpha), 0],
               [0,            0,             1]])
X = np.dot(X,R)
fe = FE(frame='3D')

fe.add_mat(E=1,I=1,A=1,G=1,J=1)


fe.add_nodes(X)
fe.add_elements()
fe.grid()


fe.add_nodes([0.5,-0.3,0])
fe.add_elements(n=[fe.nndo-12,fe.nndo])



fe.plot_nodes()




fe.initalise()
fe.addBC('fix',[0,fe.nel-1])  # fix left hand node

fe.addBC('v',[fe.nel])
fe.addBC('u',[fe.nel])

fe.add_force(0.5,[0,F,0])  # l, [Fx,Fy,Fz], global

fe.solve()
fe.plot_nodes()


fe.interpolate()


pl.plot(fe.shape['U'][:,0],fe.shape['U'][:,1])

pl.axis('equal')

print(fe.shape['u'][:,1].max())


'''
pl.figure()
pl.plot(fe.shape['x'],fe.shape['u'][:,1])
pl.plot(fe.shape['x'],fe.shape['u'][:,2])

pl.figure()
pl.plot(fe.shape['x'],fe.shape['d2u'][:,1])
pl.plot(fe.shape['x'],fe.shape['d2u'][:,2])


print(F*fe.L[-1]/(fe.mat['A'][0]*fe.mat['E'][0]),fe.shape['u'][:,0].max())

'''