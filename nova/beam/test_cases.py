import numpy as np
import pylab as pl
from finite_element import FE
import seaborn as sns
rc = {'figure.figsize':[6,6*12/16],'savefig.dpi':140, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
        

L,F,alpha = 1,150,0*np.pi/180
Nel = 12
X = np.zeros((Nel+1,3))
X[:,0] = np.linspace(0,L,Nel+1)
R = np.array([[np.cos(alpha),-np.sin(alpha),0],
               [np.sin(alpha),np.cos(alpha), 0],
               [0,            0,             1]])
X = np.dot(X,R)
fe = FE(frame='3D')

fe.add_mat(0,E=1,I=1,A=1,G=1,J=1,rho=1)


fe.add_nodes(X)
fe.add_elements(part_name='beam')

'''
fe.add_nodes([0.1,-0.1,0])
fe.add_elements(n=[fe.part['beam']['el'][-12],fe.nndo],part_name='strut')
'''

fe.initalise()


fe.addBC('pin',[0,-1],part='beam')  
#fe.addBC('fix',[-1],part='strut')  


#fe.add_load(part='beam',L=0.21,F=[0,F,0])  # F,f==global,local


#fe.add_load(el=12,W=[0,F,0])  # F,f==global,local

fe.add_weight()  # add weight to all elements

fe.solve()
fe.plot_nodes()


fe.interpolate()

for part in fe.part:
    pl.plot(fe.part[part]['U'][:,0],fe.part[part]['U'][:,1])

pl.axis('equal')



pl.figure()
for part in fe.part:
    pl.plot(fe.part[part]['l'],fe.part[part]['d2u'][:,1])

print(fe.part[part]['U'][:,1].min(),
      5*9.81*fe.mat['rho'][0]*fe.mat['A'][0]*\
      fe.part['beam']['L'][-1]**4/(384*fe.mat['E'][0]*fe.mat['Iz'][0]))