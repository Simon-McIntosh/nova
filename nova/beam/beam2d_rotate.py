import numpy as np
import pylab as pl
from finite_element import FE


import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

L = 1
Nel = 2
X = np.zeros((Nel+1,3))
X[:,0] = np.linspace(0,L,Nel+1)
#X[:,1] = np.linspace(0,L/30,Nel+1)

fe = FE(X,frame='1D')
fe.addBC('fix',[0])  # fix left hand node
fe.add_nodal_load('v',fe.nel,5)
fe.solve()
fe.plot()
print(fe.K)

print('1D',fe.L[-2]*L**3/(3*fe.E*fe.Iz),fe.D['y'][-1])

fe = FE(X,frame='2D')
fe.addBC('fix',[0])  # fix left hand node
fe.add_nodal_load('v',fe.nel,5)
fe.solve()
fe.plot()
K = np.copy(fe.K)
for i in range(2):
    K = np.delete(K,[0,3],i)
print(K)

print('2D',fe.L[-2]*L**3/(3*fe.E*fe.Iz),fe.D['y'][-1])



           
'''
# [k][v] = [F]       
# [k][v1,t1,v2,t2]' = [P1,M1,P2,M2]'
     
L = 10
     
Nel = 51
dL = L/Nel
n = 4+2*(Nel-1)
k = np.zeros((n,n))

for i in range(Nel):
    k[2*i:2*i+4,2*i:2*i+4] += elem_k(dL)
    
k = k[2:,2:] # fix end


F = np.zeros((n-2,1))  # loading vector
P = [[0.3,-0.1],[1,-0.95],[0.5,1.5]]
for p in P:
    n = int(np.ceil(p[0]*Nel)*2)-1
    F[n] = p[1]

vs = np.linalg.lstsq(k,F)[0]*dL**3/EI
Fo = np.dot(k,np.reshape(vs,(len(vs),1)))

pl.plot(vs[::2],'o-')
'''