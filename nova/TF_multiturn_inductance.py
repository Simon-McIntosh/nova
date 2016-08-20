import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.coils import PF,TF
from etna.coil_geom import configure,coil_cage
from etna.coil_apdl import coil_map
from inductance import neumann
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from time import time
import sys
import datetime
import pickle

import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

cgeom = configure('TF',Ndp=0,Nloop=0)
cmap = coil_map(cgeom)
cgeom.segment(cmap.loops,Nseg=100)

nTF = 18
theta = np.linspace(0,2*np.pi,nTF,endpoint=False)
 
r_coil = cgeom.pn*np.sqrt(134/np.pi)
r_strand = cgeom.r_cond
#r_strand = np.sqrt(235.3e-6/np.pi)
cage = coil_cage(cgeom.profile,cmap.loops,theta,r_coil,r_strand)
neu = neumann()
    
fig = pl.figure()
ax = fig.gca(projection='3d')
N = cmap.Ncond+len(theta)  
#N = 18
M = np.zeros((N,N))
to = time()
for i in np.arange(0,N):
    X,nturns,r = cage.load(i)
    neu.setX(X)
    neu.setr(r)
    ax.plot(np.append(X[:,0],X[0,0]),np.append(X[:,1],X[0,1]),
            np.append(X[:,2],X[0,2]))

    for j in np.arange(i,N):
        X_,nturns_ = cage.load(j)[:2]
        neu.setX_(X_)
        M[i,j] = nturns*nturns_*neu.calculate()
        if i != j:
            M[j,i] = M[i,j]  # copy to lower

    t1 = time()
    t_fraction = 2*((i+1)*(N-(i+1)/2))/N**2
    t_wall = t1-to
    t_total = t_wall/t_fraction
    t_remain = t_total-t_wall
    t_str = '\rtotal: '+str(datetime.timedelta(seconds=np.round(t_total)))+'\t'
    t_str += 'complete:{:3.0f}%\t'.format(100*t_fraction)
    t_str += 'remaining: '+str(datetime.timedelta(seconds=np.round(t_remain)))
    sys.stdout.write(t_str)
    sys.stdout.flush()

filename = 'TFtmp_{:1.0f}_{:1.0f}'.format(N,cgeom.Nseg)
#with open('D:/Code/Etna/Data/Inductance/'+filename+'.pkl', 'wb') as output:
#    pickle.dump(M,output,-1)


ax.auto_scale_xyz([-10,10],[-10,10],[-15,5])   
ax.axis('off')

np.set_printoptions(threshold=100,edgeitems=5,precision=3,linewidth=150)
print('')
print(M)
print(np.sum(M))




