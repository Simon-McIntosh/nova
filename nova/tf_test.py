import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.coils import PF,TF
from etna.coil_geom import configure
from etna.coil_apdl import coil_map
import numpy as np
from mpl_toolkits.mplot3d import Axes3D





cmap = coil_map(configure('TF',Ndp=3,Nloop=1))
cmap.wrap()


import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

setup = Setup('SXex')
sf = SF(setup.filename)
rb = RB(setup,sf)


pf = PF(sf.eqdsk)

tf = TF()

xCoil = [ 5.63566835,  2.5318409 ,  4.4570691 ,  4.52603387, -2.61044312]
x,z,l = tf.drawTF(xCoil)
y = np.zeros(np.shape(x))

Xo = np.zeros((len(x)-1,3))
Xo[:,0] = x[:-1]+np.diff(x)/2
Xo[:,2] = z[:-1]+np.diff(z)/2

def dloop(X):
    dX = np.diff(np.append(X,X[0,:,None].T,axis=0),axis=0)
    return dX
    
def rotate(theta):
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
                   [np.sin(theta),-np.cos(theta),0],
                   [0,0,1]])
    return Rz
    
def inductance(X,X_):
    dX,dX_,L = dloop(X),dloop(X_),0
    for i in range(len(X)):
        for j in range(len(X_)):
            if np.linalg.norm(X[i,:]-X_[j,:]) != 0:
                L += np.dot(dX[i,:],dX_[j,:])/np.linalg.norm(X[i,:]-X_[j,:])
            else:
                L += np.linalg.norm(dX[i,:])/2
    L *= mu_o/(4*np.pi)   
    return(L)
   
nTF = 18
theta = np.linspace(0,2*np.pi,nTF,endpoint=False)
theta =theta[:int(np.floor(nTF/2+1))]
  
fig = pl.figure()
ax = fig.gca(projection='3d')
ax.plot(Xo[:,0],Xo[:,1],Xo[:,2])

for t in theta:
    Rz = rotate(t)
    X = np.dot(Xo,Rz)
    print(t,134**2*inductance(X,Xo))
    ax.plot(X[:,0],X[:,1],X[:,2])


ax.auto_scale_xyz([-10,10],[-10,10],[-15,5])   
'''



nTF = 18
theta = np.linspace(0,2*np.pi,nTF)
for t in theta:
    
    

#ax.plot(x,y,z)


ax.axis('off')

#pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 

#sns.despine()
#tf.TFcoil(False)
'''