import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.coils import PF,TF
from etna.coil_geom import configure
from etna.coil_apdl import coil_map
from inductance import rotate,induce
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

cgeom = configure('TF',Ndp=0,Nloop=0)
cmap = coil_map(cgeom)
cmap.wrap()

#for offset,shift in zip(cmap.loops['x'],cmap.loops['y']):
#    pl.plot(offset,shift,'o')
'''
X = cgeom.loop(np.mean(cmap.loops['x']),0)  # 
X_ = np.dot(X,rotate(np.pi/4))
X_ = np.copy(X)
L = induce(X,X_,0.2)  # cgeom.r_cond
print(134**2*L)


fig = pl.figure()
ax = fig.gca(projection='3d')
pl.plot(X[:,0],X[:,1],X[:,2])  
'''
  
'''
fig = pl.figure()
ax = fig.gca(projection='3d')
M = np.zeros((cmap.Ncond,cmap.Ncond))
for i in range(cmap.Ncond):
    print(i)
    X = cgeom.loop(cmap.loops['x'][i],cmap.loops['y'][i])
    ax.plot(X[:,0],X[:,1],X[:,2])
    for j in np.arange(i,cmap.Ncond):
        X_ = cgeom.loop(cmap.loops['x'][j],cmap.loops['y'][j])
        M[i,j] = induce(X,X_,0.22978)
        #M[j,i] = M[i,j]  # copy to lower

print(M)
print(np.sum(M)*134**2)
ax.auto_scale_xyz([2,11],[-7.5,7.5],[-7.5,7.5]) 

    #for offset,shift in zip(cmap.loops['x'],cmap.loops['y']):
        #x,y,z = cgeom.loop()
'''


'''
mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

setup = Setup('SXex')
sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)
tf = TF()

xCoil = [ 5.63566835,  2.5318409 ,  4.4570691 ,  4.52603387, -2.61044312]
x,z,l = tf.drawTF(xCoil)
'''

'''
Xo = np.zeros((len(x)-1,3))
Xo[:,0] = x[:-1]+np.diff(x)/2
Xo[:,2] = z[:-1]+np.diff(z)/2


fig = pl.figure()
ax = fig.gca(projection='3d')
'''

X = cgeom.loop(np.mean(cmap.loops['x']),0) 

nTF = 18
theta = np.linspace(0,2*np.pi,nTF,endpoint=False)
theta =theta[:int(np.floor(nTF/2+1))]
  
fig = pl.figure()
ax = fig.gca(projection='3d')
ax.plot(X[:,0],X[:,1],X[:,2])

for t in theta:
    Rz = rotate(t)
    X_ = np.dot(X,Rz)
    print('{:1.3f}'.format(134**2*induce(X,X_,cgeom.pn*np.sqrt(134)/np.pi)))
    ax.plot(X_[:,0],X_[:,1],X_[:,2])


ax.auto_scale_xyz([-10,10],[-10,10],[-15,5])   


