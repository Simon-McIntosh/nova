import cross_coil as cc
import pylab as pl
import numpy as np
mu_o = 4*np.pi*1e-7
import seaborn as sns
rc = {'figure.figsize':[12,12*1/2],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

coil = {}
coil['c1'] = {'r':8,'z':-1,'I':30e6}
coil['c2'] = {'r':10,'z':1,'I':-10e6}
limit = [2,16,-3,3]

coil1 = {}
coil1['c1'] = {'r':9,'z':0,'I':20e6}

        
ro,zo,n = limit[:2],limit[2:],5e3
dro,dzo = (ro[-1]-ro[0]),(zo[-1]-zo[0])
ar = dro/dzo
nz = int(np.sqrt(n/ar))
nr = int(n/nz)
r = np.linspace(ro[0],ro[1],nr)
z = np.linspace(zo[0],zo[1],nz)
r2d,z2d = np.meshgrid(r,z,indexing='ij')
        
        
psi = 0
for c in coil:
    psi += mu_o*coil[c]['I']*cc.green(r2d,z2d,coil[c]['r'],coil[c]['z'])

psi1 = 0
for c in coil1:
    psi1 += mu_o*coil1[c]['I']*cc.green(r2d,z2d,coil1[c]['r'],coil1[c]['z'])


levels = np.linspace(-50,300,50)
#pl.figure(figsize=0.5*np.array((0.5*np.diff(limit[:2]),np.diff(limit[2:]))))

pl.subplot(2,1,1)
pl.contour(r2d,z2d,psi,50,levels=levels)
pl.axis('equal')
pl.axis('off')

pl.subplot(2,1,2)
pl.contour(r2d,z2d,psi1,50,levels=levels)
pl.axis('equal')
pl.axis('off')

#pl.savefig('../Figs/two_coils.png',dpi=300)

'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm



fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')
psi_scale = 2e-2
ax.plot_surface(r2d,z2d,psi_scale*psi,rstride=1,cstride=1,alpha=0.5,
                cmap=cm.coolwarm)
cset = ax.contour(r2d,z2d,psi_scale*psi,100,zdir='z',
                  offset=-1,cmap=cm.coolwarm)

for c in coil:
    if coil[c]['I'] > 0:
        marker = 'ro'
    else:
        marker = 'bo'
    ax.plot([coil[c]['r']],[coil[c]['z']],[0],marker,markersize=6)

ax.set_xlabel('X')
ax.set_xlim(limit[0],limit[1])
ax.set_ylabel('Y')
ax.set_ylim(limit[2],limit[3])
ax.set_zlabel('Z')
ax.set_zlim(0, 3)

ax.view_init(elev=40., azim=-90)
ax.set_axis_off()
'''