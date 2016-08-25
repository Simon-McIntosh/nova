import cross_coil as cc
import pylab as pl
import numpy as np
mu_o = 4*np.pi*1e-7
import seaborn as sns
rc = {'figure.figsize':[8,8],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2')

coil = {'r':8,'z':-10,'I':30e6,'rc':0.01}

limit = [-12,12,-12,12]

ro,zo,n = limit[:2],limit[2:],1e4
dro,dzo = (ro[-1]-ro[0]),(zo[-1]-zo[0])
ar = dro/dzo
nz = int(np.sqrt(n/ar))
nr = int(n/nz)
r = np.linspace(ro[0],ro[1],nr)
z = np.linspace(zo[0],zo[1],nz)
r2d,z2d = np.meshgrid(r,z,indexing='ij')
Br = np.zeros(np.shape(r2d))
Bz = np.zeros(np.shape(r2d))
  
Brl = np.zeros(np.shape(r2d))
Bzl = np.zeros(np.shape(r2d))     

for i,ri in enumerate(r):
    for j,zj in enumerate(z):
        #B = mu_o*coil['I']*cc.green_feild(ri,zj,coil['r'],coil['z'])
        #Br[i,j] = B[0]
        #Bz[i,j] = B[1]
        
        B = cc.green_feild_loop(coil,(ri,zj,-1.5))
        Brl[i,j] = B[0]
        Bzl[i,j] = B[1]
print(B)

pl.figure()
pl.axis('equal')
#pl.streamplot(r,z,Br.T,Bz.T) 
Bmag = np.sqrt(Brl.T**2+Bzl.T**2)
#pl.streamplot(r,z,Brl.T,Bzl.T,color=Bmag,cmap=pl.cm.Spectral) 
pl.quiver(r,z,Brl.T,Bzl.T)

'''
B = cc.green_feild_loop(coil,(r,0,z))

print(B)

levels = np.linspace(-50,300,50)
'''

#pl.figure(figsize=0.5*np.array((0.5*np.diff(limit[:2]),np.diff(limit[2:]))))
'''
pl.figure()
pl.contour(r2d,z2d,psi,50,levels=levels)


c = pl.contour(r2d,z2d,psi_loop,50,levels=levels)
for cs in c.collections:
    cs.set_linestyle('--')
    cs.set_color(color[0])

pl.axis('equal')
pl.axis('off')


#pl.savefig('../Figs/two_coils.png',dpi=300)


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