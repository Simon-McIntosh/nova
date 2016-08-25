import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
from nova.config import Setup
from nova.streamfunction import SF
from nova.coils import TF
from amigo import geom

rc = {'figure.figsize':[8,4],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2')

setup = Setup('SXex')
sf = SF(setup.filename)
tf = TF(setup=setup)





mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

r = np.linspace(1,18,40)
z = np.linspace(-8,8,40)
R,Z = np.meshgrid(r,z,indexing='ij')

B = np.zeros(np.append(np.shape(R),3))

# prepare coil loop
Nseg = 10

loop = np.zeros((Nseg,3))
(loop[:,0],loop[:,2]) = geom.rzSLine(tf.Rmid,tf.Zmid,Np=Nseg)



for i,ri in enumerate(r):
    for k,zk in enumerate(z):
        
        for t in np.linspace(0,2*np.pi,tf.nTF):
            loopR = np.dot(loop,geom.rotate(t))
            B[i,k,:] += mu_o*tf.Iturn*cc.green_feild_loop(loopR,(ri,0,zk,))

    

pl.contour(R,Z,B[:,:,1],50)
pl.plot(tf.Rmid,tf.Zmid)
pl.plot(sf.rbdry,sf.zbdry)
pl.axis('equal')

