import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
from nova.config import Setup
from nova.streamfunction import SF
from nova.coils import TF
from amigo import geom
from amigo.addtext import linelabel

rc = {'figure.figsize':[8,4],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2')


mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]


# prepare coil loop
Nl = 100  # coil segment number
Ns = 100  # sep segment number

text = linelabel(Ndiv=10,value='1.1f',postfix='%',loc='max')

for conf in ['SN','SXex','SFp','SFm','X','SX']:
    setup = Setup(conf)
    sf = SF(setup.filename)
    tf = TF(setup=setup)
    tf.energy()
    
    B = np.zeros((Ns,2))
    loop = np.zeros((Nl,3))
    (loop[:,0],loop[:,2]) = geom.rzSLine(tf.Rmid,tf.Zmid,Np=Nl)
    r95,z95 = sf.get_boundary(alpha=0.95)
    sep = np.zeros((Ns,3))
    (sep[:,0],sep[:,2]) = geom.rzSLine(r95,z95,Np=Ns)

    for j,tsep in enumerate([0,np.pi/tf.nTF]):
        sepR = np.dot(sep,geom.rotate(tsep))
        for i,s in enumerate(sepR):     
            for tcoil in np.linspace(0,2*np.pi,tf.nTF):
                loopR = np.dot(loop,geom.rotate(tcoil))
                B[i,j] += mu_o*tf.Iturn*cc.\
                green_feild_loop(loopR,s,smooth=False)[1]
    
    ripple = 1e2*(B[:,0]-B[:,1])/(B[:,1]+B[:,0])
    pl.plot(ripple)
    text.add(conf)
    
    print(conf,'ripple {:1.1f}% E {:1.0f}GJ'.format(np.max(ripple),
                                                    1e-9*tf.Ecage))

text.plot()
sns.despine()