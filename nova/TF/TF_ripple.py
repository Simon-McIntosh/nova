import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF,loop_vol
import amigo.geom as geom
import numpy as np

import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
        
nTF = 18

'''
cgeom = configure('TF',Ndp=0,Nloop=0)  # ITER refernace
cmap = coil_map(cgeom)
X = cgeom.profile(np.mean(cmap.loops['x']),0)

tf = TF(R=X[:,0],Z=X[:,2])
tf.energy(nTF,Iturn=134*68e3)
pl.plot(tf.Rmid,tf.Zmid)
'''

setup = Setup('SN')
sf = SF(setup.filename)
pf = PF(sf.eqdsk)
rb = RB(setup,sf)


eq = EQ(sf,pf,sigma=0.2,boundary=rb.get_fw(expand=0.25),n=7.5e4)  
eq.plotj(trim=True)
pl.plot(sf.rbdry,sf.zbdry,color=0.75*np.ones(3),lw=1.5)


for conf in ['SN','SXex']:  #  
    setup = Setup(conf)
    sf = SF(setup.filename)
    tf = TF(setup=setup,nTF=18)
    tf.fill()
    
    L = geom.length(tf.Rmid,tf.Zmid,norm=False)[-1]
    V = loop_vol(tf.Rmid,tf.Zmid)
    print('L {:1.2f}m, V {:1.0f}m3'.format(L,V))

pl.axis('equal')

    
    