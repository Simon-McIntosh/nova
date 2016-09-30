import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF,loop_vol
import amigo.geom as geom
import numpy as np
from nova.TF.ripple import ripple

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

#setup = Setup('SN')
#sf = SF(setup.filename)



#eq = EQ(sf,pf,sigma=0.2,boundary=rb.get_fw(expand=0.25),n=7.5e4)  
#eq.plotj(trim=True)
#pl.plot(sf.rbdry,sf.zbdry,color=0.75*np.ones(3),lw=1.5)


for conf in ['SN','X','SFm','SX','SXex']:  # 
    print(conf) 
    setup = Setup(conf)
    sf = SF(setup.filename)
    pf = PF(sf.eqdsk)
    rb = RB(setup,sf)
    rb.firstwall(calc=False,plot=True,debug=False)
    rb.vessel()
    pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
    tf = TF(nTF=18,shape={'vessel':rb.loop,'pf':pf,'fit':False,'setup':setup,
               'plot':True,'config':conf,'coil_type':'A'})
    tf.fill()
    
    pl.plot(tf.Rcl,tf.Zcl)
    coil = {'Rcl':tf.Rcl,'Zcl':tf.Zcl,
            'nTF':tf.nTF,'Iturn':1}
    rp = ripple(plasma={'config':'SN'},coil=coil)
    rp.plot_loops()
    print(conf,'ripple',rp.get_ripple())
    L = geom.length(tf.Rcl,tf.Zcl,norm=False)[-1]
    V = loop_vol(tf.Rcl,tf.Zcl)
    print('L {:1.2f}m, V {:1.0f}m3'.format(L,V))

pl.axis('equal')

    
    