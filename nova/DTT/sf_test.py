import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF
from nova.inverse import INV
from nova.TF.ripple import ripple
import numpy as np
import scipy

import seaborn as sns
rc = {'figure.figsize':[6,6*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

config = 'SFm'
setup = Setup(config)
sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)


levels = sf.contour()


eq = EQ(sf,pf,dCoil=0.5,sigma=0,boundary=sf.get_sep(expand=1.25),n=1e4)  



#eq.set_Vcoil()
                                  
#eq.gen(sf.Mpoint[1],Zerr=5e-4)    
eq.gen_opp(z=sf.Mpoint[1],Zerr=5e-4)
#eq.resample(n=1e4)
#eq.gen_opp(z=sf.Mpoint[1],Zerr=5e-4)

#print('cc {:1.3f}KA'.format(1e-3*eq.cc))

sf.contour(levels=levels,color=sns.color_palette('Set2',1)[0])
pf.plot(coils=eq.coil,label=False) 


pl.figure()
pl.plot(eq.Zerr)
pl.xlabel(r'itteration number, $i$')
pl.ylabel('vertical position error, m')
sns.despine()

'''


cc = []
for z in np.linspace(0.8,1,10):
    eq = EQ(sf,pf,sigma=0,boundary=rb.get_fw(expand=0.25),n=2e3)  
    cc.append(eq.gen(z,Verr=1e-3))
    

eq.get_plasma_coil()

pf.plot(coils=eq.plasma_coil,label=False)


'''
'''
sf.contour()
pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
rb.firstwall(calc=False,plot=True,debug=False)

import time
tic = time.time()
rb.vessel()
print(time.time()-tic)

#rb.trim_sol(plot=True)

#sp = sf.shape_parameters()
#print(sp)


tf = TF(nTF=18,shape={'vessel':rb.loop,'pf':pf,'fit':True,'setup':setup,
               'plot':True,'config':config,'coil_type':'S'})  
coil = {'Rcl':tf.Rcl,'Zcl':tf.Zcl,
        'nTF':tf.nTF,'Iturn':tf.Iturn}
rp = ripple(plasma={'config':config},coil=coil)
print(config,'ripple',rp.get_ripple())
tf.fill()

#tf.coil.plot()
'''

