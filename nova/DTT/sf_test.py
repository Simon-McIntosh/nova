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
rc = {'figure.figsize':[12,12*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

config = 'SXex'
setup = Setup(config)

sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)

#eq = EQ(sf,pf,dCoil=0.5,sigma=0,boundary=rb.get_fw(expand=1.5),n=1e4)  


#eq.gen_opp(z=sf.Mpoint[1],Zerr=5e-4)


sf.contour()
pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
rb.firstwall(calc=True,plot=True,debug=False)

rb.vessel()


rb.trim_sol(plot=True)


'''
tf = TF(nTF=18,shape={'vessel':rb.loop,'pf':pf,'fit':True,'setup':setup,
               'plot':True,'config':config,'coil_type':'S'})  
coil = {'Rcl':tf.Rcl,'Zcl':tf.Zcl,
        'nTF':tf.nTF,'Iturn':tf.Iturn}
rp = ripple(plasma={'config':config},coil=coil)
print(config,'ripple',rp.get_ripple())
tf.fill()

#tf.coil.plot()

'''