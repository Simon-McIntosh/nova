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
from time import time
import amigo.geom as geom
from nova.loops import Profile
from nova.shape import Shape
from nova.TF.DEMOxlsx import DEMO

import seaborn as sns
rc = {'figure.figsize':[12,12*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

config = 'DEMO_SNc'
setup = Setup(config)

sf = SF(setup.filename)
pf = PF(sf.eqdsk)
pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
levels = sf.contour()


rb = RB(setup,sf)
rb.firstwall(calc=True,plot=False,debug=False)
'''
demo = DEMO()
demo.fill_loops()
demo.get_ports(plot=True)
'''


eq = EQ(sf,pf,dCoil=0.5,sigma=0,boundary=rb.get_fw(expand=0.25),n=2e4)  
eq.gen_opp()

sf.contour(levels=levels)

rb = RB(setup,sf)
rb.firstwall(calc=True,plot=True,debug=False)
pl.axis('equal')

sf.eqwrite(pf,config=config,CREATE=True)

#pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 


'''
profile = Profile(config,family='S',part='TF')
shp = Shape(profile,objective='L',nTF=18)
    
rb.vessel()
rvv,zvv = geom.rzSLine(rb.loop.R,rb.loop.Z,30)
rvv,zvv = geom.offset(rvv,zvv,0.2)
shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
#shp.plot_bounds()
#shp.minimise()
shp.update()
#shp.tf.fill()



#rb.trim_sol(plot=True)
'''
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