import pylab as pl
from nova.config import Setup,select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF
from nova.inverse import INV,scenario
import numpy as np
from time import time
import amigo.geom as geom
from nova import loops
from nova.loops import Profile,plot_oppvar
from nova.shape import Shape
from nova.DEMOxlsx import DEMO
from nova.force import force_feild
from nova.firstwall import divertor,main_chamber
from amigo.IO import trim_dir
from nova.coil_cage import coil_cage

import seaborn as sns
rc = {'figure.figsize':[5,5*16/12],'savefig.dpi':150, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=5/8,rc=rc)


nTF,ripple = 18,True

plot_demoTF = True

config = {'TF':'demo_nTF','eq':'DEMO_SN_SOF'}    
config,setup = select(config,nTF=nTF,update=False)     
sf = SF(setup.filename) 
pf = PF(sf.eqdsk) 
  
   
pf.plot(coils=pf.coil,color=0.75*np.ones(3),label=False,plasma=True)




demo = DEMO()
profile = Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L')
shp = Shape(profile,eqconf=config['eq'],ny=3)
shp.add_vessel(demo.parts['Vessel']['out'])
#shp.minimise(ripple=ripple,verbose=False)

tf = TF(x_in=demo.parts['TF_Coil']['in'],nTF=nTF,sf=sf)
tf.x['out'] = demo.parts['TF_Coil']['out']

tf = shp.tf


inv = INV(pf,tf,dCoil=0.5)
sc = scenario(inv,sf)
sc.flat_top()

sc.output()


inv.solve_slsqp(inv.swing['flux'][0])
inv.eq.run(update=False)


cage = coil_cage(nTF=nTF,rc=tf.rc,plasma={'config':config['eq']},ny=3)
cage.set_TFcoil(tf.x['cl'])
cage.output()            


demo.fill_part('Blanket')
demo.fill_part('Vessel')
sf.contour()
pf.plot(coils=pf.coil,label=False,current=True,plasma=True,alpha=0.75)
pf.plot(coils=pf.sub_coil)
inv.ff.plot(scale=3)


#if plot_demoTF:
#    demo.fill_part('TF_Coil',color=0.75*np.ones(3))
tf.fill()


'''
#cage = shp.cage
#shp.update()
#shp.tf.fill()
#shp.loop.plot({'flat':0.3,'tilt':13})
#shp.loop.plot()
#demo.fill_part('TF_Coil',alpha=0.8)
#shp.cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)
#shp.cage.pattern(plot=True)
#plot_oppvar(shp.loop.xo,shp.loop.oppvar)

    
#rb.generate(mc,plot=True,DN=False,debug=False)
#rb.get_sol(plot=True)
'''
