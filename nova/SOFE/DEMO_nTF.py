from nova.loops import Profile,plot_oppvar,plot_variables
from nova.shape import Shape
from nova.DEMOxlsx import DEMO
from nova.config import select
import pylab as pl
import numpy as np

nTF = 18
family='S'
ripple = True

config = {'TF':'demo','eq':'DEMO_SN_SOF'}
config,setup = select(config,nTF=nTF)


demo = DEMO()

profile = Profile(config['TF'],family=family,part='TF',nTF=nTF) 

shp = Shape(profile,eqconf=config['eq'],ny=3)
shp.add_vessel(demo.parts['Vessel']['out'])
shp.minimise(ripple=ripple,verbose=True)



cage = shp.cage

#x_in = demo.parts['TF_Coil']['in']
#tf = TF(x_in=x_in,nTF=nTF)  
#x = tf.get_loops(x_in)
#cage = coil_cage(nTF=18,rc=tf.rc,plasma={'config':config['eq']},ny=3)
#cage.set_TFcoil(x['cl'],smooth=True)

cage.output()


fig,ax = pl.subplots(1,1,figsize=(8,10))
pl.plot([3,18],[-10,10],'ko',alpha=0)
demo.fill_part('Blanket')
demo.fill_part('Vessel')
shp.tf.fill()
#demo.fill_part('TF_Coil',color=0.75*np.ones(3))

#cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)  # 2e5
pl.axis('off')

#pl.savefig('../Figs/ripple_referance')
