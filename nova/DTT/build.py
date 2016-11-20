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
from nova.loops import Profile,plot_oppvar
from nova.shape import Shape
from nova.TF.DEMOxlsx import DEMO
from nova.force import force_feild

import seaborn as sns
rc = {'figure.figsize':[8,8*16/12],'savefig.dpi':80, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)


nTF = 13
config = {'TF':'dtt','eq':'SFm'}
config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)
setup = Setup(config['eq'])

sf = SF(setup.filename)
pf = PF(sf.eqdsk)


#pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
levels = sf.contour()


rb = RB(setup,sf)
rb.firstwall(plot=True,debug=False)

profile = Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L')

shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'])  # 

rvv,zvv = geom.rzSLine(rb.segment['vessel']['r'],rb.segment['vessel']['z'],80)
rvv,zvv = geom.offset(rvv,zvv,0.2)
rmin = np.min(rvv)
rvv[rvv<=rmin+0.12] = rmin+0.12
#shp.loop.oppvar.remove('flat')
shp.loop.set_l({'value':0.8,'lb':0.65,'ub':1.8})  # 1/tesion
shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
shp.add_bound({'r':np.min(rvv)-0.01,'z':0},'interior')  # vessel
shp.minimise(ripple=True,verbose=True)


tf = TF(profile,sf=sf)
tf.fill()
#shp.plot_bounds()
#shp.loop.plot()
#plot_oppvar(shp.loop.xo,shp.loop.oppvar)


