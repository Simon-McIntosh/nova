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


nPF,nCS,nTF = 4,3,16
config = {'TF':'dtt','eq':'SN'}
config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)

#config['eq'] = 'SNdtt{:d}_{:d}PF_{:d}CS'.format(nTF,nPF,nCS)
#config['eq'] = 'SFmdtt18_5PF_3CS'
setup = Setup(config['eq'])

'''
setup.targets['inner1'] = {'L2D':[1.1],'open':True,'dR':0}
setup.targets['inner2'] = {'L2D':[1.2-0.08],'open':False,'dR':-1}
setup.targets['outer1'] = {'L2D':[1.85],'open':False,'dR':-1}
setup.targets['outer2'] = {'L2D':[1.0],'open':True,'dR':0}  
setup.firstwall['div_ex'] = 0.18
'''

sf = SF(setup.filename)
pf = PF(sf.eqdsk)


pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
levels = sf.contour()


rb = RB(setup,sf)
rb.firstwall(plot=True,debug=False)
rb.trim_sol()

profile = Profile(config['TF'],family='S',part='TF',
                  nTF=nTF,obj='L',load=True)

shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'],ny=1)
shp.add_vessel(rb.segment['vessel'])
#shp.minimise(ripple=True,verbose=True)

tf = TF(profile,sf=sf)
tf.fill()

pl.savefig('../../Figs/'+config['eq']+'_coils.png')

sf.eqwrite(pf,config=config['eq'],CREATE=True)

#shp.plot_bounds()
#shp.loop.plot()
plot_oppvar(shp.loop.xo,shp.loop.oppvar)


