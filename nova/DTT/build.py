import pylab as pl
from nova.config import Setup,select
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
import json
from amigo.IO import trim_dir
from nova.shelf import PKL

import seaborn as sns
rc = {'figure.figsize':[8,8*16/12],'savefig.dpi':140, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

config,setup = select(base={'TF':'dtt','eq':'SN'},nTF=18,nPF=5,nCS=3)
#config,setup = select(base={'TF':'dtt','eq':'SX'},nTF=18,nPF=5,nCS=3)
config,setup = select(base={'TF':'dtt','eq':'DN'},nTF=18,nPF=4,nCS=3)

sf = SF(setup.filename)  
pf = PF(sf.eqdsk)
pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
levels = sf.contour()

rb = RB(setup,sf)

sf.get_Xpsi(select='lower')
rb.update_sf()  # update streamfunction

sf.sol(plot=True)

rb.firstwall(plot=True,debug=False)
'''
rb.trim_sol()

nTF = 18#config['nTF']
profile = Profile(config['TF'],family='S',part='TF',
                  nTF=nTF,obj='L',load=True)
<<<<<<< HEAD

shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'],ny=1)
shp.add_vessel(rb.segment['vessel'])
shp.minimise(ripple=False,verbose=True)

tf = TF(profile,sf=sf)
tf.fill()

#pl.savefig('../../Figs/'+config['eq']+'_coils.png')
#sf.eqwrite(pf,config=config['eq'],CREATE=True)

#shp.plot_bounds()
#shp.loop.plot()
plot_oppvar(shp.loop.xo,shp.loop.oppvar)

=======
shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'],ny=3)
shp.add_vessel(rb.segment['vessel_outer'])
shp.loop.set_l({'value':0.8,'lb':0.75,'ub':1.8})  # 1/tesion
shp.loop.xo['lower'] = {'value':0.7,'lb':0.5,'ub':1}  # vertical shift
shp.minimise(ripple=False,verbose=True)
tf = TF(profile,sf=sf)
tf.fill()
pl.tight_layout()
sf.eqwrite(pf,config=config['eq'],CREATE=True)

'''
'''
pkl = PKL(config['eq'],directory='../../Movies/')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
for i,(flux,S) in enumerate(zip(inv.swing['flux'],['SOF','MOF','EOF'])):
    inv.fix_flux(flux)  # swing
    inv.solve_slsqp()
    inv.eq.gen_opp()
    sf.eqwrite(inv.eq.pf,config=config['eq']+'_{}'.format(S),CREATE=True)


data = {}
for loop in ['first_wall','blanket','vessel_inner','vessel_outer']:
    data[loop] = {}
    for var in rb.segment[loop]:
        data[loop][var] = list(rb.segment[loop][var])
for loop,label in zip(['in','out'],['TF_inner','TF_outer']):
    data[label] = {}
    for var in tf.x[loop]:
        data[label][var] = list(tf.x[loop][var])
datadir = trim_dir('../../../Data/') 
with open(datadir+'{}.json'.format(config['eq']),'w') as f:
    json.dump(data,f,indent=4)

'''
>>>>>>> 7794d945fa648939cd2adb2c0bb1b0614495cdcd
