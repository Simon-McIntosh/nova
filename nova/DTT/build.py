import pylab as pl
from nova.config import Setup,select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF
from nova.inverse import INV
from nova.TF.ripple import ripple
import numpy as np
from time import time
import amigo.geom as geom
from nova.loops import Profile,plot_oppvar
from nova.shape import Shape
from nova.DEMOxlsx import DEMO
from nova.force import force_feild
import json
from amigo.IO import trim_dir
from nova.shelf import PKL

import seaborn as sns
rc = {'figure.figsize':[5,5*16/12],'savefig.dpi':150, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

#config,setup = select(base={'TF':'dtt','eq':'SN'},nTF=18,nPF=5,nCS=3)
#config,setup = select(base={'TF':'dtt','eq':'SX'},nTF=18,nPF=5,nCS=3)
#config,setup = select(base={'TF':'dtt','eq':'DEMO_FW_SOF'},nTF=18)
config,setup = select(base={'TF':'dtt','eq':'dtt_SN'},nTF=18)

if 'DN' in setup.configuration:
    DN = True
else:
    DN = False

sf = SF(setup.filename)  
sf.contour()
sf.shape_parameters()  #verbose=True

print(sf.eqdsk['bcentr']*sf.eqdsk['rcentr'])

pf = PF(sf.eqdsk)


'''
eq = EQ(sf,pf,dCoil=1.5,sigma=0,n=5e3,boundary=sf.get_sep(expand=1.1),
        zmin=-abs(sf.Xpoint[1])-2,zmax=abs(sf.Xpoint[1])+2) 
#eq.gen_opp(Zerr=5e-4)
eq.gen_bal(Zerr=5e-4,tol=1e-4)
'''


pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
sf.contour()

rb = RB(setup,sf)

rb.firstwall(symetric=DN,DN=DN,plot=True,debug=False)  # ,mode='eqdsk'
rb.trim_sol()

if DN:
    sf.get_Xpsi(select='upper')  # upper X-point
    rb.trim_sol()


nTF = config['nTF']
profile = Profile(config['TF'],family='S',part='TF',
                  nTF=nTF,obj='L',load=False,symetric=DN)
shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'],ny=3)
shp.add_vessel(rb.segment['vessel_outer'])
#shp.plot_bounds()
shp.loop.adjust_xo('l',lb=0.75)  # 1/tesion
shp.loop.adjust_xo('upper',lb=0.5)
shp.loop.adjust_xo('lower',lb=0.5)
shp.minimise(ripple=False,verbose=True)
shp.update()
tf = TF(profile,sf=sf)
tf.fill()

pl.plot(3,-11.5,'o',alpha=0)
pl.plot(16,7.5,'o',alpha=0)
pl.tight_layout()


'''
demo = DEMO()
demo.fill_part('Blanket',alpha=1)
demo.fill_part('Vessel',alpha=1)
demo.fill_part('TF_Coil',alpha=1)
'''  


sf.eqwrite(pf,config=config['eq'],CREATE=True)

'''
pkl = PKL(config['eq'],directory='../../Movies/')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
for i,(flux,S) in enumerate(zip(inv.swing['flux'],['SOF','MOF','EOF'])):
    inv.fix_flux(flux)  # swing
    inv.solve_slsqp()
    inv.eq.gen_opp()
    sf.eqwrite(inv.eq.pf,config=config['eq']+'_{}'.format(S),CREATE=True)
'''


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
