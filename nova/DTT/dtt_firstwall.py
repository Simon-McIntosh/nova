import pylab as pl
from nova.config import Setup,select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF
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

import seaborn as sns
rc = {'figure.figsize':[5,5*16/12],'savefig.dpi':150, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=5/8,rc=rc)


eq_names = ['DEMO_SN_SOF','DEMO_SN_EOF']  #    # 'DTT_SN'
mc = main_chamber('DEMO',date='2017_03_07')  
mc.generate(eq_names,psi_n=1.07,flux_fit=True,plot=False,symetric=False)
mc.load_data(plot=False)  # load from file
mc.shp.plot_bounds()


nTF,ripple = 18,True
config = {'TF':'dtt','eq':eq_names[0]}    
config,setup = select(config,nTF=nTF,update=False)     
sf = SF(setup.filename)   
rb = RB(sf,setup)
rb.generate(mc,plot=True,DN=False,debug=False)
rb.get_sol(plot=True)

profile = Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L')
shp = Shape(profile,eqconf=config['eq'],ny=3)
shp.add_vessel(rb.segment['vessel_outer'])
#shp.minimise(ripple=ripple,verbose=True)
shp.tf.fill()


pf = PF(sf.eqdsk)
eq = EQ(sf,pf,dCoil=0.5,sigma=0,boundary=sf.get_sep(expand=1.05),n=2.5e3) 
eq.gen_opp()

inv = INV(sf,eq,shp.tf)

sc = scenario(inv)
sc.flat_top()


inv.solve_slsqp(inv.swing['flux'][0])
eq.gen_opp()
#eq.run()
sf.contour()


pf.plot(coils=pf.coil,label=True,current=True)
pf.plot(coils=eq.coil,label=False,plasma=True,current=False) 
ff = force_feild(eq.pf.index,eq.pf.coil,eq.coil,eq.plasma_coil,
                 multi_filament=True)
ff.plot(scale=1.5)

pl.figure(figsize=([5*16/12,5]))
pl.plot(inv.swing['flux']*2*np.pi,inv.swing['rms'],'.-')
