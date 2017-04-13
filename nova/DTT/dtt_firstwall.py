import pylab as pl
from nova.config import Setup, select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF
from nova.inverse import INV, scenario
import numpy as np
from time import time
import amigo.geom as geom
from nova import loops
from nova.loops import Profile, plot_oppvar
from nova.shape import Shape
from nova.DEMOxlsx import DEMO
from nova.force import force_feild
from nova.firstwall import divertor, main_chamber
from amigo.IO import trim_dir

import seaborn as sns
rc = {'figure.figsize': [5, 5 * 16 / 12], 'savefig.dpi': 200,
      'savefig.jpeg_quality': 200, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 1.5}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=5 / 8, rc=rc)

machine = 'demo'
nTF, ripple = 18, True

if machine == 'demo':
    eq_names = ['DEMO_SN_SOF', 'DN', 'DEMO_SN_EOF']
    date = '2017_03_10'
elif machine == 'dtt':
    eq_names = ['DTT_SN']
    date = '2017_03_08'
else:
    raise ValueError('list machine type')

mc = main_chamber(machine, date=date)
mc.generate(eq_names, psi_n=1.07, flux_fit=True, plot=False, symetric=False)
mc.load_data(plot=False)  # load from file
mc.shp.plot_bounds()


config = {'TF': machine, 'eq': eq_names[0]}
config, setup = select(config, nTF=nTF, update=False)
sf = SF(setup.filename)
sf.contour()


rb = RB(sf, setup)
rb.generate(mc, plot=True, DN=False, debug=False)
rb.get_sol(plot=True)


# pl.plot(rb.segment['divertor']['r'],rb.segment['divertor']['z'])


profile = Profile(config['TF'], family='S', part='TF', nTF=nTF, obj='L')
shp = Shape(profile, eqconf=config['eq'], ny=3)
shp.add_vessel(rb.segment['vessel_outer'])
# shp.minimise(ripple=ripple,verbose=True)
shp.tf.fill()

'''
rb.write_json(tf=shp.tf)

to = time()
pf = PF(sf.eqdsk)
eq = EQ(sf,pf,dCoil=0.5,boundary=sf.get_sep(expand=1.05),n=2.5e3) 

#eq.gen_opp()
#sf.contour()
#pf.plot(coils=pf.coil,label=True,current=True)

inv = INV(sf,eq,shp.tf)
sc = scenario(inv)
sc.flat_top()


inv.solve_slsqp(inv.swing['flux'][0])
eq.run()
#eq.gen_opp()

print('t',time()-to)

sf.contour()

pf.plot(coils=pf.coil,label=True,current=True)
pf.plot(coils=eq.coil,label=False,plasma=True,current=False) 
#inv.ff.plot(scale=1.5)


#pl.figure(figsize=([5*16/12,5]))
#pl.plot(inv.swing['flux']*2*np.pi,inv.swing['rms'],'.-')
'''
