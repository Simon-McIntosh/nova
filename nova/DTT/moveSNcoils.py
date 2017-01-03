from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup,select
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.coils import PF,TF
from nova import loops
from nova.loops import Profile
import pylab as pl
from nova.shelf import PKL

import seaborn as sns
rc = {'figure.figsize':[7*10/16,7],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

base = {'TF':'dtt','eq':'SN'}
config,setup = select(base=base,update=False,nTF=18,nPF=5,nCS=3)
sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)
tf = TF(Profile(config['TF'],family='S',part='TF',
                nTF=config['nTF'],obj='L',load=True))
tf.fill()
eq = EQ(sf,pf,dCoil=1.0,sigma=0,n=2e4,
        boundary=sf.get_sep(expand=1.1),zmin=sf.Xpoint[1]-2) 
eq.gen_opp()

inv = INV(sf,eq,tf)
Lpf = inv.grid_PF(nPF=config['nPF'])
Lcs = inv.grid_CS(nCS=config['nCS'],Zbound=[-8,8],gap=0.1,fdr=1)
L = np.append(Lpf,Lcs)
inv.update_coils()
inv.fit_PF(offset=0.3)

inv.fix_boundary_psi(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)
        
inv.set_swing()
inv.update_limits(LCS=[-11.5,10.5])

L = inv.optimize(L)
#inv.plot_fix(tails=True)
inv.fix_flux(inv.swing['flux'][1]) 
inv.solve_slsqp()

eq.gen_opp()
sf.contour(boundary=True,plot_vac=False)
pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
rb.firstwall(mode='calc',plot=True,debug=False)
rb.trim_sol()

#loops.plot_variables(inv.Io,scale=1,postfix='MA')
#loops.plot_variables(inv.Lo,scale=1)

sf.eqwrite(pf,config=config['eq'])

pkl = PKL(config['eq'],directory='../../Movies/')
pkl.write(data={'sf':sf,'eq':eq,'inv':inv})  # pickle data