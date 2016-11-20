from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.coils import PF,TF
from nova import loops
from nova.loops import Profile

import seaborn as sns
rc = {'figure.figsize':[7*10/16,7],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))


nPF,nCS,nTF = 3,3,18
config = {'TF':'dtt','eq':'SN'}
config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)
setup = Setup(config['eq'])
sf = SF(setup.filename)

rb = RB(setup,sf)
pf = PF(sf.eqdsk)
tf = TF(Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L',load=True))
tf.fill()

eq = EQ(sf,pf,dCoil=2.0,sigma=0,n=1e4,
        boundary=sf.get_sep(expand=1.5),zmin=sf.Xpoint[1]-3) 
eq.gen_opp()
inv = INV(sf,eq,tf)
Lpf = inv.grid_PF(nPF=nPF)
Lcs = inv.grid_CS(nCS=nCS,Zbound=[-12,8],gap=0.1)
L = np.append(Lpf,Lcs)
inv.update_coils()
inv.update_coils()
inv.fit_PF(offset=0.3)

inv.fix_boundary_psi(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)
        
inv.set_swing()
inv.update_limits(LCS=[-9.5,9.5])

L = inv.optimize(L)
#inv.plot_fix(tails=True)
inv.fix_flux(inv.swing['flux'][0]) 
inv.solve_slsqp()

eq.gen_opp()
sf.contour(boundary=True,plot_vac=False)
pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
rb.firstwall(mode='calc',plot=True,debug=False)

loops.plot_variables(inv.Io,scale=1,postfix='MA')
loops.plot_variables(inv.Lo,scale=1)

sf.eqwrite(pf,config=config['TF']+'_{:d}PF_{:d}CS'.format(inv.nPF,inv.nCS))
