import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
import nova.cross_coil as cc
from nova.coils import PF,TF
from time import time
from nova import loops
from DEMOxlsx import DEMO
from nova.loops import Profile

import seaborn as sns
rc = {'figure.figsize':[7*10/16,7],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))


nTF = 16

config = {'TF':'SN','eq':'DEMO_SNb'}
setup = Setup(config['eq'])
sf = SF(setup.filename)

rb = RB(setup,sf)
pf = PF(sf.eqdsk)
tf = TF(Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L',load=True))

#pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 

demo = DEMO()
demo.fill_part('Vessel')
demo.fill_part('Blanket')
demo.fill_part('TF_Coil')
demo.plot_ports()
demo.plot_limiter()  
tf.fill()

eq = EQ(sf,pf,dCoil=2.0,sigma=0,boundary=sf.get_sep(expand=1.5),n=5e3) 
eq.gen_opp()
inv = INV(sf,eq,tf)
L = inv.grid_coils()
inv.update_coils()
inv.fit_PF(offset=0.3)

inv.fix_boundary_psi(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)
        
inv.set_swing()
inv.update_limits(LCS=[-9.5,9.5])

inv.initialize_log()
inv.set_background()
inv.get_weight()
inv.set_Lo(L)  # set position bounds
Lnorm = loops.normalize_variables(inv.Lo)
inv.update_position(Lnorm,update_area=True)

eq.gen_opp()
rb.firstwall(mode='calc',plot=True,debug=False)

pf.plot(coils=pf.coil,label=True,plasma=True,current=True) 
sf.contour(boundary=True)

inv.plot_fix(tails=True)
tf.fill()

loops.plot_variables(inv.Io,scale=1,postfix='MA')
loops.plot_variables(inv.Lo,scale=1)

sf.eqwrite(pf,config=config['TF']+'_{:d}PF_{:d}TF'.format(inv.nPF,nTF))
