import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup,select
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
import nova.cross_coil as cc
from nova.coils import PF,TF
from time import time
from nova import loops
from nova.DEMOxlsx import DEMO
from nova.loops import Profile

import seaborn as sns
rc = {'figure.figsize':[7*10/16,7],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))


nTF,nPF,nCS = 18,6,5
config = {'TF':'demo','eq':'SN'}
config,setup = select(config,nTF=nTF,nPF=nPF,nCS=nCS,update=False)
    

sf = SF(setup.filename)

rb = RB(setup,sf)
pf = PF(sf.eqdsk)
tf = TF(Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L',load=True))

#pf.plot(coils=pf.coil,label=False,plasma=False,current=True) 

demo = DEMO()
demo.fill_part('Vessel')
demo.fill_part('Blanket')
#demo.fill_part('TF_Coil')
demo.plot_ports()
demo.plot_limiter() 

pl.axis('off')
#tf.fill()

sf.cpasma *= 1.1

eq = EQ(sf,pf,dCoil=0.5,sigma=0,boundary=sf.get_sep(expand=1.5),n=5e3) 

eq.gen_opp()

sf.contour()

eq.plotb()


'''
inv = INV(sf,eq,tf)
L = inv.grid_coils(offset=0.3)
#pf.plot(coils=pf.coil,label=False,plasma=False,current=True) 

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

#pf.plot(coils=pf.coil,label=False,current=True) 
pf.plot(coils=eq.coil,label=False,plasma=True,current=False) 
sf.contour(boundary=True)
inv.plot_fix(tails=True)

inv.ff.plot(scale=1.5)

sf.eqwrite(pf,config=config['eq'])
'''