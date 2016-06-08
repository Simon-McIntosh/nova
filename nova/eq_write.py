import pylab as pl
from streamfunction import SF
from elliptic import EQ
from inverse import INV
#from eqConfig import Config
from itertools import cycle
import numpy as np
from radial_build import RB
from shelf import PKL
from eqConfig import Config

import seaborn as sns
rc = {'figure.figsize':[6*12/16,6],'savefig.dpi':175, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)
Color = cycle(sns.color_palette('Set2'))
pl.figure()
pl.axis('equal')
pl.axis('off')

pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
sf.config = 'SXex'

inv.swing_fix(np.mean(inv.Swing))
inv.solve_slsqp()
eq.run(update=False)

conf = Config('SXex')
conf.TF(sf)
sf.conf = conf
rb = RB(conf,sf,Np=500)
rb.divertor_outline(False,plot=True,debug=False)
eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb},n=5e3)
eq.set_sf_psi()  # set psi
eq.run()
sf.set_plasma({'r':eq.r,'z':eq.z,'psi':eq.psi},contour=True)
    
 
inv.plot_fix()

sf.eqwrite()

sf = SF(conf,sample=1,Nova=True)
sf.contour(lw=0.5)

sf.plot_coils(next(Color),coils=sf.coil,label=True,plasma=False,current=False)
