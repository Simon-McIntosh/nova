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
ar = 2/5
import seaborn as sns
rc = {'figure.figsize':[8,8*ar],'savefig.dpi':125, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)
Color = cycle(sns.color_palette('Set2'))
fig = pl.figure()
pl.axis('equal')
pl.axis('off')
pl.ylim((-7.6,-7.5))

rt,zt = 9,-7.4
delta = 2.4
pl.xlim(rt+ar*delta*np.array([-1,1]))
pl.ylim(zt+delta*np.array([-1,1]))

pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
sf.config = 'SXex'

conf = Config('SXex')
sf.conf = conf
conf.TF(sf)

inv.swing_fix(-38)
inv.solve_slsqp()
eq.run(update=False)
    
sf.contour(lw=1)


#inv.rb.sol.plot()
#sf.sol()
#Rsol,Zsol = inv.rb.sol.legs('outer')

from eqConfig import Config
conf = Config('SXex')

conf.TF(sf)
sf.conf = conf

rb = RB(conf,sf,Np=100)
rb.divertor_outline(False,plot=True,debug=False)
rb.trim_sol(plot=True)

pl.savefig('../Figs/divertor_rad.png',dpi=300)