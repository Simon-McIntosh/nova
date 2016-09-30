import pylab as pl
import numpy as np
from matplotlib.animation import FuncAnimation
from shelf import PKL
import pickle
import seaborn as sns
rc = {'figure.figsize':[15,15*7.5/15],'savefig.dpi':250, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.75}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=1,rc=rc)
color = sns.color_palette('Set2')
from addtext import linelabel

from itertools import cycle
Color = cycle(sns.color_palette('Set2'))
from radial_build import RB
from eqConfig import Config
conf = Config('SXex')


pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
conf.TF(sf)
rb = RB(conf,sf,Np=500)

swing = 90
inv.swing_fix(swing)
inv.solve_slsqp()

#eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb},n=5e3)
eq.set_sf_psi()  # set psi
eq.run(update=False)


                
print(np.sqrt(np.mean(dr_bndry**2)))

        