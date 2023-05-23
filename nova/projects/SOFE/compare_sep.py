from amigo.addtext import linelabel
import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup, select
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
import nova.cross_coil as cc
from nova.coils import PF, TF
from time import time
from nova import loops
from nova.DEMOxlsx import DEMO
from nova.loops import Profile
from nova.force import force_feild

import seaborn as sns
rc = {'figure.figsize': [7 * 10 / 16, 7], 'savefig.dpi': 250,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 1}
sns.set(context='poster', style='white', font='sans-serif', palette='Set2',
        font_scale=0.75, rc=rc)
Color = cycle(sns.color_palette('Set2'))

text = linelabel(Ndiv=20, value='')

nTF = 18

for eq in ['DEMO_SN_SOF', 'DEMO_SN_EOF']:
    config = {'TF': 'demo', 'eq': eq}
    config, setup = select(config, nTF=nTF, update=False)
    sf = SF(setup.filename)
    sf.get_boundary(plot=True, alpha=1 - 1e-5)
    text.add(eq)
    sf.sol(plot=True)


text.plot()
pl.axis('equal')
pl.axis('off')
