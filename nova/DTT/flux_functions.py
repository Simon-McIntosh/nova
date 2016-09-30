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
rc = {'figure.figsize':[6,6*12/16],'savefig.dpi':125, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)
Color = cycle(sns.color_palette('Set2'))
conf = Config('SN')
sf = SF(conf,sample=1,Nova=False)
#sf.contour(lw=0.5)

pl.figure()
pl.plot(np.linspace(0,1,len(sf.eq['pressure'])),sf.eq['pressure'])

pl.figure()
pl.plot(np.linspace(0,1,len(sf.eq['pressure'])),sf.eq['fpol'])
#sf.eqwrite()

