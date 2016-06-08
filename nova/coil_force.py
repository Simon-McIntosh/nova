import pylab as pl
import numpy as np
from streamfunction import SF
from elliptic import grid
import cross_coil as cc
from eqConfig import Config
from itertools import cycle
import seaborn as sns
rc = {'figure.figsize':[3.14*12/16,3.14],'savefig.dpi':250, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))


pl.figure()
pl.axis('equal')
pl.axis('off')

config = 'SFm'  # SN,X,SX,SF
conf = Config(config,inside=False)
sf = SF(conf)
sf.contour()

sf.get_boundary(alpha=0.5)