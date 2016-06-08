import pylab as pl
import numpy as np
from itertools import cycle
from eqConfig import Config
from radial_build import RB
from streamfunction import SF
import seaborn as sns
rc = {'figure.figsize':[3.14*9/6,3.14],'savefig.dpi':200, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

pl.figure()
#pl.figure(figsize=(12,8))
pl.axis('equal')
pl.axis('off')
#pl.xlim([4,14]),pl.ylim([-13,8])
pl.xlim([6,11]),pl.ylim([-6.5,-4.5])

config = 'SFp'  # SN,X,SX,SF,SX2
conf = Config(config,inside=False)
sf = SF(conf,sample=1)
rb = RB(conf,sf,Np=360)
rb.divertor_outline(True,color=next(Color))

config = 'SFm'  # SN,X,SX,SF,SX2
conf = Config(config,inside=False)
sf = SF(conf,sample=1)
rb = RB(conf,sf,Np=360)
rb.divertor_outline(True,color=next(Color))