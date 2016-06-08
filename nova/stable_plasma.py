import pylab as pl
import numpy as np
from streamfunction import SF
from radial_build import RB
from scipy.interpolate import interp1d as interp1
from elliptic import EQ
#import cross_coil as cc
from eqConfig import Config
from itertools import cycle
import seaborn as sns
rc = {'figure.figsize':[3.14*12/16,3.14],'savefig.dpi':300, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

pl.figure()
pl.axis('equal')
pl.axis('off')

conf = Config('SFm')
sf = SF(conf)
sf.contour(Nstd=1.5,color=next(Color))

eq = EQ([4,13],[-6,6],1e4,sf)
eq.Vtarget = sf.Mpoint[1]

rbdry,zbdry = sf.get_boundary()  # update boundary
pl.plot(rbdry,zbdry)
pl.plot(sf.Mpoint[0],sf.Mpoint[1],'o',markersize=1)

eq.gen(Verr=1e-4)

sf.contour(levels=sf.cs.levels,color=next(Color))
rbdry,zbdry = sf.get_boundary()  # update boundary
pl.plot(rbdry,zbdry)
pl.plot(sf.Mpoint[0],sf.Mpoint[1],'ko',markersize=1)

pl.imshow(eq.b.reshape((eq.nr,eq.nz)).T, cmap='RdBu',
          vmin=eq.b.min(),vmax=eq.b.max(),
          extent=[eq.r[0],eq.r[-1],eq.z[0],eq.z[-1]],
          interpolation='nearest', origin='lower')


