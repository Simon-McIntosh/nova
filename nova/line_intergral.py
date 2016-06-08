import pylab as pl
from streamfunction import SF
from elliptic import EQ
from inverse import INV
from eqConfig import Config
from itertools import cycle
import numpy as np
from radial_build import RB
import copy
from shelf import PKL
import cross_coil as cc
import scipy as sp
from surface import bernstein
from scipy.interpolate import interp1d
import scipy.optimize as op

import seaborn as sns
rc = {'figure.figsize':[7*12/14,7],'savefig.dpi':110, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = cycle(sns.color_palette('Set2'))
pl.figure()
pl.axis('equal')
pl.axis('off')


eqdsk = 'vde'
eqdsk = 'SN'

sf = SF(Config(eqdsk))
sf.eq['ncoil'] = 0

if eqdsk == 'vde':
    eq = EQ(sf,dCoil=1,limit=[4.25,8,-4.5,2],n=1e3)
    #eq = EQ(sf,dCoil=1,limit=[3,9,-6,6],n=5e3)
else:
    eq = EQ(sf,dCoil=1,limit=[5,13,-5.5,5],n=5e4)

    
eq.set_eq_psi()
levels = eq.sf.contour()

I = 0
R,Z = sf.get_boundary(1-1e-4)
tR,tZ,R,Z = cc.tangent(R,Z,norm=False)
for r,z,tr,tz in zip(R,Z,tR,tZ):
    B = sf.Bcoil((r,z))
    t = np.array([tr,tz])
    I += np.dot(B,t)
I /= cc.mu_o   


print(I*1e-6)
