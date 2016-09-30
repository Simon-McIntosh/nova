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

#pkl = PKL('moveSX')

import seaborn as sns
rc = {'figure.figsize':[7*12/16,7],'savefig.dpi':175, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))
pl.figure()
pl.axis('equal')
pl.axis('off')

conf = Config('SN')
sf = SF(conf)
conf.TF(sf)
rb = RB(conf,sf,Np=500)



a = (sf.LFPr-sf.HFPr)/2
R = (sf.LFPr+sf.HFPr)/2
AR = R/a
r95,z95 = sf.get_boundary(alpha=0.95)

ru = r95[np.argmax(z95)]  # triangularity
rl = r95[np.argmin(z95)]
del_u = (R-ru)/a
del_l = (R-rl)/a
kappa = (np.max(z95)-np.min(z95))/(2*a)


pl.plot(sf.Mpoint[0],sf.Mpoint[1],'o')
pl.plot(sf.LFPr,sf.LFPz,'o')
pl.plot(sf.HFPr,sf.HFPz,'o')
pl.plot(r95,z95)
pl.plot(ru,z95[np.argmax(z95)],'o')
pl.plot(rl,z95[np.argmin(z95)],'o')

print('R',R,'a',a,'AR',AR,'del_u',del_u,'del_l',del_l,'kappa',kappa)

Pvol = rb.loop_vol(rb.Rp,rb.Zp,plot=False)
print(Pvol)

sf.contour()
#pl.plot(sf.rbdry,sf.zbdry)
#pl.plot(rb.Rp,rb.Zp)