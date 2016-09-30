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
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq

import seaborn as sns
rc = {'figure.figsize':[11*10/14,11],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=1.0,rc=rc)
color = sns.color_palette('Set1',12)
pl.figure()
#pl.axis('equal')
pl.axis('off')


eqdsk = 'vde'
#eqdsk = 'SN'

sf = SF(Config(eqdsk))
sf.eq['ncoil'] = 0

if eqdsk == 'vde':
    #eq = EQ(sf,dCoil=1,limit=[4.25,8,-4.5,2],n=1e5)
    #eq = EQ(sf,dCoil=1,delta=0.25,n=5e4)  #,limit=[3,9,-6,6]
    eq = EQ(sf,limit=[4,8,-4.5,1.5],n=1e5)
    #eq = EQ(sf,dCoil=1,n=1e5)
    #eq = EQ(sf,dCoil=1,limit=[-1+sf.Xpoint[0],1+sf.Xpoint[0],
    #                          -1+sf.Xpoint[1],2+sf.Xpoint[1]],n=1e3)
else:
    #eq = EQ(sf,dCoil=1,limit=[4,13,-5.5,4.75],n=5e3)
    eq = EQ(sf,delta=5)

#eq.set_eq_psi()

levels = sf.contour(lw=0.75,Nstd=5,Nlevel=50)
cref = sf.cfeild  # store referance contour
pl.plot(sf.eq['xlim']*1e-2,sf.eq['ylim']*1e-2,'k',alpha=0.75,
        linewidth=1.5)
pl.plot(sf.eq['rbdry'],sf.eq['zbdry'],'k',alpha=0.9,
        linewidth=1.5,color=color[1])
        
rbdry,zbdry = sf.get_boundary(alpha=1-1e-3)
pl.plot(rbdry,zbdry,'k',alpha=0.9,linewidth=1.5,color=color[3])
  
#pl.axis([-1+sf.Xpoint[0],3+sf.Xpoint[0],-0.5+sf.Xpoint[1],1.5+sf.Xpoint[1]])      

sigma = 0.1  # smoothing standard deviation [m]
eq.set_plasma_current()  # extract plasma current from line sep intergtal
eq.GSoper()  # apply GS operator
eq.set_fluxfunctions(sigma=0.1,update=False)

'''
eq.sparseBC()  # far field no smoothing
eq.edgeBC()  # edge
eq.psi = eq.solve()     
eq.set_eq_psi()
#sf.contour(levels=levels,color=color[0],Xnorm=False,lw=0.5)
#sf.cfeild = cref
#sf.contour(levels=levels,lw=1.5,Xnorm=False)

#eq.plotb()
eq.plotj()
pl.tight_layout()
pl.savefig('../Figs/vde_j_zoom')
'''

'''
#pl.figure()
#pl.axis('equal')
#pl.axis('off')
eq.sparseBC(sigma=sigma)  # smooth far feild
eq.edgeBC()  # re-set edge

eq.psi = eq.solve()     
eq.set_eq_psi()
sf.contour(levels=levels,color=color[0],Xnorm=False,lw=0.5)
sf.cfeild = cref
sf.contour(levels=levels,lw=1.5,Xnorm=False)
eq.plotj()
pl.tight_layout()
pl.savefig('../Figs/vde_j_smooth_zoom')
'''

#pl.figure()
#pl.axis('equal')
#pl.axis('off')
eq.sparseBC(sigma=sigma)  # smooth far feild
eq.coreBC()  # insert flux function core
eq.edgeBC()  # re-set edge

eq.psi = eq.solve()     
eq.set_eq_psi()
sf.contour(levels=levels,color=color[0],Xnorm=False,lw=0.5)
sf.cfeild = cref
sf.contour(levels=levels,lw=1.5,Xnorm=False)
eq.plotj()
pl.tight_layout()
pl.savefig('../Figs/vde_j_eq_fit_zoom')

sf.eqwrite(config='vde_highres')




 


'''
pl.figure()
pl.subplot(2,1,1)
pl.plot(eq.psi_norm,eq.FFp)
pl.plot(eq.psi_norm,sf.FFprime(psi_norm),'--')
pl.subplot(2,1,2)
pl.plot(eq.psi_norm,eq.Pp)
pl.plot(eq.psi_norm,sf.Pprime(eq.psi_norm),'--')
'''