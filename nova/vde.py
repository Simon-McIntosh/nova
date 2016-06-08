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
#eqdsk = 'SN'

sf = SF(Config(eqdsk))




sf.eq['ncoil'] = 0

if eqdsk == 'vde':
    eq = EQ(sf,dCoil=1,limit=[4.25,8,-4.5,2],n=8e3)
    #eq = EQ(sf,dCoil=1,limit=[3,9,-6,6],n=5e3)
else:
    eq = EQ(sf,dCoil=1,limit=[5,13,-5.5,5],n=1e3)

eq.set_eq_psi()


levels = eq.sf.contour()


psi_o = eq.psi.copy()
psi_flux = np.linspace(0,1,500) 
eq.edgeBC()

    
def flux_fit(b,*args):
    
    eq.fP = b[0]*1e2
    eq.fF = b[1]*1e2
    '''
    b *= b_norm
    sf.Pprime = bern.spline(b[:bern.n+1])
    sf.FFprime = bern.spline(b[bern.n+1:])
    
    #sf.FFprime = bern.spline(b[:bern.n+1])
    b /= b_norm
    '''
    eq.coreBC()
    psi = eq.solve()
    err = np.sqrt(np.mean((psi-psi_o)**2))
    print(err)
    return err

'''
n=2
bern = bernstein(psi_flux,n=n)  # set up bezier curves
bPprime = bern.fit(sf.Pprime(psi_flux))[1]
bFFprime = bern.fit(sf.FFprime(psi_flux))[1]
b_norm = np.append(bPprime,bFFprime)
bo = np.ones(len(b_norm))
opp = op.minimize(flux_fit,bo,options={'disp':True},method='L-BFGS-B')
'''

opp = op.minimize(flux_fit,[1e-2,1e-2],options={'disp':True},method='SLSQP',tol=1e-12)
print('fun:{:1.8f}'.format(opp.fun))
#eq.coreBC()               

eq.psi = eq.solve()     
eq.set_eq_psi()

eq.plotb()
sf.contour(levels=levels,color=next(color),Xnorm=False)
eq.sf.Bquiver()



fig,ax = pl.subplots(2,sharex=True)
ax[0].plot(psi_flux,sf.Pprime(psi_flux))
ax[0].plot(sf.eq['pnorm'],sf.eq['pprime'],'.')
ax[1].plot(psi_flux,sf.FFprime(psi_flux))
ax[1].plot(sf.eq['pnorm'],sf.eq['ffprim'],'.')
fig.subplots_adjust(hspace=0.1)
sns.despine()

