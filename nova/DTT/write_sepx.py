import pylab as pl
import numpy as np
from scipy.interpolate import interp1d as interp1
from itertools import cycle
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')
sns.set_palette('Set2')
Color = cycle(sns.color_palette('Set2'))
from eqConfig import Config
from radial_build import RB
from streamfunction import SF

config = 'SN'  # SN,X,SX,SF

pl.figure(figsize=(9,12))

pl.axis('equal')
pl.xlim([5,12]),pl.ylim([-6,6])

conf = Config(config,inside=False)
sf = SF(conf,sample=1)

sf.contour()

N = 100
sf.get_boundary(alpha=1)
rb,zb = sf.eq['rbdry'],sf.eq['zbdry']
L = sf.length(rb,zb)
Lpoints = np.linspace(1/(2*(N-1)),1-1/(2*(N-1)),N)
R = interp1(L,rb)(Lpoints)
Z = interp1(L,zb)(Lpoints)

   
with open('../Data/SN_ref_bdry.txt','w') as f:
    f.write('Npoints = {:1.0f}\n\n'.format(N))
    f.write('R[m]\t\tZ[m]\n')
    for i in range(N):
        f.write('{:1.6f}\t{:1.6f}\n'.format(R[i],Z[i]))

pl.plot(rb,zb,'r')
pl.plot(R,Z,'bo')
pl.axis('off')
pl.savefig('../Figs/SN_bdry_image.png',dpi=200)

nf = 100
psi_ff = np.linspace(0,1,nf)

pl.figure(figsize=(12,9))
ax1 = pl.subplot(211)
ax2 = pl.subplot(212,sharex=ax1)

FFprime = sf.FFprime(psi_ff)
Pprime = sf.Pprime(psi_ff)

ff=np.polyfit(psi_ff,FFprime,4)
FFfit = np.zeros(np.shape(psi_ff))
for i in range(len(ff)):
    FFfit += ff[-i-1]*psi_ff**i

p=np.polyfit(psi_ff,Pprime,4)
Pfit = np.zeros(np.shape(psi_ff))
for i in range(len(p)):
    Pfit += p[-i-1]*psi_ff**i
    
C = next(Color)
ax1.plot(psi_ff,FFprime,color=C)
ax1.plot(psi_ff,FFfit,'k--')
C = next(Color)
ax2.plot(psi_ff,Pprime,color=C)
ax2.plot(psi_ff,Pfit,'k--')
pl.setp(ax1.get_xticklabels(), visible=False)
ax2.set_xlabel(r'$\phi^\prime$')
ax1.set_ylabel(r'$FF$ $^\prime$')
ax2.set_ylabel(r'$P$ $^\prime$')
pl.tight_layout()
pl.savefig('../Figs/SN_fluxfunctions.png',dpi=200)

with open('../Data/SN_ref_fluxfunctions.txt','w') as f:
    f.write('Npoints = {:1.0f}\n\n'.format(nf))
    f.write('phi\t\tFFprime\t\tPprime\n')
    for i in range(N):
        f.write('{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(psi_ff[i],
                                                     FFprime[i],Pprime[i]))

'''
pl.figure()
pl.plot(eq['pprime'][-N:])
pl.plot(Pp[-N:])
'''     