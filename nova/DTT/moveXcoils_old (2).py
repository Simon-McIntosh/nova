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
from scipy.interpolate import interp1d as interp1

pkl = PKL('moveX')

import seaborn as sns
rc = {'figure.figsize':[7*12/16,7],'savefig.dpi':150, #*12/16
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

pl.plot(sf.rbdry,sf.zbdry)
eq = EQ(sf,dCoil=1,limit=[4,15.5,-13,5.5],n=1e4)


eq.get_plasma_coil()
#eq.set_coil_psi()
#eq.get_Xpsi()
#eq.get_Mpsi()

inv = INV(sf,eq,configTF='SN',config='SN')
inv.fix_boundary_psi(N=25,alpha=0.99,factor=1)  # add boundary points

inv.add_null(factor=1,point=sf.Xpoint)
inv.add_alpha(1,factor=1,point=sf.Xpoint)
#inv.add_alpha(1,factor=1,point=(12,-8.5)) 
#inv.add_alpha(0.95,factor=1,point=(8.5,-8.5))
#inv.add_alpha(1,factor=1,Lin=0.2,norm=-0.5)


sf.sol()
'''
Np = [4,6]
for key,npt in zip(inv.rb.targets.keys(),Np):
    leg = sf.legs[key]
    R,Z = leg['R'][0],leg['Z'][0]
    L = sf.length(R,Z)
    Lp = np.linspace(1/(2*npt),1-1/(2*npt),npt)
    Rp,Zp = interp1(L,R)(Lp),interp1(L,Z)(Lp)
    for r,z in zip(Rp,Zp):
        inv.add_alpha(1,factor=1,point=(r,z))

L,norm = 0.225,-0.5     
inv.add_null(factor=10,Lin=L,norm=norm)
inv.add_alpha(1,factor=10,Lin=L,norm=norm)
'''
inv.get_weight()        
inv.plot_fix()

Lpf = inv.grid_PF(nPF=5)
Lcs = inv.grid_CS(nCS=7)
Lo = np.append(Lpf,Lcs)
#Lo = Lpf

inv.eq.coils(dCoil=inv.eq.dCoil)  # re-grid 
inv.update_coils()  

Lo = inv.optimize(Lo)[1]  
#inv.optimize_rms(Lo) 

print('Isum',inv.Isum*1e-6,'rms',inv.rms)

sf.conf = Config('SXex')

eq.run(update=False)  
sf.contour()

inv.update_coils(plot=True)
sf.plot_coils(next(Color),coils=sf.coil,label=False,plasma=False,current=True) 
sf.plot_coils(next(Color),coils=eq.coil,label=False,plasma=False) 

Vtarget = sf.Mpoint[1]  # height of magnetic centre
#eq.gen(Vtarget=Vtarget,Nmax=1)
#eq.run()

#sf.contour()
eq.plasma()
eq.plotb()
pl.plot(sf.rbdry,sf.zbdry,'--')

#sf.eqwrite(config='SXex')
pkl.write(data={'sf':sf,'eq':eq,'inv':inv})  # pickle data
#sf,eq,inv = pkl.fetch(['sf','eq','inv'])

