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

pkl = PKL('moveSX_Dev2')

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

conf = Config('X')
sf.conf = conf
conf.TF(sf)
#rb = RB(conf,sf,Np=200)
#rb.divertor_outline(False,plot=False,debug=False)
eq = EQ(sf,dCoil=0.5,limit=[5,12,-8,5.5],n=1e4)
eq.set_sf_psi()  # set psi
eq.gen()

inv = INV(sf,eq,configTF='X',config='X')
Lpf = inv.grid_PF(nPF=5)
Lcs = inv.grid_CS(nCS=5)
Lo = np.append(Lpf,Lcs)
inv.eq.coils()  # re-grid
inv.update_coils()

inv.fix_boundary_psi(N=31,alpha=1-1e-4,factor=1)  # add boundary points
#inv.fix_boundary_feild(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)

R,arg = 1.5,20
target = (R,arg)
pl.plot(sf.Xpoint[0]+R*np.sin(arg*np.pi/180),
        sf.Xpoint[1]-R*np.cos(arg*np.pi/180),'s')

#inv.add_alpha(1,factor=3,polar=target)
#inv.add_B(0,[-15],factor=3,polar=target)

'''
inv.plot_fix()
inv.plot_coils()
sf.contour()
'''


Vtarget = sf.Mpoint[1]  # height of magnetic centre
Lo = inv.optimize(Lo)[1]

#sf.conf = Config('X')
#inv.write_swing()

eq.run(update=False)  
sf.contour()

#pkl.write(data={'sf':sf,'eq':eq,'inv':inv})  # pickle data
#sf,eq,inv = pkl.fetch(['sf','eq','inv'])


inv.plot_coils()
sf.plot_coils(coils=sf.coil,label=False,plasma=False,current=False) 
#sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False) 

#inv.plot_fix()



#eq.gen(Vtarget=Vtarget,Nmax=1)


#eq.run()

#sf.contour()
#eq.plasma()
#eq.plotb()
#sf.eqwrite(config='SXex')
#pl.plot(sf.rbdry,sf.zbdry,'--')



'''
for swing in np.linspace(-20,80,5):
    pl.figure()
    pl.axis('equal')
    pl.axis('off')


    inv.swing_fix(swing)
    inv.solve() 
    
    inv.update_coils(plot=True)
    sf.plot_coils(Color,coils=sf.coil,label=False,plasma=False,current=True) 
    sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False) 
 
    eq.run()
    
    sf.contour()
    eq.plasma()
    #eq.plotb()
    #sf.eqwrite(config='SXex')
    pl.plot(sf.rbdry,sf.zbdry,'--')
    inv.plot_fix()
'''


#inv.rb.sol.plot()
#sf.sol()
#Rsol,Zsol = inv.rb.sol.legs('outer')
'''
from eqConfig import Config
conf = Config('SXex')

conf.TF(sf)
rb = RB(conf,sf,Np=100)
rb.divertor_outline(True,plot=True,debug=False)


rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-3)
rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
        referance='length',color=next(Color))
rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                 alpha=0.7,color=next(Color))
rb.VVfill(dt=conf.VV,ref_o=0.25*np.pi,dref=0.25*np.pi,offset=0.5/10*np.pi,
          alpha=0.7,loop=True,color=next(Color))  # ref_o=0.385
'''

'''
print('L3D',inv.rb.sol.connection('outer',0)[-1][-1])
print('R',Rsol[-1])
print('R/X',Rsol[-1]/sf.Xpoint[0])
print('Itotal',inv.Itotal*1e-6,'MA')
print('R',rb.targets['outer']['Rsol'][-1],'Z',
      rb.targets['outer']['Zsol'][-1])
'''

'''
conf.TFopp = 'V'
rb.set_TFbound()  # TF boundary conditions
rb.TFbound['ro_min'] -= 0.5
#rb.plot_TFbounds()          
rb.TFopp(True,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()
'''

'''
pl.figure(figsize=(3.14,3.14*12/16))
pl.semilogy(I,LS,'o-',markersize=2.5)
pl.xlabel(r'current sum $I$ MA')
pl.ylabel('error m')

pl.figure(figsize=(3.14,3.14*12/16))
pl.plot(np.log10(Mix),Fun,'o-',markersize=2.5)

pl.figure(figsize=(3.14,3.14*12/16))
pl.plot(np.log10(Mix),I,'o-',markersize=2.5)

pl.figure(figsize=(3.14,3.14*12/16))
pl.plot(np.log10(Mix),LS,'o-',markersize=2.5)
'''