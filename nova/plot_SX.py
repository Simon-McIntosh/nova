import pylab as pl
from streamfunction import SF
from elliptic import EQ
from inverse import INV
#from eqConfig import Config
from itertools import cycle
import numpy as np
from radial_build import RB
from shelf import PKL
from eqConfig import Config

import seaborn as sns
rc = {'figure.figsize':[8*12/16,8],'savefig.dpi':125, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)
Color = cycle(sns.color_palette('Set2'))
fig = pl.figure()
pl.axis('equal')
pl.axis('off')

pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
sf.config = 'SXex'

conf = Config('SXex')
sf.conf = conf
conf.TF(sf)
#rb = RB(conf,sf,Np=100)
#rb.divertor_outline(True,plot=False,debug=True)
#eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb},n=5e3)
#eq.set_sf_psi()  # set psi
#eq.gen()

inv.swing_fix(np.mean(inv.Swing[0]))
inv.solve_slsqp()
eq.run(update=False)
#eq.gen(Vtarget=inv.Vtarget)
    
sf.contour(lw=1)
#sf.eqwrite()

#inv.plot_coils()
#sf.plot_coils(next(Color),coils=sf.coil,label=True,plasma=False,current=False) 
sf.plot_coils(next(Color),coils=eq.coil,label=False,plasma=False) 

inv.plot_fix()



#inv.rb.sol.plot()
#sf.sol()
#Rsol,Zsol = inv.rb.sol.legs('outer')

from eqConfig import Config
conf = Config('SXex')

conf.TF(sf)
sf.conf = conf


rb = RB(conf,sf,Np=100)
'''
rb.divertor_outline(False,plot=True,debug=False)
#rb.trim_sol(plot=True)

#R,Z = rb.inloop(rb.Rb[::-1],rb.Zb[::-1],
#                sf.legs['outer']['R'][0],sf.legs['outer']['Z'][0])                                 
#print('Rex',sf.legs['outer']['R'][0][-1]/sf.Xpoint[0])

rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-3)

rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
        referance='length',color=next(Color))
rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))

rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                 alpha=0.7,color=next(Color))
rb.VVfill(dt=conf.VV,ref_o=0.25*np.pi,dref=0.25*np.pi,offset=0.5/10*np.pi,
          alpha=0.7,loop=True,color=next(Color))  # ref_o=0.385

inv.set_force_feild(multi_filament=True)
inv.get_force() 
#inv.plot_force()

a = (sf.LFPr-sf.HFPr)/2
R = (sf.LFPr+sf.HFPr)/2
AR = R/a
r95,z95 = sf.get_boundary(alpha=0.95)

ru = r95[np.argmax(z95)]  # triangularity
rl = r95[np.argmin(z95)]
del_u = (R-ru)/a
del_l = (R-rl)/a
kappa = (np.max(z95)-np.min(z95))/(2*a)

print('R',R,'a',a,'AR',AR,'del_u',del_u,'del_l',del_l,'kappa',kappa)

Pvol = rb.loop_vol(rb.Rp,rb.Zp,plot=False)
print(Pvol)


'''
rb.TFopp(False,config='SXex')  #  load TF outline
rb.TFfill()  # construct TF geometory



'''
conf.TFopp = 'L'
rb.set_TFbound()  # TF boundary conditions
#rb.TFbound['ro_min'] -= 0.5
rb.plot_TFbounds()          
rb.TFopp(True,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()  # construct TF geometory
'''

fig.tight_layout(rect=[-0.3,-0.54,1.1,1.5])
pl.savefig('../Figs/SXcoils_const.png',dpi=300)