import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF
from nova.inverse import INV
from nova.TF.ripple import ripple
import numpy as np
import scipy
from time import time
import amigo.geom as geom
from nova.loops import Profile,plot_oppvar
from nova.shape import Shape
from nova.TF.DEMOxlsx import DEMO
from nova.force import force_feild

import seaborn as sns
rc = {'figure.figsize':[8,8*16/12],'savefig.dpi':80, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

nPF,nTF = 3,18
config = {'TF':'SN_dtt','eq':'SN_{:d}PF_{:d}TF'.format(nPF,nTF)}
config = {'TF':'SN_dtt','eq':'DEMO_SNb'}

setup = Setup(config['eq'])

sf = SF(setup.filename)
pf = PF(sf.eqdsk)


pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
levels = sf.contour()


rb = RB(setup,sf)
rb.firstwall(plot=False,debug=False)

profile = Profile(config['TF'],family='S',part='TF',nTF=18,obj='L',load=True)

shp = Shape(profile,obj='L',eqconf=config['eq'],load=True)  # ,nTF=18

rvv,zvv = geom.rzSLine(rb.segment['vessel']['r'],rb.segment['vessel']['z'],31)
rvv,zvv = geom.offset(rvv,zvv,0.2)
rmin = np.min(rvv)
rvv[rvv<=rmin+0.12] = rmin+0.12
shp.loop.xo['r1'] = {'value':4.486,'lb':np.min(rvv),'ub':8}  # inner radius
shp.loop.xo['upper'] = {'value':0.33,'lb':0.5,'ub':1}  
shp.loop.xo['lower'] = {'value':0.33,'lb':0.5,'ub':1}
shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
shp.plot_bounds()
shp.minimise()
#shp.loop.plot()

tf = TF(profile,sf=sf)
tf.fill()

demo = DEMO() 
#demo.fill_part('Blanket')
#demo.fill_part('Vessel')
#demo.plot_limiter()

'''
rb.vessel()
rb.trim_sol()

profile = Profile(config['TF'],family='S',part='TF')
shp = Shape(profile,obj='L',nTF=18,eqconf=config['eq'])

#shp.cage.pattern(plot=True)

rvv,zvv = geom.rzSLine(rb.loop.R,rb.loop.Z,20)
rvv,zvv = geom.offset(rvv,zvv,0.2)
rmin = np.min(rvv)
rvv[rvv<=rmin+0.12] = rmin+0.12
shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
shp.plot_bounds()
'''
#shp.minimise()
#shp.update()
#shp.tf.fill()
#shp.cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)
#plot_oppvar(shp.loop.xo,shp.loop.oppvar)
    
#demo = DEMO()
#demo.fill_loops()
#demo.get_ports(plot=True)

'''

eq = EQ(sf,pf,dCoil=0.25,sigma=0,boundary=rb.get_fw(expand=0.25),n=1e4)  
eq.gen_opp()

#ff = force_feild(pf.index,pf.coil,eq.coil,eq.plasma_coil,plot=True)

pf.plot(coils=eq.coil,label=False,plasma=True,current=False) 
sf.contour(levels=levels)

#pl.plot(sf.eqdsk['xlim'],sf.eqdsk['ylim'])

#rb = RB(setup,sf)
#rb.firstwall(calc=True,plot=True,debug=False)
#pl.axis('equal')

sf.eqwrite(pf,config=config,CREATE=True)

#pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
'''

'''
profile = Profile(config,family='S',part='TF')
shp = Shape(profile,objective='L',nTF=18)
    
rb.vessel()
rvv,zvv = geom.rzSLine(rb.loop.R,rb.loop.Z,30)
rvv,zvv = geom.offset(rvv,zvv,0.2)
shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
#shp.plot_bounds()
#shp.minimise()
shp.update()
#shp.tf.fill()



#rb.trim_sol(plot=True)
'''
'''
tf = TF(nTF=18,shape={'vessel':rb.loop,'pf':pf,'fit':True,'setup':setup,
               'plot':True,'config':config,'coil_type':'S'})  
coil = {'Rcl':tf.Rcl,'Zcl':tf.Zcl,
        'nTF':tf.nTF,'Iturn':tf.Iturn}
rp = ripple(plasma={'config':config},coil=coil)
print(config,'ripple',rp.get_ripple())
tf.fill()

#tf.coil.plot()

'''