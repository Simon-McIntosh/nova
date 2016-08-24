import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF

import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

setup = Setup('SN')

sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)




#eq = EQ(sf,pf,sigma=0.1,boundary=rb.get_fw(expand=0.25),n=2e3)  

#eq.plotj()

#pf.plot(coils=eq.coil,label=False,plasma=False) 

sf.contour()


pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 
rb.firstwall(calc=False,plot=True,debug=False)
rb.vessel()

#rb.trim_sol(plot=True)

shape = sf.shape_parameters()

tf = TF(setup,fit=True,loop=rb.loop,pf=pf)
#print(shape)
