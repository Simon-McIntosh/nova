import pylab as pl
from config import Setup
from streamfunction import SF
from radial_build import RB
from elliptic import EQ
from coils import PF

import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

pl.clf()


setup = Setup('SXex')
sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)



#eq = EQ(sf,pf,sigma=0.1,boundary=rb.get_fw(expand=0.25),n=2e4)  
#eq.plotj()

#pf.plot(coils=eq.coil,label=False,plasma=False) 

sf.contour()

pf.plot(coils=pf.coil,label=True,plasma=False,current=False) 

rb.firstwall(calc=True,plot=True,debug=True)
'''
#rb.vessel()
#rb.TFcoil(False)
#rb.trim_sol(plot=True)

'''
