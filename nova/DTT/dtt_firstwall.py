import pylab as pl
from nova.config import Setup,select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF
from nova.inverse import INV
from nova.TF.ripple import ripple
import numpy as np
from time import time
import amigo.geom as geom
from nova.loops import Profile,plot_oppvar
from nova.shape import Shape
from nova.DEMOxlsx import DEMO
from nova.force import force_feild
from nova.firstwall import divertor,main_chamber

from amigo.IO import trim_dir
from nova.shelf import PKL

import seaborn as sns
rc = {'figure.figsize':[5,5*16/12],'savefig.dpi':150, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)



eq_names = ['DEMO_SN_SOF','DEMO_SN_EOF'] 
mc = main_chamber('DTT',date='2017_03_06')  
mc.generate(eq_names,psi_n=1.07,flux_fit=True,plot=True)
mc.load_data(plot=True)  # load from file

setup = Setup(eq_names[0])             
sf = SF(setup.filename)   


div = divertor(sf,setup)
div.place(debug=False)
r,z = div.join(mc)

pl.plot(r,z)


rb = RB(sf,setup)
rb.firstwall()

