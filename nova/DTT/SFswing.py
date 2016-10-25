import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
from eqConfig import Config
import pylab as pl
import sys
from nova.loops import Profile
from nova.config import Setup
from nova.coils import PF,TF

import seaborn as sns
ar = 0.75  # figure aspect
rc = {'figure.figsize':[8/ar,8],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)
config = 'SFm'
pkl = PKL(config)
sf,eq,inv = pkl.fetch(['sf','eq','inv'])



inv.set_swing(array=np.linspace(-0.5,0.5,10))
inv.swing_flux()
inv.solve_slsqp()  

pl.plot(2*np.pi*inv.swing['flux'],1e3*inv.swing['rms'])

pl.figure()
config = 'SFm'
setup = Setup(config)
rb = RB(setup,sf)
rb.firstwall(calc=False,plot=True,debug=False)
profile = Profile(config,family='S',part='TF',nTF=18,objective='L')
tf = TF(profile)
tf.fill()
rb.vessel()
inv.eq.pf.plot(coils=inv.eq.pf.coil,label=True,plasma=True,current=True) 
sf.contour(boundary=False)

sf.eqwrite(inv.eq.pf,config='SFm_100Vs')
