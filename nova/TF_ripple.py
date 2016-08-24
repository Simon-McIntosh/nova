import pylab as pl
from nova.config import Setup
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.elliptic import EQ
from nova.coils import PF,TF,loop_vol
import amigo.geom as geom
import numpy as np
from inductance import neumann
from time import time
import sys
import datetime
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

for conf in ['SN','SXex']:
    setup = Setup(conf)
    tf = TF(setup)
    
    L = geom.length(tf.Rmid,tf.Zmid,norm=False)[-1]
    V = loop_vol(tf.Rmid,tf.Zmid)
    print('L {:1.2f}m, V {:1.0f}m3'.format(L,V))
    
    pl.plot(tf.Rmid,tf.Zmid)
    pl.axis('equal')
    
 



    
    