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
from time import time
import sys
import datetime


import seaborn as sns
ar = 1.25  # figure aspect
rc = {'figure.figsize':[8/ar,8],'savefig.dpi':100, 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)

config = 'SF'
config = 'SNdtt18_6PF_5CS'
config = 'SNdtt18_5PF_5CS'
config = 'SNdtt18_4PF_5CS'
config = 'SXdtt_18TF_5PF_3CS'
pkl = PKL(config,directory='../../Movies/')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

print(config)
print('rms {:1.1f}mm'.format(1e3*inv.rms))
print('Isum {:1.1f}MA'.format(inv.Isum))


FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=5, bitrate=-1,codec='libx264',
                      extra_args=['-pix_fmt','yuv420p'])

inv.set_swing(array=np.linspace(0.5,-0.5,80))
inv.swing_flux()

pl.plot(inv.swing['flux'],inv.swing['rms'])