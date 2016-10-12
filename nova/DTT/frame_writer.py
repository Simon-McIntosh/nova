import numpy as np
import matplotlib
matplotlib.use("Agg")
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

import seaborn as sns
ar = 16/8  # figure aspect
rc = {'figure.figsize':[8/ar,8],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)
config = 'SX54'
pkl = PKL(config)
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=5, bitrate=1000)

def animate(index): 
    #color = cycle(sns.color_palette('Set3',2+len(inv.pf.coil)))
    pl.sca(ax)
    ax.cla()
    ax.set_axis_off()
    
    pl.plot(-1,-13.5,'o',alpha=0)
    pl.plot(22,8,'o',alpha=0)
    
    inv.update_position(inv.log['Lo'][index]['norm'],update_area=True)
    inv.swing_flux(inv.swing['flux'][1])  
    inv.solve_slsqp()  

    eq.run(update=False,verbose=False)
    sf.contour(lw=1.25,boundary=False)
    inv.plot_fix()
    
    if inv.log['plasma_iter'][index] != inv.log['plasma_iter'][index-1] or\
    index ==inv.log['position_iter'][-1]-1:
        inv.eq.pf.plasma_coil = inv.log['plasma_coil'][inv.log['plasma_iter'][index]-1]
        plasma = True
        print('GS {:1.0f} of {:1.0f}'.format(inv.log['plasma_iter'][index],
                                             inv.log['plasma_iter'][-1]))
    else:
        plasma = False
    inv.eq.pf.plot(coils=inv.eq.pf.coil,label=False,plasma=plasma,current=True) 
    text = 'Plasma update: {:1.0f}\n'.format(inv.log['plasma_iter'][index])
    text += 'Position update: {:1.0f}\n'.format(index)
    text += 'Current update: {:1.0f}\n'.format(inv.log['current_iter'][index])
    pl.text(11.5,6.5,text)
    
    inv.tf.fill()
    
fig,ax = pl.subplots()
ax.axis('equal')
with writer.saving(fig,'../../Figs/{}.wmv'.format(config),100):  # wmv
    for i in np.arange(0,inv.log['position_iter'][-1]):  # 
        animate(i)
        writer.grab_frame()
       