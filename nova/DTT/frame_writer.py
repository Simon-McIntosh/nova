import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from streamfunction import SF
from elliptic import EQ
from inverse import INV
#from eqConfig import Config
from itertools import cycle
import numpy as np
from radial_build import RB
from shelf import PKL
from eqConfig import Config
import pylab as pl

import seaborn as sns
ar = 16/12  # figure aspect
rc = {'figure.figsize':[8/ar,8],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)

pkl = PKL('moveSX_dev')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
sf.config = 'SXex'
'''
conf = Config('SXex')
sf.conf = conf
conf.TF(sf)
rb = RB(conf,sf,Np=100)
rb.divertor_outline(False,plot=False,debug=False)
eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb},n=5e3)
eq.set_sf_psi()  # set psi
eq.gen()
'''

FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=5, bitrate=1000)

fig,ax = pl.subplots()
inv.plot_fix()

def animate(index): 
    color = cycle(sns.color_palette('Set3',2+len(inv.sf.coil)))
    pl.sca(ax)
    ax.cla()
    ax.set_axis_off()
    
    scale = 1.05
    ylim,yoffset = np.array([-13.5,8]),0.25
    xlim,xoffset = np.array([0,np.diff(ylim)/ar]),2.25
    ax.set_ylim(yoffset+scale*ylim)
    ax.set_xlim(xoffset+scale*xlim)

    inv.update_position(inv.log['Lo'][index])
    inv.swing_fix(np.mean(inv.Swing))  
    inv.solve_slsqp()
    inv.update_area()  
    eq.run(update=False)
    sf.contour(lw=1.25)
    inv.plot_fix()

    if inv.log['plasma_iter'][index] != inv.log['plasma_iter'][index-1] or\
    index ==inv.log['position_iter'][-1]-1:
        sf.plasma_coil = inv.log['plasma_coil'][inv.log['plasma_iter'][index]-1]
        plasma = True
        print('GS {:1.0f} of {:1.0f}'.format(inv.log['plasma_iter'][index],
                                             inv.log['plasma_iter'][-1]))
    else:
        plasma = False
    sf.plot_coils(color=color,coils=eq.coil,label=False,plasma=plasma)
    
    text = 'Plasma update: {:1.0f}\n'.format(inv.log['plasma_iter'][index])
    text += 'Position update: {:1.0f}\n'.format(index)
    text += 'Current update: {:1.0f}\n'.format(inv.log['current_iter'][index])
    pl.text(11.5,6.5,text)
    
    inv.rb.TFfill()
    
    
with writer.saving(fig,'../Figs/SXdev1.wmv',100):
    for i in np.arange(0,inv.log['position_iter'][-1]): #inv.log['position_iter'][-1]
        animate(i)
        writer.grab_frame()
        
        
'''
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='SXex sweep')
writer = FFMpegWriter(fps=1, bitrate=5000, metadata=metadata)
anim.save('../Figs/SX_animation.avi',dpi=75,
          savefig_kwargs={'bboxinches':'tight'},
          writer=writer)
'''