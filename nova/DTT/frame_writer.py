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
ar = 1.25#14/10.0 # figure aspect
rc = {'figure.figsize':[8/ar,8],'savefig.dpi':100, 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=1.5,rc=rc)

config = 'SF'
config = 'SNdtt18_6PF_5CS'
config = 'SNdtt18_5PF_5CS'
config = 'SNdtt18_4PF_5CS'
pkl = PKL(config,directory='../../Movies/')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

print(config)
print('rms {:1.1f}mm'.format(1e3*inv.rms))
print('Isum {:1.1f}MA'.format(inv.Isum))

'''
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=5, bitrate=-1,codec='libx264',
                      extra_args=['-pix_fmt','yuv420p'])

def animate(index): 
    #color = cycle(sns.color_palette('Set3',2+len(inv.pf.coil)))
    pl.sca(ax)
    ax.cla()
    #ax.set_axis_off()
    
    pl.plot(2,-13,'o',alpha=0)
    pl.plot(19,11,'o',alpha=0)
    
    inv.update_position(inv.log['Lo'][index]['norm'],update_area=True)
    inv.swing_flux(inv.swing['flux'][0])  
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
    inv.eq.pf.plot(coils=inv.eq.pf.coil,label=False,plasma=plasma,current=False) 
    text = 'Plasma update: {:1.0f}\n'.format(inv.log['plasma_iter'][index])
    text += 'Position update: {:1.0f}\n'.format(index)
    text += 'Current update: {:1.0f}\n'.format(inv.log['current_iter'][index])
    #pl.text(11.5,9.0,text)
    inv.tf.fill()
    pl.xlim([2,19])
    pl.ylim([-13,11])
    
    if index%1 == 0 and index > 0:
        elapsed = time()-to
        remain = int((nS-index)/index*elapsed)
        prog_str = '\r{:1.0e}'.format(index)
        prog_str += ' elapsed {:0>8}s'.format(str(\
        datetime.timedelta(seconds=int(elapsed))))
        prog_str += ' remain {:0>8}s'.format(str(\
        datetime.timedelta(seconds=remain)))
        prog_str += ' complete {:1.1f}%'.format(1e2*i/nS)
        nh = int(i/nS*width)
        prog_str += ' |'+nh*'#'+(width-nh)*'-'+'|'
        sys.stdout.write(prog_str)
        sys.stdout.flush()
    
to = time()
nS = inv.log['position_iter'][-1]
width = 35
    
fig,ax = pl.subplots()

with writer.saving(fig,'../../Movies/{}.mp4'.format(config),100):  
    for i in np.arange(0,nS):  
        animate(i)
        #pl.tight_layout()
        writer.grab_frame()
'''