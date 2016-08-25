import pylab as pl
from streamfunction import SF
from elliptic import EQ
from inverse import INV
#from eqConfig import Config
from itertools import cycle
import numpy as np
from radial_build import RB
from shelf import PKL
from eqConfig import Config
from matplotlib import animation

import seaborn as sns
rc = {'figure.figsize':[8*12/16,8],'savefig.dpi':100, #*12/16
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
eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb},n=3e3)
eq.set_sf_psi()  # set psi
eq.gen()
'''

fig,ax1 = pl.subplots()

def animate(index,data,ax1): 
    print(index,inv.log['plasma_iter'][index])
    color = cycle(sns.color_palette('Set3',2+len(inv.sf.coil)))
    pl.sca(ax1)
    ax1.cla()
    ax1.set_axis_off()
    ax1.set_xlim([2,18])
    ax1.set_ylim((-15.0, 10.0))
    #pl.axis('equal')

    pl.tight_layout()
    
    #sf.coil = inv.log['sf_coil'][index]
    #sf.plasma_coil = inv.log['plasma_coil'][inv.log['plasma_iter'][index]-1]
    #eq.b = inv.log['b'][inv.log['plasma_iter'][index]-1]
    inv.eq.coil = inv.log['eq_coil'][index]
    inv.coil = inv.log['inv_coil'][index]

    #inv.set_force_feild(state='both')  # update force feild
    #inv.swing_fix(inv.Swing[0])
    #inv.solve_slsqp()

    #eq.run(update=False)
    #sf.contour(lw=1.25)

    inv.plot_coils()
    #sf.plot_coils(color=color,coils=sf.coil,label=True,plasma=False) 
    sf.plot_coils(color=color,coils=eq.coil,label=False,plasma=False) 
    #inv.plot_fix()
    #inv.rb.TFopp(False)  # L==length, V==volume
    inv.rb.TFfill()

    #971
anim = animation.FuncAnimation(fig,animate,frames=np.arange(1,10),fargs=([],ax1))

FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='SXex sweep')
writer = FFMpegWriter(fps=1, bitrate=5000, metadata=metadata)
anim.save('../Figs/SX_animation.avi',dpi=75,
          savefig_kwargs={'bboxinches':'tight'},
          writer=writer)
