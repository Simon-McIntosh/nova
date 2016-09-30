import pylab as pl
import numpy as np
from itertools import cycle
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')
sns.set_palette('Set2')
Color = cycle(sns.color_palette('Set2'))
from eqConfig import Config
from radial_build import RB
from streamfunction import SF
import matplotlib.gridspec as gridspec

config = 'SX'  # SN,X,SX,SF

#pl.figure(figsize=(8.5,12))

#fig,ax = pl.subplots(1,3,figsize=(12,4.5))
fig = pl.figure(figsize=(15,5))
gd = gridspec.GridSpec(1,4,wspace=0.0, hspace=0.0)


for i,config in enumerate(['SN','X','SF','SX']):
    sns.set_palette('Set2')

    ax = pl.Subplot(fig,gd[i])
    fig.add_subplot(ax)
    fig.sca(ax)
    pl.axis('equal')
    pl.axis('off')
    pl.tight_layout()
    #pl.xlim([4,14]),pl.ylim([-13,8])
    pl.xlim([3,17]),pl.ylim([-12,10])
    
    conf = Config(config,inside=False)
    sf = SF(conf)  
    sf.contour()
    sf.plot_coils(Color)
    conf.TF(sf)
    rb = RB(conf,sf,Np=200)
    rb.divertor_outline(True)
    rb.fill(dt=conf.tfw,alpha=0.7,color=next(Color))
    rb.fill(dt=conf.BB,alpha=0.7,color=next(Color))
    rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
    rb.BBsheild_fill(dt=conf.sheild,ref_o=2/8*np.pi,alpha=0.7,
                     color=next(Color))
    rb.fill(dt=conf.VV,alpha=0.7,color=next(Color),loop=True)
    rb.set_TFbound()  # TF boundary conditions
    #rb.TFbound['ro_min'] -= 0.25
    #rb.plot_TFbounds()
    rb.TFopp(False,objF='L')  # L==length, V==volume
    rb.TFfill()
    
    pl.text(16,9.5,config,fontsize=36,ha='right',va='top')
    '''
    Color = cycle(sns.color_palette('Set2'))
    for leg in conf.targets.keys():
        R,Z = rb.sol.legs(leg)
        pl.plot(R,Z,color=next(Color))
    pl.tight_layout()
    
    col_labels=['volume','length']
    row_labels=['TF','LCFS','ratio']
    table_vals=[[r'{:1.0f}m$^3$'.format(rb.TFvol),r'{:1.1f}m'.format(rb.TFlength)],
                [r'{:1.0f}m$^3$'.format(rb.Pvol),r'{:1.1f}m'.format(rb.Plength)],
                ['{:1.2f}'.format(rb.Rvol),r'{:1.2f}'.format(rb.Rlength)]]
    row_colours = ['k']*len(row_labels)
    pl.table(cellText=table_vals,colWidths = [0.1]*3,rowLabels=row_labels,
             colLabels=col_labels,loc='bottom right',
             alpha=1,bbox=[0.8, 0.05, 0.2, 0.15])
    '''

pl.savefig('../Figs/DTT_bdrys.png',dpi=200)
