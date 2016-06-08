import pylab as pl
import numpy as np
from matplotlib import gridspec
import matplotlib
from itertools import cycle
from eqConfig import Config
from radial_build import RB
from streamfunction import SF
import pickle
import seaborn as sns
rc = {'figure.figsize':[4*3.14,4*3.14*3.25/10.5],'savefig.dpi':200, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

#config = 'X'  # SN,X,SX,SF,SX2

pl.figure()
pl.axis('equal')
pl.axis('off')

#LS = [50,100,130]
fig = pl.figure(1)
fs = matplotlib.rcParams['legend.fontsize']
nr,nc = 1,5
gs = gridspec.GridSpec(nr,nc,wspace=0,hspace=0)
ax = [[] for i in range(nr)]
for i in range(nr):
    for j in range(nc):
        ax[i].append(pl.subplot(gs[i,j]))
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
        #ax[i][j].axis('equal')
        ax[i][j].set_xlim([2.6000000000000001, 17.699999999999999])
        ax[i][j].set_ylim([-14.292150967273809, 9.0768966517738079])
        
        ax[i][j].axis('off')

title = ['SN','X','SF','SX','SXex']
for j,config in enumerate(['SN','X','SFm','SX8','SXex']):
    print(config)
    pl.sca(ax[0][j])
    pl.title(title[j])
    Color = cycle(sns.color_palette('Set2'))
    conf = Config(config)
    sf = SF(conf)
    
    sf.contour(lw=0.3)
    sf.plot_coils(next(Color),label=False)
    conf.TF(sf)
    rb = RB(conf,sf,Np=550)
    rb.divertor_outline(False)

    rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-4)
    rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
            referance='length',color=next(Color))
    rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
    rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                     alpha=0.7,color=next(Color))
    rb.VVfill(dt=conf.VV,ref_o=0.35*np.pi,dref=0.25*np.pi,offset=1/10*np.pi,
              alpha=0.7,loop=True,color=next(Color))

    rb.set_TFbound()  # TF boundary conditions
    rb.TFbound['ro_min'] -= 0.5
    rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
    rb.TFfill()

    col_labels=['volume','length']
    row_labels=['TF','LCFS','ratio']
    table_vals=[[r'{:1.0f}m$^3$'.format(rb.TFvol),r'{:1.1f}m'.format(rb.TFlength)],
                [r'{:1.0f}m$^3$'.format(rb.Pvol),r'{:1.1f}m'.format(rb.Plength)],
                ['{:1.2f}'.format(rb.Rvol),r'{:1.2f}'.format(rb.Rlength)]]        
    cell_colours = np.chararray(np.shape(table_vals))
    for i in range(np.shape(table_vals)[0]):
        for j in range(np.shape(table_vals)[1]):
            cell_colours[i,j] = 'w'
    pl.table(cellText=table_vals,colWidths=[0.2]*3,rowLabels=row_labels,
             colLabels=col_labels,loc='bottom right',
             alpha=1,bbox=[0.575, 0.02, 0.4, 0.17],fontsize=18,
             cellColours=cell_colours.decode())

pl.savefig('../Figs/prototypes_radial_build_numbers.png')