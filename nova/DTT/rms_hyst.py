import pickle
import numpy as np
from sklearn.decomposition import PCA
import pylab as pl
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec

import seaborn as sns
rc = {'figure.figsize':[6,6*10/16],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
colors = sns.color_palette('Set2',5)


nPF,nCS,nTF = 4,3,18
config = {'TF':'dtt','eq':'SN'}
config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)

nS = 3e5
filename = 'rms_cube_{}_{:1.0e}'.format(config['TF'],nS).replace('+','')
#filename = 'rms_cube_5L'
with open('../../BigData/{}.pkl'.format(filename), 'rb') as input:
    cube = pickle.load(input)
    rms = pickle.load(input)
#cube[:,:nPF] = np.sort(cube[:,:nPF])
#cube[:,nPF:] = np.sort(cube[:,nPF:])

pca = PCA(n_components=(nPF+nCS+1))
pca.fit(cube[np.argsort(rms)[:50]])
Leig = pca.components_


nr,nc = 3,3
fig = pl.figure()
grid = gridspec.GridSpec(nr,nc,wspace=0.1,hspace=0.1)

label = ['Coil0','Coil1','Coil2','Coil3','CS0','CS1','CS2','CS3']
ax = [[] for _ in range(nr*nc)]    
for i,c in enumerate(cube.T):
    ax[i] = pl.subplot(grid[i])
    ax[i].axis('off')
    pl.plot(c,rms,'k.',ms=0.5)
    pl.text(0,1,label[i],color=0.9*np.ones(3))
pl.savefig('../../Figs/'+filename+'_points'+'.png')
#grid.tight_layout(fig)  

'''

pl.hist(1e3*rms,1000,linewidth=0.01,color=colors[0])
pl.xlabel('rms mm')
pl.ylabel('frequency')
sns.despine()
pl.savefig('../../Figs/'+filename+'_hyst_full'+'.png')


pl.figure()
pl.hist(1e3*rms,1000,linewidth=0.05,color=colors[1])
pl.xlim([0,500])
pl.ylim([0,500])
pl.xlabel('rms mm')
pl.ylabel('frequency')
sns.despine()
pl.savefig('../../Figs/'+filename+'_hyst_part'+'.png')

pl.figure()
pl.hist(1e3*rms[rms<0.2],10,linewidth=0.1,color=colors[2])
pl.xlabel('rms mm')
pl.ylabel('frequency')
sns.despine()
pl.savefig('../../Figs/'+filename+'_hyst_zoom'+'.png')
'''