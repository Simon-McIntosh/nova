import numpy as np
import pylab as pl
#from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import matplotlib.cm as cm
import pickle
from nova import loops
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from itertools import cycle
import matplotlib.gridspec as gridspec

from pyDOE import lhs
import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
from nova.coils import PF,TF
import pickle
from nova import loops
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from time import time
from nova.loops import Profile
import sys
import datetime
import matplotlib.animation as manimation
import matplotlib


import seaborn as sns
rc = {'figure.figsize':[6,6*12/16],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

#pkl = PKL('moveSX_Dev3')

nPF,nCS,nTF = 4,3,18
config = {'TF':'dtt','eq':'SN'}
config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)
setup = Setup(config['eq'])
sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)

profile = Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L',
                  load=True)
    
tf = TF(profile,sf=sf)
eq = EQ(sf,pf,dCoil=2.0,limit=[5.2,13,-6.5,6],n=5e3)  # 
eq.get_plasma_coil()
eq.run(update=False)

inv = INV(sf,eq,tf)
Lpf = inv.grid_PF(nPF=nPF)
Lcs = inv.grid_CS(nCS=nCS,Zbound=[-12,8],gap=0.1)
Lo = np.append(Lpf,Lcs)
inv.update_coils()
inv.fit_PF(offset=0.3)

'''
inv.fix_boundary_psi(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=3,point=sf.Xpoint)
Rex,arg = 1.5,40
R = sf.Xpoint[0]*(Rex-1)/np.sin(arg*np.pi/180)
target = (R,arg)
inv.add_alpha(1,factor=1,polar=target)  # 20
inv.add_B(0,[-15],factor=1,polar=target)  # -30
'''
inv.fix_boundary_psi(N=25,alpha=1-1e-4,factor=1)  # add boundary points
#inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)

inv.set_background()
inv.get_weight()
inv.set_foreground()
inv.fix_flux(inv.swing['flux'][0])

inv.set_swing()
inv.plot_fix(tails=True)
inv.fix_flux(inv.swing['flux'][1]) 
inv.set_Lo(Lo)
inv.initialize_log()


'''
config = 'SXex'
setup = Setup(config)
sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)
tf = TF(config,coil_type='S')
eq = EQ(sf,pf,dCoil=2.0,sigma=0,limit=[5.2,12,-10,5],n=1e3) 
eq.get_plasma_coil()
eq.run(update=False)

inv = INV(sf,eq,tf)
Lpf = inv.grid_PF(nPF=5)
Lcs = inv.grid_CS(nCS=3,Zbound=[-9,7],gap=0.1)
Lo = np.append(Lpf,Lcs)
inv.update_coils()

inv.fit_PF(offset=0.3)
inv.fix_boundary_psi(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=3,point=sf.Xpoint)
Rex,arg = 1.5,40
R = sf.Xpoint[0]*(Rex-1)/np.sin(arg*np.pi/180)
target = (R,arg)
inv.add_alpha(1,factor=1,polar=target)  # 20
inv.add_B(0,[-15],factor=1,polar=target)  # -30

inv.set_background()
inv.get_weight()
inv.set_foreground()
inv.fix_flux(inv.swing['flux'][0])

inv.set_swing(centre=15)
inv.set_Lo(Lo)
inv.initialize_log()


nPF = 5

'''

nS = 3e5
filename = 'rms_cube_{}_{:1.0e}'.format(config['TF'],nS).replace('+','')
with open('../../BigData/{}.pkl'.format(filename), 'rb') as input:
    cube = pickle.load(input)
    rms = pickle.load(input)
    
nPF = 4
nL = np.shape(cube)[1]   

cube[:,:nPF] = np.sort(cube[:,:nPF])
cube[:,nPF:] = np.sort(cube[:,nPF:])

index = np.argsort(rms)
rms = rms[index]
cube = cube[index,:]

#print(np.where(rms>0.2)[0])

pca = PCA(n_components=nL)
pca.fit(cube[:200])
Leig = pca.components_

nSearch,nSelect = 50,3
search = cube[:nSearch]
select = np.zeros((nSelect,nL))
index = np.zeros(nSelect,dtype=int)
delta = np.zeros(nSelect)
select_rms = np.zeros(nSelect)

for j in range(nSelect):
    if j == 0:
        select[0] = cube[0]
        search = np.delete(search,0,axis=0)    
    else:
        for i in range(j):  # compare to previously selected
            error = np.sqrt(np.mean((select[i]-search)**2,axis=1))
            index[i] = np.argmax(error)
            delta[i] = error[index[i]]
        imax = np.argmax(delta[:j])
        select_rms[j] = delta[imax]
        select[j] = search[index[imax]]
        search = np.delete(search,index[imax],axis=0)


fig = pl.figure(figsize=(8,3))
grid = gridspec.GridSpec(1,nSelect,wspace=0.0,hspace=0.0)
ax = [[] for _ in range(nSelect)]    
for i in range(nSelect):
    ax[i] = pl.subplot(grid[i])
color = cycle(sns.color_palette('Set2',nSelect))
pl.sca(ax[0])       
levels = sf.contour()  # freeze contour levels
pl.cla()
Mpoint = np.copy(sf.Mpoint)
for i in range(nSelect):
    loops.denormalize_variables(select[i],inv.Lo)
    rms = inv.optimize(inv.Lo['value'])
    Lopp = loops.normalize_variables(inv.Lo)
    
    pl.sca(ax[i])
    pf.plot(coils=pf.coil,color=next(color))
    tf.fill()
    eq.run(update=False)
    sf.contour(levels=levels,plot_vac=True,lw=0.5)
    pl.text(Mpoint[0],Mpoint[1],'{:1.0f}mm'.format(1e3*inv.rms),
            fontsize=10, va='center',ha='center')
    print()
    print(select[j])
    print(Lopp)

pl.savefig('../../Figs/Decorolate_pca.png')