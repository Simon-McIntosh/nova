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

import seaborn as sns
rc = {'figure.figsize':[6,6*12/16],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

#pkl = PKL('moveSX_Dev3')

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

'''
nS = 1e6

to = time()
cube = lhs(inv.nL,samples=int(nS))
rms = np.zeros(int(nS))
for i,c in enumerate(cube):
    if i%500 == 0:
        print(i)
    rms[i] = inv.update_position(c)
'''
#with open('../../Data/rms_cube_5.pkl', 'wb') as output:
#    pickle.dump(cube,output,-1)
#    pickle.dump(rms,output,-1)    
#print('time {:1.0f}s'.format(time()-to))


with open('../../Data/rms_cube_5L.pkl', 'rb') as input:
    cube = pickle.load(input)
    rms = pickle.load(input)
    
cube[:,:inv.nPF] = np.sort(cube[:,:inv.nPF])
cube[:,inv.nPF:] = np.sort(cube[:,inv.nPF:])

import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=20, bitrate=1000)

pca = PCA(n_components=inv.nL)
pca.fit(cube[np.argsort(rms)[:200]])
Leig = pca.components_

index = np.argsort(rms)
Lo = cube[index[200],:]

loops.denormalize_variables(Lo,inv.Lo)
inv.optimize(inv.Lo['value'])
Lopp = loops.normalize_variables(inv.Lo)



N = 10
perturb = np.linspace(0,0.1,N)
perturb = np.append(perturb,np.linspace(0.1,-0.1,2*N))
perturb = np.append(perturb,np.linspace(-0.1,0,N))


nr,nc = 2,4
fig = pl.figure(figsize=(10.5,9))
grid = gridspec.GridSpec(nr,nc,wspace=0.0,hspace=0.0)

ax = [[] for _ in range(nr*nc)]    
for i in range(nr*nc):
    ax[i] = pl.subplot(grid[i])
color = cycle(sns.color_palette('Set2',nr*nc))    

pl.sca(ax[0])       
levels = sf.contour()  # freeze contour levels

with writer.saving(fig,'../../Figs/coils_pca.wmv',100):  # wmv
    for p in perturb:
        print(p)
        for eig in range(nr*nc):  # inv.nL
            c = next(color)
        
            Lnorm = Lopp + p*Leig[eig]
            inv.update_position(Lnorm)
            pl.sca(ax[eig])
            pl.cla()
            pl.axis('equal')
            #pl.axis('off')
            #inv.plot_fix()
            pf.plot(coils=pf.coil,color=c)
            tf.fill()
            eq.run(update=False)
            sf.contour(levels=levels,plot_vac=True,lw=0.5)
            pl.plot(sf.Xpoint[0],sf.Xpoint[1],'x',color=0.25*np.ones(3),
                    mew=1.5,ms=4)
            pl.plot(2.4,-15.5,'o',alpha=0)
            pl.plot(18,8.5,'o',alpha=0)
            pl.text(sf.Mpoint[0],sf.Mpoint[1],
                    '{:1.0f}'.format(eig),fontsize=10,
                    va='center',ha='center')
        writer.grab_frame()
    


#Lnorm = cube[np.argmin(rms),:]
#rms_o = inv.update_position(Lnorm)
