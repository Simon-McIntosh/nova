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
inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
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

nS = 3e5
filename = 'rms_cube_{}_{:1.0e}'.format(config['TF'],nS).replace('+','')
'''
width = 35
to = time()
cube = lhs(inv.nC+1,samples=int(nS))
rms = np.zeros(int(nS))
for i,c in enumerate(cube):
    if i%10 == 0 and i > 0:
        elapsed = time()-to
        remain = int((nS-i)/i*elapsed)
        prog_str = '\r{:1.0e}'.format(i)
        prog_str += ' elapsed {:0>8}s'.format(str(\
        datetime.timedelta(seconds=int(elapsed))))
        prog_str += ' remain {:0>8}s'.format(str(\
        datetime.timedelta(seconds=remain)))
        prog_str += ' complete {:1.1f}%'.format(1e2*i/nS)
        nh = int(i/nS*width)
        prog_str += ' |'+nh*'#'+(width-nh)*'-'+'|'
        sys.stdout.write(prog_str)
        sys.stdout.flush()
    rms[i] = inv.update_position(c)
with open('../../BigData/{}.pkl'.format(filename), 'wb') as output:
    pickle.dump(cube,output,-1)
    pickle.dump(rms,output,-1)    
print('')
'''
with open('../../BigData/{}.pkl'.format(filename), 'rb') as input:
    cube = pickle.load(input)
    rms = pickle.load(input)
#cube[:,:inv.nPF] = np.sort(cube[:,:inv.nPF])
#cube[:,inv.nPF:] = np.sort(cube[:,inv.nPF:])

pca = PCA(n_components=inv.nL)
pca.fit(cube[np.argsort(rms)[:50]])
Leig = pca.components_

#with open('../../Data/{}_Leig.pkl'.format(config),'wb') as output:
#    pickle.dump(Leig,output,-1)

index = np.argsort(rms)
Lo = cube[index[0],:]
loops.denormalize_variables(Lo,inv.Lo)
inv.optimize(inv.Lo['value'])
Lopp = loops.normalize_variables(inv.Lo)

#Lopp = cube[index[0],:]
#Lopp = [ 0.08480096,  0.37595893,  0.686798  ,  0.9607147 ,  0.05430536,
#        0.24645849,  0.40159608,  0.87988169]

N,L = 5,0.1
perturb = np.linspace(0,L,N,endpoint=False)
perturb = np.append(perturb,np.linspace(L,-L,2*N,endpoint=False))
perturb = np.append(perturb,np.linspace(-L,0,N,endpoint=False))

nr,nc = 2,4

figwidth = 15  # 25
ar = 25/16.5
matplotlib.rcParams['figure.figsize'] = (figwidth,figwidth/ar)

fig = pl.figure()

grid = gridspec.GridSpec(nr,nc,wspace=0.0,hspace=0.0)

ax = [[] for _ in range(nr*nc)]    
for i in range(nr*nc):
    ax[i] = pl.subplot(grid[i])
    ax[i].axis('equal')
#grid.tight_layout(fig)  
  
color = cycle(sns.color_palette('Set2',nr*nc))    

pl.sca(ax[0])       
levels = sf.contour()  # freeze contour levels
Mpoint = np.copy(sf.Mpoint)


moviename = '../../Movies/{}'.format(filename)
moviename += '.mp4'
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=10, bitrate=5000,codec='libx264',
                      extra_args=['-pix_fmt','yuv420p'])

with writer.saving(fig,moviename,100):  
    for p in perturb:
        print(p)
        for eig in range(nr*nc):
            c = next(color)
            if eig < len(Leig): 
                Lnorm = Lopp + p*Leig[eig]
                inv.update_position(Lnorm)
                pl.sca(ax[eig])
                pl.cla()
                
                #pl.axis('off')
                #inv.plot_fix()
                pf.plot(coils=pf.coil,color=c)
                tf.fill()
                eq.run(update=False)
                sf.contour(levels=levels,plot_vac=True,lw=0.5)
                pl.plot(sf.Xpoint[0],sf.Xpoint[1],'x',color=0.25*np.ones(3),
                        mew=1.5,ms=4)
                pl.plot(2,-10,'o',alpha=0)
                pl.plot(18,10,'o',alpha=0)
                pl.text(Mpoint[0],Mpoint[1],'{:1.0f}'.format(eig),fontsize=10,
                        va='center',ha='center')
            else:
                pl.sca(ax[eig])
                pl.cla()
                pl.axis('off')
        writer.grab_frame()
    


#Lnorm = cube[np.argmin(rms),:]
#rms_o = inv.update_position(Lnorm)
