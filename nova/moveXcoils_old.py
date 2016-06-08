import pylab as pl
import numpy as np
from streamfunction import SF
from radial_build import RB
from scipy.interpolate import interp1d as interp1
from elliptic import EQ
import cross_coil as cc
from eqConfig import Config
from itertools import cycle
import seaborn as sns
rc = {'figure.figsize':[3.14*12/16,3.14],'savefig.dpi':400, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))


pl.figure()
pl.axis('equal')
pl.axis('off')

conf = Config('X')
sf = SF(conf)
rb = RB(conf,sf,Np=150)
eq = EQ([4,12.5],[-9.5,6.5],5e4,sf)
sf.plasma_coils(N=11,dL=0.25)
eq.coils(delta=0.25)  # multi-filiment coils 

r,z = sf.get_boundary(alpha=0.95)
L = sf.length(r,z)
Lc = np.linspace(0,1,21)[:-1]

fix = {}
fix['r'],fix['z'] = interp1(L,r)(Lc),interp1(L,z)(Lc)
fix['value'] = eq.Pcoil(fix['r'],fix['z'])
fix['BC'] = np.array(['psi']*len(fix['r'])) 

rx,zx = 8.9,-8

fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],eq.Pcoil(sf.Xpoint[0],sf.Xpoint[1]))
fix['BC'] = np.append(fix['BC'],'psi')


fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Br')

fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Bz')
 
sf.coil['Coil10']['r']-=2 
sf.coil['Coil10']['z']+=0.9
 
sf.coil['Coil11']['z']-=1.5 
sf.coil['Coil12']['r']+=0.8 
sf.coil['Coil12']['z']-=1.6 
eq.coils(delta=0.25)  # multi-filiment coils 

conf.TF(sf)
rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()

sf.plot_coils(Color,coils=sf.coil,label=True,plasma=False)  
sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False)  
eq.coil_currents(fix)
    
eq.update_psi() # levels=sf.cs.levels
sf.contour(Nstd=1.5)

#sf.sol(plot=True)

conf = Config('Xex')
rb = RB(conf,sf,Np=550)
rb.divertor_outline(True)

rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-3)
rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
        referance='length',color=next(Color))
rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                 alpha=0.7,color=next(Color))
rb.VVfill(dt=conf.VV,ref_o=0.385*np.pi,dref=0.15*np.pi,offset=1/10*np.pi,
          alpha=0.7,loop=True,color=next(Color))

print(rb.targets['outer']['theta']*180/np.pi)


    def copy_coil(self,coil_read):
        coil_copy = {}
        for strand in coil_read.keys():
            coil_copy[strand] = {}
            coil_copy[strand] = coil_read[strand].copy()
        return coil_copy
        
                    self.log['sf_coil'].append(self.copy_coil(self.sf.coil).copy())
            self.log['eq_coil'].append(self.copy_coil(self.eq.coil).copy())
            self.log['b'].append(self.eq.b.copy())
            
            self.log['inv_coil'].append([])
            self.log['inv_coil'][-1] = {}
            for state in ['active','passive']:
                self.log['inv_coil'][-1][state] = \
                self.copy_coil(self.coil[state]).copy()