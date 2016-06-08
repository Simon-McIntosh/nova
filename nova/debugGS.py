import pylab as pl
import numpy as np
from streamfunction import SF
from radial_build import RB
from scipy.interpolate import interp1d as interp1
from elliptic import EQ
from inverse import INV
from scipy.interpolate import RectBivariateSpline
#import cross_coil as cc
from eqConfig import Config
from itertools import cycle
import seaborn as sns
rc = {'figure.figsize':[3.14,3.14*12/16],'savefig.dpi':250, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))

pl.figure()
pl.axis('equal')
pl.axis('off')
'''
levels = np.array([-23.42853409, -20.08160064, -16.7346672 , -13.38773376,
                   -10.04080032,  -6.69386688,  -3.34693344,   0.        ,
                   3.34693344,   6.69386688,  10.04080032,  13.38773376,
                   16.7346672 ,  20.08160064,  23.42853409])
'''

conf = Config('SN')
'''
sf = SF(conf)
eq = EQ([5,13],[-6,9],1e3,sf)
eq.get_plasma_coil()
eq.set_coil_psi_plasma()
#eq.plotb()    
sf.contour(levels=levels,color=next(Color),linetype='-')
#levels = np.copy(sf.cs.levels)
print(sf.Xpsi,sf.Mpsi)
#pl.plot(sf.Xpoint[0],sf.Xpoint[1],'o',markersize=1.5)
#pl.plot(eq.b[eq.plasma_index[:eq.Nplasma]])
'''
'''
from scipy.interpolate import griddata
r,z = np.zeros(eq.Nplasma),np.zeros(eq.Nplasma)
for indx in range(eq.Nplasma):
    i,j = eq.ij(eq.plasma_index[indx])
    r[indx] = eq.r[i]
    z[indx] = eq.z[j]
    
rbv = np.linspace(r.min(),r.max(),200)
zbv = 0*np.ones(len(rbv))
b = griddata((r,z),eq.b[eq.plasma_index[:eq.Nplasma]],
             (rbv,zbv), method='linear')
pl.plot(rbv,b)
zbv = np.linspace(z.min(),z.max(),200)
rbv = np.mean(r)*np.ones(len(zbv))
b = griddata((r,z),eq.b[eq.plasma_index[:eq.Nplasma]],
             (rbv,zbv), method='linear')
pl.plot(zbv,b)
'''



sf = SF(conf)
eq = EQ([5,13],[-6,5],1e3,sf)
#eq.get_plasma_coil()


#sf.plot_coils(Color,coils=eq.coil,label=False,plasma=True) 
      
#eq.set_coil_psi()
eq.plasma()
eq.sf.update_plasma({'r':eq.r,'z':eq.z,'psi':eq.psi_plasma})

'''
r,z = np.zeros(eq.Nplasma),np.zeros(eq.Nplasma)
for indx in range(eq.Nplasma):
    i,j = eq.ij(eq.plasma_index[indx])
    r[indx] = eq.r[i]
    z[indx] = eq.z[j]
    
rbv = np.linspace(r.min(),r.max(),200)
zbv = 0*np.ones(len(rbv))
b = griddata((r,z),eq.b[eq.plasma_index[:eq.Nplasma]],
             (rbv,zbv), method='linear')
pl.plot(rbv,b,'--')
zbv = np.linspace(z.min(),z.max(),200)
rbv = np.mean(r)*np.ones(len(zbv))
b = griddata((r,z),eq.b[eq.plasma_index[:eq.Nplasma]],
             (rbv,zbv), method='linear')
pl.plot(zbv,b,'--')
'''

sf.contour(color=next(Color),linetype='--')
levels = np.copy(sf.cs.levels)
#pl.plot(eq.b[eq.plasma_index[:eq.Nplasma]])



sf = SF(conf)
eq = EQ([5,16],[-6,5],1e3,sf)
#eq.get_plasma_coil()
#eq.set_coil_psi()

eq.plasma()
print(eq.b[eq.plasma_index[:eq.Nplasma]].min(),
      eq.b[eq.plasma_index[:eq.Nplasma]].max())
eq.sf.update_plasma({'r':eq.r,'z':eq.z,'psi':eq.psi_plasma})
sf.contour(levels=levels)


#eq.plotb()

#inv = INV(sf,eq)
#inv.plot_coils()
#sf.plot_coils(Color,coils=eq.coil,label=False,plasma=True) 
