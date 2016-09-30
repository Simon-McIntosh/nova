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

config = 'X'  # SN,X,SX,SF
conf = Config(config,inside=False)
sf = SF(conf)
rb = RB(conf,sf,Np=150)
eq = EQ([2,16],[-10,9],5000,sf)
sf.plasma_coils(N=7,dL=0.25)
eq.coils(delta=0.25)  # multi-filiment coils 
#sf.contour()


r,z = sf.get_boundary(alpha=0.95)
L = sf.length(r,z)
Lc = np.linspace(0,1,21)[:-1]

fix = {}
fix['r'],fix['z'] = interp1(L,r)(Lc),interp1(L,z)(Lc)
fix['value'] = eq.Pcoil(fix['r'],fix['z'])
fix['BC'] = np.array(['psi']*len(fix['r'])) 
'''
fix['r'] = np.append(fix['r'],8.6089790197)
fix['z'] = np.append(fix['z'],-7.12320200125)
fix['value'] = np.append(fix['value'],eq.Pcoil(8.608,-7.123))
fix['BC'] = np.append(fix['BC'],'psi')
'''
fix['r'] = np.append(fix['r'],8.6089790197)
fix['z'] = np.append(fix['z'],-7.12320200125)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Br')

fix['r'] = np.append(fix['r'],8.6089790197)
fix['z'] = np.append(fix['z'],-7.12320200125)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Bz')

#rc = np.append(rc,sf.coil['Coil12']['r'])
#zc = np.append(zc,(sf.coil['Coil12']['z']+sf.coil['Coil13']['z'])/2)
#psi_c = np.append(psi_c,eq.Pcoil(sf.Xpoint[0],sf.Xpoint[1]))

#sf.coil['Coil11']['r']+=3 
sf.coil['Coil11']['z']-=1.5 

sf.coil['Coil12']['r']+=1.0 
sf.coil['Coil12']['z']-=1.5 
#del sf.coil['Coil11']
#del sf.coil['Coil12']
#del sf.coil['Coil13']


conf.TF(sf)
rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()

'''
r,z = rb.offset(rb.R,rb.Z,0.75)
r,z = rb.rzInterp(r,z,Np=35,ends=False)

keys = list(sf.coil.keys())
for key in keys:
    if 'Coil' in key:
        del sf.coil[key]
#sf.coil = {}
dr,dz=1,1
for i,(rcoil,zcoil) in enumerate(zip(r,z)):
    sf.coil['Coil'+str(i)] = {}
    sf.coil['Coil'+str(i)]['I'] = 1
    sf.coil['Coil'+str(i)]['r'] = rcoil
    sf.coil['Coil'+str(i)]['z'] = zcoil
    sf.coil['Coil'+str(i)]['dr'] = dr
    sf.coil['Coil'+str(i)]['dz'] = dz
sf.plot_coils(Color,coils=sf.coil,label=True,plasma=False) 
'''

eq.coils(delta=0.25)  # multi-filiment coils 


#sf.coil['Coil10']['r']+=1 
#sf.coil['Coil5']['r']+=1 

sf.plot_coils(Color,coils=sf.coil,label=True,plasma=False)  

  
sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False)  


eq.coil_currents(fix)
    
#sf.coil['Coil11']['I']*=0.8 

eq.update_psi() # levels=sf.cs.levels
sf.contour(Nstd=1.5)

sf.get_Xpsi()
rx,zx = sf.Xpoint
#rx+=3

pl.plot(rx,zx,'o',markersize=1.5)
Bp = sf.Bcoil([rx,zx])
Bx = np.sqrt(Bp[0]**2+Bp[1]**2)
print(Bp[0],Bp[1],sf.Xpsi)


Br,Bz,PsiX = 0,0,0
mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

pl.plot(rx,zx,'o',markersize=1.5)
for sub_name in eq.coil.keys():
    r,z = eq.coil[sub_name]['r'],eq.coil[sub_name]['z']
    I = eq.coil[sub_name]['I']
    Br += mu_o*I*eq.green_feild(rx,zx,r,z,'Br')
    Bz += mu_o*I*eq.green_feild(rx,zx,r,z,'Bz')
    PsiX += 2*np.pi*mu_o*I*eq.greens(rx,zx,r,z)
                                               
print(Br,Bz,PsiX)

#sf.sol(plot=True)
#sf.get_legs()

#pl.plot(sf.Xpoint[0],sf.Xpoint[1],'o',markersize=2)
#sf.plot_coils(Color,coils=eq.coil,label=False,plasma=True)


'''
for coil in eq.coil.keys():
        if 'Coil11' in coil:
            ro,zo = eq.coil[coil]['r'],eq.coil[coil]['z']
            Fr,Fz = cc.Fcoil(eq.coil,coil)
            Fmax = np.max(np.sqrt(Fr**2+Fz**2))
            pl.arrow(eq.coil[coil]['r'], eq.coil[coil]['z'], 
             Fr/Fmax, Fz/Fmax, fc="r", ec="r",
             head_width=0.1, head_length=0.15)
             
print('Fr',Fr,'Fz',Fz)
'''

'''
eq.coils(delta=0.25)  # multi-filiment coils
Fr_sum,Fz_sum,M = 0,0,0
with open('../Data/Coil11_forces.vi','w') as f:
    f.write('r[m]\tz[m]\tFr[MN]\tFz[MN]\n')
    for coil in eq.coil.keys():
        if 'Coil11' in coil:
            Fr,Fz = cc.Fcoil(eq.coil, coil)
            Fr_sum += Fr[0]
            Fz_sum += Fz[0]
            r,z = eq.coil[coil]['r'][0],eq.coil[coil]['z'][0]
            R = np.array([r-ro,z-zo])
            F = np.array([Fr[0],Fz[0]])
            M += np.cross(R,F)  # moment
            f.write('{:1.3f}\t'.format(eq.coil[coil]['r'][0]))
            f.write('{:1.3f}\t'.format(eq.coil[coil]['z'][0]))
            f.write('{:1.2f}\t'.format(Fr[0]*1e-6))
            f.write('{:1.2f}\n'.format(Fz[0]*1e-6))
            pl.arrow(eq.coil[coil]['r'][0], eq.coil[coil]['z'][0], 
             Fr[0]/Fmax, Fz[0]/Fmax, fc="k", ec="k",
             head_width=0.075, head_length=0.1)   

pl.arrow(ro,zo,Fr_sum/Fmax, Fz_sum/Fmax, fc="b", ec="b",
         head_width=0.1, head_length=0.15)            
sf.plot_coils(Color,coils=eq.coil,label=False)

#pl.xlim([10,14])
#pl.ylim([-7,-5])
pl.xlabel(r'$R$ [m]')
pl.ylabel(r'$Z$ [m]')
pl.savefig('../Figs/Coil_vectors.png',dpi=200)
'''