import pylab as pl
import numpy as np
from streamfunction import SF
from radial_build import RB
from scipy.interpolate import interp1d as interp1
from elliptic import EQ
#import cross_coil as cc
from eqConfig import Config
from itertools import cycle
import seaborn as sns
rc = {'figure.figsize':[3.14*12/16,3.14],'savefig.dpi':350, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))


pl.figure()
pl.axis('equal')
pl.axis('off')

conf = Config('SX')
sf = SF(conf)
'''
zcs_space = np.linspace(6,-8,13)
rcs_space = 2.9*np.ones(len(zcs_space))
pl.plot(rcs_space,zcs_space,'-o',markersize=1.5)

for i in range(5):
    del sf.coil['Coil{:1.0f}'.format(i)]    
   
for i in range(len(zcs_space)):
    name = 'Coilcs{:1.0f}'.format(i)
    sf.coil[name] = {'r':rcs_space[i],'z':zcs_space[i],'dr':1,
                     'dz':abs(np.diff(zcs_space))[0],'I':1}
'''    
   
rb = RB(conf,sf,Np=150)
eq = EQ([4,15.5],[-12,6.5],5e4,sf)
eq.get_plasma_coil()
#sf.plasma_coils(N=11,dL=0.25)
eq.coils(delta=0.25)  # multi-filiment coils 

r,z = sf.get_boundary(alpha=0.99)
L = sf.length(r,z)
Lc = np.linspace(0,1,1)[:-1]

fix = {}
fix['r'],fix['z'] = interp1(L,r)(Lc),interp1(L,z)(Lc)
fix['value'] = eq.psi_coil(fix['r'],fix['z'])
fix['BC'] = np.array(['psi']*len(fix['r'])) 
'''
r,z = sf.get_boundary(alpha=0.45)
L = sf.length(r,z)
Lc = np.linspace(0,1,31)[:-1]
fix['r'] = np.append(fix['r'],interp1(L,r)(Lc))
fix['z'] = np.append(fix['z'],interp1(L,z)(Lc))
fix['value'] = np.append(fix['value'],eq.psi_coil(interp1(L,r)(Lc),
                                               interp1(L,z)(Lc)))
fix['BC'] = np.append(fix['BC'],np.array(['psi']*len(Lc)))
'''



'''
rx,zx = sf.Xpoint
rx += 1
zx -= 1
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],eq.Pcoil(sf.Xpoint[0],sf.Xpoint[1]))
fix['BC'] = np.append(fix['BC'],'psi')
'''

'''

'''


rx,zx = sf.Xpoint
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Br')
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Bz')


'''
zx = 6
Xpsi = eq.Pcoil(sf.Xpoint[0],sf.Xpoint[1])
Mpsi = eq.Pcoil(sf.Mpoint[0],sf.Mpoint[1])
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],1.05*(Xpsi-Mpsi))
fix['BC'] = np.append(fix['BC'],'psi')
'''


rx,zx = 5.5,5.5
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Br')

fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Bz')

 
'''
eq = EQ([4,12.5],[-9.5,6.5],1e4,sf)
sf.plasma_coils(N=11,dL=0.25)
eq.coils(delta=0.25)  # multi-filiment coils
'''
sf.coil['Coil5']['r']+=2 
sf.coil['Coil5']['z']-=1.5
'''
sf.coil['Coil6']['r']+=0 
sf.coil['Coil6']['z']-=3.2

sf.coil['Coil7']['r']-=0.6 
sf.coil['Coil7']['z']-=2.5

sf.coil['Coil8']['r']-=0.2 
sf.coil['Coil8']['z']+=0.5
'''

sf.coil['Coil13']['r'] = 4.5
sf.coil['Coil13']['z'] = -10.5
sf.coil['Coil13']['dr'] = 1
sf.coil['Coil13']['dz'] = 1
sf.coil['Coil13']['I'] = 1

sf.coil['Coil10']['r']+=3.0
sf.coil['Coil10']['z']-=0.75
sf.coil['Coil10']['dr']=1
sf.coil['Coil10']['dz']=1

sf.coil['Coil8']['dr']=1
sf.coil['Coil8']['dz']=2
sf.coil['Coil8'] ['r']+=0
sf.coil['Coil8'] ['z']+=1.5

#sf.coil['Coil11']['z']-=1.5 
#sf.coil['Coil11']['r']+=5.5 
#sf.coil['Coil12']['r']-=1.5 
#sf.coil['Coil12']['z']-=5.5

sf.coil['Coil9']['r']-=4.5#4.5 
sf.coil['Coil9']['z']+=2.5#3
sf.coil['Coil9']['dr'] = 2
sf.coil['Coil9']['dz'] = 0.5
'''
sf.coil['Coil13'] = {'r':sf.coil['Coil9']['r'],
                     'z':sf.coil['Coil9']['z']-2,
                     'dr':2,
                     'dz':sf.coil['Coil9']['dz'],'I':1}
'''

fx = 0.65
rx = sf.Xpoint[0]+fx*(sf.coil['Coil10']['r']-sf.Xpoint[0])
zx = sf.Xpoint[1]+fx*(sf.coil['Coil10']['z']-sf.Xpoint[1])
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Br')

fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Bz')


fx = 0.75
rx = sf.coil['Coil4']['r']+fx*(sf.coil['Coil13']['r']-sf.coil['Coil4']['r'])
zx = sf.coil['Coil4']['z']+fx*(sf.coil['Coil13']['z']-sf.coil['Coil4']['z'])
'''
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],eq.Pcoil(sf.Xpoint[0],sf.Xpoint[1]))
fix['BC'] = np.append(fix['BC'],'psi')
'''
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Br')


sf.coil['Coil11'] = {'r':sf.coil['Coil9']['r']+5.35,
                     'z':sf.coil['Coil9']['z']-1.5,'dr':1.0,'dz':1.0,'I':1}
sf.coil['Coil12'] = {'r':sf.coil['Coil11']['r']-1.5,
                     'z':sf.coil['Coil11']['z']-1.75,'dr':1.0,'dz':1,'I':1}
'''
fx = 0.5
rx = sf.coil['Coil9']['r']+fx*(sf.coil['Coil11']['r']-sf.coil['Coil9']['r'])
zx = sf.coil['Coil9']['z']+fx*(sf.coil['Coil11']['z']-sf.coil['Coil9']['z'])
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
'''

fx = 0.5
rx = sf.coil['Coil12']['r']+fx*(sf.coil['Coil11']['r']-sf.coil['Coil12']['r'])
zx = sf.coil['Coil12']['z']+fx*(sf.coil['Coil11']['z']-sf.coil['Coil12']['z'])
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],eq.psi_coil(sf.Xpoint[0],sf.Xpoint[1]))
fix['BC'] = np.append(fix['BC'],'psi')

fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],5e2)
fix['BC'] = np.append(fix['BC'],'Br')
'''
fix['r'] = np.append(fix['r'],rx)
fix['z'] = np.append(fix['z'],zx)
fix['value'] = np.append(fix['value'],0)
fix['BC'] = np.append(fix['BC'],'Bz')
'''

del sf.coil['Coil9']
#del sf.coil['Coil10']
#del sf.coil['Coil11']
#del sf.coil['Coil12']
#del sf.coil['Coil13']



eq.coils(delta=0.25)  # multi-filiment coils 

conf.TF(sf)
rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()

eq.coil_currents(fix)
sf.plot_coils(Color,coils=sf.coil,label=False,plasma=False,current=True)  
sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False)  

    
eq.update_psi() # levels=sf.cs.levels
sf.contour(Nstd=1.5)

#sf.sol(plot=True)


eq = EQ([3,14],[-10,10],8e3,sf)
#eq = EQ([2,16],[-10,10],2e3,sf)

Mtarget = sf.Mpoint[1]
pl.plot(sf.Mpoint[0],sf.Mpoint[1],'o',markersize=1)

sf.contour(Nstd=1.5)


#sf.coil['Coil8']['I'] *= 1.25
#eq.update_psi()

#sf.contour(Nstd=1.5)

rbdry,zbdry = sf.get_boundary()  # update boundary
pl.plot(rbdry,zbdry,'r')


'''
conf = Config('SXex')
rb = RB(conf,sf,Np=550)
rb.divertor_outline(True)
'''
'''
rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color),s=2e-3)
rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
        referance='length',color=next(Color))
rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,offset=1/10*np.pi,
                 alpha=0.7,color=next(Color))
rb.VVfill(dt=conf.VV,ref_o=0.385*np.pi,dref=0.15*np.pi,offset=1/10*np.pi,
          alpha=0.7,loop=True,color=next(Color))

print(rb.targets['outer']['theta']*180/np.pi)
'''