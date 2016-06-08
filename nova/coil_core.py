import numpy as np
import pylab as pl
from cross_coil import Bcoil, Bcoil_b
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

radius = 0.5
delta = 0.1
nr = np.int(radius/delta)
r_line = np.linspace(0,radius,nr)

rm,zm,theta,radius = np.array([]), np.array([]), np.array([]), np.array([])
rc,zc = 2,0

coil = {}
coil['extra'] = {'I':1e6, 'r':2, 'z':2, 'rc':0.5}
I = 1e6

for i,rL in enumerate(r_line):
    if rL == 0:
        rm = np.array([rc])
        zm = np.array([zc])
        radius = np.zeros(1)
        theta = np.zeros(1)
    if rL > 0:
        n_theta = np.int(2*np.pi*rL/delta)
        thetaL = np.linspace(0,2*np.pi,n_theta,endpoint=False)
        rm = np.append(rm,rc+rL*np.cos(thetaL))
        zm = np.append(zm,zc+rL*np.sin(thetaL))
        theta = np.append(theta, thetaL)
        radius = np.append(radius, rL*np.ones(n_theta))
Icoil = I/len(theta)
for i,r,z in zip(range(len(theta)), rm, zm):        
    coil[str(i)] = {'I':Icoil, 'r':r, 'z':z, 'rc':delta}
    
from streamfunction import SF
sf = SF(coil, res=0.05, rlim=[0.3,3.5], zlim=[-1.5,1.5])

pl.figure(figsize=(9,9))
sf.Bcontor('r', Nstd=3)
pl.plot(rm, zm, 'bo')     
pl.axis('equal') 
pl.xlim([1.5,2.5])   
pl.grid()
pl.savefig(fig_path+'coil_Br.png', dpi=300)

pl.figure(figsize=(9,9))
sf.Bcontor('z', Nstd=3)
pl.plot(rm, zm, 'bo')         
pl.axis('equal')       
pl.xlim([1.5,2.5])       
pl.grid()
pl.savefig(fig_path+'coil_Bz.png', dpi=300)

pl.figure(figsize=(9,9))
sf.Bstreamfunction()
sf.contour(Nstd=3)

nx = 30
#r = np.linspace(0.5, 3.5, nx)
z = np.linspace(-1.5, 1.5, nx)
r = 2*np.ones(nx)

Br,Bz = [],[]
Brb,Bzb = [],[]
for i in range(nx):
    B = Bcoil(coil, [r[i],z[i]])
    Br.append(B[0])
    Bz.append(B[1])
    
    Bb = Bcoil_b(coil, [r[i],0,z[i]])
    Brb.append(Bb[0])
    Bzb.append(Bb[2])
    
coil = {}    
coil['extra'] = {'I':1e6, 'r':2, 'z':2, 'rc':0.5}
coil['centre'] = {'I':I, 'r':2, 'z':0, 'rc':0.5}
Bcr,Bcz = [],[]
for i in range(nx):
    B = Bcoil(coil, [r[i],z[i]])
    Bcr.append(B[0])
    Bcz.append(B[1])

        
pl.figure(figsize=(9,6))
pl.plot(z, Br, 'b-')
pl.plot(z, Bz, 'b--')
pl.plot(z, Bcr, 'k-')
pl.plot(z, Bcz, 'k--')
pl.plot(z, Brb, 'r-')
pl.plot(z, Bzb, 'r--')
#pl.ylim((0,0.2))
#pl.xlim((1,3))
pl.grid()
pl.savefig(fig_path+'coil_Btrace.png', dpi=300)


