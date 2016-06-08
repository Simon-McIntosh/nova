import numpy as np
import pylab as pl
from streamfunction import SF

zo = 0  # vertical position
ro = 9.06  # major radius
A = 4.44  # aspect ratio
k = 1.61  # elongation
delta = 0.28  # triangularity

def plasma_contor(zo,ro,A,k,delta):
    theta = np.linspace(0,2*np.pi,50)
    R = ro+ro/A*np.cos(theta+delta*np.sin(theta))
    Z = zo+ro/A*k*np.sin(theta)
    return (R,Z)



pl.figure(figsize=(14,10))
pl.axis('equal')

r = np.linspace(0.1,ro/A,7)  # minor radius

for Ar in ro/r:
    R,Z = plasma_contor(zo,ro,Ar,k,delta)
    pl.plot(R,Z,'b-')

Jmax = 2e7  # [A/m^2]
coil = {}

for i,z in enumerate([-1,1]):
    name = 'plasma-'+str(i)
    coil[name] = {}
    coil[name]['I'] = 5e3
    coil[name]['r'] = ro
    coil[name]['z'] = z
    coil[name]['rc'] = (abs(coil[name]['I'])/(Jmax*np.pi))**0.5

sf = SF(coil, res=0.1, rlim=[7,18], zlim=[-4,4]) 
sf.contour(label=True) 





