import numpy as np
import pylab as pl
from cross_coil import Pcoil, mu_o, Bmin
from config import layout
from extract import extract_geom
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
fig_path = '../Figs/'

config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 0.3  # node spaceing [m]
geom.plasma_coils(I, delta)
coil = geom.coil

fig = pl.figure(figsize=(9, 12))
ax = fig.add_subplot(111)



Xpoint = Bmin(mu_o, coil, [8,-4])
Xphi = Pcoil(mu_o, coil, Xpoint)
C=pl.contour(rm, zm, phi, levels=[Xphi])

p = C.collections[0].get_paths()
N = len(p)
for i in range(N):
    v = p[i].vertices
    z = v[:,1]
    if (z>0).any() and (z<0).any():
        LCFS = i
        
v = p[LCFS].vertices
r,z = (v[:,0],v[:,1])
index = z>Xpoint[1]
r,z = (r[index], z[index])
R = np.mean([np.min(r), np.max(r)])
theta = np.unwrap(np.arctan2(z,r-R))

pl.plot(r, z, 'b-')

from scipy.interpolate import interp1d
fLFSr = interp1d(theta, r)  #, kind='cubic'
fLFSz = interp1d(theta, z)
LFSr,LFSz = fLFSr(0),fLFSz(0)
pl.plot(LFSr,LFSz, 'ko')

LCFS_phi = Pcoil(mu_o, coil, [LFSr-1e-3,LFSz])
pl.contour(rm, zm, phi, levels=[LCFS_phi], colors='r')

sol_phi = []
for dsol in np.linspace(0,10,100)*1e-3:
    sol_phi.append(Pcoil(mu_o, coil, [LFSr+dsol,LFSz]))

pl.contour(rm, zm, phi, levels=sol_phi)

#pl.contour(rm, zm, phi, levels=np.linspace(Xphi, Xphi-0.5))

xnull = np.array([[8,-4],[9,4],[11,-4],[10,-8],
                  [11.5,-6],[4.5,-7],[8,-9.5],[14,2]])
for xo in xnull:
    xmin = Bmin(mu_o, coil, xo)               
    pl.plot(xmin[0], xmin[1], 'co')      


plot = layout(geom)
plot.plasma()
#plot.plasma_coils()
plot.sol()  # scrape-off layer 
plot.plates()  # divertor plates
plot.TF()  # TF coil
plot.coil_fill()  
#plot.coil_sheild() 
#plot.coil_label() 
pl.axis('equal')
pl.xlim(rlim)
pl.ylim(zlim)
pl.savefig(fig_path+config+'_sol.png', dpi=600)
         