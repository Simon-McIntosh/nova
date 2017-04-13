import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.coils import PF, TF
from nova import loops
from nova.loops import Profile
from nova.shelf import PKL
from amigo import geom
from scipy.interpolate import griddata

import seaborn as sns
rc = {'figure.figsize': [7 * 10 / 16, 7], 'savefig.dpi': 150,  # *12/16
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 0.75}
sns.set(context='paper', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)
Color = cycle(sns.color_palette('Set2'))

nPF, nCS, nTF = 5, 5, 18
config = {'TF': 'dtt', 'eq': 'SFm'}
config['TF'] = '{}{}{:d}'.format(config['eq'], config['TF'], nTF)
setup = Setup(config['eq'])
sf = SF(setup.filename)

rb = RB(setup, sf)
pf = PF(sf.eqdsk)
tf = TF(Profile(config['TF'], family='S',
                part='TF', nTF=nTF, obj='L', load=True))

eq = EQ(sf, pf, dCoil=2.0, sigma=0, n=1e4,
        boundary=sf.get_sep(expand=0.75),
        rmin=sf.Xpoint[0] - 3.5, zmin=sf.Xpoint[1] - 3.5)
eq.gen_opp()

theta, dXpoint = -5 * np.pi / 180, [0.5, -1]
pivot = np.argmin(sf.rbdry)  # rotate/translate eqdsk
rp, zp = sf.rbdry[pivot], sf.zbdry[pivot]
TRT = geom.rotate2D(theta, xo=rp, yo=zp)[1]  # trial rotate/translate
zref = sf.Mpoint[1]
dz = zref - np.dot(TRT, np.append(sf.Mpoint, 1))[1]  # correct z-shift
R, TRT = geom.rotate2D(theta, xo=rp, yo=zp, dy=dz)

points = np.zeros((eq.nr * eq.nz, 2))
values = eq.psi.flatten()
for i, r in enumerate(eq.r):
    for j, z in enumerate(eq.z):
        # rotate psi grid
        points[i * eq.nz + j, :] = np.dot(TRT, np.array([r, z, 1]))
sf.xo = np.dot(TRT, np.append(sf.xo, 1))  # rotate/translate X-point sead
sf.xo[0] -= 0.5
dXpoint = np.dot(R, dXpoint)  # rotate X-point deta
eq.psi = griddata(points, values, (eq.r2d, eq.z2d),
                  method='cubic', fill_value=0)
eq.set_eq_psi()
eq.plasma()  # update core-b and plasma coils

# eq.plotb()
'''
sf.contour(boundary=False)
pl.plot(sf.xo[0],sf.xo[1],'o')
pl.plot(sf.Xpoint[0],sf.Xpoint[1],'d')
inv.plot_fix(tails=True)
'''
# pf.plot(coils=pf.coil,label=True,plasma=True,current=True)

inv = INV(sf, eq, tf)
Lpf = inv.grid_PF(nPF=nPF)
Lcs = inv.grid_CS(nCS=nCS, Zbound=[-10.5, 7], gap=0.1)
L = np.append(Lpf, Lcs)
inv.update_coils()
inv.update_coils()
inv.fit_PF(offset=0.3)

inv.fix_boundary_psi(N=25, alpha=1 - 1e-4, factor=1)  # add boundary points
# inv.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1, point=sf.Xpoint)
inv.add_alpha(1, factor=1, point=sf.Xpoint, label='psi_x')

'''
#sf.get_Xpsi(xo=sf.Xpoint+dXpoint)
target = (0.75,30)
inv.add_null(factor=1,polar=target)
#inv.add_alpha(1+0.1,factor=1,polar=target,label='psi_x')
#sf.get_Xpsi(xo=sf.Xpoint-dXpoint)
'''

sf.get_Xpsi(xo=sf.Xpoint + dXpoint)
inv.add_null(factor=1, point=sf.Xpoint)
inv.add_alpha(1, factor=1, point=sf.Xpoint, label='psi_x')
sf.get_Xpsi(xo=sf.Xpoint - dXpoint)

inv.set_swing(width=150)  # width=363
inv.update_limits(LCS=[-11.5, 11.5])

for i in range(2):
    L = inv.optimize(L)
    inv.fix_flux(inv.swing['flux'][1])
    inv.solve_slsqp()
    eq.gen_opp()
    # eq.get_Vcoil()
    # eq.gen(ztarget=zref)

sf.contour(boundary=False, plot_vac=True)
pf.plot(coils=pf.coil, label=True, plasma=True, current=True)
# inv.plot_fix(tails=True)
rb.firstwall(mode='calc', plot=True, debug=False)
rb.trim_sol()
tf.fill()

loops.plot_variables(inv.Io, scale=1, postfix='MA')
loops.plot_variables(inv.Lo, scale=1)

sf.eqwrite(pf, config=config['TF'] + '_{:d}PF_{:d}CS'.format(inv.nPF, inv.nCS),
           CREATE=True)

#pkl = PKL('SF',directory='../../Movies/')
# pkl.write(data={'sf':sf,'eq':eq,'inv':inv})  # pickle data


'''
inv.fix_flux(inv.swing['flux'][1]) 
inv.solve_slsqp()
#eq.run()
eq.gen_opp()

pf.plot(coils=inv.eq.pf.coil,label=True,plasma=True,current=True)
tf.fill()
sf.contour(boundary=False)
pl.plot(sf.Xpoint[0],sf.Xpoint[1],'d')
inv.plot_fix(tails=True)
'''
