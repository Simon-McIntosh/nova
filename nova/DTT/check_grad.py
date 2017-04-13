import pylab as pl
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
import nova.cross_coil as cc
from nova.coils import PF, TF
from time import time
from nova import loops
import scipy.optimize as op
from amigo.addtext import linelabel
from nova.loops import Profile

config = 'SN'
setup = Setup(config)

sf = SF(setup.filename)
rb = RB(setup, sf)
pf = PF(sf.eqdsk)

profile = Profile(config, family='S', part='TF', nTF=18, objective='L')
tf = TF(profile)

eq = EQ(sf, pf, dCoil=2.5, sigma=0, boundary=sf.get_sep(expand=0.5), n=5e2)
eq.get_plasma_coil()
eq.run(update=False)

inv = INV(sf, eq, tf)
inv.initialize_log()

Lpf = inv.grid_PF(nPF=5)
Lcs = inv.grid_CS(nCS=5)
Lo = np.append(Lpf, Lcs)
inv.update_coils()

inv.fit_PF(offset=0.3)


inv.fix_boundary_psi(N=31, alpha=1 - 1e-4, factor=1)  # add boundary points
# inv.fix_boundary_feild(N=11,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1, point=sf.Xpoint)

inv.set_background()
inv.get_weight()
inv.set_foreground()
inv.set_force_feild(state='both')
inv.set_target()

inv.solve()

inv.set_Io()  # set coil current and bounds


#I = np.copy(inv.I.reshape(-1))
#Inorm = (Inorm+30e6)/60e6

eps = 1e-8
inv.I -= 0.05


inv.get_force()


#jac_approx = op.approx_fprime(inv.I,inv.set_force,eps)
# print(jac_approx)
# print(inv.set_force(inv.I,grad=True))


#jac_approx = op.approx_fprime(inv.I.reshape(-1),inv.get_r,eps)

jac_approx = op.approx_fprime(inv.I, inv.get_rms, eps)
print(jac_approx)

# print(inv.rms)

grad = np.zeros(inv.nC)
rms = inv.frms(inv.I, grad)
print(grad)
print('rms_frms {:1.6f}'.format(rms))

# inv.get_rms()


inv.initialize_log()
inv.set_Lo(Lo)  # set position bounds
Lnorm = loops.normalize_variables(inv.Lo)

inv.update_position(Lnorm)

for name in inv.PF_coils:
    print('dA {:1.4f}'.format(pf.coil[name]['dr'] * pf.coil[name]['dz']))

print('L grad2', op.approx_fprime(Lnorm, inv.update_position, 1e-6))

for name in inv.PF_coils:
    print('dA {:1.4f}'.format(pf.coil[name]['dr'] * pf.coil[name]['dz']))

print('L grad', op.approx_fprime(Lnorm, inv.update_position, 1e-6))

for name in inv.PF_coils:
    print('dA {:1.4f}'.format(pf.coil[name]['dr'] * pf.coil[name]['dz']))

print('L grad3', op.approx_fprime(Lnorm, inv.update_position, 1e-6))


Neps = 7
jac = np.zeros((Neps, inv.nL))
eps = 10**np.linspace(-8, -2, Neps)


for i, e in enumerate(eps):
    #j1[i] = op.approx_fprime(alpha,inv.get_rms,e)[j]
    inv.update_position(Lnorm)
    jac[i, :] = op.approx_fprime(Lnorm, inv.update_position, e)

text = linelabel(Ndiv=30, value='')
for name, j in zip(inv.Lo['name'], jac.T):
    pl.plot(eps, j)
    text.add(name)

pl.xscale('log')
text.plot()
# pl.yscale('log')

print('L grad', op.approx_fprime(Lnorm, inv.update_position, 1e-6))
