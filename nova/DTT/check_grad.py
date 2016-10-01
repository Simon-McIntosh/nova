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
from nova.coils import PF,TF
from time import time
from nova import loops
import scipy.optimize as op

config = 'SXex'
setup = Setup(config)

sf = SF(setup.filename)
rb = RB(setup,sf)
pf = PF(sf.eqdsk)

tf = TF(config,coil_type='S')

eq = EQ(sf,pf,dCoil=2.5,sigma=0,boundary=sf.get_sep(expand=0.5),n=5e2) 
eq.get_plasma_coil()
eq.run(update=False)

inv = INV(sf,eq,tf)

Lpf = inv.grid_PF(nPF=5)
Lcs = inv.grid_CS(nCS=5)
Lo = np.append(Lpf,Lcs)
inv.update_coils()

inv.fit_PF(offset=0.3)


inv.fix_boundary_psi(N=31,alpha=1-1e-4,factor=1)  # add boundary points
#inv.fix_boundary_feild(N=11,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=1,point=sf.Xpoint)

inv.set_background()
inv.get_weight()
inv.set_foreground()
inv.set_target()
inv.solve()


inv.set_Io()  # set coil current and bounds
Inorm = loops.set_oppvar(inv.Io,inv.adjust_coils)[0]

#I = np.copy(inv.I.reshape(-1))
#I = (I+30e6)/60e6


jac_approx = op.approx_fprime(Inorm,inv.frms,1e-9)
print(jac_approx)
print(inv.rms)

eps = 1e-4
for i in range(len(Inorm)):
    In = np.copy(Inorm)
    In_ = np.copy(Inorm)
    In_[i] += eps
    print((inv.frms(In_)-inv.frms(In))/eps)
'''
jac = inv.fprime(Inorm)
rms = inv.frms(Inorm)


print(jac)

print(rms,inv.rms)
#print(op.check_grad(inv.frms,inv.fprime,Inorm,epsilon=10))
'''


















