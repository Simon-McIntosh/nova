
import numpy as np

from nova.electromagnetic.coilset import CoilSet

cs = CoilSet()

ri = '0.801 0.860 0.918 0.977 1.035 1.088 1.142 1.195 1.248 1.300 1.377'
ro = '0.853 0.911 0.970 1.028 1.081 1.134 1.188 1.241 1.294 1.346 1.799'
dz = ' 1.679 1.605 1.688 1.607 1.689 1.652 1.655 1.655 1.678 1.654 1.564'
N = '30.75 30.75 30.75 30.75 34 34 34 34 34 34 274'

ri = np.array([float(ri) for ri in ri.split()])
ro = np.array([float(ro) for ro in ro.split()])
dz = np.array([float(dz) for dz in dz.split()])
N = np.array([float(N) for N in N.split()])

x = (ri+ro) / 2
z = 0
dl = ro - ri
dt = dz

cs.add_coil(x[:-1], z, dl[:-1], dt[:-1], Nt=N[:-1], dCoil=-1,
            turn_section='square',
            turn_fraction=0.881, label='CSMC_inner')
cs.add_coil(x[-1], z, dl[-1], dt[-1], Nt=N[-1], dCoil=-1,
            turn_section='square',
            turn_fraction=0.881, name='CSMC_outer')

cs.add_mpc(cs.coil.index)
cs.plot(True, label='full')