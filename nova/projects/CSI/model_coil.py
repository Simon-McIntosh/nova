import numpy as np

from nova.frame.coilset import CoilSet

cs = CoilSet(
    dCoil=-1,
    turn_fraction=0.665,
    turn_section="skin",
    skin_fraction=0.75,
    biot_instances="grid",
)

# self.biot_instances =

ri = "0.801 0.860 0.918 0.977 1.035 1.088 1.142 1.195 1.248 1.300 1.377"
ro = "0.853 0.911 0.970 1.028 1.081 1.134 1.188 1.241 1.294 1.346 1.799"
dz = " 1.679 1.605 1.688 1.607 1.689 1.652 1.655 1.655 1.678 1.654 1.564"
N = "30.75 30.75 30.75 30.75 34 34 34 34 34 34 274"

ri = np.array([float(ri) for ri in ri.split()])
ro = np.array([float(ro) for ro in ro.split()])
dz = np.array([float(dz) for dz in dz.split()])
N = np.array([float(N) for N in N.split()])

# cs.add_coil((0.701 + 0.775) / 2, 0, 0.775 - 0.701, 1.674, Nt=31, label='CSI')

cs.add_coil(
    (1.525 + 1.427) / 4,
    0,
    0.049,
    1.589,
    Nt=8.66,
    label="CSI",
    dCoil=0.049,
    turn_section="rectangle",
)

cs.add_coil(
    (0.8005 + 1.0285) / 2, 0, 1.0285 - 0.8005, 1.674, Nt=4 * 31, label="CSMC_inner"
)
cs.add_coil(
    (1.0325 + 1.3465) / 2, 0, 1.3465 - 1.0325, 1.666, Nt=6 * 34, label="CSMC_inner"
)
cs.add_coil(
    (1.37685 + 1.79915) / 2,
    0,
    1.79915 - 1.37685,
    1.666,
    Nt=8 * 34,
    Nf=8 * 34,
    name="CSMC_outer",
)

cs.Ic = 45e3

cs.plot(True, label="full")
cs.grid.generate_grid(limit=[0.25, 2, -1.2, 1.2])
cs.grid.plot_flux()

"""


CSMC 1-4 0.8005 1.0285 1.674 4x31
CSMC 5-10 1.0325 1.3465 1.666 6x34
CSMC 11-18 1.37685 1.79915 1.666 8x34
CS Insert (1998) 0.701 0.775 1.7712 31


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
"""
