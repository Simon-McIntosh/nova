"""Develop boundary transition elements."""

import numpy as np
from nova.frame.coilset import CoilSet

from scipy.optimize import newton_krylov

coilset = CoilSet(
    filename="boundary", dcoil=-1, dplasma=-500, tplasma="r", nlevelset=1e3, nwall=5
)


def build(coilset):
    coilset.firstwall.insert({"e": [6.2, 0.0, 2.7, 3.0]})
    coilset.coil.insert(5.4, [2, -2], 0.25, 0.25)

    coilset.plasma.solve()

    coilset.sloc["coil", "Ic"] = 3e3
    coilset.sloc["plasma", "Ic"] = 2e3
    coilset.store()


try:
    coilset.load()
except FileNotFoundError:
    build(coilset)

seperatrix = {"e": [5.8, 0.005, 1, 1.6]}
coilset.plasma.separatrix = seperatrix


def psi_residual(psi):
    """Return psi residual."""
    coilset.plasma.psi = psi
    coilset.plasma.separatrix = coilset.plasma.psi_lcfs
    coilset.plasmagrid.version["psi"] = None
    coilset.plasmawall.version["psi"] = None
    return np.r_[coilset.plasmagrid.psi, coilset.plasmawall.psi] - psi


newton_krylov(
    psi_residual,
    coilset.plasma.psi,
    verbose=True,
    rdiff=1e-8,
)

coilset.plasma.plot(attr="psi")
coilset.plasma.lcfs.plot()
coilset.plasmagrid.plot()
coilset.plasma.axes.plot(*coilset.levelset(coilset.plasma.psi_lcfs).T)

"""
for _ in range(10):
    coilset.plasma.separatrix = coilset.plasma.psi_lcfs
    sep = coilset.plasma.separatrix
    sep[:, 1] += 1e-3
    coilset.plasma.separatrix = sep

    coilset.plasma.set_axes("2d")
    levels = coilset.plasma.plot(attr="psi")
    coilset.plasma.lcfs.plot()
    coilset.plasmagrid.plot()
    coilset.plasma.axes.plot(*coilset.levelset(coilset.plasma.psi_lcfs).T)
"""

"""
gradient calculation for element ramp.

dpsi_dr, dpsi_dz = np.gradient(
    coilset.levelset.psi_,
    coilset.levelset.data.x,
    coilset.levelset.data.z,
)

br = -dpsi_dz / (2 * np.pi * coilset.levelset.data.x2d)
bz = dpsi_dr / (2 * np.pi * coilset.levelset.data.x2d)

coilset.plasma.axes.contour(
    coilset.levelset.data.x2d, coilset.levelset.data.z2d, bz, levels
)

grad_psi = np.linalg.norm([dpsi_dr, dpsi_dz], axis=0)

grad_psi_b = (
    2
    * np.pi
    * coilset.levelset.data.x2d
    * np.linalg.norm([coilset.levelset.bz_, -coilset.levelset.br_], axis=0)
)

coilset.plasma.set_axes("2d")
coilset.plasma.axes.contour(
    coilset.levelset.data.x2d, coilset.levelset.data.z2d, grad_psi
)
coilset.plasma.axes.contour(
    coilset.levelset.data.x2d, coilset.levelset.data.z2d, grad_psi_b
)


# coilset.plasma.axes.plot(*coilset.plasma.separatrix.T)
# coilset.plasma.axes.plot(*coilset.loc["plasma", ["x", "z"]].values.T, ".")
"""
