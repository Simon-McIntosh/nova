"""Develop boundary transition elements."""

import numpy as np

from scipy.constants import mu_0
from scipy.optimize import newton_krylov

from nova.imas.operate import Operate

kwargs = {
    "pulse": 135013,
    "run": 2,
    "machine": "iter",
    "pf_passive": True,
    "pf_active": True,
}


operate = Operate(
    **kwargs,
    dplasma=-2000,
    tplasma="h",
    nwall=-0.2,
    ngrid=None,
    nlevelset=2e3,
)

operate.time = 2.5
# operate.plasma.plot()

"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

kernel = ConstantKernel(1e-8, (1e-11, 1e-3)) * RBF(
    length_scale=1.0, length_scale_bounds=(0.1, 10.0)
)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

_slice = operate.plasma._slice
gpr.fit(
    np.c_[operate.aloc["x"][_slice], operate.aloc["z"][_slice]],
    operate.aloc["nturn"][_slice],
)

operate.plasma.plot("_nturn", colors="k")


operate.plasma._nturn = gpr.predict(
    np.c_[operate.plasma._radius, operate.plasma._height]
)

operate.plasma.plot("_nturn", colors="C0", linestyles="--")

operate.grid['psi'] = gpr.predict(
    np.c_[operate.grid.data.x2d.data.flatten(), operate.grid.data.z2d.data.flatten()]
)
operate.grid.plot(nulls=False)
"""

"""    

coilset = CoilSet(
    filename="boundary", dcoil=-1, dplasma=-500, tplasma="h", nlevelset=1e3, nwall=5
)


def build(coilset):
    coilset.firstwall.insert({"e": [6.2, 0.0, 2.7, 3.0]})
    coilset.coil.insert(6.2, [2, -2], 0.25, 0.25, link=True)
    coilset.coil.insert(8, [-0.5, 0.5], 0.25, 0.25, link=True)
    coilset.coil.insert(7, [-1.5, 1.5], 0.25, 0.25, link=True, factor=-1)

    coilset.plasma.solve()
    coilset.store()


try:
    coilset.load()
except FileNotFoundError:
    build(coilset)

seperatrix = {"e": [6.2, 0, 1, 1.6]}
coilset.plasma.separatrix = seperatrix

coilset.sloc[1, "Ic"] = 1e2
coilset.sloc[2, "Ic"] = -600
coilset.sloc["plasma", "Ic"] = 2e3

"""


def nturn_residual(nturn):
    """Return nturn residual."""

    operate.plasma.nturn_[:] = nturn  # / np.sum(nturn)

    operate.plasma.update_aloc_hash("nturn")

    psi_axis = operate.plasma.psi_axis
    psi_boundary = operate.plasma.psi_boundary

    # calculate psi_norm
    psi_norm = (operate.plasmagrid.psi - psi_axis) / (psi_boundary - psi_axis)

    # set seperatrix
    operate.plasma.separatrix = operate.plasma.psi_lcfs

    # update plasma current
    psi_norm = psi_norm[operate.plasma.ionize]
    radius = operate.plasma.radius
    current_density = radius * operate.p_prime(psi_norm) + operate.ff_prime(
        psi_norm
    ) / (mu_0 * radius)
    current_density *= -2 * np.pi
    current = current_density * operate.plasma.area

    operate.plasma.nturn = current / current.sum()

    return operate.plasma.nturn_ - nturn


# operate.plasma.separatrix = {"e": [6.2, 0, 3, 4.6]}


# psi_residual(operate.plasma.psi)


def psi_residual(psi):
    """Return psi residual."""
    operate.plasma.psi = psi  # update flux map
    with operate.plasma.profile(operate.p_prime, operate.ff_prime):
        operate.plasma.separatrix = operate.plasma.psi_lcfs
        """
        psi_axis = self.psi_axis
        psi_boundary = self.psi_boundary

        # calculate psi_norm
        psi_norm = (self.grid.psi - psi_axis) / (psi_boundary - psi_axis)

        # set seperatrix
        self.separatrix = self.psi_lcfs

        # update plasma current
        psi_norm = psi_norm[self.ionize]

        current_density = self.radius * operate.p_prime(psi_norm) + operate.ff_prime(
            psi_norm
        ) / (mu_0 * self.radius)
        current_density *= -2 * np.pi
        current = current_density * self.area

        self.nturn = current / current.sum()
        """

    return operate.plasma.psi - psi


# operate.sloc["PF6", "Ic"] *= 1
# operate.sloc["PF4", "Ic"] *= 0.95

# operate.set_axes("1d")

# newton_krylov(nturn_residual, operate.plasma.nturn_, verbose=True, rdiff=1e-4)
# newton_krylov(psi_residual, operate.plasma.psi, verbose=True)


# operate.plot()
# operate.plasma.lcfs.plot()


levels = operate.plot_2d(colors="gray", label="DINA")
levels = -levels[::-1]

# operate.plasma.plot(levels=levels, colors="black")
# operate.axes.plot(*operate.levelset(operate.plasma.psi_lcfs).T)
operate.plasma.wall.plot(limitflux=True)

# newton_krylov(nturn_residual, operate.plasma.nturn_, verbose=True)
operate.plasma.psi = newton_krylov(psi_residual, operate.plasma.psi, verbose=True)
with operate.plasma.profile(operate.p_prime, operate.ff_prime):
    operate.plasma.separatrix = operate.plasma.psi_lcfs
operate.plasma.plot(levels=levels, colors="C6", label="NOVA")

operate.plot()
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
