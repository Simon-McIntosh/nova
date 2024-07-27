"""Develop boundary transition elements."""

from nova.imas.operate import Operate

kwargs = {
    "pulse": 135013,
    "run": 2,
    "machine": "iter",
    "pf_passive": True,
    "pf_active": True,
}

"""
kwargs = {
    "pulse": 57410,
    "run": 0,
    "machine": "west",
    "pf_passive": {"occurrence": 0},
    "pf_active": {"occurrence": 0},
}
"""


operate = Operate(
    **kwargs,
    tplasma="h",
    nwall=-0.2,
    ngrid=None,
    nlevelset=2e3,
)


# psi_residual(operate.plasma.psi)

operate.time = 250

# operate.sloc["PF6", "Ic"] *= 1
# operate.sloc["PF4", "Ic"] *= 0.95


# operate.plasma.separatrix = {"e": [6.2, 0.5, 3, 4.6]}
operate.plasma.solve_flux(verbose=True)

levels = operate.plot_2d(label="DINA")
levels = -levels[::-1]

# operate.plasma.plot(levels=levels, colors="black")
# operate.axes.plot(*operate.levelset(operate.plasma.psi_lcfs).T)


operate.plasma.wall.plot(limitflux=True)
operate.plasma.plot(colors="C6", label="NOVA", levels=levels, nulls=False)


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
