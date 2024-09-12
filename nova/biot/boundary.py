"""Develop boundary transition elements."""

import jax
import jax.numpy as jnp
import optimistix as optx

from nova.jax.plasma import Plasma

from nova.imas.operate import Operate

jax.config.update("jax_enable_x64", True)


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

# kwargs = {"pulse": 17151, "run": 4, "machine": "aug"}


operate = Operate(
    **kwargs,
    tplasma="r",
    nwall=-0.2,
    ngrid=None,
    nlevelset=None,
)


# psi_residual(operate.plasma.psi)

operate.time = 300

operate.sloc["PF6", "Ic"] *= 1.1
# operate.sloc["PF4", "Ic"] *= 5.5


# operate.plasma.separatrix = {"e": [6.2, 0.5, 3, 4.6]}

operate.plasmagrid["psi"] *= 0.01

operate.plasma.solve_flux(verbose=True, f_rtol=1e-12)


def residual(psi):
    operate.plasma.psi = psi
    return operate.plasma.psi - psi


levels = -operate.plot_2d(label="DINA")[::-1]
# operate.plot()
# operate.plasma.wall.plot(limitflux=True)


# operate.plasma.plot(levels=levels, colors="black")
# operate.axes.plot(*operate.levelset(operate.plasma.psi_lcfs).T)

operate.plasma.plot(colors="C6", label="NOVA", levels=levels, nulls=True)
# operate.plasma.plot(colors="C0", levels=[operate.plasma.psi_lcfs], nulls=True)

operate.plasmagrid.plot(colors="C0", levels=[operate.plasma.psi_lcfs])


plasma = Plasma(
    operate.plasmagrid.target,
    operate.plasmawall.target,
    operate.p_prime,
    operate.ff_prime,
    jnp.array(operate.saloc["Ic"]),
    jnp.array(operate.plasma.area_),
    operate.polarity,
)


# jac = jax.jit(jax.jacfwd(plasma.residual))

psi = jnp.array(operate.plasma.psi)


# Often import when doing scientific work


def fn(psi, args):
    return plasma.residual(psi)


solver = optx.Newton(rtol=1e-3, atol=1e-3)
y0 = jnp.array(psi)
sol = optx.root_find(fn, solver, y0)

operate.plasmagrid["psi"] = sol.value[: plasma.grid.node_number]
operate.plasma.plot(colors="C0", label="optx", levels=levels, nulls=True)
operate.plasmagrid.plot(colors="C0", levels=[operate.plasma.psi_lcfs])


# sol = scipy.optimize.root(plasma.residual, psi, jac=jac, tol=1e-3)
# print(sol)

# def residual(psi):
#    return plasma.residual(psi), jac(psi)

# import jaxopt

# jaxopt.ScipyRootFinding(optimality_fun=plasma.residual, method="hybr").run(psi)


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
