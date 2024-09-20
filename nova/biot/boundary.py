"""Develop boundary transition elements."""

import logging
from timer import timer

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from nova.jax.plasma import Plasma
from nova.imas.operate import Operate

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, force=True)

timer.set_level(logging.INFO)
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", default=".nova", help="set cache dir")
args = parser.parse_args()

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
    tplasma="h",
    nwall=-0.2,
    ngrid=5e3,
    limit=[3.0, 9.0, -6.0, 6.0],
    nlevelset=None,
    ngap=21,
    dirname=args.dirname,
)


def flux_residual(psi):
    """Return flux residual."""
    operate.plasma.psi = psi
    return operate.plasma.psi - psi


operate.time = 300

# operate.sloc["PF6", "Ic"] *= 1.3
# operate.sloc["PF4", "Ic"] *= 5.5

operate.plasma.separatrix = {"e": [6.2, 0.5, 3, 4.6]}


plasma = Plasma(
    operate.plasmagrid.target,
    operate.plasmawall.target,
    operate.p_prime,
    operate.ff_prime,
    jnp.array(operate.saloc["Ic"]),
    jnp.array(operate.plasma.area_),
    operate.plasma_index,
    operate.polarity,
)

# plot DINA solution
levels = -operate.plot_2d(
    colors="C0",
    label=f"DINA ({kwargs['pulse']}|{kwargs['run']} time={operate.time:1.0f}s)",
)[::-1]

# solve Newton-Krylov
# with timer("plasma.solve_flux"):
#    operate.plasma.solve_flux(verbose=True, f_rtol=1e-1, f_tol=1e-1)

print(f"Krylov solve {optx.two_norm(flux_residual(operate.plasma.psi))}")

operate.grid.plot(
    colors="k",
    label="Jacobian-Free Newton-Krylov",
    levels=levels,
    nulls=False,
)

# Newton solve
operate.time = 300
solver = optx.Newton(rtol=1e-1, atol=1e-1)
psi = jnp.array(operate.plasma.psi)

with timer("optx.root_find"):
    sol = optx.root_find(lambda psi, args: plasma.residual(psi), solver, psi)

print(f"Newton L2 {optx.two_norm(flux_residual(sol.value))}")

# plot optx sol
plasma_current = plasma.plasma_current(sol.value)
operate.plasma.ionize = np.array(plasma_current, bool)
operate.plasma.nturn = np.array(plasma_current)[plasma_current != 0]
operate.plasma.nturn /= np.sum(operate.plasma.nturn)
operate.grid.plot(
    colors="C6",
    linestyles="--",
    label="Auto-diff Newton (jax, optimistix)",
    levels=levels,
    nulls=False,
)

operate.grid.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14))

# operate.plasmagrid["psi"] = sol.value[: plasma.grid.node_number]
# operate.plasma.plot()
# operate.plasmagrid.plot(colors="C0", levels=[operate.plasma.psi_lcfs])


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
