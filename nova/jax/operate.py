"""Manage jax backed operator classes."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import jax
import jax.numpy as jnp
from scipy.constants import mu_0
import xarray

from nova.jax.basis import Basis
from nova.jax.tree_util import Pytree
from nova.jax.target import Target
from nova.jax.topology import Topology


class MatrixData(NamedTuple):
    """EM coupling data for jax backed computations."""

    plasma_target: jnp.ndarray | None = None
    source_plasma: jnp.ndarray | None = None
    plasma_plasma: jnp.ndarray | None = None
    force_index: jnp.ndarray | None = None


@dataclass
@jax.tree_util.register_pytree_node_class
class Operator(Pytree):
    """Manage EM influence matrices."""

    source_target: jnp.ndarray
    matrix_data: MatrixData
    source_plasma_index: int = -1
    target_plasma_index: int = -1
    classname: str = ""

    @property
    def target(self):
        """Return target attributes."""
        return (
            self.source_target,
            self.matrix_data.plasma_target,
            self.source_plasma_index,
        )

    @jax.jit
    def evaluate(self, source_target, source_current):
        """Return source-target interaction."""
        result = source_target @ source_current
        if self.classname == "Force":
            return source_current[self.matrix_data.force_index] * result
        return result

    @jax.jit
    def evaluate_external(self, source_current):
        """Return source-target interaction excluding plasma."""
        source_current = source_current.at[self.source_plasma_index].set(0.0)
        return self.evaluate(self.source_target, source_current)

    @jax.jit
    def update_plasma_turns(self, plasma_nturn):
        """Update plasma turns inplace."""
        source_target = self.source_target
        if update_source := self.source_plasma_index != -1:
            source_target = source_target.at[:, self.source_plasma_index].set(
                self.matrix_data.plasma_target @ plasma_nturn
            )
        if update_target := self.target_plasma_index != -1:
            source_target = source_target.at[self.target_plasma_index, :].set(
                plasma_nturn @ self.matrix_data.source_plasma
            )
        if update_source and update_target:
            source_target = source_target.at[
                self.target_plasma_index, self.source_plasma_index
            ].set(plasma_nturn @ self.matrix_data.plasma_plasma @ plasma_nturn)
        return source_target

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (
            self.source_target,
            self.matrix_data,
        )
        aux_data = {
            "source_plasma_index": self.source_plasma_index,
            "target_plasma_index": self.target_plasma_index,
            "classname": self.classname,
        }
        return (children, aux_data)


@dataclass
class Operators:
    """Generate EM coupling matricies."""

    data: xarray.Dataset = field(repr=False)

    def __getitem__(self, attr: str) -> Operator:
        """Retrun jax Operator instance."""
        plasma_dataset = {}
        if source_plasma := self.data.source_plasma_index != -1:
            plasma_dataset["plasma_target"] = jnp.array(self.data[f"{attr}_"])
        if target_plasma := self.data.target_plasma_index != -1:
            plasma_dataset["source_plasma"] = jnp.array(self.data[f"_{attr}"])
        if source_plasma and target_plasma:
            plasma_dataset["plasma_plasma"] = jnp.array(self.data[f"_{attr}_"])
        try:
            plasma_dataset["force_index"] = jnp.array(self.data["index"])
        except KeyError:
            pass

        return Operator(
            jnp.array(self.data[attr]),
            MatrixData(**plasma_dataset),
            self.data.source_plasma_index,
            self.data.target_plasma_index,
            self.data.classname,
        )


@dataclass
@jax.tree_util.register_pytree_node_class
class Plasma(Pytree):
    """Update plasma current."""

    grid: Target
    wall: Target
    p_prime: Basis
    ff_prime: Basis
    source_current: jnp.ndarray = field(repr=False)
    area: jnp.ndarray = field(repr=False)
    polarity: int

    def __post_init__(self):
        """Generate topology instance."""
        self.topology = Topology(self.grid.null, self.wall.null)

    @jax.jit
    def __call__(self, psi: jnp.ndarray):
        """Return total poloidal flux."""
        return self.external + self.internal(psi)

    @jax.jit
    def residual(self, psi):
        """Return poloidal flux residual."""
        return psi - self(psi)

    @jax.jit
    def plasma_current(self, psi):
        """Return plasma current calculated from poloidal flux."""
        psi_norm, ionize = self.topology.update(psi, self.polarity)
        current_density = self.grid.coordinate[:, 0] * self.p_prime(
            psi_norm
        ) + self.ff_prime(psi_norm) / (mu_0 * self.grid.coordinate[:, 0])
        current_density *= -2 * jnp.pi
        plasma_current = current_density * self.area
        return jnp.where(ionize, plasma_current, 0)

    @cached_property
    def external(self):
        """Return external flux map."""
        return jnp.r_[
            self.grid.external(self.source_current),
            self.wall.external(self.source_current),
        ]

    @jax.jit
    def internal(self, psi):
        """Return internal (plasma generated) flux map."""
        plasma_current = self.plasma_current(psi)
        return jnp.r_[
            self.grid.internal(plasma_current), self.wall.internal(plasma_current)
        ]

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (
            self.grid,
            self.wall,
            self.p_prime,
            self.ff_prime,
            self.source_current,
            self.area,
        )
        aux_data = {"polarity": self.polarity}
        return (children, aux_data)


if __name__ == "__main__":

    from nova.imas.operate import Operate

    # plasmagrid = xarray.open_dataset("plasmagrid.nc")

    operate = Operate(
        135013,
        2,
        pf_active=True,
        pf_passive=False,
        wall=True,
        tplasma="h",
        ngrid=2e3,
        nwall=3,
    )

    operator = Operators(operate.plasmagrid.data)["Psi"]

    operate.time = 300

    plasma = Plasma(
        operate.plasmagrid.target,
        operate.plasmawall.target,
        operate.p_prime,
        operate.ff_prime,
        jnp.array(operate.saloc["Ic"]),
        jnp.array(operate.plasma.area_),
        operate.polarity,
    )

    levels = operate.plasma.plot(colors="C0")

    # operate.sloc["PF4", "Ic"] *= 1.2

    psi = jnp.array(operate.plasma.psi)
    # operate.plasma.grid["psi"] = 0.1 * plasma(psi)[: plasma.grid.node_number]

    operate.plasma.solve_flux(verbose=True, f_rtol=1e-6)
    operate.plasma.plot(levels=levels, colors="C2")

    # psi = jnp.array(operate.plasma.psi)
    # operate.plasma.grid["psi"] = plasma(psi)[: plasma.grid.node_number]

    # operate.plasma.plot(levels=levels, colors="C2")

    jac = jax.jit(jax.jacfwd(plasma.residual))

    # sol = root(plasma.residual, psi, jac=jac, tol=1e-3)
    # operate.plasma.grid["psi"] = sol.x[: plasma.grid.node_number]
    # operate.plasma.plot(levels=levels, colors="C3")

    # print(sol)
    # import jaxopt

    # root = jaxopt.ScipyRootFinding(optimality_fun=plasma.residual, method="hybr")
    # root.run(psi)
