r"""Field, current and force conservation receipts on a structured lattice.

The forward solve returns a flux map; these are the checks that decide
whether that map is an equilibrium rather than a converged iteration. All of
them are evaluated by central differences on the same uniform
:class:`FluxLattice` the flux was solved on, with the sign chain taken from
:mod:`nova.equilibrium.convention` and never re-derived here.

Four residuals, each measuring a different failure:

``divergence_b``
    :math:`\nabla \cdot B` vanishes identically for a flux-function
    representation, so a non-zero value is a broken flux-to-field relation,
    not a physics error. It is reported at the differencing floor.

``divergence_j``
    :math:`\nabla \cdot J` likewise vanishes identically once the poloidal
    current follows :math:`F(\psi)`; it catches a toroidal-field function
    that is not single valued on a surface.

``grad_shafranov_residual``
    :math:`\Delta^\star \Phi - 4 \pi^2 (\mu_0 R^2 p' + FF')` is the real
    equilibrium residual, and the only one of the four that a converged but
    wrong solve can fail.

``force_residual``
    :math:`J \times B - \nabla p` with the toroidal current density taken
    from the FIELD, :math:`\mu_0 j_\phi = -\Delta^\star \Phi / (2 \pi R)`,
    rather than from the source. Taking it from the source would make the
    cancellation algebraic and the check empty; taking it from the field
    makes the residual proportional to the Grad-Shafranov residual times
    :math:`\nabla \Phi`, which is what a force-balance receipt should
    measure.

Every residual is restricted to cells where the source is declared and the
stencil is complete, so a one-sided difference at the lattice border never
enters a reported number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.convention import (
    TOTAL_FLUX_FACTOR,
    grad_shafranov_source,
)
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.observation import (
    core_field_function_squared,
    core_pressure,
)

__all__ = [
    "ConservationLedger",
    "FluxLattice",
    "conservation_ledger",
    "delta_star",
    "poloidal_field",
]

#: Cells trimmed from each lattice border before a residual is reported. Two
#: central differences reach two cells, so a narrower margin would report the
#: wrap-around of the differencing stencil rather than a physical residual.
STENCIL_MARGIN = 2


@dataclass(frozen=True)
class FluxLattice:
    """Uniform structured node lattice the flux map is carried on.

    Nodes are flattened in C order over ``(radius, height)``, matching the
    hexagonal stencil the topology read is built from, so one flat vector
    indexes the lattice, the coupling operators and the domain labels alike.
    """

    radius: np.ndarray
    height: np.ndarray

    def __post_init__(self):
        """Validate a uniformly spaced, strictly increasing lattice."""
        for name in ("radius", "height"):
            axis = np.asarray(getattr(self, name), dtype=np.float64)
            if axis.ndim != 1 or axis.size < 2 * STENCIL_MARGIN + 3:
                raise ValueError(
                    f"{name} must be one dimensional with at least "
                    f"{2 * STENCIL_MARGIN + 3} nodes"
                )
            step = np.diff(axis)
            if np.any(step <= 0) or not np.allclose(step, step[0], rtol=1e-12):
                raise ValueError(f"{name} must be uniformly spaced and increasing")
            object.__setattr__(self, name, axis)
        if self.radius[0] <= 0.0:
            raise ValueError("radius must be strictly positive")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the node lattice extent as ``(radial, vertical)``."""
        return (self.radius.size, self.height.size)

    @property
    def node_count(self) -> int:
        """Return the flattened node count."""
        return self.radius.size * self.height.size

    @property
    def radial_step(self) -> float:
        """Return the radial node spacing [m]."""
        return float(self.radius[1] - self.radius[0])

    @property
    def vertical_step(self) -> float:
        """Return the vertical node spacing [m]."""
        return float(self.height[1] - self.height[0])

    @property
    def coordinate(self) -> np.ndarray:
        """Return the flattened ``(radius, height)`` node coordinates."""
        radius, height = np.meshgrid(self.radius, self.height, indexing="ij")
        return np.c_[radius.ravel(), height.ravel()]

    @property
    def cell_area(self) -> np.ndarray:
        """Return the per-node control area [m^2]."""
        return np.full(self.node_count, self.radial_step * self.vertical_step)

    @property
    def node_radius(self) -> np.ndarray:
        """Return the flattened node radius [m]."""
        return self.coordinate[:, 0]

    def reshape(self, flat):
        """Return a flat node vector as a lattice-shaped array."""
        return jnp.reshape(flat, self.shape)

    def flatten(self, lattice_shaped):
        """Return a lattice-shaped array as a flat node vector."""
        return jnp.reshape(lattice_shaped, (self.node_count,))

    def interior(self, margin: int = STENCIL_MARGIN) -> jax.Array:
        """Return the flat mask of nodes with a complete difference stencil."""
        mask = np.zeros(self.shape, dtype=bool)
        mask[margin:-margin, margin:-margin] = True
        return jnp.asarray(mask.reshape(-1))


def _central(field, spacing: float, axis: int):
    """Return the second-order central first difference along one axis."""
    return (jnp.roll(field, -1, axis) - jnp.roll(field, 1, axis)) / (2.0 * spacing)


def _second(field, spacing: float, axis: int):
    """Return the second-order central second difference along one axis."""
    return (
        jnp.roll(field, -1, axis) - 2.0 * field + jnp.roll(field, 1, axis)
    ) / spacing**2


def _erode(mask, margin: int):
    """Return a lattice mask shrunk by ``margin`` four-neighbour steps."""
    eroded = mask
    for _ in range(margin):
        eroded = (
            eroded
            & jnp.roll(eroded, 1, 0)
            & jnp.roll(eroded, -1, 0)
            & jnp.roll(eroded, 1, 1)
            & jnp.roll(eroded, -1, 1)
        )
    return eroded


def delta_star(lattice: FluxLattice, flux) -> jax.Array:
    """Return the elliptic operator value [Wb/m^2] of one flux map."""
    flux2d = lattice.reshape(flux)
    radius = lattice.reshape(lattice.node_radius)
    value = (
        _second(flux2d, lattice.radial_step, 0)
        - _central(flux2d, lattice.radial_step, 0) / radius
        + _second(flux2d, lattice.vertical_step, 1)
    )
    return lattice.flatten(value)


def poloidal_field(lattice: FluxLattice, flux) -> tuple[jax.Array, jax.Array]:
    """Return the radial and vertical poloidal field [T] of one flux map."""
    flux2d = lattice.reshape(flux)
    radius = lattice.reshape(lattice.node_radius)
    scale = TOTAL_FLUX_FACTOR * radius
    radial = -_central(flux2d, lattice.vertical_step, 1) / scale
    vertical = _central(flux2d, lattice.radial_step, 0) / scale
    return lattice.flatten(radial), lattice.flatten(vertical)


def _axisymmetric_divergence(lattice: FluxLattice, radial, vertical) -> jax.Array:
    """Return the divergence of one axisymmetric poloidal vector field."""
    radius = lattice.reshape(lattice.node_radius)
    radial2d = lattice.reshape(radial)
    vertical2d = lattice.reshape(vertical)
    value = _central(radius * radial2d, lattice.radial_step, 0) / radius + _central(
        vertical2d, lattice.vertical_step, 1
    )
    return lattice.flatten(value)


class ConservationLedger(NamedTuple):
    """Sup-norm conservation residuals and the scales they are read against."""

    grad_shafranov_residual: jax.Array
    grad_shafranov_scale: jax.Array
    force_residual: jax.Array
    force_scale: jax.Array
    divergence_b: jax.Array
    divergence_b_scale: jax.Array
    divergence_j: jax.Array
    divergence_j_scale: jax.Array
    checked_cells: jax.Array

    @property
    def relative_grad_shafranov(self) -> jax.Array:
        """Return the Grad-Shafranov residual against its own drive."""
        return self.grad_shafranov_residual / self.grad_shafranov_scale

    @property
    def relative_force(self) -> jax.Array:
        """Return the force residual against the pressure gradient scale."""
        return self.force_residual / self.force_scale

    @property
    def relative_divergence_b(self) -> jax.Array:
        """Return the magnetic divergence against the field gradient scale."""
        return self.divergence_b / self.divergence_b_scale

    @property
    def relative_divergence_j(self) -> jax.Array:
        """Return the current divergence against its own gradient scale."""
        return self.divergence_j / self.divergence_j_scale


def _sup(values, mask) -> jax.Array:
    """Return the sup-norm of one field over the checked cells."""
    return jnp.max(jnp.where(mask, jnp.abs(values), 0.0))


def _guard(scale) -> jax.Array:
    """Return a scale that never divides by zero."""
    return jnp.where(scale > 0.0, scale, 1.0)


def conservation_ledger(
    lattice: FluxLattice,
    flux,
    source,
    masks: DomainMasks,
    flux_span,
) -> ConservationLedger:
    """Return the conservation receipts of one converged flux map.

    Residuals are read on the labelled core eroded by the difference stencil
    width, so every reported number comes from a complete central stencil
    inside the domain the source is declared on.
    """
    radius = jnp.asarray(lattice.node_radius)
    checked = lattice.flatten(_erode(lattice.reshape(masks.core), STENCIL_MARGIN))
    checked = checked & lattice.interior()

    elliptic = delta_star(lattice, flux)
    drive = grad_shafranov_source(
        radius,
        source.core.p_prime(masks.psi_norm),
        source.core.ff_prime(masks.psi_norm),
    )

    radial_field, vertical_field = poloidal_field(lattice, flux)
    divergence_b = _axisymmetric_divergence(lattice, radial_field, vertical_field)

    radius2d = lattice.reshape(radius)
    squared = lattice.reshape(core_field_function_squared(source, masks, flux_span))
    # the poloidal current is carried by the toroidal-field function; its sign
    # branch is the toroidal-field direction and flips J entirely, so the
    # positive branch is taken and the divergence it is read for is unchanged
    field_function = jnp.sqrt(jnp.maximum(squared, 0.0))
    radial_current = -_central(field_function, lattice.vertical_step, 1) / (
        mu_0 * radius2d
    )
    vertical_current = _central(field_function, lattice.radial_step, 0) / (
        mu_0 * radius2d
    )
    divergence_j = _axisymmetric_divergence(
        lattice,
        lattice.flatten(radial_current),
        lattice.flatten(vertical_current),
    )

    # the force components need only the SQUARED field function: both
    # J_R B_phi and J_Z B_phi reduce to gradients of F^2 over 2 mu_0 R^2
    diamagnetic_radial = lattice.flatten(
        _central(squared, lattice.radial_step, 0) / (2.0 * mu_0 * radius2d**2)
    )
    diamagnetic_vertical = lattice.flatten(
        _central(squared, lattice.vertical_step, 1) / (2.0 * mu_0 * radius2d**2)
    )
    pressure = lattice.reshape(core_pressure(source, masks, flux_span))
    pressure_radial = lattice.flatten(_central(pressure, lattice.radial_step, 0))
    pressure_vertical = lattice.flatten(_central(pressure, lattice.vertical_step, 1))

    toroidal_current = -elliptic / (TOTAL_FLUX_FACTOR * mu_0 * radius)
    force = jnp.hypot(
        toroidal_current * vertical_field - diamagnetic_radial - pressure_radial,
        -toroidal_current * radial_field - diamagnetic_vertical - pressure_vertical,
    )
    pressure_gradient = jnp.hypot(pressure_radial, pressure_vertical)

    field_scale = jnp.maximum(
        _sup(
            lattice.flatten(
                _central(lattice.reshape(radial_field), lattice.vertical_step, 1)
            ),
            checked,
        ),
        _sup(
            lattice.flatten(
                _central(lattice.reshape(vertical_field), lattice.radial_step, 0)
            ),
            checked,
        ),
    )
    current_scale = jnp.maximum(
        _sup(
            lattice.flatten(_central(radial_current, lattice.vertical_step, 1)), checked
        ),
        _sup(
            lattice.flatten(_central(vertical_current, lattice.radial_step, 0)), checked
        ),
    )
    return ConservationLedger(
        grad_shafranov_residual=_sup(elliptic - drive, checked),
        grad_shafranov_scale=_guard(_sup(drive, checked)),
        force_residual=_sup(force, checked),
        force_scale=_guard(_sup(pressure_gradient, checked)),
        divergence_b=_sup(divergence_b, checked),
        divergence_b_scale=_guard(field_scale),
        divergence_j=_sup(divergence_j, checked),
        divergence_j_scale=_guard(current_scale),
        checked_cells=jnp.sum(checked),
    )
