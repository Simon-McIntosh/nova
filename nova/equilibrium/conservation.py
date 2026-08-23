r"""Field, current and force conservation receipts of a solved flux map.

The forward solve returns a flux map; these are the checks that decide
whether that map is an equilibrium rather than a converged iteration. All of
them are evaluated by differentiating that map on the mesh it was solved on,
with the sign chain taken from :mod:`nova.equilibrium.convention` and never
re-derived here.

The mesh enters through one contract, :class:`FluxMesh`: a gradient, an
elliptic operator, the cells that carry a complete stencil, and an erosion of
a cell mask by that stencil. :class:`FluxLattice` meets it by central
differences on a uniform raster and
:class:`~nova.equilibrium.stencil_mesh.StencilMesh` by a least-squares
quadratic on neighbour rings, so the same receipts are published on the
production hexagonal plasma mesh as on a structured one.

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
    wrong solve can fail. The pressure gradient is the one the declared
    closure evaluates at fixed major radius, so a rotating source is read
    against its own drive rather than against a flux-function idealisation
    of it.

``force_residual``
    :math:`J \times B - \nabla p` plus whatever body force the closure
    declares, with the toroidal current density taken from the FIELD,
    :math:`\mu_0 j_\phi = -\Delta^\star \Phi / (2 \pi R)`, rather than from
    the source. Taking it from the source would make the cancellation
    algebraic and the check empty; taking it from the field makes the
    residual proportional to the Grad-Shafranov residual times
    :math:`\nabla \Phi`, which is what a force-balance receipt should
    measure. The body force is zero for a static closure and the centrifugal
    density :math:`\rho R \Omega^2` for a rotating one, which is what makes
    the outboard pressure pile-up a balance rather than a residual.

The two identically-vanishing residuals are read against the mesh, not
against machine epsilon. Central differences on a raster commute exactly, so
the mixed derivative cancels term by term and ``divergence_b`` lands at
round-off. A least-squares stencil fit does not commute, so on
:class:`~nova.equilibrium.stencil_mesh.StencilMesh` the same residual lands
at the truncation floor of the second fit and falls at second order under
refinement rather than staying at round-off. It is still four to five decades
below the physical residuals, so it still discriminates a broken
flux-to-field relation from a converged one — but its floor is a property of
the mesh and has to be read as such.

Every residual is restricted to cells where the source is declared and the
stencil is complete, so a one-sided difference at the mesh border never
enters a reported number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.convention import (
    TOTAL_FLUX_FACTOR,
    delta_star_from_current_density,
)
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.observation import (
    declared_body_force,
    declared_field_function_squared,
    declared_pressure,
    layout_invariant_sum,
)

__all__ = [
    "ConservationLedger",
    "FluxLattice",
    "FluxMesh",
    "conservation_ledger",
    "delta_star",
    "poloidal_field",
]

#: Cells trimmed from each mesh border before a residual is reported. Two
#: differences reach two cells, so a narrower margin would report the
#: wrap-around of the differencing stencil rather than a physical residual.
STENCIL_MARGIN = 2


class FluxMesh(Protocol):
    """The differential contract a flux map is read for its receipts on.

    Cell fields are flat vectors in one mesh-wide order, and every operator
    returns a vector in that same order, so one index reads the mesh, the
    coupling operators and the domain labels alike.
    """

    @property
    def node_count(self) -> int:
        """Return the cell count."""

    @property
    def node_radius(self) -> np.ndarray:
        """Return the major radius [m] of every cell."""

    @property
    def cell_area(self) -> np.ndarray:
        """Return the poloidal cross-section [m^2] of every cell."""

    def gradient(self, field) -> tuple[jax.Array, jax.Array]:
        """Return the radial and vertical derivative of one cell field."""

    def delta_star(self, flux) -> jax.Array:
        """Return the elliptic operator value [Wb/m^2] of one flux map."""

    def erode(self, mask, margin: int) -> jax.Array:
        """Return a cell mask shrunk by ``margin`` stencil steps."""

    def interior(self, margin: int = STENCIL_MARGIN) -> jax.Array:
        """Return the mask of cells with a complete difference stencil."""


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

    def gradient(self, field) -> tuple[jax.Array, jax.Array]:
        """Return the radial and vertical central difference of one field."""
        value = self.reshape(field)
        return (
            self.flatten(_central(value, self.radial_step, 0)),
            self.flatten(_central(value, self.vertical_step, 1)),
        )

    def delta_star(self, flux) -> jax.Array:
        """Return the elliptic operator value [Wb/m^2] of one flux map."""
        flux2d = self.reshape(flux)
        radius = self.reshape(self.node_radius)
        value = (
            _second(flux2d, self.radial_step, 0)
            - _central(flux2d, self.radial_step, 0) / radius
            + _second(flux2d, self.vertical_step, 1)
        )
        return self.flatten(value)

    def erode(self, mask, margin: int) -> jax.Array:
        """Return a flat node mask shrunk by ``margin`` four-neighbour steps."""
        return self.flatten(_erode(self.reshape(mask), margin))


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


def delta_star(mesh: FluxMesh, flux) -> jax.Array:
    """Return the elliptic operator value [Wb/m^2] of one flux map."""
    return mesh.delta_star(flux)


def poloidal_field(mesh: FluxMesh, flux) -> tuple[jax.Array, jax.Array]:
    """Return the radial and vertical poloidal field [T] of one flux map."""
    radial, vertical = _layout_invariant_gradient(mesh, flux)
    scale = TOTAL_FLUX_FACTOR * jnp.asarray(mesh.node_radius)
    return -vertical / scale, radial / scale


def _axisymmetric_divergence(mesh: FluxMesh, radial, vertical) -> jax.Array:
    """Return the divergence of one axisymmetric poloidal vector field."""
    radius = jnp.asarray(mesh.node_radius)
    return (
        _layout_invariant_gradient(mesh, radius * radial)[0] / radius
        + _layout_invariant_gradient(mesh, vertical)[1]
    )


def _layout_invariant_gradient(mesh: FluxMesh, field) -> tuple[jax.Array, jax.Array]:
    """Apply fitted ring derivatives with one association under transformations."""

    value = jax.lax.optimization_barrier(jnp.asarray(field))
    if not all(
        hasattr(mesh, name)
        for name in ("stencil", "radial_weight", "vertical_weight", "node_count")
    ):
        return mesh.gradient(value)
    stencil = jnp.asarray(mesh.stencil)
    gathered = value[stencil]
    radial = layout_invariant_sum(
        jnp.asarray(mesh.radial_weight, dtype=value.dtype) * gathered,
        axis=1,
    )
    vertical = layout_invariant_sum(
        jnp.asarray(mesh.vertical_weight, dtype=value.dtype) * gathered,
        axis=1,
    )
    centre = stencil[:, 0]
    return (
        jnp.zeros(mesh.node_count, dtype=value.dtype).at[centre].set(radial),
        jnp.zeros(mesh.node_count, dtype=value.dtype).at[centre].set(vertical),
    )


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
    mesh: FluxMesh,
    flux,
    source,
    masks: DomainMasks,
    flux_span,
) -> ConservationLedger:
    """Return the conservation receipts of one converged flux map.

    Residuals are read on the union of the domains the source declares a
    closure on, eroded by the difference stencil width, so every reported
    number comes from a complete central stencil inside declared support. A
    source that declares the core alone is therefore read exactly where it
    always was; a source carrying a separatrix continuation is read across the
    seam as well, which is where a continuation that is smooth in the profile
    but not in the equilibrium would show up.
    """
    radius = jnp.asarray(mesh.node_radius)
    support = source.declared_support(masks)
    checked = mesh.erode(support, STENCIL_MARGIN) & mesh.interior()

    elliptic = delta_star(mesh, flux)
    drive = delta_star_from_current_density(
        radius, source.current_density(radius, masks)
    )

    radial_field, vertical_field = poloidal_field(mesh, flux)
    divergence_b = _axisymmetric_divergence(mesh, radial_field, vertical_field)

    squared = declared_field_function_squared(source, masks, flux_span)
    # the poloidal current is carried by the toroidal-field function; its sign
    # branch is the toroidal-field direction and flips J entirely, so the
    # positive branch is taken and the divergence it is read for is unchanged
    field_function = jnp.sqrt(jnp.maximum(squared, 0.0))
    function_radial, function_vertical = _layout_invariant_gradient(
        mesh, field_function
    )
    radial_current = -function_vertical / (mu_0 * radius)
    vertical_current = function_radial / (mu_0 * radius)
    divergence_j = _axisymmetric_divergence(mesh, radial_current, vertical_current)

    # the force components need only the SQUARED field function: both
    # J_R B_phi and J_Z B_phi reduce to gradients of F^2 over 2 mu_0 R^2
    squared_radial, squared_vertical = _layout_invariant_gradient(mesh, squared)
    diamagnetic_radial = squared_radial / (2.0 * mu_0 * radius**2)
    diamagnetic_vertical = squared_vertical / (2.0 * mu_0 * radius**2)
    cell_pressure = declared_pressure(source, masks, radius, flux_span)
    pressure_radial, pressure_vertical = _layout_invariant_gradient(mesh, cell_pressure)
    body_force = declared_body_force(source, masks, radius, cell_pressure)

    toroidal_current = -elliptic / (TOTAL_FLUX_FACTOR * mu_0 * radius)
    force = jnp.hypot(
        toroidal_current * vertical_field
        - diamagnetic_radial
        - pressure_radial
        + body_force,
        -toroidal_current * radial_field - diamagnetic_vertical - pressure_vertical,
    )
    pressure_gradient = jnp.hypot(pressure_radial, pressure_vertical)

    field_scale = jnp.maximum(
        _sup(_layout_invariant_gradient(mesh, radial_field)[1], checked),
        _sup(_layout_invariant_gradient(mesh, vertical_field)[0], checked),
    )
    current_scale = jnp.maximum(
        _sup(_layout_invariant_gradient(mesh, radial_current)[1], checked),
        _sup(_layout_invariant_gradient(mesh, vertical_current)[0], checked),
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
