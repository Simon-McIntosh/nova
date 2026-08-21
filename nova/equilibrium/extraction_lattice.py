r"""Exact flux evaluation on a structured extraction lattice.

The extraction lattice is a target, not a representation of the solved
field.  This module therefore rebuilds the target-by-source Green blocks from
the source sections that defined a free-boundary solve and contracts them with
the converged source currents.  No value from the solved nodal flux map enters
the evaluation.

Rectangular sections use :func:`nova.biot.greens.hybrid_greens`, the same
finite-section/filament route used by the structured forward fixture.  General
polygonal cells use
:func:`nova.biot.polygonanalytic.polygon_analytic_flux_moments`, including the
linear source moments carried by :class:`nova.equilibrium.ForwardFluxOperator`.
Those are the existing coil and plasma-cell Green authorities identified by
the geometry-service reuse map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.greens import hybrid_greens
from nova.biot.polygonanalytic import (
    polygon_analytic_flux,
    polygon_analytic_flux_moments,
)
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward import ForwardEquilibrium
from nova.equilibrium.stencil_mesh import CellCurrentMoments

SectionKernel = Literal["hybrid_rectangle", "analytic_polygon"]

__all__ = ["GreenSourceRepresentation", "evaluate_forward_equilibrium"]


class _SourceMoments(NamedTuple):
    uniform: np.ndarray
    radial: np.ndarray
    vertical: np.ndarray


@dataclass(frozen=True)
class GreenSourceRepresentation:
    """Geometry and currents retained from one forward-map construction.

    ``external_sections`` and ``plasma_sections`` are counter-clockwise
    ``(R, Z)`` polygons, one per source column.  The kernel name records which
    Green route built that source block.  ``hybrid_rectangle`` is accepted only
    for axis-aligned rectangles and uniform current; ``analytic_polygon``
    carries arbitrary polygonal sections and the optional linear plasma-current
    moments.

    A forward result deliberately contains observations rather than machine
    geometry.  Keeping this record beside it is what makes later evaluation
    exact: reconstructing sections from sampled flux values would be an inverse
    problem and interpolation would violate the extraction seam.
    """

    external_sections: tuple[np.ndarray, ...]
    external_current: np.ndarray
    plasma_sections: tuple[np.ndarray, ...]
    external_kernel: SectionKernel = "analytic_polygon"
    plasma_kernel: SectionKernel = "analytic_polygon"
    plasma_current_moments: CellCurrentMoments | None = None

    def __post_init__(self) -> None:
        """Validate immutable source geometry without changing its precision."""
        external = _validated_sections(self.external_sections, "external")
        plasma = _validated_sections(self.plasma_sections, "plasma")
        current = np.asarray(self.external_current, dtype=np.float64)
        if current.ndim != 1 or current.size != len(external):
            raise ValueError(
                "external_current must have one value per external section"
            )
        if not np.all(np.isfinite(current)):
            raise ValueError("external_current must be finite")
        for name in ("external_kernel", "plasma_kernel"):
            if getattr(self, name) not in ("hybrid_rectangle", "analytic_polygon"):
                raise ValueError(f"unsupported {name} {getattr(self, name)!r}")
        if self.external_kernel == "hybrid_rectangle":
            tuple(_rectangle_descriptor(section) for section in external)
        if self.plasma_kernel == "hybrid_rectangle":
            tuple(_rectangle_descriptor(section) for section in plasma)
        object.__setattr__(self, "external_sections", external)
        object.__setattr__(self, "plasma_sections", plasma)
        current.setflags(write=False)
        object.__setattr__(self, "external_current", current)


def _validated_sections(sections, family: str) -> tuple[np.ndarray, ...]:
    """Return finite, positive-radius, read-only section polygons."""
    validated = []
    for position, section in enumerate(sections):
        vertices = np.asarray(section, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
            raise ValueError(
                f"{family} section {position} must have shape (vertices, 2)"
            )
        if not np.all(np.isfinite(vertices)) or np.any(vertices[:, 0] <= 0.0):
            raise ValueError(
                f"{family} section {position} must have finite positive radius"
            )
        copy = np.ascontiguousarray(vertices)
        copy.setflags(write=False)
        validated.append(copy)
    return tuple(validated)


def _rectangle_descriptor(section: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(R, Z, dR, dZ)`` for one axis-aligned rectangle."""
    radius = np.unique(section[:, 0])
    height = np.unique(section[:, 1])
    if radius.size != 2 or height.size != 2:
        raise ValueError("hybrid_rectangle sections must be axis-aligned rectangles")
    corners = {(float(r), float(z)) for r in radius for z in height}
    if set(map(tuple, section)) != corners:
        raise ValueError("hybrid_rectangle sections must contain their four corners")
    width = float(radius[1] - radius[0])
    thickness = float(height[1] - height[0])
    if width <= 0.0 or thickness <= 0.0:
        raise ValueError("hybrid_rectangle dimensions must be positive")
    return (
        float(np.mean(radius)),
        float(np.mean(height)),
        width,
        thickness,
    )


def _uniform_source_block(
    target_r: np.ndarray,
    target_z: np.ndarray,
    sections: tuple[np.ndarray, ...],
    kernel: SectionKernel,
) -> np.ndarray:
    """Return one exact uniform-current flux column per source section."""
    if not sections:
        return np.zeros((target_r.size, 0), dtype=np.float64)
    if kernel == "hybrid_rectangle":
        columns = [
            hybrid_greens(target_r, target_z, *(_rectangle_descriptor(section)))[0]
            for section in sections
        ]
    else:
        columns = [
            polygon_analytic_flux(target_r, target_z, section) for section in sections
        ]
    return np.stack(columns, axis=1)


def _plasma_source_blocks(
    target_r: np.ndarray,
    target_z: np.ndarray,
    sources: GreenSourceRepresentation,
    result: ForwardEquilibrium,
) -> tuple[np.ndarray, _SourceMoments]:
    """Return plasma Green blocks and converged current coefficients."""
    cell_current = np.asarray(result.cell_current, dtype=np.float64)
    if cell_current.shape != (len(sources.plasma_sections),):
        raise ValueError("the result needs one cell current per plasma section")

    carried = sources.plasma_current_moments
    if carried is None:
        moments = _SourceMoments(
            cell_current,
            np.zeros_like(cell_current),
            np.zeros_like(cell_current),
        )
    else:
        moments = _SourceMoments(
            *(np.asarray(value, dtype=np.float64) for value in carried)
        )
        if any(value.shape != cell_current.shape for value in moments):
            raise ValueError("plasma current moments must align with plasma sections")
        tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * max(float(np.max(np.abs(cell_current), initial=0.0)), 1.0)
        )
        if not np.allclose(moments.uniform, cell_current, rtol=0.0, atol=tolerance):
            raise ValueError(
                "the retained uniform plasma moments do not match the result"
            )

    if sources.plasma_kernel == "hybrid_rectangle":
        if np.any(moments.radial != 0.0) or np.any(moments.vertical != 0.0):
            raise ValueError(
                "hybrid_rectangle cannot evaluate linear plasma-current moments"
            )
        uniform = _uniform_source_block(
            target_r, target_z, sources.plasma_sections, sources.plasma_kernel
        )
        return uniform, moments

    columns = [
        polygon_analytic_flux_moments(target_r, target_z, section)
        for section in sources.plasma_sections
    ]
    uniform, radial, vertical = (
        np.stack([column[index] for column in columns], axis=1) for index in range(3)
    )
    return np.stack((uniform, radial, vertical)), moments


def evaluate_forward_equilibrium(
    result: ForwardEquilibrium,
    lattice: FluxLattice,
    sources: GreenSourceRepresentation,
) -> jax.Array:
    """Evaluate one converged equilibrium exactly on ``lattice``.

    The returned array has shape ``(height, radius)`` so it can be passed
    directly to :func:`nova.equilibrium.extract_flux_surface_geometry` beside
    ``lattice.radius`` and ``lattice.height``.  The calculation contracts only
    newly evaluated Green blocks with retained source currents; it never reads
    ``result.flux`` and performs no interpolation.
    """
    if not isinstance(lattice, FluxLattice):
        raise TypeError("lattice must be a FluxLattice")
    if not bool(np.asarray(result.finite.passed)):
        raise ValueError("the forward result contains a non-finite field")
    residual = float(np.asarray(result.fixed_point.residual))
    if not np.isfinite(residual):
        raise ValueError("the forward result has no finite convergence residual")

    mesh_r, mesh_z = np.meshgrid(lattice.radius, lattice.height, indexing="xy")
    target_r = mesh_r.reshape(-1)
    target_z = mesh_z.reshape(-1)
    external_block = _uniform_source_block(
        target_r, target_z, sources.external_sections, sources.external_kernel
    )
    external_flux = external_block @ sources.external_current

    plasma_blocks, moments = _plasma_source_blocks(target_r, target_z, sources, result)
    if sources.plasma_kernel == "hybrid_rectangle":
        plasma_flux = plasma_blocks @ moments.uniform
    else:
        plasma_flux = (
            plasma_blocks[0] @ moments.uniform
            + plasma_blocks[1] @ moments.radial
            + plasma_blocks[2] @ moments.vertical
        )
    evaluated = external_flux + plasma_flux
    return jnp.asarray(evaluated.reshape(lattice.height.size, lattice.radius.size))
