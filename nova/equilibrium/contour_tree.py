"""Contour-tree topology conventions for equilibrium states.

Topology orders the signed, raw poloidal flux ``sigma * psi`` in webers, where
``sigma`` is derived from the state's declared COCOS sign tuple and the sign of
its plasma current through :mod:`nova.io.cocos`; the magnetic axis is therefore
a maximum of the signed flux.  Normalised flux is never a topology operand.

The domain is the interior of the multi-unit vessel polygon under true polygon
containment, with that polygon as its boundary.  The carrier is the hex plasma
cells: this read never resamples onto a raster.  Results map to DD 4.1.0 under
COCOS 17 as ``contour_tree.node`` and ``contour_tree.edges`` (with
``critical_type`` on raw ``psi``), ``boundary.type``, ``boundary.psi``,
``global_quantities.psi_magnetic_axis``, and ``boundary.closest_wall_point``.

The fixed capacities are 256 nodes and 255 edges.  A capacity overflow is a
visible refusal rather than a truncated topology result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Protocol, TypeAlias

from nova.io.cocos import convention

ContourTreeNode: TypeAlias = tuple[int, float, float, float]
"""One DD node as ``(critical_type, radius, height, psi)`` on raw flux."""

ContourTreeEdge: TypeAlias = tuple[int, int]
"""One pair of indices into :attr:`ContourTreeResult.node`."""


class FluxCurrentSignError(ValueError):
    """The current sign and raw flux ordering disagree with declared COCOS."""


class ContourTreeState(Protocol):
    """State fields required to orient a contour-tree topology read."""

    cocos: int
    plasma_current: float
    psi_magnetic_axis: float
    boundary_psi: float


def _require_flux_current_consistency(
    state: ContourTreeState, *, sigma_bp: int, current_sign: int
) -> None:
    """Refuse a state whose raw axis-to-boundary ordering contradicts COCOS."""

    axis = float(state.psi_magnetic_axis)
    boundary = float(state.boundary_psi)
    difference = boundary - axis
    if not isfinite(axis) or not isfinite(boundary) or difference == 0.0:
        raise FluxCurrentSignError(
            "axis and boundary flux must be distinct finite webers"
        )
    flux_sign = 1 if difference > 0.0 else -1
    expected = current_sign * sigma_bp
    if flux_sign != expected:
        raise FluxCurrentSignError(
            "plasma current sign and raw flux ordering disagree with declared COCOS"
        )


@dataclass(frozen=True, slots=True)
class ContourTreeResult:
    """DD 4.1.0 topology fields emitted under COCOS 17.

    ``node`` contains raw-psi critical-point rows, and ``edges`` contains their
    index pairs.  The remaining fields map directly to the DD paths named in
    the module contract.
    """

    node: tuple[ContourTreeNode, ...]
    edges: tuple[ContourTreeEdge, ...]
    boundary_type: int
    boundary_psi: float
    psi_magnetic_axis: float
    closest_wall_point: tuple[float, float]

    node_capacity: ClassVar[int] = 256
    edge_capacity: ClassVar[int] = 255

    def __post_init__(self) -> None:
        """Refuse invalid DD values and fixed-capacity overflow visibly."""

        if len(self.node) > self.node_capacity:
            raise ValueError(f"contour tree has more than {self.node_capacity} nodes")
        if len(self.edges) > self.edge_capacity:
            raise ValueError(f"contour tree has more than {self.edge_capacity} edges")
        if self.boundary_type not in (0, 1):
            raise ValueError("boundary type must be 0 (limiter) or 1 (diverted)")
        if not isfinite(self.boundary_psi) or not isfinite(self.psi_magnetic_axis):
            raise ValueError("contour-tree flux values must be finite raw webers")
        if len(self.closest_wall_point) != 2 or not all(
            isfinite(value) for value in self.closest_wall_point
        ):
            raise ValueError("closest wall point must be one finite radius-height pair")
        for critical_type, radius, height, psi in self.node:
            if critical_type not in (0, 1, 2):
                raise ValueError(
                    "critical type must be 0 (minimum), 1 (saddle), or 2 (maximum)"
                )
            if not all(isfinite(value) for value in (radius, height, psi)):
                raise ValueError("contour-tree nodes must carry finite raw values")
        for start, end in self.edges:
            if not (0 <= start < len(self.node) and 0 <= end < len(self.node)):
                raise ValueError("contour-tree edges must reference declared node rows")


def sigma_for_state(state: ContourTreeState) -> int:
    """Return the signed-flux multiplier that makes the state axis a maximum."""

    declared = convention(state.cocos)
    current = float(state.plasma_current)
    if not isfinite(current) or current == 0.0:
        raise FluxCurrentSignError("plasma current must be finite and nonzero")
    current_sign = 1 if current > 0.0 else -1
    _require_flux_current_consistency(
        state, sigma_bp=declared.sigma_bp, current_sign=current_sign
    )
    return -current_sign * declared.sigma_bp


__all__ = [
    "ContourTreeEdge",
    "ContourTreeNode",
    "ContourTreeResult",
    "ContourTreeState",
    "FluxCurrentSignError",
    "sigma_for_state",
]
