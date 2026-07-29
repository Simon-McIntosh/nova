r"""Flux coupling between 3-D sources and closed observation loops.

The Biot element tier exposes the Cartesian vector potential without ``mu_0``.
This module restores physical units once, evaluates it along a fixed-shape loop
quadrature, and applies

.. math::

   M = \oint_\mathcal{C} \mathbf{A}\cdot\mathrm{d}\mathbf{l}.

The construction works for every source segment understood by
:class:`nova.biot.solve.Solve`; polygon loops provide the general path and a
circle-specific quadrature preserves the exact circular geometry used by the
Maxwell mutual-inductance oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0
from scipy.special import ellipe, ellipk

from nova.biot.biotframe import Target
from nova.biot.solve import Solve


@dataclass(frozen=True)
class LoopQuadrature:
    """Points and weighted line elements for one closed observation loop."""

    points: np.ndarray
    differentials: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        differentials = np.asarray(self.differentials, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("loop quadrature points must have shape (n, 3)")
        if differentials.shape != points.shape:
            raise ValueError("loop differentials must match the point shape")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "differentials", differentials)

    def transformed(
        self, translation: np.ndarray, rotation: np.ndarray
    ) -> LoopQuadrature:
        """Return the same loop under a rigid-body transform."""
        return LoopQuadrature(
            transform_points(self.points, translation, rotation),
            transform_vectors(self.differentials, rotation),
        )


def _orthogonal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-handed orthonormal pair spanning a plane."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def polygon_loop_quadrature(vertices: np.ndarray, *, order: int = 12) -> LoopQuadrature:
    """Gauss-Legendre quadrature along a closed polygonal loop.

    The final vertex may repeat the first; otherwise the closing edge is added.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("loop vertices must have shape (n >= 3, 3)")
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    nodes, weights = leggauss(order)
    start = vertices
    stop = np.roll(vertices, -1, axis=0)
    midpoint = 0.5 * (start + stop)
    half_edge = 0.5 * (stop - start)
    points = midpoint[:, None, :] + nodes[None, :, None] * half_edge[:, None, :]
    differentials = weights[None, :, None] * half_edge[:, None, :]
    return LoopQuadrature(points.reshape(-1, 3), differentials.reshape(-1, 3))


def circular_loop_quadrature(
    radius: float,
    *,
    center: np.ndarray = np.zeros(3),
    normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    panels: int = 8,
    order: int = 12,
) -> LoopQuadrature:
    """Fixed-shape quadrature on an exact circular loop."""
    if radius <= 0:
        raise ValueError("loop radius must be positive")
    center = np.asarray(center, dtype=float)
    first, second = _orthogonal_basis(normal)
    nodes, weights = leggauss(order)
    edges = np.linspace(0.0, 2.0 * np.pi, panels + 1)
    midpoint = 0.5 * (edges[:-1] + edges[1:])
    half_width = 0.5 * np.diff(edges)
    angle = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    points = (
        center + radius * cosine[..., None] * first + radius * sine[..., None] * second
    )
    tangent = radius * (-sine[..., None] * first + cosine[..., None] * second)
    differentials = tangent * half_width[:, None, None] * weights[None, :, None]
    return LoopQuadrature(points.reshape(-1, 3), differentials.reshape(-1, 3))


def frame_vector_potential(
    source,
    points: np.ndarray,
    *,
    turns: bool = True,
    reduce: bool = True,
) -> np.ndarray:
    """Physical Cartesian vector potential at points, per source ampere.

    Returns shape ``(point, source, xyz)``. The source axis is reduced through
    the frame links when ``reduce`` is true.
    """
    points = np.asarray(points, dtype=float)
    target = Target(
        {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}, available=[]
    )
    solved = Solve(
        source,
        target,
        turns=[turns, False],
        reduce=[reduce, False],
        attrs=["Ax", "Ay", "Az"],
    )
    potential = np.stack(
        [np.asarray(solved.data[name]) for name in ("Ax", "Ay", "Az")], axis=-1
    )
    return mu_0 * potential


def integrate_vector_potential(evaluator, quadrature: LoopQuadrature) -> np.ndarray:
    """Integrate a vector-potential evaluator around one loop."""
    potential = np.asarray(evaluator(quadrature.points))
    if potential.shape[0] != len(quadrature.points) or potential.shape[-1] != 3:
        raise ValueError("vector-potential evaluator must return shape (point, ..., 3)")
    return np.einsum(
        "q...c,qc->...", potential, quadrature.differentials, optimize=True
    )


def frame_loop_coupling(
    source,
    loops: LoopQuadrature | list[LoopQuadrature],
    *,
    turns: bool = True,
    reduce: bool = True,
) -> np.ndarray:
    """Flux coupling from a source frame into one or more observation loops.

    The result has shape ``(loop, source)`` for a list and ``(source,)`` for a
    single loop.
    """

    def evaluator(points: np.ndarray) -> np.ndarray:
        return frame_vector_potential(source, points, turns=turns, reduce=reduce)

    if isinstance(loops, LoopQuadrature):
        return integrate_vector_potential(evaluator, loops)
    return np.stack([integrate_vector_potential(evaluator, loop) for loop in loops])


def circular_mutual_inductance(
    source_radius: float,
    target_radius: float,
    separation: float,
) -> float:
    """Maxwell mutual inductance of two coaxial circular filaments [H]."""
    k2 = (
        4.0
        * source_radius
        * target_radius
        / ((source_radius + target_radius) ** 2 + separation**2)
    )
    k = np.sqrt(k2)
    return float(
        mu_0
        * np.sqrt(source_radius * target_radius)
        * ((2.0 / k - k) * ellipk(k2) - 2.0 / k * ellipe(k2))
    )


def transform_vectors(vectors, rotation):
    """Rotate vectors, supporting leading transform-batch dimensions."""
    namespace = getattr(vectors, "__array_namespace__", lambda: np)()
    return namespace.einsum("...ij,pj->...pi", rotation, vectors)


def transform_points(points, translation, rotation):
    """Rigidly transform points with fixed shapes and batchable leading axes."""
    return transform_vectors(points, rotation) + translation[..., None, :]


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3×3 right-handed rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    identity = np.eye(3)
    return (
        np.cos(angle) * identity
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
    )
