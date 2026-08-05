"""Flux linkage and nominal resistance of the passive conductors, from geometry.

A free decay measures ``L dI/dt + R I = 0``, so a time constant says nothing
about resistance until the inductance is known.  This module supplies the
inductance exactly -- it is pure geometry, and the geometry is measured -- which
leaves resistance as the single unknown a decay can identify.

**Each disjoint poloidal section is its own toroidal circuit, and that is not an
assumption.**  An axisymmetric conducting section closes on itself once around
the torus whether or not anything welds it to its neighbour, so separate
sections are separate circuits by default and a *parallel* connection is what
would need a source.  Lumping a family's sections into one loop is therefore not
the cautious choice it looks like: it asserts a galvanic grouping, and it leaves
no basis for a resistance at all wherever a family resolves into several parts.
One circuit per section removes the question -- the sections are independent
rings, the mutual couplings between them carry whatever the geometry implies, and
no wiring is claimed anywhere.

**Both sections are integrated, because centroid linking is wrong here.**  The
mutual inductance of two conductors is the flux one links per ampere of the
other, and linking centroid to centroid is exact only when the flux is uniform
across both cross-sections.  These conductors are thin shells and thin case
plates -- a three-millimetre plate sees a flux that varies substantially across
its own thickness, and the self term's flux is logarithmically peaked at the
conductor itself.  So the source side is integrated exactly by the polygon-section
kernel and the observer side is averaged over an interior quadrature of the
receiving section.

**The observer quadrature resolves the thin direction, and the rule is
machine-agnostic.**  A section's thickness is taken as twice its area over its
perimeter -- exact for a long strip, conservative for a compact block -- and the
grid pitch is a fraction of that thickness, so a bent shell is resolved across
its wall rather than across its bounding box.  Expressing the pitch as a fraction
of each section's own thickness rather than as a metre-level cell size is what
lets the rule transfer to another machine unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import shapely

from nova.biot.polygon import polygon_greens
from nova.imas.mast_passive_response import (
    CASE_FAMILY,
    PassiveError,
    case_grouping,
    passive_sections,
)
from nova.imas.mast_seed_parameters import STAINLESS_STEEL, passive_material
from nova.imas.mast_vacuum_response import ProbeTarget

CELLS_ACROSS_THICKNESS = 4
"""Observer quadrature cells across a section's own thickness.

Four cells resolve the flux gradient across a thin wall, which is the error a
centroid link makes and the reason the observer side is integrated at all.  The
convergence of the linkage under refinement is measured rather than assumed:
:func:`linkage_convergence` reports how far the matrix moves between two
resolutions, so the value chosen here is defended by a number.
"""

QUADRATURE_POINT_TARGET = 1200
"""Points one section contributes before its pitch is allowed to coarsen.

A four-metre shell resolved across a two-centimetre wall would otherwise carry
tens of thousands of points and dominate the cost of every column.  Reaching the
target coarsens the pitch in both directions at once, which costs resolution
mostly along the length -- the direction the linked flux varies slowly along.
The count is a target and not a hard ceiling: the grid is laid on the bounding
box in whole cells, so a section can land slightly over it.
"""


@dataclass(frozen=True)
class PassiveTurn:
    """One closed toroidal conductor: a single poloidal section of the registry.

    ``enclosed_coil`` names the poloidal-field coil a case plate surrounds and is
    empty for the vessel and structural sections.  It is carried because the
    decay experiments excite one coil at a time, so which coil a plate sits
    inside is what makes a case circuit reachable at all.
    """

    name: str
    family: str
    enclosed_coil: str
    vertices: np.ndarray

    @property
    def area(self) -> float:
        """Return the section's poloidal cross-section area [m^2]."""

        return abs(float(shapely.Polygon(self.vertices).area))

    @property
    def perimeter(self) -> float:
        """Return the section outline's length [m]."""

        return float(shapely.Polygon(self.vertices).exterior.length)

    @property
    def thickness(self) -> float:
        """Return twice the area over the perimeter [m].

        Exact for a long strip, where the perimeter is twice the length and the
        area is length times thickness.  For a compact block it returns half the
        side, which only asks the quadrature for more resolution than the shape
        needs -- the error is in the safe direction.
        """

        perimeter = self.perimeter
        if perimeter <= 0.0:
            raise PassiveError(f"section {self.name!r} has no outline")
        return 2.0 * self.area / perimeter

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the section's area centroid ``(r, z)`` [m]."""

        point = shapely.Polygon(self.vertices).centroid
        return (float(point.x), float(point.y))


def passive_turns(geometry: Mapping[str, Any]) -> tuple[PassiveTurn, ...]:
    """Return one closed toroidal circuit per disjoint registry section.

    Case plates are named by the coil they enclose, which nearest-coil geometry
    settles without a fit, so a case circuit can be matched to the experiment
    that drives the coil inside it.
    """

    plate_coil: dict[bytes, str] = {}
    for name, plates in case_grouping(geometry).items():
        coil = name.removeprefix(f"{CASE_FAMILY}_")
        for vertices in plates:
            plate_coil[np.asarray(vertices, dtype=float).tobytes()] = coil

    by_stem: dict[str, list[tuple[str, np.ndarray]]] = {}
    for family, parts in sorted(passive_sections(geometry).items()):
        for vertices in parts:
            vertices = np.asarray(vertices, dtype=float)
            coil = plate_coil.get(vertices.tobytes(), "")
            if family == CASE_FAMILY and not coil:
                continue
            stem = f"{family}_{coil}" if coil else family
            by_stem.setdefault(stem, []).append((family, vertices))

    turns: list[PassiveTurn] = []
    for stem in sorted(by_stem):
        rows = sorted(
            by_stem[stem], key=lambda row: tuple(row[1].mean(axis=0).tolist())
        )
        for index, (family, vertices) in enumerate(rows):
            name = stem if len(rows) == 1 else f"{stem}_{index:02d}"
            turns.append(
                PassiveTurn(
                    name=name,
                    family=family,
                    enclosed_coil=stem.removeprefix(f"{CASE_FAMILY}_")
                    if family == CASE_FAMILY
                    else "",
                    vertices=vertices,
                )
            )
    if not turns:
        raise PassiveError("the registry carries no passive sections")
    return tuple(turns)


def section_quadrature(
    turn: PassiveTurn,
    *,
    cells_across: int = CELLS_ACROSS_THICKNESS,
    point_target: int = QUADRATURE_POINT_TARGET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interior quadrature points and equal-area weights for one section.

    The pitch resolves the section's thickness unless doing so would pass the
    point target, in which case it coarsens to the pitch the target allows.
    Cells whose centre lies outside the outline are dropped, so a bent shell is
    sampled inside its wall and not across the void its bounding box spans.
    The weights sum to one, which makes the reduction against them a section
    mean.
    """

    polygon = shapely.Polygon(turn.vertices)
    area = abs(polygon.area)
    if area <= 0.0:
        raise PassiveError(f"section {turn.name!r} has no cross-section")
    pitch = max(
        turn.thickness / max(cells_across, 1),
        math.sqrt(area / max(point_target, 1)),
    )
    r0, z0, r1, z1 = polygon.bounds
    count_r = max(int(math.ceil((r1 - r0) / pitch)), 1)
    count_z = max(int(math.ceil((z1 - z0) / pitch)), 1)
    centres_r = r0 + (r1 - r0) * (np.arange(count_r) + 0.5) / count_r
    centres_z = z0 + (z1 - z0) * (np.arange(count_z) + 0.5) / count_z
    mesh_r, mesh_z = np.meshgrid(centres_r, centres_z)
    mesh_r = mesh_r.ravel()
    mesh_z = mesh_z.ravel()
    inside = shapely.contains_xy(polygon, mesh_r, mesh_z)
    if not inside.any():
        centre = polygon.representative_point()
        return (
            np.array([float(centre.x)]),
            np.array([float(centre.y)]),
            np.array([1.0]),
        )
    point_r = mesh_r[inside]
    point_z = mesh_z[inside]
    weight = np.full(point_r.size, 1.0 / point_r.size)
    return (point_r, point_z, weight)


def quadrature_grid(
    turns: Sequence[PassiveTurn],
    *,
    cells_across: int = CELLS_ACROSS_THICKNESS,
    point_target: int = QUADRATURE_POINT_TARGET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every section's quadrature into one point set.

    Returns ``(r, z, weight, owner)``; ``owner`` maps each point back to its
    circuit index so one kernel evaluation reduces to a whole linkage column
    with :func:`numpy.bincount`.
    """

    point_r: list[np.ndarray] = []
    point_z: list[np.ndarray] = []
    weight: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    for index, turn in enumerate(turns):
        sub_r, sub_z, sub_w = section_quadrature(
            turn, cells_across=cells_across, point_target=point_target
        )
        point_r.append(sub_r)
        point_z.append(sub_z)
        weight.append(sub_w)
        owner.append(np.full(sub_r.size, index, dtype=np.int64))
    return (
        np.concatenate(point_r),
        np.concatenate(point_z),
        np.concatenate(weight),
        np.concatenate(owner),
    )


@dataclass(frozen=True)
class Linkage:
    """The passive circuits' flux-linkage matrix and what it cost to get right.

    ``matrix`` [H] is symmetric; ``reciprocity_residual`` is how far the raw
    analytic-source, quadrature-observer estimate was from symmetric before it
    was averaged, relative to the largest self term.  The physical matrix is
    symmetric by reciprocity, so that residual measures the observer quadrature
    error directly and is the honest error bar on the inductance.
    """

    names: tuple[str, ...]
    matrix: np.ndarray
    reciprocity_residual: float
    quadrature_points: np.ndarray


def linkage_matrix(
    turns: Sequence[PassiveTurn],
    *,
    cells_across: int = CELLS_ACROSS_THICKNESS,
    point_target: int = QUADRATURE_POINT_TARGET,
) -> Linkage:
    """Return the flux linkage of the passive circuits [H].

    ``matrix[i, j]`` is the poloidal flux circuit ``i`` links per ampere of
    circuit ``j``, self terms included -- the polygon kernel is smooth inside the
    conductor, so a self term is the same calculation as a mutual one.  Both
    triangles are computed and averaged rather than one being mirrored, which
    both symmetrises the estimate and measures its own error.
    """

    point_r, point_z, weight, owner = quadrature_grid(
        turns, cells_across=cells_across, point_target=point_target
    )
    count = len(turns)
    raw = np.zeros((count, count))
    for column, turn in enumerate(turns):
        psi, _, _ = polygon_greens(point_r, point_z, turn.vertices)
        raw[:, column] = np.bincount(owner, weights=weight * psi, minlength=count)
    scale = float(np.abs(np.diag(raw)).max())
    return Linkage(
        names=tuple(turn.name for turn in turns),
        matrix=0.5 * (raw + raw.T),
        reciprocity_residual=float(np.abs(raw - raw.T).max() / scale),
        quadrature_points=np.bincount(owner, minlength=count),
    )


def linkage_convergence(
    turns: Sequence[PassiveTurn],
    *,
    coarse: int = 2,
    fine: int = CELLS_ACROSS_THICKNESS,
) -> dict[str, float]:
    """Measure how far the linkage moves between two observer resolutions.

    The quantity that matters for a decay fit is not the matrix entry but the
    mode it produces, and the modes come from ``L`` through a generalised
    eigenproblem, so the relative movement of the eigenvalues is reported
    alongside the entrywise movement.
    """

    refined = linkage_matrix(turns, cells_across=fine).matrix
    rough = linkage_matrix(turns, cells_across=coarse).matrix
    scale = float(np.abs(np.diag(refined)).max())
    diagonal = np.abs(np.diag(refined) - np.diag(rough)) / np.abs(np.diag(refined))
    refined_values = np.linalg.eigvalsh(refined)
    rough_values = np.linalg.eigvalsh(rough)
    return {
        "cells_across_coarse": coarse,
        "cells_across_fine": fine,
        "eigenvalue_shift_max": float(
            np.max(np.abs(refined_values - rough_values) / np.abs(refined_values))
        ),
        "entry_shift_max": float(np.abs(refined - rough).max() / scale),
        "self_term_shift_max": float(diagonal.max()),
        "self_term_shift_median": float(np.median(diagonal)),
    }


def nominal_resistance(turns: Sequence[PassiveTurn]) -> np.ndarray:
    """Return each circuit's nominal toroidal-ring resistance [ohm].

    ``rho * 2 pi r / A`` at the section's own area and centroid radius.  A family
    whose material no source fixes is seeded at the vessel steel so the circuit
    has a starting resistance at all; :func:`unsourced_material` names those
    circuits, and a fit that moves one of them is fitting a resistivity nobody
    published rather than refining one somebody did.
    """

    values = np.zeros(len(turns))
    for index, turn in enumerate(turns):
        material = passive_material(turn.family) or STAINLESS_STEEL
        values[index] = material.loop_resistance(turn.area, turn.centroid[0])
    return values


def unsourced_material(turns: Sequence[PassiveTurn]) -> tuple[str, ...]:
    """Return the circuits whose material no source fixes."""

    return tuple(turn.name for turn in turns if passive_material(turn.family) is None)


def probe_coupling(
    turns: Sequence[PassiveTurn],
    targets: Sequence[ProbeTarget],
) -> np.ndarray:
    """Field each circuit produces along each probe's sensitive axis [T/A]."""

    target_r = np.asarray([target.r for target in targets], dtype=float)
    target_z = np.asarray([target.z for target in targets], dtype=float)
    cosine = np.asarray([target.radial_cosine for target in targets], dtype=float)
    sine = np.asarray([target.axial_sine for target in targets], dtype=float)
    coupling = np.zeros((len(targets), len(turns)))
    for column, turn in enumerate(turns):
        _, radial, axial = polygon_greens(target_r, target_z, turn.vertices)
        coupling[:, column] = cosine * radial + sine * axial
    return coupling


def coil_coupling(
    coil_outlines: Mapping[str, Sequence[np.ndarray]],
    turns: Sequence[PassiveTurn],
    *,
    cells_across: int = CELLS_ACROSS_THICKNESS,
    point_target: int = QUADRATURE_POINT_TARGET,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Flux each passive circuit links per ampere-turn of each coil [Wb/A].

    The driving term of the decay problem.  A coil's winding packs are summed at
    unit current each, so the column is per ampere in one turn and a shot's
    ampere-turn drive scales it directly.
    """

    point_r, point_z, weight, owner = quadrature_grid(
        turns, cells_across=cells_across, point_target=point_target
    )
    families = tuple(sorted(coil_outlines))
    linkage = np.zeros((len(turns), len(families)))
    for column, family in enumerate(families):
        for vertices in coil_outlines[family]:
            psi, _, _ = polygon_greens(point_r, point_z, np.asarray(vertices))
            linkage[:, column] += np.bincount(
                owner, weights=weight * psi, minlength=len(turns)
            )
    return (families, linkage)


def linkage_provenance(
    turns: Sequence[PassiveTurn],
    linkage: Linkage,
    *,
    physical_digest: str,
    shot: int,
) -> dict[str, Any]:
    """Return the record that pins a linkage matrix to the geometry it came from.

    A committed inductance is only reusable if the reader can tell which
    configuration produced it, so the physical digest, the registry selection and
    the per-circuit sections travel with the matrix.
    """

    resistance = nominal_resistance(turns)
    return {
        "cells_across_thickness": CELLS_ACROSS_THICKNESS,
        "circuit_count": len(turns),
        "circuits": [
            {
                "area": turn.area,
                "centroid_r": turn.centroid[0],
                "centroid_z": turn.centroid[1],
                "enclosed_coil": turn.enclosed_coil,
                "family": turn.family,
                "name": turn.name,
                "nominal_resistance": float(resistance[index]),
                "nominal_time_constant": float(
                    linkage.matrix[index, index] / resistance[index]
                ),
                "quadrature_points": int(linkage.quadrature_points[index]),
                "self_inductance": float(linkage.matrix[index, index]),
                "thickness": turn.thickness,
            }
            for index, turn in enumerate(turns)
        ],
        "physical_digest": physical_digest,
        "quadrature_points": int(linkage.quadrature_points.sum()),
        "reciprocity_residual": linkage.reciprocity_residual,
        "registry_shot": shot,
        "unsourced_material": list(unsourced_material(turns)),
    }
