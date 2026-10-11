"""Device-batched assembly of the hex-carrier coupling blocks.

The certificate machine's twelve flux coupling blocks map each cell's current
moment triplet onto a family of target points.  Each block is a dense
``(target, section)`` matrix, so assembling it on the host is an O(cells^2)
Python/numpy reduction that grows without bound as the cell count climbs.

This module assembles the same three flux rows ``(uniform, radial, vertical)``
through one traced, compiled program per block shape: the exact moment kernel
:func:`nova.biot.tiledassembly.tile_evaluator` already carries for the broad
read's coupling.

The uniform row reproduces the host analytic builder to round-off (1.4e-07
norm-relative at 550 cells, 4.2e-09 at 2,000).  The two moment rows do not: they
differ from the host builder by up to 1.05e-03 norm-relative at 550 cells and
6.6e-05 at 2,000, because the packed kernel accumulates each target tile in the
reflected edge pairing :func:`nova.biot.polygon.pad_batch` fixes, an order that
cannot reproduce the host builder's per-section reduction about each centroid.
That gap is open, and it is not yet established which side carries the error:
an independent reference evaluation deviates from both paths on the moment
rows, so a repair may land in either, starting from the module that owns the
moment kernel, :mod:`nova.biot.polygonanalytic`.
"""

from __future__ import annotations

import numpy as np

from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import _horizontal_reflection, _section_centroid
from nova.biot.tiledassembly import TilePlan, tile_evaluator
from nova.jax.config import Precision

FLUX_ROWS = (0, 1, 2)


def _reflection_geometry(
    sections: list[np.ndarray], edge_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the packed reflection axis and per-edge partner for every section.

    A section symmetric about a horizontal axis has each edge paired with its
    mirror; the packed kernel uses that pairing to cancel the source-coordinate
    part odd about the axis.  Sections with no reflection carry a NaN axis and
    pair every edge with itself, which leaves the reduction unchanged.
    """
    axis = np.full(len(sections), np.nan, dtype=np.float64)
    partner = np.repeat(
        np.arange(edge_count, dtype=np.intp)[:, None], len(sections), axis=1
    )
    for column, vertices in enumerate(sections):
        reflection = _horizontal_reflection(vertices)
        if reflection is None:
            continue
        axis_value, vertex_partner = reflection
        axis[column] = axis_value
        count = len(vertices)
        for index in range(count):
            partner[index, column] = vertex_partner[(index + 1) % count]
    return axis, partner


def build_flux_blocks(
    targets: np.ndarray,
    polygons: tuple[np.ndarray, ...],
    expansion_points: np.ndarray,
    *,
    precision: Precision = Precision.DOUBLE,
    tile_side: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(uniform, radial, vertical)`` flux blocks for one family.

    ``targets`` is ``(T, 2)``, ``polygons`` carries one packed section per
    source column, and ``expansion_points`` is ``(S, 2)``.  Each returned block
    is ``(T, S)`` in Wb/A, matching
    :func:`nova.biot.polygonanalytic.polygon_analytic_flux_moments_batched`.
    """
    target = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    target_r = np.ascontiguousarray(target[:, 0])
    target_z = np.ascontiguousarray(target[:, 1])
    sections = [
        np.ascontiguousarray(np.asarray(polygon, dtype=np.float64))
        for polygon in polygons
    ]
    expansion = np.asarray(expansion_points, dtype=np.float64)
    if expansion.shape != (len(sections), 2):
        raise ValueError(
            "expansion_points must hold one finite (R, Z) pair per section"
        )

    edge, weight, norm = pad_batch(sections)
    section_centre = np.column_stack(
        [_section_centroid(section) for section in sections]
    )
    reflection_axis, reflection_partner = _reflection_geometry(sections, edge.shape[0])

    n_target = target_r.size
    n_source = len(sections)
    expansion_axis_first = np.ascontiguousarray(expansion.T)
    target_tile = min(int(tile_side), n_target)
    source_tile = min(int(tile_side), n_source)
    plan = TilePlan(
        target_tile=target_tile,
        source_tile=source_tile,
        block=target_tile * source_tile,
        n_panels=16,
        n_nodes=48,
    )
    evaluate = tile_evaluator(
        plan,
        batched=True,
        kernel="moments",
        precision=precision,
        edge_count=int(edge.shape[0]),
    )
    rows = np.empty((len(FLUX_ROWS), n_target, n_source), dtype=np.float64)
    for target_slice, source_slice in plan.tiles(n_target, n_source):
        tile = evaluate(
            target_r[target_slice],
            target_z[target_slice],
            edge[..., source_slice],
            weight[:, source_slice],
            norm[source_slice],
            section_centre[:, source_slice],
            expansion_axis_first[:, source_slice],
            reflection_axis[source_slice],
            reflection_partner[:, source_slice],
        )
        for row, component in zip(rows, FLUX_ROWS, strict=True):
            row[target_slice, source_slice] = tile[component]
    return rows[0], rows[1], rows[2]
