"""Assemble the three flux rows of a cell coupling without unused field rows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import _conditioned_flux_moments, _section_centroid
from nova.jax.config import Precision, resolve_precision

FLUX_ROWS = (0, 1, 2)
PARALLEL_PAIR_FLOOR = 8192


@jax.jit
def _flux_pairs(radius, height, edge, weight, norm, centre, expansion):
    """Evaluate only the three area-integrated flux rows for packed pairs."""
    return jnp.stack(
        _conditioned_flux_moments(
            jnp, radius, height, edge, weight, norm, centre, expansion
        )
    )


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

    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = np.float32 if resolved is Precision.SINGLE else np.float64
    edge, weight, norm = pad_batch(sections)
    section_centre = np.column_stack(
        [_section_centroid(section) for section in sections]
    )
    n_target = target_r.size
    n_source = len(sections)
    if tile_side < 1:
        raise ValueError("tile_side must be positive")
    batch = 2 * int(tile_side)
    present = ~np.signbit(weight)
    extent = np.max(
        np.where(
            present,
            (edge[:, 0] - section_centre[0]) ** 2
            + (edge[:, 1] - section_centre[1]) ** 2,
            0.0,
        ),
        axis=0,
    )
    near = (np.abs(target_r[:, None]) - section_centre[0]) ** 2 + (
        target_z[:, None] - section_centre[1]
    ) ** 2 < 16.0 * extent
    rows = np.empty((len(FLUX_ROWS), n_target * n_source), dtype=np.float64)
    chunks = []
    for nearby in (False, True):
        indices = np.flatnonzero(near.ravel() == nearby)
        for start in range(0, len(indices), batch):
            chunks.append(indices[start : start + batch])

    def evaluate_chunk(selected):
        padded = np.pad(selected, (0, batch - len(selected)), mode="edge")
        targets = padded // n_source
        sources = padded % n_source
        values = np.asarray(
            _flux_pairs(
                target_r[targets].astype(dtype),
                target_z[targets].astype(dtype),
                edge[:, :, sources].astype(dtype),
                weight[:, sources].astype(dtype),
                norm[sources].astype(dtype),
                section_centre[:, sources].astype(dtype),
                expansion.T[:, sources].astype(dtype),
            )
        )
        return selected, values[:, : len(selected)]

    if chunks:
        selected, values = evaluate_chunk(chunks[0])
        rows[:, selected] = values
        if n_target * n_source < PARALLEL_PAIR_FLOOR:
            completed = map(evaluate_chunk, chunks[1:])
            for selected, values in completed:
                rows[:, selected] = values
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for selected, values in pool.map(evaluate_chunk, chunks[1:]):
                    rows[:, selected] = values
    return tuple(row.reshape(n_target, n_source) for row in rows)
