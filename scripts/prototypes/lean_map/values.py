"""Precompute the dense values-only plasma map at interior Gauss points."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scripts.analytic_oracle_fixtures.coupling_build import build_flux_blocks
from scripts.prototypes.read_route.jets import (
    VALUES_KIND,
    load_operand,
    operand_cache_directory,
    operand_digest,
    store_operand,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DenseValues:
    matrix: jax.Array
    exterior: jax.Array

    def apply(self, moments):
        return self.exterior + jnp.einsum("tcm,cm->t", self.matrix, moments)


def source_bytes() -> tuple[bytes, ...]:
    """Return the bytes of the two source files a value key is computed over."""
    from nova.biot import polygonanalytic
    from scripts.analytic_oracle_fixtures import coupling_build

    return (
        Path(coupling_build.__file__).read_bytes(),
        *(Path(polygonanalytic.__file__).read_bytes(),),
    )


def values_digest(points, polygons, centres) -> str:
    """Key the value matrix on its geometry and on both source files.

    ``points`` and ``centres`` carry the Gauss rule, ``polygons`` the source
    columns; a change to any of them, or to the analytic block builder or the
    polygon kernel, moves the key and forces a rebuild.
    """
    lengths = np.array([len(polygon) for polygon in polygons], dtype=np.int64)
    return operand_digest(
        np.asarray(points, dtype=np.float64).tobytes(),
        np.asarray(centres, dtype=np.float64).tobytes(),
        lengths.tobytes(),
        *(np.asarray(polygon, dtype=np.float64).tobytes() for polygon in polygons),
        *source_bytes(),
    )


def build_value_matrix(points, polygons, centres, persist=True, receipt=None):
    """Return the ``(target, cell, moment)`` flux matrix, cached when persistent.

    The exterior is not cached: it depends on the field and the reference
    moments and is cheap to pose once the matrix is in hand.
    """
    shape = (len(points), len(centres), 3)
    directory = None
    cached = None
    load_seconds = None
    if persist:
        digest = values_digest(points, polygons, centres)
        directory = operand_cache_directory(VALUES_KIND, digest)
        if receipt is not None:
            receipt["digest"] = digest
            receipt["cache_path"] = str(directory / "matrix.npy")
        load_started = perf_counter()
        cached = load_operand(directory, shape)
        if cached is not None:
            load_seconds = perf_counter() - load_started
    matrix_bytes = int(np.prod(shape) * 8)
    if cached is not None:
        matrix, build_seconds = cached
        matrix = np.asarray(matrix)
    else:
        build_started = perf_counter()
        blocks = build_flux_blocks(points, polygons, centres)
        matrix = np.stack(blocks, axis=-1)
        matrix_bytes = matrix.nbytes
        build_seconds = perf_counter() - build_started
        if directory is not None:
            store_operand(directory, matrix, build_seconds)
    if receipt is not None:
        receipt.update(
            cache_hit=cached is not None,
            build_seconds=build_seconds,
            load_seconds=load_seconds,
            matrix_bytes=matrix_bytes,
        )
    return matrix


def build_values(machine, rule, total_field, reference_moments, persist=True):
    """Pose the analytic exterior on a fixed matrix of cell-source responses."""
    points = np.asarray(rule.points).reshape(-1, 2)
    centres = np.asarray(rule.centre)
    polygons = tuple(
        np.asarray(item, dtype=np.float64) for item in machine.cell_polygons
    )
    receipt: dict[str, Any] = {}
    matrix = build_value_matrix(points, polygons, centres, persist, receipt)
    reference = np.asarray(reference_moments)
    analytic = np.asarray(jax.jit(jax.vmap(total_field.value))(jnp.asarray(points)))
    if np.ptp(analytic) <= 0:
        raise AssertionError("the analytic Gauss field is uniform")
    exterior = analytic - np.einsum("tcm,cm->t", matrix, reference)
    values = DenseValues(jnp.asarray(matrix), jnp.asarray(exterior))
    jax.block_until_ready(values)
    reconstruction = np.asarray(values.apply(jnp.asarray(reference)))
    calibration_error = float(np.max(np.abs(reconstruction - analytic)))
    receipt.update(
        target_points=len(points),
        source_cells=len(centres),
        assembly_seconds=(receipt.get("load_seconds") or 0.0)
        + (receipt.get("build_seconds") or 0.0),
        calibration_sup_absolute=calibration_error,
        analytic_flux_range=float(np.ptp(analytic)),
    )
    return values, receipt