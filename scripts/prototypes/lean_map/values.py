"""Precompute the dense values-only plasma map at interior Gauss points."""

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from scripts.analytic_oracle_fixtures.coupling_build import build_flux_blocks


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DenseValues:
    matrix: jax.Array
    exterior: jax.Array

    def apply(self, moments):
        return self.exterior + jnp.einsum("tcm,cm->t", self.matrix, moments)


def build_values(machine, rule, total_field, reference_moments):
    """Pose the analytic exterior on a fixed matrix of cell-source responses."""
    points = np.asarray(rule.points).reshape(-1, 2)
    centres = np.asarray(rule.centre)
    started = perf_counter()
    blocks = build_flux_blocks(points, machine.cell_polygons, centres)
    matrix = np.stack(blocks, axis=-1)
    assembly_seconds = perf_counter() - started
    reference = np.asarray(reference_moments)
    analytic = np.asarray(jax.jit(jax.vmap(total_field.value))(jnp.asarray(points)))
    if np.ptp(analytic) <= 0:
        raise AssertionError("the analytic Gauss field is uniform")
    exterior = analytic - np.einsum("tcm,cm->t", matrix, reference)
    values = DenseValues(jnp.asarray(matrix), jnp.asarray(exterior))
    jax.block_until_ready(values)
    reconstruction = np.asarray(values.apply(jnp.asarray(reference)))
    calibration_error = float(np.max(np.abs(reconstruction - analytic)))
    return values, {
        "target_points": len(points),
        "source_cells": len(centres),
        "matrix_bytes": matrix.nbytes,
        "assembly_seconds": assembly_seconds,
        "calibration_sup_absolute": calibration_error,
        "analytic_flux_range": float(np.ptp(analytic)),
    }
