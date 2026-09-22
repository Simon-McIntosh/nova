"""Localize floating-point changes in a frozen polynomial evaluator."""

import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import measure_bernstein as prior
from nova.jax.config import configure_dtypes
import nova.linalg.interpolant as module
from reference_clip import polygon_moments

configure_dtypes()
output = Path(__file__).resolve().parent / "bernstein-reference"
result = {}
reference = prior.baseline_module("nova/linalg/interpolant.py", "probe_baseline")
for order in (3, 6):
    coordinate = jnp.asarray([0.01, 0.17, 0.5, 0.82, 0.99])
    baseline = np.asarray(
        reference.Bernstein(order=order).coefficent_matrix(coordinate)
    )
    candidate = np.asarray(module.Bernstein(order=order).coefficent_matrix(coordinate))
    direct = np.asarray(reference.Bernstein(order=order).binom(jnp.arange(order + 1)))
    frozen = np.asarray(module._binomial_coefficients(order, True))
    result[str(order)] = {
        "basis_equal": bool(np.array_equal(baseline, candidate)),
        "basis_differing_entries": int(np.count_nonzero(baseline != candidate)),
        "basis_max_absolute": float(np.max(np.abs(baseline - candidate))),
        "coefficient_equal": bool(np.array_equal(direct, frozen)),
        "baseline_coefficient_hex": [float(x).hex() for x in direct],
        "frozen_coefficient_hex": [float(x).hex() for x in frozen],
    }
for name in ("production-patch-reference", "bicubic-reference"):
    arrays = np.load(output / (name + "-arrays.npz"))
    before = arrays["baseline_vertices"]
    after = arrays["candidate_vertices"]
    cells = arrays["cell_ids"]
    counts = [len(arrays[f"reference_polygon_{cell}"]) for cell in cells]
    reduced = {}
    for arm in ("baseline", "candidate"):
        values = np.stack(
            [
                polygon_moments(points[:count], centre)
                for points, count, centre in zip(
                    arrays[arm + "_vertices"], counts, arrays["centres"], strict=True
                )
            ]
        )
        exact = arrays["reference_moments"]
        stored = arrays[arm + "_moments"]
        reduced[arm] = {
            "stored_vertex_reference_relative": [
                float(x)
                for x in np.sqrt(np.sum((values - exact) ** 2, axis=0))
                / np.sqrt(np.sum(exact**2, axis=0))
            ],
            "device_reduction_relative": [
                float(x)
                for x in np.sqrt(np.sum((stored - values) ** 2, axis=0))
                / np.sqrt(np.sum(exact**2, axis=0))
            ],
        }
    result[name] = {
        "vertices_equal": bool(np.array_equal(before, after)),
        "vertex_differing_entries": int(np.count_nonzero(before != after)),
        "maximum_vertex_difference": float(np.max(np.abs(before - after))),
        "reductions": reduced,
    }
(output / "rounding-probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
