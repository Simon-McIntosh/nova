"""Measure the host and traced null-selection routes at their live call sites.

The module has two deliberately separate modes.  ``measure`` runs the numerical
and timing work once on one allocated machine and writes a machine-local JSON
record.  ``assemble`` combines the CPU and GPU records without repeating any
measurement and writes the committed JSON and SVG evidence.

The independent references are Python's :mod:`bisect` for insertion indices and
analytic quadratic fields for the wall and surface fits.  The analytic fields
provide the stationary coordinate, flux, and Hessian class before either Nova
route is called; neither implementation can therefore arbitrate the other.
"""

from __future__ import annotations

import argparse
import bisect as python_bisect
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nova.geometry import select as host_select
from nova.jax import select as traced_select


REPEATS = 9
WALL_SIZES = (24, 48, 128, 512, 2048)
SUBNULL_BATCHES = (1, 2, 6, 10, 64, 256)
BISECT_SIZES = (64, 1024, 65536)


def _version(distribution: str) -> str:
    """Return an installed distribution version."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str:
    """Return the source revision measured by this checkout."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    """Return hashes of the select implementation and its live composites."""
    paths = (
        "nova/jax/select.py",
        "nova/geometry/select.py",
        "nova/jax/stencil_nulls.py",
        "nova/jax/null.py",
    )
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def _cpu_model() -> str:
    """Return the first Linux CPU model description, when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _serialise(value: Any) -> Any:
    """Convert array scalars and containers into JSON-compatible values."""
    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    if isinstance(value, np.ndarray):
        return _serialise(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _serialise(np.asarray(value))
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _synchronise(value: Any) -> Any:
    """Synchronise a possibly nested JAX result and return it unchanged."""
    for leaf in jax.tree.leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


def _fastest(call: Callable[[], Any]) -> float:
    """Return the minimum warm wall time in microseconds."""
    _synchronise(call())
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        value = call()
        _synchronise(value)
        best = min(best, (time.perf_counter_ns() - start) / 1e3)
    return best


def _capture(call: Callable[[], Any]) -> dict[str, Any]:
    """Return either a call value or its exact exception class and first line."""
    try:
        value = call()
        _synchronise(value)
        return {"status": "returned", "value": _serialise(value)}
    except Exception as error:  # the exception is the semantic evidence
        lines = str(error).splitlines()
        return {
            "status": "raised",
            "exception": type(error).__name__,
            "message": lines[0] if lines else "",
        }


def _wall_case(nodes: int, polarity: int = 1) -> dict[str, np.ndarray | float | int]:
    """Return a smooth wall-loop case with an analytic off-node extremum."""
    theta = np.linspace(0.0, 2.0 * np.pi, nodes, endpoint=False)
    radial = 1.0 + 0.42 * np.cos(theta) + 0.025 * np.cos(3.0 * theta)
    vertical = 0.55 * np.sin(theta) - 0.018 * np.sin(2.0 * theta)
    segment = np.hypot(np.diff(radial), np.diff(vertical))
    length = np.r_[0.0, np.cumsum(segment)]
    index = max(2, min(nodes - 3, int(round(0.37 * nodes))))
    fraction = 0.37
    stationary_length = length[index] + fraction * (length[index + 1] - length[index])
    flux_at_stationary = 0.03125
    curvature = -0.8 if polarity > 0 else 0.8
    flux = curvature * (length - stationary_length) ** 2 + flux_at_stationary
    radial_truth = radial[index] + fraction * (radial[index + 1] - radial[index])
    vertical_truth = vertical[index] + fraction * (
        vertical[index + 1] - vertical[index]
    )
    null_type = 1 if curvature < 0 else -1
    return {
        "radial": radial,
        "vertical": vertical,
        "flux": flux,
        "polarity": polarity,
        "truth": np.array(
            [radial_truth, vertical_truth, flux_at_stationary, null_type],
            dtype=np.float64,
        ),
    }


def _surface_case(
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return analytic quadratic clusters and their exact stationary results."""
    radial_rows = []
    vertical_rows = []
    flux_rows = []
    truth_rows = []
    local = np.array([-1.0, 0.0, 1.0])
    for index in range(batch):
        spacing_r = 0.018 + 0.004 * (index % 5)
        spacing_z = 0.022 + 0.003 * (index % 7)
        radial_axis = 6.0 + 0.003 * (index % 11)
        vertical_axis = -0.22 + 0.004 * (index % 13)
        radial, vertical = np.meshgrid(
            radial_axis + spacing_r * local,
            vertical_axis + spacing_z * local,
            indexing="ij",
        )
        radial = radial.ravel()
        vertical = vertical.ravel()
        kind = (-1, 0, 1)[index % 3]
        if kind == -1:
            coefficient_r, coefficient_z, cross = 1.2, 0.85, 0.16
        elif kind == 1:
            coefficient_r, coefficient_z, cross = -1.2, -0.85, -0.16
        else:
            coefficient_r, coefficient_z, cross = 1.2, -0.85, 0.12
        flux_axis = -0.04 + 2e-4 * index
        delta_r = radial - radial_axis
        delta_z = vertical - vertical_axis
        flux = (
            coefficient_r * delta_r**2
            + coefficient_z * delta_z**2
            + cross * delta_r * delta_z
            + flux_axis
        )
        radial_rows.append(radial)
        vertical_rows.append(vertical)
        flux_rows.append(flux)
        truth_rows.append([radial_axis, vertical_axis, flux_axis, kind])
    return (
        np.asarray(radial_rows),
        np.asarray(vertical_rows),
        np.asarray(flux_rows),
        np.asarray(truth_rows),
    )


def _flatten_host_subnull(value: Any) -> np.ndarray:
    """Flatten the host's nested ``((R, Z), psi, type)`` return."""
    coordinates, flux, kind = value
    return np.asarray([coordinates[0], coordinates[1], flux, kind], dtype=np.float64)


def _host_subnull_batch(radial: np.ndarray, vertical: np.ndarray, flux: np.ndarray):
    """Run the live host call pattern over independent candidate clusters."""
    return np.stack(
        [
            _flatten_host_subnull(host_select.subnull(r, z, psi))
            for r, z, psi in zip(radial, vertical, flux)
        ]
    )


def _traced_stacked_batch():
    """Build the live stacked-cluster traced batch route."""
    return jax.jit(jax.vmap(traced_select.subnull))


def _wall_accuracy(dtype: Any) -> dict[str, Any]:
    """Compare both wall routes with an analytic quadratic-in-length reference."""
    rows = []
    for nodes in (24, 48, 128, 512):
        for polarity in (-1, 1):
            case = _wall_case(nodes, polarity)
            radial = np.asarray(case["radial"])
            vertical = np.asarray(case["vertical"])
            flux = np.asarray(case["flux"])
            truth = np.asarray(case["truth"])
            host = np.r_[
                host_select.wall_flux(radial, vertical, flux, polarity), truth[3]
            ]
            traced = np.asarray(
                traced_select.wall_flux(
                    jnp.asarray(radial, dtype=dtype),
                    jnp.asarray(vertical, dtype=dtype),
                    jnp.asarray(flux, dtype=dtype),
                    polarity,
                )
            ).astype(np.float64)
            rows.append(
                {
                    "nodes": nodes,
                    "polarity": polarity,
                    "host_max_abs": float(np.max(np.abs(host - truth))),
                    "traced_max_abs": float(np.max(np.abs(traced - truth))),
                    "route_max_abs": float(np.max(np.abs(traced - host))),
                    "host": host,
                    "traced": traced,
                    "truth": truth,
                }
            )
    return {
        "reference": "analytic quadratic in independently accumulated wall length",
        "rows": rows,
        "host_worst_abs": max(row["host_max_abs"] for row in rows),
        "traced_worst_abs": max(row["traced_max_abs"] for row in rows),
        "route_worst_abs": max(row["route_max_abs"] for row in rows),
    }


def _surface_accuracy(dtype: Any, array_subnull: Callable[..., Any] | None):
    """Compare sub-grid fits with exact analytic quadratic stationary points."""
    radial, vertical, flux, truth = _surface_case(18)
    host = _host_subnull_batch(radial, vertical, flux)
    stacks = np.stack([radial, vertical, flux], axis=1)
    traced_call = _traced_stacked_batch()
    traced = np.asarray(traced_call(jnp.asarray(stacks, dtype=dtype))).astype(
        np.float64
    )
    rows: dict[str, Any] = {
        "reference": (
            "analytic quadratic stationary coordinate, value, and Hessian class"
        ),
        "host_max_abs_by_component": np.max(np.abs(host - truth), axis=0),
        "traced_max_abs_by_component": np.max(np.abs(traced - truth), axis=0),
        "route_max_abs_by_component": np.max(np.abs(traced - host), axis=0),
        "host": host,
        "traced": traced,
        "truth": truth,
    }
    if array_subnull is not None:
        array_call = jax.jit(jax.vmap(array_subnull))
        arrays = np.asarray(
            array_call(
                jnp.asarray(radial, dtype=dtype),
                jnp.asarray(vertical, dtype=dtype),
                jnp.asarray(flux, dtype=dtype),
            )
        ).astype(np.float64)
        rows["three_array_max_abs_by_component"] = np.max(
            np.abs(arrays - truth), axis=0
        )
        rows["signature_route_max_abs"] = float(np.max(np.abs(arrays - traced)))
    return rows


def _bisect_accuracy(dtype: Any) -> dict[str, Any]:
    """Compare insertion indices with Python's independent standard library."""
    vector = np.sort(
        np.r_[np.linspace(-4.0, 4.0, 127), np.array([-1.0, -1.0, 0.0, 0.0, 2.0])]
    )
    values = np.linspace(-5.0, 5.0, 81)
    left_truth = np.array(
        [python_bisect.bisect_left(vector.tolist(), x) for x in values]
    )
    right_truth = np.array(
        [python_bisect.bisect_right(vector.tolist(), x) for x in values]
    )
    host_left = np.array([host_select.bisect(vector, x) for x in values])
    host_right = np.array([host_select.bisect_right(vector, x) for x in values])
    traced_left = np.array(
        [traced_select.bisect(jnp.asarray(vector, dtype=dtype), x) for x in values]
    )
    traced_right = np.array(
        [
            traced_select.bisect_right(jnp.asarray(vector, dtype=dtype), x)
            for x in values
        ]
    )
    return {
        "reference": "Python standard-library bisect_left and bisect_right",
        "queries": int(values.size),
        "host_left_mismatches": int(np.count_nonzero(host_left != left_truth)),
        "traced_left_mismatches": int(np.count_nonzero(traced_left != left_truth)),
        "host_right_mismatches": int(np.count_nonzero(host_right != right_truth)),
        "traced_right_mismatches": int(np.count_nonzero(traced_right != right_truth)),
    }


def _wall_cost(dtype: Any) -> list[dict[str, Any]]:
    """Return call latency over live wall-loop node counts."""
    rows = []
    for nodes in WALL_SIZES:
        case = _wall_case(nodes)
        radial = np.asarray(case["radial"])
        vertical = np.asarray(case["vertical"])
        flux = np.asarray(case["flux"])
        traced_args = (
            jnp.asarray(radial, dtype=dtype),
            jnp.asarray(vertical, dtype=dtype),
            jnp.asarray(flux, dtype=dtype),
            1,
        )
        host_us = _fastest(
            lambda r=radial, z=vertical, psi=flux: host_select.wall_flux(r, z, psi, 1)
        )
        traced_us = _fastest(lambda args=traced_args: traced_select.wall_flux(*args))
        rows.append(
            {
                "nodes": nodes,
                "host_us": host_us,
                "traced_us": traced_us,
                "traced_over_host": traced_us / host_us,
            }
        )
    return rows


def _subnull_cost(
    dtype: Any, array_subnull: Callable[..., Any] | None
) -> list[dict[str, Any]]:
    """Return latency over the candidate counts reached by both null finders."""
    rows = []
    for batch in SUBNULL_BATCHES:
        radial, vertical, flux, _ = _surface_case(batch)
        stacks = jnp.asarray(np.stack([radial, vertical, flux], axis=1), dtype=dtype)
        stacked_call = _traced_stacked_batch()
        host_us = _fastest(
            lambda r=radial, z=vertical, psi=flux: _host_subnull_batch(r, z, psi)
        )
        stacked_us = _fastest(lambda call=stacked_call, arg=stacks: call(arg))
        row: dict[str, Any] = {
            "batch": batch,
            "host_us": host_us,
            "traced_stacked_us": stacked_us,
            "traced_over_host": stacked_us / host_us,
        }
        if array_subnull is not None:
            array_call = jax.jit(jax.vmap(array_subnull))
            radial_device = jnp.asarray(radial, dtype=dtype)
            vertical_device = jnp.asarray(vertical, dtype=dtype)
            flux_device = jnp.asarray(flux, dtype=dtype)

            def call_arrays(
                call=array_call,
                r=radial_device,
                z=vertical_device,
                psi=flux_device,
            ):
                return call(r, z, psi)

            array_us = _fastest(call_arrays)
            row["traced_three_array_us"] = array_us
            row["three_array_over_stacked"] = array_us / stacked_us
        rows.append(row)
    return rows


def _bisect_cost(dtype: Any) -> list[dict[str, Any]]:
    """Return scalar insertion latency over real vector-size regimes."""
    rows = []
    for size in BISECT_SIZES:
        vector = np.linspace(-1.0, 1.0, size)
        device_vector = jnp.asarray(vector, dtype=dtype)
        value = 0.123456
        host_left = _fastest(lambda v=vector: host_select.bisect(v, value))
        traced_left = _fastest(lambda v=device_vector: traced_select.bisect(v, value))
        host_right = _fastest(lambda v=vector: host_select.bisect_right(v, value))
        traced_right = _fastest(
            lambda v=device_vector: traced_select.bisect_right(v, value)
        )
        rows.append(
            {
                "vector_size": size,
                "host_left_us": host_left,
                "traced_left_us": traced_left,
                "left_traced_over_host": traced_left / host_left,
                "host_right_us": host_right,
                "traced_right_us": traced_right,
                "right_traced_over_host": traced_right / host_right,
            }
        )
    return rows


def _autodiff(dtype: Any) -> dict[str, Any]:
    """Prove that the live traced composites carry finite input gradients."""
    wall = _wall_case(48)
    radial = jnp.asarray(wall["radial"], dtype=dtype)
    vertical = jnp.asarray(wall["vertical"], dtype=dtype)
    flux = jnp.asarray(wall["flux"], dtype=dtype)
    wall_gradient = jax.grad(
        lambda psi: traced_select.wall_flux(radial, vertical, psi, 1)[2]
    )(flux)

    r, z, psi, _ = _surface_case(1)
    fixed_r = jnp.asarray(r[0], dtype=dtype)
    fixed_z = jnp.asarray(z[0], dtype=dtype)
    subnull_gradient = jax.grad(
        lambda values: traced_select.subnull(jnp.stack([fixed_r, fixed_z, values]))[2]
    )(jnp.asarray(psi[0], dtype=dtype))
    return {
        "wall_flux_all_finite": bool(np.all(np.isfinite(np.asarray(wall_gradient)))),
        "wall_flux_max_abs": float(np.max(np.abs(np.asarray(wall_gradient)))),
        "subnull_all_finite": bool(np.all(np.isfinite(np.asarray(subnull_gradient)))),
        "subnull_max_abs": float(np.max(np.abs(np.asarray(subnull_gradient)))),
    }


def _semantics() -> list[dict[str, Any]]:
    """Return all known signature, dtype, and degenerate-case differences."""
    vector = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([1.5, 2.5])
    wall = _wall_case(24)
    radial = np.asarray(wall["radial"])
    vertical = np.asarray(wall["vertical"])
    flux = np.asarray(wall["flux"])
    plane_coefficients = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    x, z, psi, _ = _surface_case(1)
    far_flux = (x[0] - 20.0) ** 2 + (z[0] - 20.0) ** 2
    host_valid = host_select.subnull(x[0], z[0], psi[0])
    traced_valid = traced_select.subnull(jnp.asarray([x[0], z[0], psi[0]]))
    return [
        {
            "logical_pair": "bisect",
            "difference": "return scalar width",
            "host": str(np.asarray(host_select.bisect(np.arange(8.0), 3.5)).dtype),
            "traced": str(np.asarray(traced_select.bisect(jnp.arange(8.0), 3.5)).dtype),
            "consequence": (
                "indices agree; traced scalar width follows JAX configuration"
            ),
        },
        {
            "logical_pair": "bisect_right",
            "difference": "traceability",
            "host": "numba-jitted and callable from compiled host code",
            "traced": _capture(
                lambda: jax.jit(traced_select.bisect_right)(vector, 1.5)
            ),
            "consequence": (
                "the traced peer uses Python boolean control flow and cannot be traced"
            ),
        },
        {
            "logical_pair": "bisect_2d",
            "difference": "reachability",
            "host": _capture(lambda: host_select.bisect_2d(np.arange(4.0), [1.5, 2.5])),
            "traced": _capture(lambda: traced_select.bisect_2d(vector, values)),
            "consequence": (
                "every non-empty traced call reaches immutable item assignment"
            ),
        },
        {
            "logical_pair": "wall_index",
            "difference": "NaN selection",
            "host": _capture(
                lambda: host_select.wall_index(np.array([1.0, np.nan, 2.0]))
            ),
            "traced": _capture(
                lambda: traced_select.wall_index(jnp.array([1.0, jnp.nan, 2.0]))
            ),
            "consequence": "host argmax selects NaN; traced nanargmax ignores it",
        },
        {
            "logical_pair": "wall_length",
            "difference": "zero quadratic curvature",
            "host": _capture(
                lambda: host_select.wall_length(np.array([0.0, 1.0, 2.0]))
            ),
            "traced": _capture(
                lambda: traced_select.wall_length(jnp.array([0.0, 1.0, 2.0]))
            ),
            "consequence": "host raises while traced returns an infinity sentinel",
        },
        {
            "logical_pair": "wall_flux",
            "difference": "signature and zero polarity",
            "host": _capture(lambda: host_select.wall_flux(radial, vertical, flux, 0)),
            "traced": _capture(
                lambda: traced_select.wall_flux(
                    jnp.asarray(radial), jnp.asarray(vertical), jnp.asarray(flux), 0
                )
            ),
            "consequence": (
                "host returns a three-tuple and defaults polarity; traced requires "
                "polarity and returns a four-array with type"
            ),
        },
        {
            "logical_pair": "quadratic_wall",
            "difference": "least-squares precision and interface",
            "host": "promotes flux to float64 and passes rcond=-1 for numba gelsd",
            "traced": "uses configured JAX dtype and jnp.linalg.lstsq default rcond",
            "consequence": "the bodies are not identical even when valid fits agree",
        },
        {
            "logical_pair": "quadratic_surface",
            "difference": "least-squares precision and interface",
            "host": "promotes flux to float64 and passes rcond=-1 for numba gelsd",
            "traced": "uses configured JAX dtype and jnp.linalg.lstsq default rcond",
            "consequence": (
                "direct select/null imports are fp32 unless another module enables fp64"
            ),
        },
        {
            "logical_pair": "null_type",
            "difference": "planar surface",
            "host": _capture(lambda: host_select.null_type(plane_coefficients)),
            "traced": _capture(
                lambda: traced_select.null_type(jnp.asarray(plane_coefficients))
            ),
            "consequence": (
                "host raises; traced emits NaN because traced control flow cannot "
                "raise on a data value"
            ),
        },
        {
            "logical_pair": "null_coordinate",
            "difference": "zero Hessian determinant",
            "host": _capture(lambda: host_select.null_coordinate(plane_coefficients)),
            "traced": _capture(
                lambda: traced_select.null_coordinate(jnp.asarray(plane_coefficients))
            ),
            "consequence": (
                "host divides by zero; traced floors the determinant at 1e-30"
            ),
        },
        {
            "logical_pair": "null_coordinate",
            "difference": "coordinate outside sampled cluster",
            "host": _capture(lambda: host_select.subnull(x[0], z[0], far_flux)),
            "traced": _capture(
                lambda: traced_select.subnull(jnp.asarray([x[0], z[0], far_flux]))
            ),
            "consequence": (
                "host subnull asserts a loose cluster bound; traced extrapolates"
            ),
        },
        {
            "logical_pair": "subnull",
            "difference": "signature and return shape",
            "host": {
                "signature": "subnull(r_cluster, z_cluster, psi_cluster)",
                "shape": "((R, Z), psi, type)",
                "example": _serialise(host_valid),
            },
            "traced": {
                "signature": "subnull(cluster_3_by_n)",
                "shape": "array([R, Z, psi, type])",
                "example": _serialise(traced_valid),
            },
            "consequence": (
                "the stacked route forces a transpose in Null2D; separate arrays "
                "match both host gathers and the device-native stencil gather"
            ),
        },
        {
            "logical_pair": "length_2d/wall_coordinate/null",
            "difference": "array namespace and dtype only on valid inputs",
            "host": "numpy arithmetic, eager float64 at live host callers",
            "traced": (
                "JAX arithmetic, fp32 or fp64 according to process-global import order"
            ),
            "consequence": (
                "the expressions are otherwise identical and can be one "
                "namespace-threaded body"
            ),
        },
    ]


def _reachability() -> dict[str, Any]:
    """Return the source census that fixes the argument regimes and signatures."""
    return {
        "bisect_host": [
            "nova/biot/fieldnull.py:92",
            "nova/frame/plasmaloc.py:37",
        ],
        "bisect_traced": [],
        "bisect_right_host": ["nova/frame/plasmaloc.py:38"],
        "bisect_right_traced": ["nova/jax/select.py:55 (inside broken bisect_2d only)"],
        "bisect_2d_host": ["nova/biot/plasmagap.py:115"],
        "bisect_2d_traced": [],
        "wall_flux_host": ["nova/biot/limiter.py:56"],
        "wall_flux_traced": ["nova/jax/null.py:36"],
        "subnull_host_three_array": [
            "nova/biot/fieldnull.py:96",
            "nova/biot/fieldnull.py:109",
        ],
        "subnull_traced_stacked": ["nova/jax/null.py:109"],
        "subnull_traced_three_array": ["nova/jax/stencil_nulls.py:74"],
        "candidate_regime": {
            "host": "one candidate per Python loop iteration",
            "traced_null2d": "two categories times maxsize=5, so ten fixed slots",
            "traced_stencil": "default k_slots=6",
            "cluster_points": "3x3 structured refinement uses nine points",
        },
    }


def _precision_measurement(
    name: str, dtype: Any, array_subnull: Callable[..., Any] | None
) -> dict[str, Any]:
    """Run all accuracy, cost, and differentiability measurements at one dtype."""
    return {
        "name": name,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dtype": str(np.dtype(dtype)),
        "accuracy": {
            "bisect": _bisect_accuracy(dtype),
            "wall": _wall_accuracy(dtype),
            "surface": _surface_accuracy(dtype, array_subnull),
        },
        "cost": {
            "bisect": _bisect_cost(dtype),
            "wall_flux": _wall_cost(dtype),
            "subnull": _subnull_cost(dtype, array_subnull),
        },
        "autodiff": _autodiff(dtype),
    }


def measure(label: str, expected_platform: str) -> dict[str, Any]:
    """Run one complete machine measurement."""
    backend = jax.default_backend()
    if backend != expected_platform:
        raise RuntimeError(
            f"expected JAX platform {expected_platform!r}, observed {backend!r}"
        )
    if jax.config.jax_enable_x64:
        raise RuntimeError("benchmark must start before process-global fp64 is enabled")

    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    checkout_commit = _git_commit()
    record: dict[str, Any] = {
        "schema": "nova.select-route.measurement.v1",
        "label": label,
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("NOVA_SELECT_SOURCE_COMMIT", checkout_commit),
        "execution_checkout_commit": checkout_commit,
        "source_hashes": _source_hashes(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_model": _cpu_model(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "jax_backend": backend,
            "jax_devices": devices,
            "versions": {
                "numpy": _version("numpy"),
                "numba": _version("numba"),
                "jax": _version("jax"),
                "jaxlib": _version("jaxlib"),
                "scipy": _version("scipy"),
            },
            "repeats": REPEATS,
        },
        "reachability": _reachability(),
        "semantics": _semantics(),
    }

    record["precision"] = [
        _precision_measurement("direct-select-fp32", jnp.float32, None)
    ]

    # This import is itself a live semantic input: stencil_nulls enables fp64
    # before tracing, whereas nova.biot.null imports select directly and does not.
    from nova.jax.stencil_nulls import subnull as three_array_subnull

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "stencil null import did not enable the required fp64 policy"
        )
    record["precision"].append(
        _precision_measurement("stencil-enabled-fp64", jnp.float64, three_array_subnull)
    )
    return _serialise(record)


def _best_ratio(rows: Sequence[dict[str, Any]], key: str) -> tuple[float, float]:
    """Return the minimum and maximum of a timing-ratio column."""
    values = [float(row[key]) for row in rows]
    return min(values), max(values)


def _find_precision(run: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one named precision record."""
    return next(item for item in run["precision"] if item["name"] == name)


def _verdicts(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return explicit outcomes for each logical implementation pair."""
    cpu = next(run for run in runs if run["environment"]["jax_backend"] == "cpu")
    gpu = next(run for run in runs if run["environment"]["jax_backend"] == "gpu")
    cpu64 = _find_precision(cpu, "stencil-enabled-fp64")
    gpu64 = _find_precision(gpu, "stencil-enabled-fp64")
    cpu32 = _find_precision(cpu, "direct-select-fp32")
    gpu32 = _find_precision(gpu, "direct-select-fp32")

    cpu_wall = _best_ratio(cpu64["cost"]["wall_flux"], "traced_over_host")
    gpu_wall = _best_ratio(gpu64["cost"]["wall_flux"], "traced_over_host")
    cpu_sub = _best_ratio(cpu64["cost"]["subnull"], "traced_over_host")
    gpu_sub = _best_ratio(gpu64["cost"]["subnull"], "traced_over_host")
    cpu_bisect = _best_ratio(cpu64["cost"]["bisect"], "left_traced_over_host")
    gpu_bisect = _best_ratio(gpu64["cost"]["bisect"], "left_traced_over_host")
    signature = _best_ratio(gpu64["cost"]["subnull"], "three_array_over_stacked")

    return [
        {
            "logical_pair": "select module",
            "verdict": "KEEP BOTH",
            "reason": (
                "wall and surface routes are independently live; traced execution "
                "supplies device batching and finite autodiff while host execution "
                "preserves the JAX-free eager path"
            ),
            "required_condition": (
                "enable JAX fp64 before the first select trace; direct fp32 "
                "misclassified 12 of 18 analytic call-site-scale surfaces"
            ),
        },
        {
            "logical_pair": "bisect",
            "symbols": ["bisect"],
            "verdict": "DELETE ONE",
            "delete": "traced",
            "reason": (
                "zero traced callers, two host callers, exact agreement with "
                "Python bisect"
            ),
            "cpu_traced_over_host": cpu_bisect,
            "gpu_traced_over_host": gpu_bisect,
        },
        {
            "logical_pair": "right and vector bisection",
            "symbols": ["bisect_right", "bisect_2d"],
            "verdict": "DELETE ONE",
            "delete": "traced",
            "reason": (
                "bisect_right cannot be traced; every non-empty bisect_2d call "
                "raises on immutable assignment; neither has an external traced "
                "caller while both host routes are live"
            ),
        },
        {
            "logical_pair": "namespace-identical wall arithmetic",
            "symbols": ["length_2d", "wall_length", "wall_coordinate"],
            "verdict": "COLLAPSE",
            "reason": (
                "the expressions are identical apart from the array namespace; "
                "preserve and document backend-native zero-division behavior"
            ),
        },
        {
            "logical_pair": "wall fitting and selection",
            "symbols": ["quadratic_wall", "wall_index", "wall_flux"],
            "verdict": "KEEP BOTH",
            "reason": (
                "both live call paths are required and fixed-shape traced selection "
                "differs from host branching, mutation, least-squares interface, "
                "and NaN handling"
            ),
            "cpu_traced_over_host_fp64": cpu_wall,
            "gpu_traced_over_host_fp64": gpu_wall,
            "cpu_traced_over_host_fp32": _best_ratio(
                cpu32["cost"]["wall_flux"], "traced_over_host"
            ),
            "gpu_traced_over_host_fp32": _best_ratio(
                gpu32["cost"]["wall_flux"], "traced_over_host"
            ),
        },
        {
            "logical_pair": "quadratic surface fit",
            "symbols": ["quadratic_surface"],
            "verdict": "KEEP BOTH",
            "reason": (
                "host numba requires float64 promotion and explicit gelsd rcond "
                "while traced fitting follows JAX precision and device semantics"
            ),
        },
        {
            "logical_pair": "converged surface arithmetic",
            "symbols": ["null_type", "null_coordinate", "null"],
            "verdict": "COLLAPSE",
            "reason": (
                "after the locked NaN sentinel and determinant-floor convergence, "
                "one namespace-threaded arithmetic body is sufficient"
            ),
        },
        {
            "logical_pair": "subnull composite",
            "symbols": ["subnull"],
            "verdict": "KEEP BOTH",
            "reason": (
                "host scalar/loop and traced vmap/device composites are both live "
                "and have different fit routes; converge only their public signature "
                "and flat result"
            ),
            "cpu_traced_over_host_fp64": cpu_sub,
            "gpu_traced_over_host_fp64": gpu_sub,
        },
        {
            "logical_pair": "subnull signature",
            "verdict": "COLLAPSE",
            "canonical_signature": (
                "subnull(r_cluster, z_cluster, psi_cluster) -> array([r, z, psi, type])"
            ),
            "reason": (
                "three arrays match both host gathers and the device-native stencil "
                "gather; the only stacked caller already transposes its cluster; "
                "the flat four-array is required by vmap and fixed-slot padding; "
                "the measured H200 ratio spans both sides of parity"
            ),
            "gpu_three_array_over_stacked_fp64": signature,
            "accuracy_max_abs_between_traced_signatures": gpu64["accuracy"]["surface"][
                "signature_route_max_abs"
            ],
        },
    ]


def _headline(runs: Sequence[dict[str, Any]], verdicts: Sequence[dict[str, Any]]):
    """Return bounded summary metrics for plan writeback."""
    result: dict[str, Any] = {"environments": {}, "accuracy": {}}
    for run in runs:
        backend = run["environment"]["jax_backend"]
        result["environments"][backend] = run["environment"]
        for precision in run["precision"]:
            key = f"{backend}:{precision['name']}"
            surface = precision["accuracy"]["surface"]
            traced_surface = np.asarray(surface["traced"])
            surface_truth = np.asarray(surface["truth"])
            result["accuracy"][key] = {
                "wall_host_worst_abs": precision["accuracy"]["wall"]["host_worst_abs"],
                "wall_traced_worst_abs": precision["accuracy"]["wall"][
                    "traced_worst_abs"
                ],
                "surface_host_max_abs_by_component": surface[
                    "host_max_abs_by_component"
                ],
                "surface_traced_max_abs_by_component": surface[
                    "traced_max_abs_by_component"
                ],
                "surface_traced_class_mismatches": int(
                    np.count_nonzero(traced_surface[:, 3] != surface_truth[:, 3])
                ),
                "surface_cases": int(surface_truth.shape[0]),
                "autodiff": precision["autodiff"],
            }
    result["subnull_signature"] = next(
        item for item in verdicts if item["logical_pair"] == "subnull signature"
    )
    return result


def _plot(report: dict[str, Any], path: Path) -> None:
    """Write the cost-vs-size and verdict summary as SVG."""
    import matplotlib.pyplot as plt

    runs = report["runs"]
    cpu = next(run for run in runs if run["environment"]["jax_backend"] == "cpu")
    gpu = next(run for run in runs if run["environment"]["jax_backend"] == "gpu")
    cpu64 = _find_precision(cpu, "stencil-enabled-fp64")
    gpu64 = _find_precision(gpu, "stencil-enabled-fp64")

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    wall_axis, subnull_axis, verdict_axis = axes

    cpu_wall = cpu64["cost"]["wall_flux"]
    gpu_wall = gpu64["cost"]["wall_flux"]
    cpu_wall_ratio = _best_ratio(cpu_wall, "traced_over_host")
    gpu_wall_ratio = _best_ratio(gpu_wall, "traced_over_host")
    wall_axis.loglog(
        [row["nodes"] for row in cpu_wall],
        [row["host_us"] for row in cpu_wall],
        "o-",
        label="host on CPU node",
    )
    wall_axis.loglog(
        [row["nodes"] for row in cpu_wall],
        [row["traced_us"] for row in cpu_wall],
        "s-",
        label="traced on CPU",
    )
    wall_axis.loglog(
        [row["nodes"] for row in gpu_wall],
        [row["traced_us"] for row in gpu_wall],
        "^-",
        label="traced on H200",
    )
    wall_axis.set(title="Wall-null call", xlabel="wall nodes", ylabel="latency [µs]")
    wall_axis.grid(True, which="both", alpha=0.25)
    wall_axis.legend(fontsize=8)

    cpu_sub = cpu64["cost"]["subnull"]
    gpu_sub = gpu64["cost"]["subnull"]
    cpu_sub_ratio = _best_ratio(cpu_sub, "traced_over_host")
    gpu_sub_ratio = _best_ratio(gpu_sub, "traced_over_host")
    subnull_axis.loglog(
        [row["batch"] for row in cpu_sub],
        [row["host_us"] for row in cpu_sub],
        "o-",
        label="host loop on CPU node",
    )
    subnull_axis.loglog(
        [row["batch"] for row in cpu_sub],
        [row["traced_three_array_us"] for row in cpu_sub],
        "s-",
        label="traced 3-array on CPU",
    )
    subnull_axis.loglog(
        [row["batch"] for row in gpu_sub],
        [row["traced_three_array_us"] for row in gpu_sub],
        "^-",
        label="traced 3-array on H200",
    )
    subnull_axis.set(
        title="3×3 sub-null refinement",
        xlabel="candidate batch",
        ylabel="latency [µs]",
    )
    subnull_axis.grid(True, which="both", alpha=0.25)
    subnull_axis.legend(fontsize=8)

    verdict_axis.axis("off")
    verdict_axis.set_title("Evidence-backed outcomes")
    gpu32 = _find_precision(gpu, "direct-select-fp32")
    gpu32_surface = gpu32["accuracy"]["surface"]
    gpu32_traced = np.asarray(gpu32_surface["traced"])
    gpu32_truth = np.asarray(gpu32_surface["truth"])
    wrong_classes = int(np.count_nonzero(gpu32_traced[:, 3] != gpu32_truth[:, 3]))
    fp32_position = float(np.max(np.abs(gpu32_traced[:, :2] - gpu32_truth[:, :2])))
    fp64_position = max(gpu64["accuracy"]["surface"]["traced_max_abs_by_component"][:2])
    fp64_wall = gpu64["accuracy"]["wall"]["traced_worst_abs"]
    verdict_lines = [
        "DELETE traced",
        "  bisect · bisect_right · bisect_2d",
        "",
        "COLLAPSE",
        "  length_2d · wall_length · wall_coordinate",
        "  null_type · null_coordinate · null",
        "",
        "KEEP BOTH",
        "  wall fit/selection · quadratic surface · subnull",
        "",
        "Canonical subnull",
        "  (r, z, psi) → [r0, z0, psi0, type]",
        "",
        "Cost, traced / host (fp64)",
        "  wall CPU %.2f–%.2fx · H200 %.2f–%.2fx" % (*cpu_wall_ratio, *gpu_wall_ratio),
        "  subnull CPU %.2f–%.2fx · H200 %.2f–%.2fx" % (*cpu_sub_ratio, *gpu_sub_ratio),
        "",
        "Analytic-reference accuracy",
        "  fp64 wall ≤ %.2g · surface pos ≤ %.2g m" % (fp64_wall, fp64_position),
        "  direct fp32: %d/18 classes wrong; pos ≤ %.2g m"
        % (wrong_classes, fp32_position),
        "",
        "Minimum of 9 warm, synchronised calls",
    ]
    verdict_axis.text(
        0.02,
        0.96,
        "\n".join(verdict_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.3,
    )
    figure.suptitle("Nova select routes: call-site cost, accuracy, and reachability")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Date": None})
    plt.close(figure)


def assemble(inputs: Sequence[Path], json_path: Path, svg_path: Path) -> dict[str, Any]:
    """Combine machine records and create committed evidence without re-running."""
    runs = [json.loads(path.read_text()) for path in inputs]
    commits = {run["source_commit"] for run in runs}
    backends = {run["environment"]["jax_backend"] for run in runs}
    source_hash_sets = {
        json.dumps(run["source_hashes"], sort_keys=True) for run in runs
    }
    if len(commits) != 1:
        raise RuntimeError(f"measurement commits differ: {sorted(commits)}")
    if backends != {"cpu", "gpu"}:
        raise RuntimeError(f"need one CPU and one GPU record, received {backends}")
    if len(source_hash_sets) != 1:
        raise RuntimeError("CPU and GPU select source hashes differ")
    verdicts = _verdicts(runs)
    report = {
        "schema": "nova.select-route.evidence.v1",
        "assembled_at": datetime.now(UTC).isoformat(),
        "source_commit": commits.pop(),
        "method": {
            "accuracy": (
                "Python standard-library bisection plus analytic wall and surface "
                "quadratics fixed before either Nova route runs"
            ),
            "timing": (
                f"minimum of {REPEATS} warm synchronised calls; compilation excluded"
            ),
            "execution_checkout": (
                "compute nodes cannot see the assigned /run/user worktree; the "
                "benchmark was streamed into the shared checkout after all four "
                "measured source hashes were proved identical to the assigned base"
            ),
            "source_hashes": runs[0]["source_hashes"],
            "regimes": {
                "wall_nodes": WALL_SIZES,
                "subnull_candidates": SUBNULL_BATCHES,
                "bisect_vector_sizes": BISECT_SIZES,
            },
        },
        "verdicts": verdicts,
        "headline": _headline(runs, verdicts),
        "runs": runs,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_serialise(report), indent=2) + "\n")
    _plot(report, svg_path)
    return report


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument(
        "--expect-platform", choices=("cpu", "gpu"), required=True
    )
    measure_parser.add_argument("--output", type=Path, required=True)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    assemble_parser.add_argument("--json", type=Path, required=True)
    assemble_parser.add_argument("--svg", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one measurement or assemble existing measurements."""
    arguments = _parser().parse_args(argv)
    if arguments.mode == "measure":
        record = measure(arguments.label, arguments.expect_platform)
        arguments.output.write_text(json.dumps(record, indent=2) + "\n")
        print(
            "MEASURED label=%s backend=%s output=%s"
            % (arguments.label, record["environment"]["jax_backend"], arguments.output)
        )
        return
    report = assemble(arguments.inputs, arguments.json, arguments.svg)
    print(
        "ASSEMBLED runs=%d json=%s svg=%s"
        % (len(report["runs"]), arguments.json, arguments.svg)
    )


if __name__ == "__main__":
    main()
