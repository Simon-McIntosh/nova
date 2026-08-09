"""Measure field-null latency, batching, transfer, and candidate fusion.

The archive-rate question is deliberately separated from single-call latency.
The benchmark exercises the archived MAST equilibrium grid shape (65 x 129),
keeps static geometry resident, and reports both dynamic-field bytes and fields
per second.  It also retains the legacy hex-stencil scan because that sequential
reduction explains the unusually slow documented H200 numbers.

The proposed kernels in this file are measurements, not production repairs:

* ``separate_stencil`` launches the current axis and X-candidate reads separately;
* ``composed_stencil`` places both current reads under one outer ``jit``;
* ``shared_classifier_svd`` shares the rectangular classifier but keeps the
  current global-coordinate ``lstsq`` refinement;
* ``normalized_matrix_fit`` shares classification and replaces each tiny SVD by
  a fixed matrix multiply on cell-local coordinates.
* ``ranked_matrix_metadata`` adds ``lax.top_k`` selection plus fixed count,
  overflow, discarded-score, source-cell, index, and tri-state metadata.  Its
  confidence and index values are schema-cost placeholders, not scientific
  claims; the production repair requires geometry-native degree and calibrated
  uncertainty.

Usage on allocated CPU and GPU nodes::

    uv run python benchmarks/fieldnull_gpu_throughput.py measure \
      --platform cpu --partial /tmp/fieldnull_cpu.json
    uv run python benchmarks/fieldnull_gpu_throughput.py measure \
      --platform gpu --partial /tmp/fieldnull_gpu.json
    uv run python benchmarks/fieldnull_gpu_throughput.py combine \
      --cpu /tmp/fieldnull_cpu.json --gpu /tmp/fieldnull_gpu.json \
      --evidence-dir docs/figures/jax-dissolution
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import socket
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from nova.geometry.hexstencil import hex_stencil
from nova.equilibrium.flux_surface_connectivity import _dilate4
from nova.biot.null import Null2D
from nova.equilibrium.stencil_nulls import (
    _refine_at,
    magnetic_axis_subgrid,
    ring_sign_changes,
    xpoint_candidates,
)


ROOT = Path(
    os.environ.get("NOVA_BENCHMARK_ROOT", Path(__file__).resolve().parents[1])
).resolve()
OUTPUT = ROOT / "docs" / "figures" / "jax-dissolution"
DEFAULT_OUTPUT = OUTPUT / "fieldnull_gpu_throughput.json"
MAST_SHAPE = (65, 129)
SLOTS = 8
REPEATS = 5
ARCHIVE_BYTES = 4.5e12
TARGET_SECONDS = 3600.0
TARGET_BYTES_PER_SECOND = ARCHIVE_BYTES / TARGET_SECONDS
TARGET_H200S = 8
TARGET_L2_SHOTS = 11573

BLOCK = tuple((dz, dr) for dz in (-1, 0, 1) for dr in (-1, 0, 1))
DESIGN = np.asarray(
    [[dr**2, dz**2, dr, dz, dr * dz, 1.0] for dz, dr in BLOCK],
    dtype=np.float64,
)
DESIGN_PINV = np.linalg.pinv(DESIGN)


def _block(tree):
    """Synchronize every device leaf without transferring it to the host."""
    return jax.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf
        ),
        tree,
    )


def _host_tree(tree):
    """Materialize every device leaf as a NumPy array."""
    return jax.tree.map(np.asarray, tree)


def _tree_bytes(tree) -> int:
    """Return the byte size of all leaves after one explicit materialization."""
    return int(sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(tree)))


def _samples(function: Callable[[], Any], repeats: int = REPEATS) -> list[float]:
    """Return synchronized wall-time samples in seconds."""
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block(function())
        values.append(time.perf_counter() - start)
    return values


def _summary_seconds(samples: list[float]) -> dict[str, float]:
    """Summarize one captured sample set."""
    return {
        "minimum_ms": 1.0e3 * float(np.min(samples)),
        "median_ms": 1.0e3 * float(np.median(samples)),
        "maximum_ms": 1.0e3 * float(np.max(samples)),
    }


def _axis_array(psi, rg, zg, inside):
    """Pack the current axis result as one fixed output row."""
    axis = magnetic_axis_subgrid(psi, rg, zg, inside)
    row = jnp.stack(
        [
            axis["r"],
            axis["z"],
            axis["psi"],
            axis["ntype"],
            axis["found"].astype(psi.dtype),
        ]
    )
    return row[jnp.newaxis, :]


def _x_array(psi, rg, zg, inside):
    """Pack current X candidates as fixed output rows."""
    candidates = xpoint_candidates(
        psi,
        rg,
        zg,
        inside,
        k_slots=SLOTS,
        material_dilate=1,
    )
    return jnp.stack(
        [
            candidates["r"],
            candidates["z"],
            candidates["psi"],
            candidates["ntype"],
            candidates["valid"].astype(psi.dtype),
        ],
        axis=1,
    )


def _composed_field(psi, rg, zg, inside):
    """Current axis plus X-candidate path under one outer compilation."""
    return jnp.concatenate(
        [_axis_array(psi, rg, zg, inside), _x_array(psi, rg, zg, inside)]
    )


def _selection(psi, inside):
    """Return one axis index and fixed X indices from one shared classifier."""
    nz, nr = psi.shape
    counts = ring_sign_changes(psi)
    edge = jnp.concatenate([psi[0], psi[-1], psi[:, 0], psi[:, -1]])
    axis_score = jnp.where(
        (counts == 0) & inside,
        jnp.abs(psi - jnp.median(edge)),
        -jnp.inf,
    )
    axis_flat = jnp.argmax(axis_score)
    axis_found = jnp.max(axis_score) > -jnp.inf

    gate = _dilate4(inside)
    x_index = jnp.where(
        ((counts == 4) & gate).reshape(-1),
        size=SLOTS,
        fill_value=-1,
    )[0]
    x_slot = x_index >= 0
    x_index = jnp.clip(x_index, 0, nz * nr - 1)
    indices = jnp.concatenate([axis_flat[jnp.newaxis], x_index])
    rows = indices // nr
    cols = indices % nr
    return rows, cols, axis_found, x_slot


def _ranked_selection(psi, inside):
    """Rank fixed X slots and expose pre-truncation capacity metadata."""
    nz, nr = psi.shape
    counts = ring_sign_changes(psi)
    edge = jnp.concatenate([psi[0], psi[-1], psi[:, 0], psi[:, -1]])
    edge_level = jnp.median(edge)
    axis_mask = (counts == 0) & inside
    axis_score = jnp.where(axis_mask, jnp.abs(psi - edge_level), -jnp.inf)
    axis_flat = jnp.argmax(axis_score)
    axis_found = jnp.any(axis_mask)

    x_mask = (counts == 4) & _dilate4(inside)
    x_score = jnp.where(x_mask, jnp.abs(psi - edge_level), -jnp.inf)
    ranked_score, ranked_index = jax.lax.top_k(x_score.reshape(-1), SLOTS + 1)
    x_index = ranked_index[:SLOTS]
    x_slot = jnp.isfinite(ranked_score[:SLOTS])
    x_count = jnp.sum(x_mask)
    axis_count = jnp.sum(axis_mask)
    overflow = (x_count > SLOTS) | (axis_count > 1)
    discarded = jnp.where(x_count > SLOTS, ranked_score[SLOTS], jnp.nan)
    indices = jnp.concatenate([axis_flat[jnp.newaxis], x_index])
    return {
        "rows": indices // nr,
        "cols": indices % nr,
        "source_cell": indices,
        "axis_found": axis_found,
        "x_slot": x_slot,
        "candidate_count": axis_count + x_count,
        "axis_candidate_count": axis_count,
        "x_candidate_count": x_count,
        "overflow": overflow,
        "discarded_score_upper_bound": discarded,
    }


def _pack_refined(subs, axis_found, x_slot):
    """Apply current output validity semantics to already refined clusters."""
    dtype = subs.dtype
    x_valid = x_slot & (jnp.abs(subs[1:, 3]) < 0.5)
    validity = jnp.concatenate(
        [axis_found[jnp.newaxis], x_valid],
    )
    numeric = jnp.concatenate(
        [
            jnp.where(axis_found, subs[:1], jnp.nan),
            jnp.where(x_valid[:, None], subs[1:], jnp.nan),
        ]
    )
    return jnp.concatenate([numeric, validity.astype(dtype)[:, None]], axis=1)


def _shared_classifier_svd_field(psi, rg, zg, inside):
    """Share classification while retaining the current tiny-SVD fit."""
    rows, cols, axis_found, x_slot = _selection(psi, inside)
    subs = jax.vmap(lambda row, col: _refine_at(psi, rg, zg, row, col))(rows, cols)
    return _pack_refined(subs, axis_found, x_slot)


def _normalized_refine(psi, rg, zg, rows, cols, design_pinv):
    """Fit all selected blocks in nondimensional cell-local coordinates."""
    nz, nr = psi.shape
    offsets_z = jnp.asarray([dz for dz, _ in BLOCK])
    offsets_r = jnp.asarray([dr for _, dr in BLOCK])
    sample_rows = jnp.clip(rows[:, None] + offsets_z[None, :], 0, nz - 1)
    sample_cols = jnp.clip(cols[:, None] + offsets_r[None, :], 0, nr - 1)
    clusters = psi[sample_rows, sample_cols]
    coefficients = jnp.einsum("ij,bj->bi", design_pinv, clusters)
    a, b, c, d, e, f = coefficients.T
    determinant = 4.0 * a * b - e**2
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(4.0 * a * b), e**2), jnp.finfo(psi.dtype).tiny
    )
    degenerate = jnp.abs(determinant) <= 64.0 * jnp.finfo(psi.dtype).eps * scale
    safe = jnp.where(degenerate, jnp.ones_like(determinant), determinant)
    local_r = (e * d - 2.0 * b * c) / safe
    local_z = (e * c - 2.0 * a * d) / safe
    value = (
        a * local_r**2
        + b * local_z**2
        + c * local_r
        + d * local_z
        + e * local_r * local_z
        + f
    )
    dr = rg[1] - rg[0]
    dz = zg[1] - zg[0]
    radius = rg[cols] + local_r * dr
    height = zg[rows] + local_z * dz
    ntype = jnp.where(
        degenerate,
        jnp.nan,
        jnp.where(
            determinant < 0,
            0.0,
            jnp.where((a > 0) & (b > 0), -1.0, 1.0),
        ),
    )
    return jnp.stack([radius, height, value, ntype], axis=1)


def _normalized_matrix_field(psi, rg, zg, inside, design_pinv):
    """Share classification and refine all slots with one fixed matrix multiply."""
    rows, cols, axis_found, x_slot = _selection(psi, inside)
    subs = _normalized_refine(psi, rg, zg, rows, cols, design_pinv)
    return _pack_refined(subs, axis_found, x_slot)


def _ranked_matrix_metadata_field(psi, rg, zg, inside, design_pinv):
    """Cost ranked selection and a fixed, tri-state metadata schema."""
    selection = _ranked_selection(psi, inside)
    subs = _normalized_refine(
        psi,
        rg,
        zg,
        selection["rows"],
        selection["cols"],
        design_pinv,
    )
    present = jnp.concatenate([selection["axis_found"][None], selection["x_slot"]])
    class_resolved = jnp.isfinite(subs[:, 3])
    expected_class = jnp.concatenate(
        [jnp.ones((1,), dtype=bool), jnp.abs(subs[1:, 3]) < 0.5]
    )
    resolved = present & class_resolved & expected_class
    state = jnp.where(resolved, 2, jnp.where(present, 1, 0))
    candidates = jnp.concatenate(
        [
            jnp.where(present[:, None], subs, jnp.nan),
            resolved.astype(psi.dtype)[:, None],
        ],
        axis=1,
    )
    index = jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.int32), -jnp.ones((SLOTS,), dtype=jnp.int32)]
    )
    index = jnp.where(present, index, 0)
    confidence = jnp.where(state == 2, 1.0, jnp.where(state == 1, 0.5, 0.0))
    return {
        "candidates": candidates,
        "index": index,
        "confidence_placeholder": confidence.astype(psi.dtype),
        "state": state,
        "source_cell": selection["source_cell"],
        "candidate_count": selection["candidate_count"],
        "axis_candidate_count": selection["axis_candidate_count"],
        "x_candidate_count": selection["x_candidate_count"],
        "overflow": selection["overflow"],
        "discarded_score_upper_bound": selection["discarded_score_upper_bound"],
        "domain_boundary_index_placeholder": jnp.sum(index),
    }


def _field_batch(batch: int, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    """Return shifted smooth fields on the archived MAST equilibrium grid."""
    nz, nr = MAST_SHAPE
    rg = np.linspace(0.2, 1.8, nr, dtype=dtype)
    zg = np.linspace(-0.8, 0.8, nz, dtype=dtype)
    rr, zz = np.meshgrid(rg, zg)
    inside = ((rr - 1.0) / 0.76) ** 2 + (zz / 0.74) ** 2 <= 1.0
    fields = []
    for index in range(batch):
        shift_r = 0.004 * np.sin(index * 0.17)
        shift_z = 0.003 * np.cos(index * 0.11)
        width = dtype.type(0.15)
        lower = np.exp(
            -(
                (rr - dtype.type(1.007 + shift_r)) ** 2
                + (zz - dtype.type(-0.30 + shift_z)) ** 2
            )
            / width**2
        )
        upper = np.exp(
            -(
                (rr - dtype.type(1.007 + shift_r)) ** 2
                + (zz - dtype.type(0.30 + shift_z)) ** 2
            )
            / width**2
        )
        fields.append((lower + upper).astype(dtype, copy=False))
    return np.stack(fields), rg, zg, inside


def _compile_single(function, argument):
    """Compile one single-output batch function and return a runner."""
    start = time.perf_counter()
    executable = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    return executable, compile_seconds


def _compile_route(name, argument, rg, zg, inside, design_pinv):
    """Compile one measured route and return its synchronized runner."""
    if name == "separate_stencil":
        axis_function = jax.vmap(lambda field: _axis_array(field, rg, zg, inside))
        x_function = jax.vmap(lambda field: _x_array(field, rg, zg, inside))
        axis_executable, axis_compile = _compile_single(axis_function, argument)
        x_executable, x_compile = _compile_single(x_function, argument)

        def run(value):
            return axis_executable(value), x_executable(value)

        return run, axis_compile + x_compile, 2
    functions = {
        "composed_stencil": jax.vmap(
            lambda field: _composed_field(field, rg, zg, inside)
        ),
        "shared_classifier_svd": jax.vmap(
            lambda field: _shared_classifier_svd_field(field, rg, zg, inside)
        ),
        "normalized_matrix_fit": jax.vmap(
            lambda field: _normalized_matrix_field(field, rg, zg, inside, design_pinv)
        ),
        "ranked_matrix_metadata": jax.vmap(
            lambda field: _ranked_matrix_metadata_field(
                field, rg, zg, inside, design_pinv
            )
        ),
    }
    executable, compile_seconds = _compile_single(functions[name], argument)
    return executable, compile_seconds, 1


def _timed_route(
    name, host_fields, rg, zg, inside, dtype
) -> tuple[dict[str, Any], Any]:
    """Measure resident and transfer-inclusive cost for one route and batch."""
    device = jax.devices()[0]
    d_fields = jax.device_put(host_fields, device)
    d_rg = jax.device_put(rg, device)
    d_zg = jax.device_put(zg, device)
    d_inside = jax.device_put(inside, device)
    d_pinv = jax.device_put(DESIGN_PINV.astype(dtype), device)
    run, compile_seconds, launches = _compile_route(
        name, d_fields, d_rg, d_zg, d_inside, d_pinv
    )
    start = time.perf_counter()
    first = run(d_fields)
    _block(first)
    first_seconds = time.perf_counter() - start
    resident = _samples(lambda: run(d_fields))

    inclusive = []
    for _ in range(REPEATS):
        dynamic = np.array(host_fields, copy=True)
        start = time.perf_counter()
        placed = jax.device_put(dynamic, device)
        result = run(placed)
        _host_tree(result)
        inclusive.append(time.perf_counter() - start)

    median = float(np.median(resident))
    batch = host_fields.shape[0]
    input_bytes = int(host_fields.nbytes)
    output_bytes = _tree_bytes(first)
    row = {
        "route": name,
        "dtype": np.dtype(dtype).name,
        "batch": batch,
        "shape": list(host_fields.shape[1:]),
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "compile_ms": 1.0e3 * compile_seconds,
        "first_execution_ms": 1.0e3 * first_seconds,
        "resident": _summary_seconds(resident),
        "transfer_inclusive": _summary_seconds(inclusive),
        "resident_fields_per_second": batch / median,
        "resident_field_bytes_per_second": input_bytes / median,
        "projected_archive_seconds_at_field_byte_rate": ARCHIVE_BYTES
        / (input_bytes / median),
        "executable_launches": launches,
        "static_geometry_resident": True,
    }
    return row, first


def _legacy_null(dtype, batches) -> list[dict[str, Any]]:
    """Measure the sequential legacy hex classifier represented in H200 evidence."""
    nz, nr = MAST_SHAPE
    fields, rg, zg, _inside = _field_batch(max(batches), dtype)
    stencil = hex_stencil((nz, nr))
    rr, zz = np.meshgrid(rg, zg)
    coordinate = np.column_stack([rr.reshape(-1), zz.reshape(-1)])
    null = Null2D(
        jnp.asarray(coordinate),
        jnp.asarray(stencil),
        jnp.asarray(coordinate[stencil]),
        SLOTS,
    )
    rows = []
    device = jax.devices()[0]
    for batch in batches:
        host = fields[:batch]
        argument = jax.device_put(host, device)

        def classify(values):
            return jax.vmap(lambda field: null.categorize(field.reshape(-1)[stencil]))(
                values
            )

        executable, compile_seconds = _compile_single(classify, argument)
        start = time.perf_counter()
        first = executable(argument)
        _block(first)
        first_seconds = time.perf_counter() - start
        samples = _samples(lambda: executable(argument))
        median = float(np.median(samples))
        rows.append(
            {
                "route": "legacy_hex_scan",
                "dtype": np.dtype(dtype).name,
                "batch": batch,
                "shape": list(MAST_SHAPE),
                "compile_ms": 1.0e3 * compile_seconds,
                "first_execution_ms": 1.0e3 * first_seconds,
                "resident": _summary_seconds(samples),
                "resident_fields_per_second": batch / median,
                "input_bytes": int(host.nbytes),
                "resident_field_bytes_per_second": host.nbytes / median,
                "algorithm": (
                    "lax.scan over every interior vertex, then fixed-size where; "
                    "categorization only, no local fits"
                ),
            }
        )
    return rows


def _materialization_metrics(
    name, host_fields, rg, zg, inside, dtype
) -> dict[str, Any]:
    """Separate input copy, synchronization, output copy, and host compaction."""
    device = jax.devices()[0]
    argument = jax.device_put(host_fields, device)
    d_rg = jax.device_put(rg, device)
    d_zg = jax.device_put(zg, device)
    d_inside = jax.device_put(inside, device)
    d_pinv = jax.device_put(DESIGN_PINV.astype(dtype), device)
    run, _compile_seconds, launches = _compile_route(
        name, argument, d_rg, d_zg, d_inside, d_pinv
    )
    resident = run(argument)
    _block(resident)

    h2d = _samples(lambda: jax.device_put(np.array(host_fields, copy=True), device))
    kernel = _samples(lambda: run(argument))
    d2h = []
    compact = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        host = _host_tree(resident)
        d2h.append(time.perf_counter() - start)
        packed = _candidate_rows(host)
        start = time.perf_counter()
        np.count_nonzero(packed[..., 4] > 0.5, axis=1)
        compact.append(time.perf_counter() - start)
    return {
        "route": name,
        "dtype": np.dtype(dtype).name,
        "batch": int(host_fields.shape[0]),
        "input_bytes": int(host_fields.nbytes),
        "fixed_output_rows_per_field": SLOTS + 1,
        "fixed_output_bytes": _tree_bytes(resident),
        "h2d": _summary_seconds(h2d),
        "kernel": _summary_seconds(kernel),
        "d2h_materialization": _summary_seconds(d2h),
        "host_validity_compaction": _summary_seconds(compact),
        "executable_launches": launches,
        "warning": (
            "host compaction is measured after materialization and therefore must not "
            "appear in a device-resident hot path"
        ),
    }


def _candidate_rows(host):
    """Return candidate rows from any measured output organization."""
    if isinstance(host, tuple):
        return np.concatenate(host, axis=1)
    if isinstance(host, dict):
        return host["candidates"]
    return host


def _json_array(array):
    """Convert an array to strict-JSON scalars, spelling non-finite as null."""
    values = np.asarray(array)
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in values
    ]


def _cache_probe() -> list[dict[str, Any]]:
    """Measure in-process dispatch caching across batch shapes and dtypes."""
    jax.clear_caches()

    def batch_function(fields, rg, zg, inside, design_pinv):
        return jax.vmap(_normalized_matrix_field, in_axes=(0, None, None, None, None))(
            fields, rg, zg, inside, design_pinv
        )

    compiled = jax.jit(batch_function)
    sequence = [
        ("float32_first", np.dtype(np.float32), 8),
        ("float32_repeat", np.dtype(np.float32), 8),
        ("float32_new_shape", np.dtype(np.float32), 9),
        ("float32_cached_shape", np.dtype(np.float32), 8),
        ("float64_new_dtype", np.dtype(np.float64), 8),
        ("float64_repeat", np.dtype(np.float64), 8),
    ]
    rows = []
    for label, dtype, batch in sequence:
        fields, rg, zg, inside = _field_batch(batch, dtype)
        args = (
            jnp.asarray(fields),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(DESIGN_PINV, dtype=dtype),
        )
        start = time.perf_counter()
        result = compiled(*args)
        _block(result)
        rows.append(
            {
                "label": label,
                "dtype": dtype.name,
                "batch": batch,
                "compile_or_dispatch_ms": 1.0e3 * (time.perf_counter() - start),
            }
        )
    return rows


def _environment(requested_platform: str) -> dict[str, Any]:
    """Return hardware and software facts needed to interpret the timings."""
    import jaxlib

    devices = jax.devices()
    if not devices or devices[0].platform != requested_platform:
        measured = [device.platform for device in devices]
        raise RuntimeError(f"requested {requested_platform}, measured {measured}")
    memory = None
    try:
        memory = devices[0].memory_stats()
    except AttributeError, RuntimeError:
        pass
    return {
        "hostname": socket.getfqdn(),
        "platform": requested_platform,
        "system": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "devices": [
            {
                "id": str(device),
                "platform": device.platform,
                "kind": getattr(device, "device_kind", "unknown"),
            }
            for device in devices
        ],
        "memory_stats": memory,
        "threads": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        },
    }


def _source_hashes() -> dict[str, str]:
    """Hash the benchmark and every production kernel it exercises."""
    paths = {
        "benchmarks/fieldnull_gpu_throughput.py": Path(__file__),
        "nova/biot/null.py": ROOT / "nova/biot/null.py",
        "nova/jax/select.py": ROOT / "nova/jax/select.py",
        "nova/equilibrium/stencil_nulls.py": ROOT / "nova/equilibrium/stencil_nulls.py",
        "nova/equilibrium/fixed_point.py": ROOT / "nova/equilibrium/fixed_point.py",
        "nova/equilibrium/profile.py": ROOT / "nova/equilibrium/profile.py",
        "nova/imas/mast_solve_inputs.py": ROOT / "nova/imas/mast_solve_inputs.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def measure(platform_name: str) -> dict[str, Any]:
    """Capture both precisions once on the selected backend."""
    environment = _environment(platform_name)
    batches = (1, 8, 32, 128) if platform_name == "cpu" else (1, 8, 32, 128, 512, 2048)
    legacy_batches = (1, 32, 128) if platform_name == "cpu" else (1, 32, 256, 1024)
    rows = []
    legacy = []
    samples: dict[str, Any] = {}
    materialization = []
    for dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        fields, rg, zg, inside = _field_batch(max(batches), dtype)
        for batch in batches:
            for route in (
                "separate_stencil",
                "composed_stencil",
                "shared_classifier_svd",
                "normalized_matrix_fit",
                "ranked_matrix_metadata",
            ):
                row, output = _timed_route(
                    route,
                    fields[:batch],
                    rg,
                    zg,
                    inside,
                    dtype,
                )
                rows.append(row)
                if batch == 1:
                    host = _host_tree(output)
                    samples[f"{dtype.name}:{route}"] = _json_array(
                        _candidate_rows(host)[0]
                    )
        legacy.extend(_legacy_null(dtype, legacy_batches))
        for route in (
            "composed_stencil",
            "normalized_matrix_fit",
            "ranked_matrix_metadata",
        ):
            materialization.append(
                _materialization_metrics(
                    route,
                    fields[: max(batches)],
                    rg,
                    zg,
                    inside,
                    dtype,
                )
            )
    return {
        "schema_version": 1,
        "benchmark": "fieldnull_gpu_throughput",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "environment": environment,
        "method": {
            "grid_shape": list(MAST_SHAPE),
            "grid_cells": int(np.prod(MAST_SHAPE)),
            "batch_sizes": list(batches),
            "legacy_batch_sizes": list(legacy_batches),
            "dtypes": ["float32", "float64"],
            "timing_repeats": REPEATS,
            "timing_statistic": "synchronized median with minimum and maximum retained",
            "transfer_policy": (
                "static coordinates, masks, and fit matrix stay resident; "
                "inclusive timing copies one dynamic field batch to the device "
                "and all fixed-slot outputs back"
            ),
        },
        "source_hashes": _source_hashes(),
        "measurements": rows,
        "legacy_hex_scan": legacy,
        "materialization": materialization,
        "cache_probe": _cache_probe(),
        "single_field_outputs": samples,
    }


def _memory_probe_row(host_fields, rg, zg, inside, dtype) -> dict[str, Any]:
    """Measure a very large resident ranked batch without duplicate transfers."""
    device = jax.devices()[0]
    argument = jax.device_put(host_fields, device)
    d_rg = jax.device_put(rg, device)
    d_zg = jax.device_put(zg, device)
    d_inside = jax.device_put(inside, device)
    d_pinv = jax.device_put(DESIGN_PINV.astype(dtype), device)
    run, compile_seconds, launches = _compile_route(
        "ranked_matrix_metadata",
        argument,
        d_rg,
        d_zg,
        d_inside,
        d_pinv,
    )
    start = time.perf_counter()
    first = run(argument)
    _block(first)
    first_seconds = time.perf_counter() - start
    resident = _samples(lambda: run(argument), repeats=3)
    median = float(np.median(resident))
    try:
        memory = device.memory_stats()
    except AttributeError, RuntimeError:
        memory = None
    return {
        "route": "ranked_matrix_metadata",
        "dtype": np.dtype(dtype).name,
        "batch": int(host_fields.shape[0]),
        "shape": list(host_fields.shape[1:]),
        "input_bytes": int(host_fields.nbytes),
        "output_bytes": _tree_bytes(first),
        "compile_ms": 1.0e3 * compile_seconds,
        "first_execution_ms": 1.0e3 * first_seconds,
        "resident": _summary_seconds(resident),
        "resident_fields_per_second": host_fields.shape[0] / median,
        "resident_field_bytes_per_second": host_fields.nbytes / median,
        "executable_launches": launches,
        "memory_stats_after_execution": memory,
        "transfer_inclusive": False,
    }


def measure_memory(platform_name: str) -> dict[str, Any]:
    """Probe large H200 batches until the bounded host allocation is exhausted."""
    if platform_name != "gpu":
        raise ValueError("the large memory probe is reserved for the H200 backend")
    environment = _environment(platform_name)
    dtype = np.dtype(np.float32)
    base, rg, zg, inside = _field_batch(1, dtype)
    batches = (8192, 32768, 131072, 524288)
    rows = []
    failure = None
    for batch in batches:
        try:
            fields = np.repeat(base, batch, axis=0)
            rows.append(_memory_probe_row(fields, rg, zg, inside, dtype))
        except RuntimeError as error:
            failure = {
                "batch": batch,
                "type": type(error).__name__,
                "message": str(error),
            }
            break
        finally:
            if "fields" in locals():
                del fields
            jax.clear_caches()
            gc.collect()
    return {
        "schema_version": 1,
        "benchmark": "fieldnull_gpu_throughput_memory_probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "environment": environment,
        "source_hashes": _source_hashes(),
        "method": {
            "batches": list(batches),
            "dtype": dtype.name,
            "route": "ranked_matrix_metadata",
            "transfer_inclusive": False,
            "host_allocation_limit": (
                "probe stops at a 17.6 GB fp32 input so the 64 GB SLURM host "
                "allocation can retain one source batch plus runtime staging"
            ),
        },
        "measurements": rows,
        "failure": failure,
    }


def _load(path: Path) -> dict[str, Any]:
    """Load a strict JSON object."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return value


def _existing_evidence(directory: Path) -> dict[str, Any]:
    """Summarize recorded H200 route artifacts without copying their bulk."""
    null_stack = _load(directory / "null_stack_route.json")
    stencil = _load(directory / "stencil_null_route.json")
    fixed = _load(directory / "fixed_point_route.json")
    null_rows = {}
    for backend in ("cpu", "gpu"):
        measurements = null_stack["runs"][backend]["measurements"]
        null_rows[backend] = {
            "null": {
                row["nodes"]: row["traced_warm_us"] for row in measurements["null"]
            },
            "fieldnull": {
                row["nodes"]: row["traced_warm_us"] for row in measurements["fieldnull"]
            },
            "topology": {
                (row["nodes"], row["batch"]): row["traced_warm_us"]
                for row in measurements["topology"]
            },
        }
    cross = []
    for family in ("null", "fieldnull"):
        for nodes, cpu_us in null_rows["cpu"][family].items():
            gpu_us = null_rows["gpu"][family][nodes]
            cross.append(
                {
                    "family": family,
                    "nodes": nodes,
                    "cpu_us": cpu_us,
                    "h200_us": gpu_us,
                    "h200_slowdown": gpu_us / cpu_us,
                }
            )
    for key, cpu_us in null_rows["cpu"]["topology"].items():
        gpu_us = null_rows["gpu"]["topology"][key]
        cross.append(
            {
                "family": "topology",
                "nodes": key[0],
                "batch": key[1],
                "cpu_us": cpu_us,
                "h200_us": gpu_us,
                "h200_slowdown": gpu_us / cpu_us,
            }
        )

    stencil_cpu = {
        (row["routine"], row.get("cells")): row["steady_us_median"]
        for row in stencil["timings"]
        if row["device"] == "cpu"
    }
    stencil_cross = []
    for row in stencil["timings"]:
        key = (row["routine"], row.get("cells"))
        if row["device"] != "gpu" or key not in stencil_cpu:
            continue
        cpu_us = stencil_cpu[key]
        stencil_cross.append(
            {
                "routine": row["routine"],
                "cells": row.get("cells"),
                "cpu_us": cpu_us,
                "h200_us": row["steady_us_median"],
                "h200_slowdown": row["steady_us_median"] / cpu_us,
            }
        )
    return {
        "answer": "yes",
        "legacy_null_stack": cross,
        "rectangular_stencil": stencil_cross,
        "fixed_point": fixed["summary"]["cross_device"],
        "interpretation": (
            "The legacy hex classifier is a sequential vertex scan, while the "
            "rectangular classifier is vectorized. Tiny local SVDs and fixed-point "
            "map launches remain latency dominated. These single-field timings do "
            "not measure bulk vmap throughput."
        ),
        "source_files": [
            str(directory / "null_stack_route.json"),
            str(directory / "stencil_null_route.json"),
            str(directory / "fixed_point_route.json"),
        ],
    }


def _find_row(run, route, dtype, batch):
    """Return one unique throughput row."""
    rows = [
        row
        for row in run["measurements"]
        if row["route"] == route and row["dtype"] == dtype and row["batch"] == batch
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {route}/{dtype}/{batch} row, found {len(rows)}")
    return rows[0]


def _cross_platform(cpu, gpu) -> list[dict[str, Any]]:
    """Compare CPU and H200 only at their shared batch sizes."""
    rows = []
    shared_batches = sorted(
        set(cpu["method"]["batch_sizes"]) & set(gpu["method"]["batch_sizes"])
    )
    for dtype in ("float32", "float64"):
        for route in (
            "separate_stencil",
            "composed_stencil",
            "shared_classifier_svd",
            "normalized_matrix_fit",
            "ranked_matrix_metadata",
        ):
            for batch in shared_batches:
                cpu_row = _find_row(cpu, route, dtype, batch)
                gpu_row = _find_row(gpu, route, dtype, batch)
                rows.append(
                    {
                        "dtype": dtype,
                        "route": route,
                        "batch": batch,
                        "cpu_fields_per_second": cpu_row["resident_fields_per_second"],
                        "h200_fields_per_second": gpu_row["resident_fields_per_second"],
                        "h200_speedup": gpu_row["resident_fields_per_second"]
                        / cpu_row["resident_fields_per_second"],
                    }
                )
    return rows


def _metadata_cost(run) -> list[dict[str, Any]]:
    """Report ranking and result-metadata overhead against the matrix kernel."""
    rows = []
    for dtype in ("float32", "float64"):
        for batch in run["method"]["batch_sizes"]:
            base = _find_row(run, "normalized_matrix_fit", dtype, batch)
            ranked = _find_row(run, "ranked_matrix_metadata", dtype, batch)
            rows.append(
                {
                    "dtype": dtype,
                    "batch": batch,
                    "matrix_fields_per_second": base["resident_fields_per_second"],
                    "ranked_metadata_fields_per_second": ranked[
                        "resident_fields_per_second"
                    ],
                    "throughput_fraction": ranked["resident_fields_per_second"]
                    / base["resident_fields_per_second"],
                    "extra_output_bytes_per_batch": ranked["output_bytes"]
                    - base["output_bytes"],
                }
            )
    return rows


def _accuracy(cpu, gpu) -> dict[str, Any]:
    """Compare candidate outputs across precision, fusion, and devices."""

    def array(run, key):
        return np.asarray(run["single_field_outputs"][key], dtype=float)

    reference = array(cpu, "float64:normalized_matrix_fit")
    valid = reference[:, 4] > 0.5
    axis_truth = np.asarray([[1.007, -0.297], [1.007, 0.303]])
    saddle_truth = np.asarray([1.007, 0.003])
    reference_axis = reference[(reference[:, 3] == 1) & valid][0]
    reference_saddle = reference[(reference[:, 3] == 0) & valid][0]
    results = {}
    for label, run in (("cpu", cpu), ("h200", gpu)):
        for dtype in ("float32", "float64"):
            for route in (
                "composed_stencil",
                "shared_classifier_svd",
                "normalized_matrix_fit",
            ):
                candidate = array(run, f"{dtype}:{route}")
                overlap = valid & (candidate[:, 4] > 0.5)
                axes = candidate[(candidate[:, 3] == 1) & (candidate[:, 4] > 0.5)]
                saddles = candidate[(candidate[:, 3] == 0) & (candidate[:, 4] > 0.5)]
                axis = axes[0] if len(axes) else None
                saddle = saddles[0] if len(saddles) else None
                same_axis_basin = bool(
                    axis is not None and np.sign(axis[1]) == np.sign(reference_axis[1])
                )
                results[f"{label}:{dtype}:{route}"] = {
                    "valid_rows": int(np.count_nonzero(candidate[:, 4] > 0.5)),
                    "validity_mismatches": int(
                        np.count_nonzero((candidate[:, 4] > 0.5) != valid)
                    ),
                    "axis_basin": (
                        "upper" if axis is not None and axis[1] > 0 else "lower"
                    )
                    if axis is not None
                    else "absent",
                    "axis_identity_matches_reference": same_axis_basin,
                    "axis_coordinate_difference_m_when_identity_matches": (
                        float(np.max(np.abs(axis[:2] - reference_axis[:2])))
                        if same_axis_basin
                        else None
                    ),
                    "axis_error_to_nearest_analytic_extremum_m": (
                        float(np.min(np.linalg.norm(axis_truth - axis[:2], axis=1)))
                        if axis is not None
                        else None
                    ),
                    "saddle_coordinate_difference_m": (
                        float(np.max(np.abs(saddle[:2] - reference_saddle[:2])))
                        if saddle is not None
                        else None
                    ),
                    "saddle_error_to_analytic_truth_m": (
                        float(np.linalg.norm(saddle[:2] - saddle_truth))
                        if saddle is not None
                        else None
                    ),
                    "saddle_flux_difference": (
                        float(abs(saddle[2] - reference_saddle[2]))
                        if saddle is not None
                        else None
                    ),
                    "type_mismatches": int(
                        np.count_nonzero(candidate[overlap, 3] != reference[overlap, 3])
                    ),
                }
    return {
        "reference": "CPU float64 normalized cell-local matrix fit",
        "reference_field": (
            "two equal Gaussian peaks; fp32 may switch between exactly tied upper "
            "and lower axis identities, so axis errors are compared to the nearest "
            "analytic extremum and cross-precision deltas require the same basin"
        ),
        "comparisons": results,
        "scientific_scope": (
            "smooth clean fields only; this throughput audit does not replace the "
            "required "
            "weak-null and correlated-noise operating curve"
        ),
    }


def _archive_target() -> dict[str, Any]:
    """State byte-rate conversions without assigning all bytes to FieldNull."""
    field_bytes = int(np.prod(MAST_SHAPE) * np.dtype(np.float32).itemsize)
    assumptions = []
    for stored_slice_bytes in (64 * 1024, 1024 * 1024, 16 * 1024 * 1024):
        assumptions.append(
            {
                "stored_bytes_per_slice": stored_slice_bytes,
                "required_slices_per_second": TARGET_BYTES_PER_SECOND
                / stored_slice_bytes,
            }
        )
    return {
        "archive_bytes_decimal": ARCHIVE_BYTES,
        "deadline_seconds": TARGET_SECONDS,
        "catalog_l2_shots": TARGET_L2_SHOTS,
        "h200_count": TARGET_H200S,
        "required_bytes_per_second": TARGET_BYTES_PER_SECOND,
        "required_decimal_gb_per_second": TARGET_BYTES_PER_SECOND / 1.0e9,
        "required_bytes_per_second_per_h200_even_shard": (
            TARGET_BYTES_PER_SECOND / TARGET_H200S
        ),
        "required_decimal_mb_per_second_per_h200_even_shard": (
            TARGET_BYTES_PER_SECOND / TARGET_H200S / 1.0e6
        ),
        "representative_grid": list(MAST_SHAPE),
        "fp32_field_bytes": field_bytes,
        "aggregate_fields_per_second_if_every_archive_byte_were_flux": (
            TARGET_BYTES_PER_SECOND / field_bytes
        ),
        "per_h200_fields_per_second_if_every_archive_byte_were_flux": (
            TARGET_BYTES_PER_SECOND / TARGET_H200S / field_bytes
        ),
        "exact_catalog_slice_count": None,
        "exact_required_slices_per_second": None,
        "slice_denominator_status": (
            "open census gate; assumed stored sizes below are conversions, not "
            "the catalog slice count"
        ),
        "stored_slice_assumptions": assumptions,
        "fieldnull_owned_share": (
            "one reconstructed flux field per solved slice; it does not own "
            "archive I/O, LZ4 decompression, clock alignment, calibration, sensor "
            "packing, Green maps, or the nonlinear solve that creates the field"
        ),
        "allocation_formula": (
            "aggregate required FieldNull fields/s = 1.25e9 * "
            "flux_field_fraction / stored_flux_bytes_per_field; divide by eight "
            "only under an even GPU shard. The all-bytes figure is an upper bound."
        ),
    }


def _production_trace() -> dict[str, Any]:
    """Record the current MAST-to-null seams established by reachability."""
    return {
        "mast_archive": {
            "path": "/work/projects/imas_gpu/mast/level1/shots/<shot>.zarr",
            "observed_shot_store_count": 17111,
            "catalog_target_l2_shots": TARGET_L2_SHOTS,
            "catalog_target_h200s": TARGET_H200S,
            "incumbent_flux_examples": [
                {
                    "shot": 22111,
                    "shape": [7, 65, 129],
                    "dtype": "float32",
                    "chunks": [7, 65, 129],
                },
                {
                    "shot": 26935,
                    "shape": [83, 65, 129],
                    "dtype": "float32",
                    "chunks": [42, 33, 65],
                    "stored_psirz_bytes": 1840076,
                },
                {
                    "shot": 30047,
                    "shape": [129, 65, 129],
                    "dtype": "float32",
                    "chunks": [65, 33, 65],
                    "stored_psirz_bytes": 2815988,
                },
            ],
            "note": (
                "efm/psirz is an incumbent reconstruction output used only to "
                "establish "
                "the realistic field shape and storage regime, not a solve input. "
                "The L1 store count is not the 11,573-shot L2 target census."
            ),
        },
        "solve_input_read": {
            "source": "nova/imas/mast_solve_inputs.py:read_solve_inputs",
            "operations": [
                "open one shot Zarr group on the host",
                "loop over current and field clock groups and every selected channel",
                "materialize each channel through NumPy as float64",
                "intersect finite masks, apply host calibration, and tensorize xarray",
            ],
            "batching": (
                "whole channels per shot, not a fixed leading solve-slice batch"
            ),
            "host_device_transfer": (
                "none specified because no solve connector consumes ShotSignals"
            ),
        },
        "missing_connector": {
            "evidence": (
                "SignalStore/IdsIngest have test-only consumers; "
                "read_solve_inputs returns "
                "ShotSignals and has no ReconstructProfile caller"
            ),
            "consequence": (
                "there is no actual end-to-end MAST archive-to-JAX FieldNull "
                "production path "
                "whose 4.5 TB wall time can honestly be measured today"
            ),
        },
        "jax_solve": {
            "source": "nova/equilibrium/profile.py:ReconstructProfile",
            "construction": (
                "geometry and Green operators are converted to device arrays once "
                "in __post_init__"
            ),
            "batch_api": (
                "solve_batch/least_squares_batch use vmap over a leading shot/time "
                "axis, "
                "but the caller must apply jit"
            ),
            "fieldnull_calls": [
                "magnetic_axis_subgrid once in _result for every returned solved field",
                "traced_smooth_boundary_read repeatedly inside each nonlinear sweep",
                "connectivity paths also call fixed-slot xpoint_candidates with "
                "eight slots",
            ],
            "shape_recompilation": (
                "grid, leading batch, dtype, topology levels, bisections, and rays "
                "are static compile keys; no production shape bucketing or "
                "persistent executable registry exists"
            ),
            "fixed_outputs": (
                "axis is scalar; X candidates are NaN-padded fixed slots and "
                "overflow is not exposed"
            ),
            "dtype": (
                "archive fields are fp32, the MAST reader promotes to host float64, "
                "and "
                "ReconstructProfile plus stencil_nulls enable fp64"
            ),
            "downstream": [
                "ProfileResult.axis and boundary_flux",
                "connectivity boundary binding, X-set deduplication, and topology "
                "masks",
                "legacy eager Flux/FieldNull plotting and contour consumers remain "
                "host-only",
            ],
        },
    }


def _recommendations(
    comparisons, gpu
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build an evidence-linked repair order and numeric gates."""
    matrix_fp32 = [
        row
        for row in gpu["measurements"]
        if row["route"] == "normalized_matrix_fit" and row["dtype"] == "float32"
    ]
    best = max(matrix_fp32, key=lambda row: row["resident_fields_per_second"])
    crossover = [
        row
        for row in comparisons
        if row["route"] == "normalized_matrix_fit"
        and row["dtype"] == "float32"
        and row["h200_speedup"] > 1.0
    ]
    crossover_batch = min((row["batch"] for row in crossover), default=None)
    repairs = [
        {
            "priority": 1,
            "repair": "batch-first shape-bucketed API",
            "mechanism": (
                "jit one vmap per grid, dtype, and bounded batch bucket; pad the "
                "final batch "
                "and remove all per-field Python dispatch"
            ),
        },
        {
            "priority": 2,
            "repair": "device-resident solve pipeline",
            "mechanism": (
                "batch decode/preprocess, use pinned asynchronous copies, and keep "
                "fixed-slot "
                "results on device through topology and reconstruction"
            ),
        },
        {
            "priority": 3,
            "repair": "geometry-native vectorized degree classifier",
            "mechanism": (
                "replace scalar sign alternation by fixed cyclic Poincare degree "
                "of the gradient, preserving the distinct rectangular and "
                "hex/unstructured neighbour contracts; share geometry-neutral "
                "scoring and result packing only"
            ),
        },
        {
            "priority": 4,
            "repair": "normalized static local fit",
            "mechanism": (
                "center and scale each uniform 3x3 block, precompute its "
                "pseudoinverse, and "
                "replace thousands of tiny SVD launches by a fused matrix multiply"
            ),
        },
        {
            "priority": 5,
            "repair": "ranked capacity, confidence, and precision contract",
            "mechanism": (
                "use audited fp32 in the hot path, retain fp64 "
                "reference/fallback, rank fixed slots with lax.top_k, and return "
                "candidate count, overflow, discarded-score bound, signed index, "
                "resolved/unresolved/absent confidence, and persistence. Cluster "
                "only same-root estimates and preserve summed index."
            ),
        },
        {
            "priority": 6,
            "repair": "archive-to-solve orchestration",
            "mechanism": (
                "join heterogeneous clocks into explicit slice batches and overlap "
                "Zarr I/O, "
                "LZ4 decode, calibration, transfer, solve, and writeback"
            ),
        },
    ]
    gates = [
        {
            "gate": "bulk throughput",
            "requirement": (
                "reserved H200 fp32 device-resident throughput at the archived "
                "65x129 shape must exceed one CPU core by at least 3x and its "
                "allocated fields/s target by 2x"
            ),
            "measured_candidate_fields_per_second": best["resident_fields_per_second"],
            "measured_crossover_batch": crossover_batch,
        },
        {
            "gate": "end-to-end",
            "requirement": (
                "with real compressed chunks, sustain at least 1.25 GB/s aggregate "
                "archive input across eight H200s (156.25 MB/s each under an even "
                "shard) and finish a representative multi-shot tranche at no more "
                "than 90% of its "
                "one-hour prorated budget"
            ),
        },
        {
            "gate": "compile stability",
            "requirement": (
                "zero recompiles inside a declared shape/dtype bucket and compile "
                "plus dispatch "
                "below 1% of wall time over a production tranche"
            ),
        },
        {
            "gate": "transfer and materialization",
            "requirement": (
                "dynamic H2D plus required D2H below 20% of pipeline wall time; "
                "no host callback "
                "or variable-length compaction in the hot path"
            ),
        },
        {
            "gate": "clean-field accuracy",
            "requirement": (
                "zero fp32 type failures; valid-count parity with fp64; "
                "localization below "
                "0.05 grid cells and 0.5 mm on the clean reference matrix"
            ),
        },
        {
            "gate": "scientific robustness",
            "requirement": (
                "publish the full weak-null/noise precision-recall curve, zero "
                "silent truncation, and explicit false-positive, false-negative, "
                "overflow, discarded-score, persistence, and topology-conservation "
                "metrics. Candidate generation recall must be 100% for reliable "
                "nonzero native degree without overflow."
            ),
        },
        {
            "gate": "capacity ordering",
            "requirement": (
                "for K-1, K, K+1, and 2K candidates, count is exact, overflow "
                "flips iff capacity is exceeded, top K is invariant to row order, "
                "and cluster index sums equal pre-cluster sums"
            ),
        },
        {
            "gate": "memory",
            "requirement": (
                "largest compiled bucket, double-buffered input, and intermediates "
                "remain below "
                "80% of one H200's available memory"
            ),
        },
    ]
    return repairs, gates


def combine(cpu, gpu, memory_probe, evidence_dir: Path) -> dict[str, Any]:
    """Combine the two captures and derive the throughput verdict."""
    if cpu["environment"]["platform"] != "cpu":
        raise ValueError("CPU input is not a CPU capture")
    if gpu["environment"]["platform"] != "gpu":
        raise ValueError("GPU input is not a GPU capture")
    if cpu["source_hashes"] != gpu["source_hashes"]:
        raise ValueError("CPU and GPU captures used different source files")
    production_names = [
        name
        for name in cpu["source_hashes"]
        if name != "benchmarks/fieldnull_gpu_throughput.py"
    ]
    if any(
        memory_probe["source_hashes"].get(name) != cpu["source_hashes"][name]
        for name in production_names
    ):
        raise ValueError("memory probe used different production kernels")
    comparisons = _cross_platform(cpu, gpu)
    repairs, gates = _recommendations(comparisons, gpu)
    capture_hashes = cpu["source_hashes"]
    assembled_hashes = dict(capture_hashes)
    assembled_hashes["benchmarks/fieldnull_gpu_throughput.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return {
        "schema_version": 1,
        "benchmark": "fieldnull_gpu_throughput",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_hashes": assembled_hashes,
        "capture_benchmark_sha256": capture_hashes[
            "benchmarks/fieldnull_gpu_throughput.py"
        ],
        "assembly_note": (
            "baseline capture kernels and timing loops are unchanged; only "
            "feature-matched accuracy assembly, execution-semantics documentation, "
            "and the separately captured bounded memory-probe command were added "
            "after those captures"
        ),
        "archive_target": _archive_target(),
        "production_trace": _production_trace(),
        "existing_h200_evidence": _existing_evidence(evidence_dir),
        "runs": {"cpu": cpu, "gpu": gpu},
        "h200_memory_probe": memory_probe,
        "cross_platform": comparisons,
        "ranking_and_metadata_cost": {
            "cpu": _metadata_cost(cpu),
            "h200": _metadata_cost(gpu),
            "scope": (
                "costs fixed arrays, lax.top_k, candidate counts, overflow, a "
                "discarded-score bound, source cells, signed index, and tri-state "
                "resolved/unresolved/absent metadata"
            ),
            "scientific_limit": (
                "the confidence and domain-index values are placeholders used only "
                "to cost the result schema; production values require native "
                "Poincare degree, covariance calibration, and index-preserving "
                "clustering"
            ),
        },
        "accuracy": _accuracy(cpu, gpu),
        "execution_semantics": {
            "device": (
                "all timed kernels are pure JAX, lowered and compiled for one "
                "declared backend; resident timings synchronize every device leaf "
                "without copying it to the host"
            ),
            "fixed_outputs": (
                "ranked results are fixed arrays plus scalar count, overflow, and "
                "discarded-score metadata, so downstream topology can remain on "
                "device without a callback or variable-length host compaction"
            ),
            "autodiff": (
                "local fitted coordinates are differentiable conditional on a "
                "fixed selected cell, but sign classification, top_k ordering, "
                "capacity overflow, and topology changes are discrete. Gradients "
                "must not be interpreted across candidate identity changes."
            ),
        },
        "roofline_attribution": {
            "archive_io_and_decompression": (
                "unmeasured end-to-end and mandatory: the current reader is host "
                "Zarr/LZ4"
            ),
            "preprocessing": (
                "unmeasured end-to-end: calibration, finite-mask intersection, "
                "clock join, "
                "sensor packing, and reconstruction dominate the raw-to-field seam"
            ),
            "transfer": (
                "measured for dynamic flux batches and fixed outputs; static "
                "geometry stays resident"
            ),
            "fieldnull_compute": (
                "measured as synchronized single and batched kernels; reported "
                "field bytes/s "
                "is an effective kernel rate, not archive read bandwidth"
            ),
            "launch_and_compile": (
                "measured separately; shape and dtype changes are demonstrated "
                "compile keys"
            ),
        },
        "bottleneck_verdict": (
            "Single-field H200 latency is genuinely slower for several documented "
            "paths, but it is not the bulk-throughput predictor. The legacy disaster "
            "is a sequential per-vertex scan; current stencil latency is tiny-SVD "
            "and launch dominated. A batched normalized matrix kernel can test the "
            "compute repair, while the one-hour archive claim remains blocked on "
            "the missing MAST-to-solve connector and an "
            "unmeasured overlapped I/O/decompression pipeline."
        ),
        "recommended_repairs": repairs,
        "acceptance_gates": gates,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run one capture or combine already captured platform results."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("measure")
    capture.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    capture.add_argument("--partial", type=Path, required=True)
    memory = subparsers.add_parser("measure-memory")
    memory.add_argument("--platform", choices=("gpu",), required=True)
    memory.add_argument("--partial", type=Path, required=True)
    merge = subparsers.add_parser("combine")
    merge.add_argument("--cpu", type=Path, required=True)
    merge.add_argument("--gpu", type=Path, required=True)
    merge.add_argument("--memory-probe", type=Path, required=True)
    merge.add_argument("--evidence-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "measure":
        _write_json(args.partial, measure(args.platform))
    elif args.command == "measure-memory":
        _write_json(args.partial, measure_memory(args.platform))
    else:
        _write_json(
            args.output,
            combine(
                _load(args.cpu),
                _load(args.gpu),
                _load(args.memory_probe),
                args.evidence_dir,
            ),
        )


if __name__ == "__main__":
    main()
