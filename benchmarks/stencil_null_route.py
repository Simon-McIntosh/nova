"""Accuracy, semantics, and device cost of the stencil critical-point route.

The routines in :mod:`nova.equilibrium.stencil_nulls` are not a host/traced pair.
The
module owns an eight-neighbour rectangular-grid classifier, a masked magnetic
axis reduction, and a fixed-slot saddle reduction.  The older field-null path
uses a six-neighbour hexagonal stencil and returns every local null; it is a
different geometry and selection contract.  The sub-grid quadratic fit has one
canonical three-array traced interface in ``nova.geometry.select``.

This benchmark measures the current call-site grid sizes on every requested JAX
device, checks positions against analytic critical points and an independently
solved two-Gaussian field, probes degenerate and overflowing candidate sets, and
compares automatic derivatives with centred differences.  It writes the raw
JSON report and an SVG summary under ``docs/figures/jax-dissolution``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize  # type: ignore[import-untyped]

from nova.geometry import select
from nova.jax.config import configure_dtypes
from nova.equilibrium.stencil_nulls import (
    magnetic_axis_subgrid,
    ring_sign_changes,
    xpoint_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/jax-dissolution"
JSON_PATH = OUTPUT / "stencil_null_route.json"
SVG_PATH = OUTPUT / "stencil_null_route.svg"

REPEATS = 12
SIZES = (
    (17, 17, "profile reconstruction test"),
    (61, 61, "batched connectivity contract"),
    (101, 81, "limited and stencil-null tests"),
    (141, 101, "diverted connectivity test"),
    (161, 121, "connectivity amplitude sweep"),
)

RECTANGULAR_RING = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
HEXAGONAL_RING = ((-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1))


def _git_revision() -> str:
    """Return the measured source revision."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _block(value: Any) -> None:
    """Synchronise every array leaf in a JAX result tree."""
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _timed_compiled(function, *args, repeats: int = REPEATS) -> dict[str, float]:
    """Measure lowering plus compilation, first execution, and steady execution."""
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*args).compile()
    compile_ms = 1.0e3 * (time.perf_counter() - start)

    start = time.perf_counter()
    result = compiled(*args)
    _block(result)
    first_us = 1.0e6 * (time.perf_counter() - start)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compiled(*args)
        _block(result)
        samples.append(1.0e6 * (time.perf_counter() - start))
    return {
        "compile_ms": compile_ms,
        "first_execution_us": first_us,
        "steady_us_min": float(np.min(samples)),
        "steady_us_median": float(np.median(samples)),
    }


def _timed_host(function, *args, repeats: int = REPEATS) -> dict[str, float]:
    """Measure a synchronous host reference after one warm call."""
    function(*args)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        samples.append(1.0e6 * (time.perf_counter() - start))
    return {
        "steady_us_min": float(np.min(samples)),
        "steady_us_median": float(np.median(samples)),
    }


def _ring_reference(psi: np.ndarray, offsets=RECTANGULAR_RING) -> np.ndarray:
    """Independent NumPy sign-change count for a closed neighbour ring."""
    ring = np.stack(
        [np.roll(psi, shift=(-dz, -dr), axis=(0, 1)) > psi for dz, dr in offsets]
    )
    changes = np.sum(ring != np.roll(ring, -1, axis=0), axis=0, dtype=np.int32)
    changes[[0, -1], :] = -1
    changes[:, [0, -1]] = -1
    return changes


def _grid(nz: int, nr: int) -> tuple[np.ndarray, ...]:
    """Return the physical grid, a smooth two-peak field, and an in-wall mask."""
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = _two_gaussian_value(rr, zz)
    inside = ((rr - 1.0) / 0.48) ** 2 + (zz / 0.74) ** 2 <= 1.0
    return rg, zg, rr, zz, psi, inside


def _two_gaussian_value(r, z, width: float = 0.15):
    """Two equal smooth flux peaks whose midpoint is a saddle."""
    return np.exp(-((r - 1.007) ** 2 + (z + 0.30) ** 2) / width**2) + np.exp(
        -((r - 1.007) ** 2 + (z - 0.30) ** 2) / width**2
    )


def _two_gaussian_gradient(point: np.ndarray, width: float = 0.15) -> np.ndarray:
    """Analytic gradient used only by the independent root solver."""
    r, z = point
    terms = []
    for centre_z in (-0.30, 0.30):
        value = np.exp(-((r - 1.007) ** 2 + (z - centre_z) ** 2) / width**2)
        terms.append(
            np.array(
                [
                    -2.0 * (r - 1.007) * value / width**2,
                    -2.0 * (z - centre_z) * value / width**2,
                ]
            )
        )
    return np.sum(terms, axis=0)


def _gaussian_truth() -> dict[str, object]:
    """Solve the nonlinear field's extrema and saddle without either route."""
    lower = scipy.optimize.root(_two_gaussian_gradient, np.array([1.007, -0.30]))
    upper = scipy.optimize.root(_two_gaussian_gradient, np.array([1.007, 0.30]))
    saddle = scipy.optimize.root(_two_gaussian_gradient, np.array([1.007, 0.0]))
    if not (lower.success and upper.success and saddle.success):
        raise RuntimeError("independent critical-point solve did not converge")
    return {
        "extrema": [lower.x.tolist(), upper.x.tolist()],
        "saddle": saddle.x.tolist(),
        "gradient_norm_max": float(
            max(
                np.linalg.norm(_two_gaussian_gradient(lower.x)),
                np.linalg.norm(_two_gaussian_gradient(upper.x)),
                np.linalg.norm(_two_gaussian_gradient(saddle.x)),
            )
        ),
    }


def _device_arrays(device, arrays: tuple[np.ndarray, ...]) -> tuple[jax.Array, ...]:
    """Place NumPy inputs explicitly on one measured device."""
    return tuple(jax.device_put(array, device) for array in arrays)


def _cost_rows(device) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Measure the three grid reductions and the canonical subnull fit."""
    rows: list[dict[str, object]] = []
    host_rows: list[dict[str, object]] = []
    for nz, nr, source in SIZES:
        rg, zg, _rr, _zz, psi, inside = _grid(nz, nr)
        d_psi, d_rg, d_zg, d_inside = _device_arrays(device, (psi, rg, zg, inside))
        functions = {
            "ring_sign_changes": (lambda p: ring_sign_changes(p), (d_psi,)),
            "magnetic_axis_subgrid": (
                lambda p, r, z, mask: magnetic_axis_subgrid(p, r, z, mask),
                (d_psi, d_rg, d_zg, d_inside),
            ),
            "xpoint_candidates": (
                lambda p, r, z, mask: xpoint_candidates(
                    p, r, z, mask, k_slots=8, material_dilate=1
                ),
                (d_psi, d_rg, d_zg, d_inside),
            ),
        }
        for name, (function, args) in functions.items():
            rows.append(
                {
                    "device": device.platform,
                    "routine": name,
                    "nz": nz,
                    "nr": nr,
                    "cells": nz * nr,
                    "call_site": source,
                }
                | _timed_compiled(function, *args)
            )
        host_rows.append(
            {
                "routine": "rectangular_ring_numpy_reference",
                "nz": nz,
                "nr": nr,
                "cells": nz * nr,
                "call_site": source,
            }
            | _timed_host(_ring_reference, psi)
        )

    rg, zg, _rr, _zz, psi, _inside = _grid(61, 61)
    centre_i, centre_j = 30, 30
    rows_i = np.repeat(np.arange(centre_i - 1, centre_i + 2), 3)
    cols_j = np.tile(np.arange(centre_j - 1, centre_j + 2), 3)
    r_cluster = rg[cols_j]
    z_cluster = zg[rows_i]
    psi_cluster = psi[rows_i, cols_j]
    d_r, d_z, d_psi = _device_arrays(device, (r_cluster, z_cluster, psi_cluster))
    rows.append(
        {
            "device": device.platform,
            "routine": "subnull_three_arrays",
            "samples": 9,
            "call_site": "canonical traced quadratic fit",
        }
        | _timed_compiled(select.traced_subnull, d_r, d_z, d_psi)
    )
    return rows, host_rows


def _accuracy_rows(device, truth: dict[str, object]) -> list[dict[str, object]]:
    """Compare discrete masks and sub-grid positions with independent truth."""
    rows: list[dict[str, object]] = []
    extrema = np.asarray(truth["extrema"])
    saddle = np.asarray(truth["saddle"])
    for nz, nr, source in SIZES:
        rg, zg, _rr, _zz, psi, inside = _grid(nz, nr)
        d_psi, d_rg, d_zg, d_inside = _device_arrays(device, (psi, rg, zg, inside))
        counts = np.asarray(ring_sign_changes(d_psi))
        rectangle = _ring_reference(psi)
        hexagonal = _ring_reference(psi, HEXAGONAL_RING)
        axis = magnetic_axis_subgrid(d_psi, d_rg, d_zg, d_inside)
        candidates = xpoint_candidates(
            d_psi, d_rg, d_zg, d_inside, k_slots=8, material_dilate=1
        )
        _block((counts, axis, candidates))
        axis_point = np.array([float(axis["r"]), float(axis["z"])])
        axis_error = float(np.min(np.linalg.norm(extrema - axis_point, axis=1)))
        valid = np.asarray(candidates["valid"])
        candidate_points = np.column_stack(
            [np.asarray(candidates["r"])[valid], np.asarray(candidates["z"])[valid]]
        )
        x_error = (
            float(np.min(np.linalg.norm(candidate_points - saddle, axis=1)))
            if candidate_points.size
            else float("nan")
        )
        rows.append(
            {
                "device": device.platform,
                "nz": nz,
                "nr": nr,
                "cells": nz * nr,
                "call_site": source,
                "spacing_max": float(max(np.diff(rg).max(), np.diff(zg).max())),
                "rectangular_reference_exact_fraction": float(
                    np.mean(counts == rectangle)
                ),
                "o_mask_difference_vs_hex_fraction": float(
                    np.mean((counts == 0) != (hexagonal == 0))
                ),
                "x_mask_difference_vs_hex_fraction": float(
                    np.mean((counts == 4) != (hexagonal == 4))
                ),
                "axis_position_error": axis_error,
                "xpoint_position_error": x_error,
                "xpoint_count": int(valid.sum()),
            }
        )
    return rows


def _quadratic_fit_checks(device) -> dict[str, object]:
    """Check the shared traced fit against independent exact quadratics."""
    r = np.array([0.91, 1.00, 1.09] * 3)
    z = np.repeat(np.array([-0.07, 0.00, 0.07]), 3)
    cases = {
        "maximum": {
            "psi": 4.0
            - 1.7 * (r - 1.013) ** 2
            - 0.8 * (z + 0.011) ** 2
            + 0.2 * (r - 1.013) * (z + 0.011),
            "truth": np.array([1.013, -0.011]),
            "type": 1.0,
        },
        "saddle": {
            "psi": 0.4
            + 1.3 * (r - 0.987) ** 2
            - 0.9 * (z - 0.019) ** 2
            + 0.3 * (r - 0.987) * (z - 0.019),
            "truth": np.array([0.987, 0.019]),
            "type": 0.0,
        },
    }
    results = {}
    d_r, d_z = _device_arrays(device, (r, z))
    for name, case in cases.items():
        d_psi = jax.device_put(case["psi"], device)
        result = np.asarray(select.traced_subnull(d_r, d_z, d_psi))
        results[name] = {
            "position_error": float(np.linalg.norm(result[:2] - case["truth"])),
            "type": float(result[3]),
            "expected_type": case["type"],
        }
    return results


def _degenerate_checks(device) -> dict[str, object]:
    """Exercise flat, planar, empty-mask, and fixed-slot overflow semantics."""
    nz = nr = 61
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)
    inside = np.ones((nz, nr), dtype=bool)
    flat = np.ones((nz, nr))
    plane = 1.0 + 0.3 * rr - 0.2 * zz
    periodic = np.sin(8.0 * np.pi * (rr - 0.5)) * np.sin(8.0 * np.pi * (zz + 0.8) / 1.6)
    d_rg, d_zg, d_inside, d_flat, d_plane, d_periodic = _device_arrays(
        device, (rg, zg, inside, flat, plane, periodic)
    )
    flat_axis = magnetic_axis_subgrid(d_flat, d_rg, d_zg, d_inside)
    plane_axis = magnetic_axis_subgrid(d_plane, d_rg, d_zg, d_inside)
    plane_x = xpoint_candidates(
        d_plane, d_rg, d_zg, d_inside, k_slots=8, material_dilate=0
    )
    empty_x = xpoint_candidates(
        d_periodic,
        d_rg,
        d_zg,
        d_inside,
        k_slots=8,
        extra_mask=jnp.zeros_like(d_inside),
        material_dilate=0,
    )
    overflow_x = xpoint_candidates(
        d_periodic, d_rg, d_zg, d_inside, k_slots=8, material_dilate=0
    )
    periodic_counts = _ring_reference(periodic)

    rows_i = np.repeat(np.arange(29, 32), 3)
    cols_j = np.tile(np.arange(29, 32), 3)
    d_r = d_rg[cols_j]
    d_z = d_zg[rows_i]
    plane_sub = np.asarray(select.traced_subnull(d_r, d_z, d_plane[rows_i, cols_j]))
    _block((flat_axis, plane_axis, plane_x, empty_x, overflow_x))
    return {
        "flat_field": {
            "axis_found": bool(flat_axis["found"]),
            "axis_type": float(flat_axis["ntype"]),
            "axis_position_finite": bool(
                np.isfinite([float(flat_axis["r"]), float(flat_axis["z"])]).all()
            ),
            "meaning": "a plateau is classified as zero sign changes",
        },
        "planar_field": {
            "axis_found": bool(plane_axis["found"]),
            "xpoint_count": int(np.asarray(plane_x["valid"]).sum()),
            "subnull_type_is_nan": bool(np.isnan(plane_sub[3])),
            "subnull_position_finite": bool(np.isfinite(plane_sub[:2]).all()),
        },
        "empty_extra_mask": {"xpoint_count": int(np.asarray(empty_x["valid"]).sum())},
        "fixed_slot_overflow": {
            "raw_rectangular_saddles": int(np.sum(periodic_counts == 4)),
            "slots": 8,
            "returned_valid": int(np.asarray(overflow_x["valid"]).sum()),
            "overflow_is_reported": False,
        },
    }


def _geometry_semantics() -> dict[str, object]:
    """Quantify where rectangular and hexagonal neighbour rings disagree."""
    nz = nr = 61
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)
    periodic = np.sin(8.0 * np.pi * (rr - 0.5)) * np.sin(8.0 * np.pi * (zz + 0.8) / 1.6)
    rectangular = _ring_reference(periodic)
    hexagonal = _ring_reference(periodic, HEXAGONAL_RING)
    return {
        "field": "smooth periodic critical-point lattice",
        "shape": [nz, nr],
        "rectangular_saddles": int(np.sum(rectangular == 4)),
        "hexagonal_saddles": int(np.sum(hexagonal == 4)),
        "saddle_mask_difference_fraction": float(
            np.mean((rectangular == 4) != (hexagonal == 4))
        ),
        "rectangular_extrema": int(np.sum(rectangular == 0)),
        "hexagonal_extrema": int(np.sum(hexagonal == 0)),
        "extremum_mask_difference_fraction": float(
            np.mean((rectangular == 0) != (hexagonal == 0))
        ),
    }


def _differentiation_checks(device) -> dict[str, object]:
    """Compare traced position derivatives with centred finite differences."""
    nz = nr = 61
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)
    inside = np.ones((nz, nr), dtype=bool)
    axis_base = 2.0 - 1.3 * (rr - 1.013) ** 2 - 0.7 * (zz + 0.021) ** 2
    x_base = 0.3 + 1.1 * (rr - 0.987) ** 2 - 0.9 * (zz - 0.017) ** 2
    direction = 0.4 * (rr - 1.0) - 0.2 * zz
    d_axis, d_x, d_direction, d_rg, d_zg, d_inside = _device_arrays(
        device, (axis_base, x_base, direction, rg, zg, inside)
    )

    def axis_r(alpha):
        return magnetic_axis_subgrid(
            d_axis + alpha * d_direction, d_rg, d_zg, d_inside
        )["r"]

    def x_r(alpha):
        candidates = xpoint_candidates(
            d_x + alpha * d_direction,
            d_rg,
            d_zg,
            d_inside,
            k_slots=1,
            material_dilate=0,
        )
        return candidates["r"][0]

    epsilon = 1.0e-5
    rows = {}
    for name, function in (
        ("magnetic_axis_subgrid", axis_r),
        ("xpoint_candidates", x_r),
    ):
        automatic = float(jax.grad(function)(0.0))
        centred = float((function(epsilon) - function(-epsilon)) / (2.0 * epsilon))
        rows[name] = {
            "automatic": automatic,
            "centred_difference": centred,
            "absolute_difference": abs(automatic - centred),
            "finite": bool(np.isfinite([automatic, centred]).all()),
            "classification_note": (
                "the selected integer vertex is nondifferentiable; "
                "the local fit carries the derivative"
            ),
        }
    return rows


def _inventory() -> list[dict[str, object]]:
    """Current logical routines, reachable callers, true peers, and decisions."""
    return [
        {
            "routine": "ring_sign_changes",
            "reachability": (
                "internal to magnetic_axis_subgrid and xpoint_candidates; "
                "directly tested"
            ),
            "true_peer": (
                "six-neighbour hexagonal classifiers in Null2D and host FieldNull"
            ),
            "semantic_difference": (
                "eight-neighbour rectangular raster versus six-neighbour "
                "structured or unstructured hex stencil"
            ),
            "verdict": "KEEP BOTH",
            "decision": (
                "keep each geometry contract; relocate the rectangular routine "
                "with the connectivity domain"
            ),
        },
        {
            "routine": "subnull",
            "reachability": "called by Null2D and exercised directly here",
            "true_peer": "none after deleting the stacked adapter",
            "semantic_difference": "one canonical traced three-array contract",
            "verdict": "COLLAPSE",
            "decision": "retain the canonical fit and its explicit host peer",
        },
        {
            "routine": "magnetic_axis_subgrid",
            "reachability": "one profile solve site and three connectivity sites",
            "true_peer": (
                "none; legacy locators return every local extremum and do not "
                "apply wall or flood-region selection"
            ),
            "semantic_difference": (
                "returns one deepest in-mask axis with fixed-shape device semantics"
            ),
            "verdict": "SINGLE-IMPLEMENTATION RELOCATION",
            "decision": (
                "move intact to equilibrium connectivity; reject degenerate "
                "plateau fits before publishing found"
            ),
        },
        {
            "routine": "xpoint_candidates",
            "reachability": (
                "one connectivity boundary site with eight slots; directly "
                "exercised by connectivity tests"
            ),
            "true_peer": (
                "none; legacy locators return unmasked local nulls with a "
                "different stencil"
            ),
            "semantic_difference": (
                "wall dilation, optional flux/flood mask, static slots, "
                "saddle recheck, masked-gradient fencing"
            ),
            "verdict": "SINGLE-IMPLEMENTATION RELOCATION",
            "decision": (
                "move intact to equilibrium connectivity; preserve fixed slots "
                "and expose overflow or prove the upstream mask bounds it"
            ),
        },
    ]


def _environment(device) -> dict[str, object]:
    """Record enough environment detail to interpret timings."""
    import jaxlib  # pylint: disable=import-outside-toplevel

    return {
        "platform": device.platform,
        "device": str(device),
        "device_kind": getattr(device, "device_kind", "unknown"),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "python": platform.python_version(),
        "x64_enabled": bool(jax.config.x64_enabled),
    }


def _summary(payload: dict[str, object]) -> dict[str, object]:
    """Derive review-facing extrema from raw rows."""
    accuracy = payload["accuracy"]
    timings = payload["timings"]
    assert isinstance(accuracy, list)
    assert isinstance(timings, list)
    return {
        "verdicts": [
            {key: row[key] for key in ("routine", "verdict", "decision")}
            for row in payload["inventory"]
        ],
        "rectangular_reference_min_exact_fraction": min(
            row["rectangular_reference_exact_fraction"] for row in accuracy
        ),
        "gaussian_axis_max_position_error": max(
            row["axis_position_error"] for row in accuracy
        ),
        "gaussian_xpoint_max_position_error": max(
            row["xpoint_position_error"] for row in accuracy
        ),
        "largest_grid_steady_us": {
            "%s:%s" % (row["device"], row["routine"]): row["steady_us_median"]
            for row in timings
            if row.get("cells") == 161 * 121
        },
        "degenerate_plateau_axis_found": any(
            checks["flat_field"]["axis_found"]
            for checks in payload["degenerate"].values()
        ),
    }


def _figure(payload: dict[str, object], destination: Path) -> None:
    """Cost curves, localization convergence, and the verdict panel."""
    timings = payload["timings"]
    accuracy = payload["accuracy"]
    inventory = payload["inventory"]
    colors = {
        "ring_sign_changes": "#4575b4",
        "magnetic_axis_subgrid": "#1a9850",
        "xpoint_candidates": "#d73027",
    }
    markers = {"cpu": "o", "gpu": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    for device in sorted({row["device"] for row in timings}):
        for routine, color in colors.items():
            rows = sorted(
                (
                    row
                    for row in timings
                    if row["device"] == device
                    and row["routine"] == routine
                    and "cells" in row
                ),
                key=lambda row: row["cells"],
            )
            axes[0].plot(
                [row["cells"] for row in rows],
                [row["steady_us_median"] for row in rows],
                marker=markers.get(device, "d"),
                color=color,
                linestyle="-" if device == "cpu" else "--",
                label="%s · %s" % (routine.replace("_", " "), device.upper()),
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("grid cells")
    axes[0].set_ylabel("steady execution (µs, median)")
    axes[0].set_title("Call-site cost by grid size")
    axes[0].grid(color="#e7e5df", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7)

    for device in sorted({row["device"] for row in accuracy}):
        rows = sorted(
            (row for row in accuracy if row["device"] == device),
            key=lambda row: row["spacing_max"],
        )
        axes[1].plot(
            [row["spacing_max"] for row in rows],
            [row["axis_position_error"] for row in rows],
            marker=markers.get(device, "d"),
            color="#1a9850",
            label="axis · %s" % device.upper(),
        )
        axes[1].plot(
            [row["spacing_max"] for row in rows],
            [row["xpoint_position_error"] for row in rows],
            marker=markers.get(device, "d"),
            color="#d73027",
            linestyle="--",
            label="X-point · %s" % device.upper(),
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("largest grid spacing (m)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_title("Non-quadratic analytic field")
    axes[1].grid(color="#e7e5df", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].axis("off")
    axes[2].set_title("Measured routing verdicts", loc="left")
    y = 0.94
    for row in inventory:
        axes[2].text(
            0.0,
            y,
            row["routine"].replace("_", " "),
            transform=axes[2].transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        axes[2].text(
            0.0,
            y - 0.06,
            row["verdict"],
            transform=axes[2].transAxes,
            fontsize=9,
            color="#1a9850" if row["verdict"] != "KEEP BOTH" else "#4575b4",
            va="top",
        )
        axes[2].text(
            0.0,
            y - 0.11,
            row["decision"],
            transform=axes[2].transAxes,
            fontsize=7.4,
            color="#4d4d49",
            va="top",
            wrap=True,
        )
        y -= 0.23
    fig.tight_layout()
    fig.savefig(destination, format="svg")
    svg = destination.read_text()
    destination.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def _json_ready(value):
    """Convert arrays and non-finite sentinels to strict JSON values."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the measurement once and write the evidence bundle."""
    configure_dtypes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platforms",
        default="cpu,gpu",
        help="comma-separated JAX platforms to measure",
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--revision",
        help="measured source revision when running from an exported snapshot",
    )
    parser.add_argument("--json", type=Path, default=JSON_PATH)
    parser.add_argument("--svg", type=Path, default=SVG_PATH)
    args = parser.parse_args(argv)

    requested = [name.strip() for name in args.platforms.split(",") if name.strip()]
    devices = []
    unavailable = []
    for name in requested:
        try:
            devices.append(jax.devices(name)[0])
        except RuntimeError as error:
            unavailable.append({"platform": name, "error": str(error)})
    if args.require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("a GPU platform was required but none is visible")

    truth = _gaussian_truth()
    payload: dict[str, object] = {
        "format": "stencil-null-route",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": args.revision or _git_revision(),
        "command": "uv run python benchmarks/stencil_null_route.py --platforms %s%s"
        % (args.platforms, " --require-gpu" if args.require_gpu else ""),
        "call_sites": [
            {"nz": nz, "nr": nr, "cells": nz * nr, "source": source}
            for nz, nr, source in SIZES
        ],
        "inventory": _inventory(),
        "geometry_semantics": _geometry_semantics(),
        "independent_truth": truth,
        "environments": [],
        "unavailable_platforms": unavailable,
        "timings": [],
        "host_reference_timings": [],
        "accuracy": [],
        "quadratic_fit_checks": {},
        "degenerate": {},
        "differentiation": {},
    }
    for device in devices:
        print("measuring", device.platform, str(device), flush=True)
        with jax.default_device(device):
            timings, host_timings = _cost_rows(device)
            payload["timings"].extend(timings)
            if device.platform == "cpu":
                payload["host_reference_timings"].extend(host_timings)
            payload["accuracy"].extend(_accuracy_rows(device, truth))
            payload["quadratic_fit_checks"][device.platform] = _quadratic_fit_checks(
                device
            )
            payload["degenerate"][device.platform] = _degenerate_checks(device)
            payload["differentiation"][device.platform] = _differentiation_checks(
                device
            )
            payload["environments"].append(_environment(device))

    payload["summary"] = _summary(payload)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n"
    )
    _figure(payload, args.svg)
    print(json.dumps(_json_ready(payload["summary"]), indent=2, allow_nan=False))
    print("wrote", args.json)
    print("wrote", args.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
