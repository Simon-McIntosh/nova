"""Measure the host and traced field-null/topology routes at their live regimes.

The four modules examined here do not form four symmetric class pairs.

``fieldnull``
    The host class is a stateful xarray/plot adapter used by the eager grid and
    IMAS readers.  The traced class is an unreferenced convenience wrapper
    around the live ``Null2D`` and ``Target`` kernels.  Their shared structured
    hex-stencil construction already lives in :mod:`nova.geometry.hexstencil`.
``null``
    ``Null1D`` and ``Null2D`` are fixed-shape differentiable kernels.  The host
    comparison is the categorisation and adaptive-length interpolation inside
    :class:`nova.biot.fieldnull.FieldNull`, not a class-compatible peer.
``target``
    The traced target evaluates coupling matrices.  The class with the same
    token in :mod:`nova.biot.biotframe` is a geometry table, so matrix products
    are compared with NumPy and extended precision rather than pretending the
    two classes implement one interface.
``topology``
    The traced topology is the pure fixed-shape form used by the differentiable
    forward operator.  Its host analogue is distributed across the stateful
    ``FieldNull``, wall selector, ``PlasmaGrid`` and ``Plasma`` methods.

One process measures one JAX platform because platform selection happens before
JAX initialises.  Capture CPU and accelerator runs separately, then merge them
into the committed JSON and SVG::

    JAX_PLATFORMS=cpu uv run python benchmarks/null_stack_route.py measure \
        --output /tmp/null_stack_cpu.json
    JAX_PLATFORMS=cuda uv run python benchmarks/null_stack_route.py measure \
        --output /tmp/null_stack_gpu.json
    uv run python benchmarks/null_stack_route.py merge \
        --cpu /tmp/null_stack_cpu.json --gpu /tmp/null_stack_gpu.json \
        --output-json docs/figures/jax-dissolution/null_stack_route.json \
        --output-svg docs/figures/jax-dissolution/null_stack_route.svg

Timing reports the minimum of repeated, synchronised calls.  First evaluation
is retained separately because traced compilation is a construction cost in a
shape-specialised workload.  Inputs are placed on device before warm timing, so
accelerator numbers are kernel dispatch plus evaluation rather than host-device
transfer.
"""

# SVG elements remain readable as complete strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import xarray as xr

from nova.biot.fieldnull import FieldNull as HostFieldNull
from nova.geometry import select as host_select
from nova.geometry.hexstencil import hex_stencil
from nova.jax.fieldnull import FieldNull as TracedFieldNull
from nova.jax.null import Null1D, Null2D
from nova.jax.target import Target
from nova.jax.topology import Topology

REPEATS = 7
FULL_GRID_SIZES = (17, 33, 65, 129)
QUICK_GRID_SIZES = (17, 33)
FULL_TARGET_SIZES = (256, 1024, 4096, 16384)
QUICK_TARGET_SIZES = (256, 1024)
MAX_NULLS = 8

ROOT = Path(
    os.environ.get("NOVA_BENCH_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
MODULE_PATHS = {
    "fieldnull": ROOT / "nova/jax/fieldnull.py",
    "null": ROOT / "nova/jax/null.py",
    "target": ROOT / "nova/jax/target.py",
    "topology": ROOT / "nova/jax/topology.py",
}


@dataclass(frozen=True)
class NullCase:
    """One analytic flux surface sampled on a structured hex-neighbour grid."""

    size: int
    x: np.ndarray
    z: np.ndarray
    x2d: np.ndarray
    z2d: np.ndarray
    coordinate: np.ndarray
    stencil: np.ndarray
    psi2d: np.ndarray
    structured: xr.Dataset
    unstructured: xr.Dataset


def _git_commit() -> str:
    """Return the exact source revision measured."""
    if override := os.environ.get("NOVA_BENCH_GIT_COMMIT"):
        return override
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    """Return content hashes for every implementation used in the comparison."""
    paths = {
        **MODULE_PATHS,
        "host_fieldnull": ROOT / "nova/biot/fieldnull.py",
        "host_select": ROOT / "nova/geometry/select.py",
        "hexstencil": ROOT / "nova/geometry/hexstencil.py",
    }
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths.values()
    }


def _block(value: Any) -> Any:
    """Synchronise every JAX leaf and return the original pytree."""
    return jax.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf
        ),
        value,
    )


def _once(call: Callable[[], Any], sync: bool = False) -> tuple[float, Any]:
    """Return elapsed seconds and result for one call."""
    start = time.perf_counter()
    result = call()
    if sync:
        _block(result)
    return time.perf_counter() - start, result


def _fastest(call: Callable[[], Any], sync: bool = False) -> float:
    """Return the least-contaminated wall seconds across repeated calls."""
    best = math.inf
    for _ in range(REPEATS):
        elapsed, _ = _once(call, sync)
        best = min(best, elapsed)
    return best


def _scaled_max_error(value: np.ndarray, truth: np.ndarray) -> float:
    """Return maximum error normalised on the reference set magnitude."""
    value = np.asarray(value, dtype=np.longdouble)
    truth = np.asarray(truth, dtype=np.longdouble)
    finite = np.isfinite(value) & np.isfinite(truth)
    if not finite.any():
        return math.inf
    scale = np.max(np.abs(truth[finite]))
    denominator = scale if scale > 0 else np.longdouble(1)
    return float(np.max(np.abs(value[finite] - truth[finite])) / denominator)


def _source_reachability() -> dict[str, Any]:
    """Return importers of each traced module from the current Python tree."""
    imports: dict[str, list[str]] = {name: [] for name in MODULE_PATHS}
    for path in sorted((*ROOT.glob("nova/**/*.py"), *ROOT.glob("tests/**/*.py"))):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError, UnicodeDecodeError:
            continue
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        for name in imports:
            module = f"nova.jax.{name}"
            if any(
                item == module or item.startswith(f"{module}.") for item in imported
            ):
                if path.resolve() != MODULE_PATHS[name].resolve():
                    imports[name].append(str(path.relative_to(ROOT)))

    result = {}
    for name, paths in imports.items():
        result[name] = {
            "production_importers": [
                path for path in paths if path.startswith("nova/")
            ],
            "test_importers": [path for path in paths if path.startswith("tests/")],
        }
    result["host_fieldnull"] = {
        "production_importers": [
            "nova/biot/grid.py",
            "nova/imas/flux.py",
        ],
        "test_importers": [],
    }
    return result


def _api_contracts() -> dict[str, Any]:
    """Return the incompatible public surfaces that constrain the verdicts."""
    return {
        "fieldnull": {
            "host_signature": str(inspect.signature(HostFieldNull)),
            "traced_signature": str(inspect.signature(TracedFieldNull)),
            "host_result": "mutable dictionaries with adaptive-length arrays",
            "traced_result": "two fixed (maxsize, 4) device arrays padded by NaN",
            "host_features": [
                "structured and unstructured xarray state",
                "optional loop filtering",
                "plot adapter",
            ],
            "traced_features": [
                "cached Null2D and Target construction",
                "device arrays",
                "fixed-size output",
            ],
            "differences": [
                "host takes subgrid and optional loop controls; traced takes maxsize",
                "host mutates dictionaries; traced mutates two device-array attributes",
                "host rows have x, z and psi; traced rows also carry null type",
                "host point counts are lengths; traced counts scan NaN padding",
                "host is eager float64; traced follows JAX device precision",
            ],
            "shared_component": "nova.geometry.hexstencil.hex_stencil",
        },
        "null": {
            "traced_signatures": {
                "Null1D": str(inspect.signature(Null1D)),
                "Null2D": str(inspect.signature(Null2D)),
            },
            "host_location": "categorisation and adaptive interpolation in nova.biot.fieldnull",
            "semantic_split": [
                "fixed size versus adaptive length",
                "NaN padding versus absent rows",
                "traceable selection versus Python control flow",
                "Null1D returns four values; host wall selection returns three",
                "zero wall polarity is all NaN traced and all zero host",
                "degenerate surface type is NaN traced and raises host",
                "traced stationary coordinates floor a degenerate determinant",
            ],
        },
        "target": {
            "traced_signature": str(inspect.signature(Target)),
            "traced_role": "evaluate source-target and plasma-target coupling matrices",
            "host_same_token_role": "store target geometry and reduction metadata",
            "common_interface": False,
            "differences": [
                "traced construction takes two coupling matrices and a null kernel",
                "host construction takes coordinate data and reduction metadata",
                "traced exposes external and internal matrix products",
                "host exposes geometry columns and source-target shape bookkeeping",
                "traced is a pytree; host is a mutable pandas-derived frame",
            ],
        },
        "topology": {
            "traced_signature": str(inspect.signature(Topology)),
            "traced_role": "pure fixed-shape normalization, boundary selection and ionization",
            "host_role": "stateful behavior distributed across PlasmaGrid and Plasma",
            "semantic_split": [
                "sentinel values versus topology exceptions",
                "fixed arrays versus adaptive null collections",
                "functional return versus mutation of plasma state",
                "update_batch has no host-class counterpart",
                "traced precision follows JAX device policy; host remains float64",
            ],
        },
    }


def _null_case(size: int) -> NullCase:
    """Return a double-well flux map with two minima and one saddle.

    The analytic surface is ``x**4/4 - x**2/2 + z**2/2``.  Its stationary
    points are minima at ``(-1, 0)`` and ``(1, 0)`` with flux ``-1/4``, and a
    saddle at ``(0, 0)`` with zero flux.  Grid extents make all three points
    grid vertices at every measured size.
    """
    x = np.linspace(-1.6, 1.6, size, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x2d, z2d = np.meshgrid(x, z, indexing="ij")
    coordinate = np.c_[x2d.ravel(), z2d.ravel()]
    stencil = hex_stencil((size, size))
    psi2d = 0.25 * x2d**4 - 0.5 * x2d**2 + 0.5 * z2d**2

    structured = xr.Dataset(coords={"x": x, "z": z})
    structured["x2d"] = (("x", "z"), x2d)
    structured["z2d"] = (("x", "z"), z2d)

    unstructured = xr.Dataset(
        coords={
            "node": np.arange(coordinate.shape[0]),
            "x": ("node", coordinate[:, 0]),
            "z": ("node", coordinate[:, 1]),
            "stencil_index": ("interior", stencil[:, 0]),
        }
    )
    unstructured["stencil"] = (("interior", "stencil_vertex"), stencil)
    return NullCase(
        size,
        x,
        z,
        x2d,
        z2d,
        coordinate,
        stencil,
        psi2d,
        structured,
        unstructured,
    )


def _host_fieldnull(data: xr.Dataset) -> HostFieldNull:
    """Construct a host finder with its xarray views linked."""
    result = HostFieldNull(data=data)
    result.load_arrays()
    return result


def _traced_fieldnull(data: xr.Dataset) -> TracedFieldNull:
    """Construct a traced wrapper and materialize its cached null kernel."""
    result = TracedFieldNull(data=data, maxsize=MAX_NULLS)
    result.load_arrays()
    _block(result.null.coordinate)
    return result


def _finite_rows(values: Any) -> np.ndarray:
    """Return finite fixed-size null rows as host arrays."""
    values = np.asarray(values)
    return values[np.isfinite(values[:, 0])]


def _null_accuracy(host: HostFieldNull, traced: TracedFieldNull) -> dict[str, Any]:
    """Compare both routes with the analytic stationary points."""
    host_rows = np.r_[
        np.c_[host.data_o["points"], host.data_o["psi"], host.data_o["null_type"]],
        np.c_[host.data_x["points"], host.data_x["psi"], host.data_x["null_type"]],
    ]
    traced_rows = np.r_[_finite_rows(traced.data_o), _finite_rows(traced.data_x)]
    truth = np.array(
        [
            [-1.0, 0.0, -0.25, -1.0],
            [1.0, 0.0, -0.25, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    def errors(rows: np.ndarray) -> tuple[float, float, bool]:
        point_errors = []
        flux_errors = []
        types_match = True
        for expected in truth:
            compatible = rows[rows[:, 3] == expected[3]]
            if not len(compatible):
                return math.inf, math.inf, False
            index = int(
                np.argmin(np.linalg.norm(compatible[:, :2] - expected[:2], axis=1))
            )
            point_errors.append(
                float(np.linalg.norm(compatible[index, :2] - expected[:2]))
            )
            flux_errors.append(abs(float(compatible[index, 2] - expected[2])))
            types_match &= compatible[index, 3] == expected[3]
        return max(point_errors), max(flux_errors), bool(types_match)

    host_point, host_flux, host_types = errors(host_rows)
    traced_point, traced_flux, traced_types = errors(traced_rows)
    route_point = 0.0
    for row in host_rows:
        compatible = traced_rows[traced_rows[:, 3] == row[3]]
        route_point = max(
            route_point,
            float(np.min(np.linalg.norm(compatible[:, :2] - row[:2], axis=1))),
        )
    return {
        "expected_count": {"o": 2, "x": 1},
        "host_count": {"o": int(host.o_point_number), "x": int(host.x_point_number)},
        "traced_count": {
            "o": int(len(_finite_rows(traced.data_o))),
            "x": int(len(_finite_rows(traced.data_x))),
        },
        "host_max_coordinate_error": host_point,
        "traced_max_coordinate_error": traced_point,
        "host_max_flux_error": host_flux,
        "traced_max_flux_error": traced_flux,
        "host_types_match": host_types,
        "traced_types_match": traced_types,
        "route_max_coordinate_deviation": route_point,
        "host_dtype": str(host_rows.dtype),
        "traced_dtype": str(traced_rows.dtype),
    }


def _measure_null_routes(
    grid_sizes: tuple[int, ...],
) -> tuple[list[dict], list[dict], dict]:
    """Measure wrapper evaluation and the categorisation kernels."""
    fieldnull_rows: list[dict] = []
    null_rows: list[dict] = []
    smallest_autodiff: dict[str, Any] = {}
    for size in grid_sizes:
        case = _null_case(size)
        psi_flat = case.psi2d.ravel()
        psi_device = jnp.asarray(psi_flat)

        for layout, data, host_psi in (
            ("structured", case.structured, case.psi2d),
            ("unstructured", case.unstructured, psi_flat),
        ):
            host_construct, host = _once(lambda data=data: _host_fieldnull(data))
            traced_construct, traced = _once(
                lambda data=data: _traced_fieldnull(data), True
            )

            host_first, _ = _once(lambda: host.update_null(host_psi))
            traced_first, _ = _once(lambda: traced.update_null(psi_device), True)
            host_warm = _fastest(lambda: host.update_null(host_psi))
            traced_warm = _fastest(lambda: traced.update_null(psi_device), True)

            fieldnull_rows.append(
                {
                    "size": size,
                    "nodes": size * size,
                    "layout": layout,
                    "host_construction_us": 1e6 * host_construct,
                    "traced_construction_us": 1e6 * traced_construct,
                    "host_first_ms": 1e3 * host_first,
                    "traced_first_ms": 1e3 * traced_first,
                    "host_warm_us": 1e6 * host_warm,
                    "traced_warm_us": 1e6 * traced_warm,
                    "accuracy": _null_accuracy(host, traced),
                }
            )

        host = _host_fieldnull(case.structured)
        host_first, host_result = _once(lambda: host.categorize_2d(case.psi2d))
        host_warm = _fastest(lambda: host.categorize_2d(case.psi2d))
        null = Null2D(
            jnp.asarray(case.coordinate),
            jnp.asarray(case.stencil),
            jnp.asarray(case.coordinate[case.stencil]),
            MAX_NULLS,
        )
        stencil_device = jnp.asarray(case.stencil)
        psi_stencil = psi_device[stencil_device]
        traced_first, traced_result = _once(lambda: null.categorize(psi_stencil), True)
        traced_warm = _fastest(lambda: null.categorize(psi_stencil), True)
        host_count = [int(np.sum(mask)) for mask in host_result]
        traced_count = [int(value) for value in np.asarray(traced_result[0])]
        null_rows.append(
            {
                "size": size,
                "nodes": size * size,
                "host_first_ms": 1e3 * host_first,
                "traced_first_ms": 1e3 * traced_first,
                "host_warm_us": 1e6 * host_warm,
                "traced_warm_us": 1e6 * traced_warm,
                "host_count": {"o": host_count[0], "x": host_count[1]},
                "traced_count": {"o": traced_count[0], "x": traced_count[1]},
            }
        )

        if size == grid_sizes[0]:
            gradient = jax.grad(lambda values: null(values)[1, 0, 0])(psi_device)
            _block(gradient)
            gradient_host = np.asarray(gradient)
            smallest_autodiff = {
                "quantity": "x coordinate of the first saddle",
                "all_finite": bool(np.isfinite(gradient_host).all()),
                "nonzero_entries": int(np.count_nonzero(gradient_host)),
                "gradient_norm": float(np.linalg.norm(gradient_host)),
                "device": str(gradient.device),
            }
    return fieldnull_rows, null_rows, smallest_autodiff


def _target_arrays(nodes: int) -> tuple[np.ndarray, ...]:
    """Return deterministic coupling matrices and currents."""
    target_index = np.arange(nodes, dtype=np.float64)[:, None]
    source_index = np.arange(32, dtype=np.float64)[None, :]
    plasma_index = np.arange(64, dtype=np.float64)[None, :]
    source = np.sin(0.003 * target_index + 0.17 * source_index) / (source_index + 1.0)
    plasma = np.cos(0.002 * target_index - 0.11 * plasma_index) / (plasma_index + 1.0)
    external_current = np.cos(np.arange(32, dtype=np.float64) * 0.13)
    plasma_current = np.sin(np.arange(64, dtype=np.float64) * 0.07)
    return source, plasma, external_current, plasma_current


def _measure_target(target_sizes: tuple[int, ...]) -> tuple[list[dict], dict]:
    """Measure the coupling evaluator against NumPy and extended precision."""
    rows = []
    autodiff: dict[str, Any] = {}
    wall = Null1D(jnp.asarray([[1.0, 0.0], [1.1, 0.1], [1.0, 0.2]]))
    for nodes in target_sizes:
        source, plasma, external_current, plasma_current = _target_arrays(nodes)

        def host_call():
            return source @ external_current, plasma @ plasma_current

        host_first, host_result = _once(host_call)
        host_warm = _fastest(host_call)

        source_device = jnp.asarray(source)
        plasma_device = jnp.asarray(plasma)
        external_device = jnp.asarray(external_current)
        plasma_current_device = jnp.asarray(plasma_current)
        construction, target = _once(
            lambda: Target(source_device, plasma_device, wall), sync=True
        )

        def traced_call():
            return (
                target.external(external_device),
                target.internal(plasma_current_device),
            )

        traced_first, traced_result = _once(traced_call, True)
        traced_warm = _fastest(traced_call, True)

        exact_external = np.asarray(source, dtype=np.longdouble) @ np.asarray(
            external_current, dtype=np.longdouble
        )
        exact_internal = np.asarray(plasma, dtype=np.longdouble) @ np.asarray(
            plasma_current, dtype=np.longdouble
        )
        rows.append(
            {
                "nodes": nodes,
                "source_columns": 32,
                "plasma_columns": 64,
                "host_construction_us": 0.0,
                "traced_construction_us": 1e6 * construction,
                "host_first_ms": 1e3 * host_first,
                "traced_first_ms": 1e3 * traced_first,
                "host_warm_us": 1e6 * host_warm,
                "traced_warm_us": 1e6 * traced_warm,
                "host_external_error": _scaled_max_error(
                    host_result[0], exact_external
                ),
                "host_internal_error": _scaled_max_error(
                    host_result[1], exact_internal
                ),
                "traced_external_error": _scaled_max_error(
                    np.asarray(traced_result[0]), exact_external
                ),
                "traced_internal_error": _scaled_max_error(
                    np.asarray(traced_result[1]), exact_internal
                ),
                "host_dtype": str(host_result[0].dtype),
                "traced_dtype": str(traced_result[0].dtype),
            }
        )
        if nodes == target_sizes[0]:
            direction = jnp.linspace(-0.5, 0.5, external_device.size)
            _, tangent = jax.jvp(target.external, (external_device,), (direction,))
            _block(tangent)
            expected = source @ np.asarray(direction, dtype=np.float64)
            autodiff = {
                "quantity": "external flux directional derivative",
                "all_finite": bool(np.isfinite(np.asarray(tangent)).all()),
                "scaled_error_against_matrix_product": _scaled_max_error(
                    np.asarray(tangent), expected
                ),
                "device": str(tangent.device),
            }
    return rows, autodiff


def _topology_case(size: int) -> tuple[np.ndarray, ...]:
    """Return a wall-limited Gaussian equilibrium with analytic anchors."""
    x = np.linspace(0.5, 1.5, size, dtype=np.float64)
    z = np.linspace(-0.6, 0.6, size, dtype=np.float64)
    x2d, z2d = np.meshgrid(x, z, indexing="ij")
    coordinate = np.c_[x2d.ravel(), z2d.ravel()]
    stencil = hex_stencil((size, size))
    theta = np.linspace(0.0, 2.0 * np.pi, max(32, 2 * size), endpoint=False)
    wall = np.c_[1.0 + 0.45 * np.cos(theta), 0.40 * np.sin(theta)]
    sigma = 0.35

    def flux(points: np.ndarray) -> np.ndarray:
        return np.exp(-((points[:, 0] - 1.0) ** 2 + points[:, 1] ** 2) / sigma**2)

    psi_grid = flux(coordinate)
    psi_wall = flux(wall)
    return coordinate, stencil, wall, psi_grid, psi_wall


def _host_topology(
    finder: HostFieldNull,
    coordinate: np.ndarray,
    wall: np.ndarray,
    psi_grid: np.ndarray,
    psi_wall: np.ndarray,
    polarity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the eager topology from its production host components."""
    finder.update_null(psi_grid.reshape(finder.data.sizes["x"], finder.data.sizes["z"]))
    o_index = int(np.nanargmax(polarity * finder.o_psi))
    data_o = np.r_[finder.o_points[o_index], finder.o_psi[o_index]]
    if finder.x_point_number:
        x_index = int(np.nanargmax(polarity * (finder.x_psi - data_o[2])))
        data_x = np.r_[finder.x_points[x_index], finder.x_psi[x_index]]
    else:
        data_x = np.array([np.nan, np.nan, np.nan])
    data_w = np.asarray(
        host_select.wall_flux(wall[:, 0], wall[:, 1], psi_wall, polarity)
    )
    if np.isfinite(data_x[0]):
        boundary = data_x if polarity * data_x[2] >= polarity * data_w[2] else data_w
    else:
        boundary = data_w
    psi_norm = (psi_grid - data_o[2]) / (boundary[2] - data_o[2])
    psi_lcfs = 0.999 * (boundary[2] - data_o[2]) + data_o[2]
    ionize = psi_grid >= psi_lcfs if polarity > 0 else psi_grid < psi_lcfs
    return psi_norm, ionize, np.r_[data_o, boundary]


def _measure_topology(grid_sizes: tuple[int, ...]) -> tuple[list[dict], dict]:
    """Measure single and batched topology evaluation."""
    rows: list[dict] = []
    autodiff: dict[str, Any] = {}
    for size in grid_sizes:
        coordinate, stencil, wall, psi_grid, psi_wall = _topology_case(size)
        x = np.unique(coordinate[:, 0])
        z = np.unique(coordinate[:, 1])
        x2d, z2d = np.meshgrid(x, z, indexing="ij")
        data = xr.Dataset(coords={"x": x, "z": z})
        data["x2d"] = (("x", "z"), x2d)
        data["z2d"] = (("x", "z"), z2d)
        host = _host_fieldnull(data)

        grid = Null2D(
            jnp.asarray(coordinate),
            jnp.asarray(stencil),
            jnp.asarray(coordinate[stencil]),
            MAX_NULLS,
        )
        traced = Topology(grid, Null1D(jnp.asarray(wall)))
        psi = np.r_[psi_grid, psi_wall]
        psi_device = jnp.asarray(psi)

        def host_call():
            return _host_topology(host, coordinate, wall, psi_grid, psi_wall, 1.0)

        def traced_call():
            return traced.update(psi_device, 1.0)

        host_first, host_result = _once(host_call)
        traced_first, traced_result = _once(traced_call, True)
        host_warm = _fastest(host_call)
        traced_warm = _fastest(traced_call, True)

        analytic_axis = 1.0
        analytic_boundary = float(np.max(psi_wall))
        analytic_norm = (psi_grid - analytic_axis) / (analytic_boundary - analytic_axis)
        analytic_lcfs = 0.999 * (analytic_boundary - analytic_axis) + analytic_axis
        analytic_ionize = psi_grid >= analytic_lcfs
        traced_norm, traced_ionize = (np.asarray(item) for item in traced_result)
        rows.append(
            {
                "size": size,
                "nodes": size * size,
                "batch": 1,
                "host_first_ms": 1e3 * host_first,
                "traced_first_ms": 1e3 * traced_first,
                "host_warm_us": 1e6 * host_warm,
                "traced_warm_us": 1e6 * traced_warm,
                "host_normalized_error": _scaled_max_error(
                    host_result[0], analytic_norm
                ),
                "traced_normalized_error": _scaled_max_error(
                    traced_norm, analytic_norm
                ),
                "host_ionize_mismatch": float(
                    np.mean(host_result[1] != analytic_ionize)
                ),
                "traced_ionize_mismatch": float(
                    np.mean(traced_ionize != analytic_ionize)
                ),
                "route_normalized_deviation": _scaled_max_error(
                    traced_norm, host_result[0]
                ),
                "route_ionize_mismatch": float(
                    np.mean(traced_ionize != host_result[1])
                ),
                "host_dtype": str(host_result[0].dtype),
                "traced_dtype": str(traced_norm.dtype),
            }
        )

        scales = np.linspace(0.8, 1.2, 5, dtype=np.float64)

        def host_batch_call():
            return [
                _host_topology(
                    host, coordinate, wall, psi_grid * scale, psi_wall * scale, 1.0
                )[:2]
                for scale in scales
            ]

        psi_batch = jnp.asarray(np.stack([psi * scale for scale in scales]))

        def traced_batch_call():
            return traced.update_batch(psi_batch, 1.0)

        host_batch_first, _ = _once(host_batch_call)
        traced_batch_first, _ = _once(traced_batch_call, True)
        rows.append(
            {
                "size": size,
                "nodes": size * size,
                "batch": 5,
                "host_first_ms": 1e3 * host_batch_first,
                "traced_first_ms": 1e3 * traced_batch_first,
                "host_warm_us": 1e6 * _fastest(host_batch_call),
                "traced_warm_us": 1e6 * _fastest(traced_batch_call, True),
            }
        )

        if size == grid_sizes[0]:
            direction = jnp.linspace(-1e-3, 1e-3, psi_device.size)
            _, tangent = jax.jvp(
                lambda values: traced.update(values, 1.0)[0],
                (psi_device,),
                (direction,),
            )
            _block(tangent)
            autodiff = {
                "quantity": "normalized flux directional derivative",
                "all_finite": bool(np.isfinite(np.asarray(tangent)).all()),
                "nonzero_entries": int(np.count_nonzero(np.asarray(tangent))),
                "tangent_norm": float(np.linalg.norm(np.asarray(tangent))),
                "device": str(tangent.device),
            }
    return rows, autodiff


def measure(profile_name: str) -> dict[str, Any]:
    """Return the complete report for the selected JAX platform."""
    grid_sizes = QUICK_GRID_SIZES if profile_name == "quick" else FULL_GRID_SIZES
    target_sizes = QUICK_TARGET_SIZES if profile_name == "quick" else FULL_TARGET_SIZES
    devices = jax.devices()
    fieldnull, null, null_autodiff = _measure_null_routes(grid_sizes)
    target, target_autodiff = _measure_target(target_sizes)
    topology, topology_autodiff = _measure_topology(grid_sizes)
    return {
        "schema_version": 1,
        "benchmark": "null_stack_route",
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": profile_name,
        "source": {
            "git_commit": _git_commit(),
            "root": str(ROOT),
            "sha256": _source_hashes(),
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "xarray": xr.__version__,
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_platform": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "device_kinds": [device.device_kind for device in devices],
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "reachability": _source_reachability(),
        "api_contracts": _api_contracts(),
        "measurements": {
            "fieldnull": fieldnull,
            "null": null,
            "target": target,
            "topology": topology,
        },
        "autodiff": {
            "null": null_autodiff,
            "target": target_autodiff,
            "topology": topology_autodiff,
        },
    }


def _range(rows: list[dict], numerator: str, denominator: str) -> list[float]:
    """Return finite ratios from measurement rows."""
    return [
        row[numerator] / row[denominator]
        for row in rows
        if row.get(denominator, 0) > 0 and math.isfinite(row.get(numerator, math.inf))
    ]


def _ratio_text(values: list[float]) -> str:
    """Format the observed ratio interval."""
    if not values:
        return "not measured"
    low, high = min(values), max(values)
    if low < 0.01:
        return f"{low:.2e}-{high:.2e}x"
    return f"{low:.2f}-{high:.2f}x"


def _synthesis(cpu: dict, gpu: dict | None) -> dict[str, Any]:
    """Attach quantitative evidence to the four module verdicts."""
    cpu_rows = cpu["measurements"]
    gpu_rows = gpu["measurements"] if gpu else {}

    def accelerator_ratios(name: str, *, batch: int | None = None) -> list[float]:
        if not gpu:
            return []
        host = cpu_rows[name]
        accelerated = gpu_rows[name]
        if batch is not None:
            host = [row for row in host if row.get("batch") == batch]
            accelerated = [row for row in accelerated if row.get("batch") == batch]
        by_key = {
            (
                row.get("size", row.get("nodes")),
                row.get("layout"),
                row.get("batch"),
            ): row
            for row in host
        }
        values = []
        for row in accelerated:
            key = (
                row.get("size", row.get("nodes")),
                row.get("layout"),
                row.get("batch"),
            )
            if key in by_key and row["traced_warm_us"] > 0:
                values.append(by_key[key]["host_warm_us"] / row["traced_warm_us"])
        return values

    fieldnull_cpu = _range(cpu_rows["fieldnull"], "traced_warm_us", "host_warm_us")
    null_cpu = _range(cpu_rows["null"], "traced_warm_us", "host_warm_us")
    target_cpu = _range(cpu_rows["target"], "traced_warm_us", "host_warm_us")
    topology_cpu = _range(
        [row for row in cpu_rows["topology"] if row["batch"] == 1],
        "traced_warm_us",
        "host_warm_us",
    )
    accuracy = [row["accuracy"] for row in cpu_rows["fieldnull"]]
    target_accuracy_rows = list(cpu_rows["target"])
    topology_accuracy_rows = [row for row in cpu_rows["topology"] if row["batch"] == 1]
    if gpu:
        target_accuracy_rows.extend(gpu_rows["target"])
        topology_accuracy_rows.extend(
            row for row in gpu_rows["topology"] if row["batch"] == 1
        )
    reachability = cpu["reachability"]
    return {
        "fieldnull": {
            "verdict": "DELETE ONE",
            "delete": "nova.jax.fieldnull.FieldNull wrapper",
            "retain": "nova.biot.fieldnull.FieldNull and the live Null/Target kernels",
            "numbers": {
                "traced_over_host_cpu_warm": _ratio_text(fieldnull_cpu),
                "host_over_traced_accelerator_warm": _ratio_text(
                    accelerator_ratios("fieldnull")
                ),
                "traced_production_importers": len(
                    reachability["fieldnull"]["production_importers"]
                ),
                "host_production_importers": len(
                    reachability["host_fieldnull"]["production_importers"]
                ),
                "max_route_coordinate_deviation": max(
                    row["route_max_coordinate_deviation"] for row in accuracy
                ),
            },
            "reason": (
                "The wrapper has no external importer and adds only xarray-to-fixed-array "
                "adaptation around live Null2D/Target kernels. The host class remains a "
                "live stateful adapter with loop filtering and adaptive results. Hex-stencil "
                "construction is already single-homed."
            ),
        },
        "null": {
            "verdict": "KEEP BOTH",
            "numbers": {
                "traced_over_host_cpu_categorize": _ratio_text(null_cpu),
                "host_over_traced_accelerator_categorize": _ratio_text(
                    accelerator_ratios("null")
                ),
                "production_importers": len(
                    reachability["null"]["production_importers"]
                ),
                "autodiff_gradient_nonzero_entries": cpu["autodiff"]["null"][
                    "nonzero_entries"
                ],
            },
            "reason": (
                "This is not a class pair. The traced fixed-size kernel supplies device "
                "execution and local derivatives; the host path supplies adaptive collections, "
                "state mutation and loop filtering. Both are live and neither interface can "
                "replace the other."
            ),
        },
        "target": {
            "verdict": "KEEP BOTH",
            "numbers": {
                "traced_over_numpy_cpu_warm": _ratio_text(target_cpu),
                "numpy_over_traced_accelerator_warm": _ratio_text(
                    accelerator_ratios("target")
                ),
                "max_traced_scaled_error": max(
                    max(row["traced_external_error"], row["traced_internal_error"])
                    for row in target_accuracy_rows
                ),
                "production_importers": len(
                    reachability["target"]["production_importers"]
                ),
            },
            "reason": (
                "The same-token host class is a geometry container, not a matrix evaluator. "
                "The traced Target is live in the differentiable forward operator and exposes "
                "correct device-linear derivatives; there is no duplicate body to collapse."
            ),
        },
        "topology": {
            "verdict": "KEEP BOTH",
            "numbers": {
                "traced_over_host_cpu_single": _ratio_text(topology_cpu),
                "host_over_traced_accelerator_single": _ratio_text(
                    accelerator_ratios("topology", batch=1)
                ),
                "host_over_traced_accelerator_batch5": _ratio_text(
                    accelerator_ratios("topology", batch=5)
                ),
                "max_route_normalized_deviation": max(
                    row["route_normalized_deviation"] for row in topology_accuracy_rows
                ),
                "production_importers": len(
                    reachability["topology"]["production_importers"]
                ),
            },
            "reason": (
                "The traced class is the pure fixed-shape topology inside the live forward "
                "operator; the host behavior is distributed across mutable plasma objects and "
                "uses adaptive null collections. Their scalar formulas agree, but their state, "
                "failure and batching contracts do not."
            ),
        },
    }


def _svg_document(report: dict[str, Any]) -> str:
    """Render four cost curves and a compact verdict panel as standalone SVG."""
    width, height = 1200, 980
    cpu = report["runs"]["cpu"]
    gpu = report["runs"].get("gpu")
    panels = [
        ("fieldnull", "Full field-null evaluation", "nodes", None),
        ("null", "Null categorisation", "nodes", None),
        ("target", "Target matrix evaluation", "nodes", None),
        ("topology", "Topology update, one map", "nodes", 1),
    ]
    colors = {
        "Host CPU": "#555555",
        "Traced CPU": "#0066aa",
        "Traced accelerator": "#d55e00",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Field-null stack route measurements</title>",
        "<desc>Warm evaluation cost by problem size and module verdict summary.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="system-ui, sans-serif" fill="#111111">',
        '<text x="40" y="38" font-size="24" font-weight="700">Field-null and topology route measurements</text>',
        '<text x="40" y="62" font-size="13" fill="#555">Minimum of seven synchronized calls; inputs resident on device; log-log axes</text>',
    ]

    def rows_for(run: dict, name: str, batch: int | None) -> list[dict]:
        rows = run["measurements"][name]
        if name == "fieldnull":
            rows = [row for row in rows if row["layout"] == "structured"]
        if batch is not None:
            rows = [row for row in rows if row.get("batch") == batch]
        return rows

    for index, (name, title, x_key, batch) in enumerate(panels):
        x0 = 45 + (index % 2) * 575
        y0 = 95 + (index // 2) * 310
        plot_x, plot_y, plot_w, plot_h = x0 + 64, y0 + 44, 475, 205
        cpu_rows = rows_for(cpu, name, batch)
        series = {
            "Host CPU": [(row[x_key], row["host_warm_us"]) for row in cpu_rows],
            "Traced CPU": [(row[x_key], row["traced_warm_us"]) for row in cpu_rows],
        }
        if gpu:
            gpu_rows = rows_for(gpu, name, batch)
            series["Traced accelerator"] = [
                (row[x_key], row["traced_warm_us"]) for row in gpu_rows
            ]
        all_points = [
            point for values in series.values() for point in values if point[1] > 0
        ]
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        min_y = 10 ** math.floor(math.log10(min_y))
        max_y = 10 ** math.ceil(math.log10(max_y))

        def sx(value: float) -> float:
            return plot_x + plot_w * (math.log(value) - math.log(min_x)) / (
                math.log(max_x) - math.log(min_x)
            )

        def sy(value: float) -> float:
            return plot_y + plot_h * (
                1
                - (math.log(value) - math.log(min_y))
                / (math.log(max_y) - math.log(min_y))
            )

        parts.extend(
            [
                f'<rect x="{x0}" y="{y0}" width="550" height="285" rx="4" fill="#fafafa" stroke="#cccccc"/>',
                f'<text x="{x0 + 16}" y="{y0 + 25}" font-size="16" font-weight="650">{title}</text>',
                f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#777"/>',
                f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#777"/>',
                f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 30}" text-anchor="middle" font-size="11">nodes</text>',
                f'<text x="{plot_x - 45}" y="{plot_y + plot_h / 2}" transform="rotate(-90 {plot_x - 45} {plot_y + plot_h / 2})" text-anchor="middle" font-size="11">warm time [microseconds]</text>',
            ]
        )
        for exponent in range(int(math.log10(min_y)), int(math.log10(max_y)) + 1):
            value = 10**exponent
            y = sy(value)
            parts.append(
                f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e4e4e4"/>'
            )
            parts.append(
                f'<text x="{plot_x - 7}" y="{y + 4:.1f}" text-anchor="end" font-size="10">1e{exponent}</text>'
            )
        for value in sorted({point[0] for point in all_points}):
            x = sx(value)
            parts.append(
                f'<line x1="{x:.1f}" y1="{plot_y + plot_h}" x2="{x:.1f}" y2="{plot_y + plot_h + 4}" stroke="#777"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{plot_y + plot_h + 15}" text-anchor="middle" font-size="9">{value}</text>'
            )
        for label, points in series.items():
            color = colors[label]
            coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
            parts.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>'
                )
        legend_x = plot_x + 8
        for offset, label in enumerate(series):
            ly = plot_y + 15 + 17 * offset
            parts.append(
                f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 20}" y2="{ly}" stroke="{colors[label]}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{legend_x + 27}" y="{ly + 4}" font-size="10">{label}</text>'
            )

    summary_y = 730
    parts.extend(
        [
            f'<rect x="45" y="{summary_y}" width="1125" height="210" rx="4" fill="#f7f7f7" stroke="#bbbbbb"/>',
            f'<text x="65" y="{summary_y + 28}" font-size="17" font-weight="700">Verdicts</text>',
        ]
    )
    verdict_colors = {
        "DELETE ONE": "#b2182b",
        "KEEP BOTH": "#2166ac",
        "COLLAPSE": "#1b7837",
    }
    verdicts = report["verdicts"]
    summary_evidence = {
        "fieldnull": (
            f"CPU traced/host {verdicts['fieldnull']['numbers']['traced_over_host_cpu_warm']}; "
            f"CPU host/accelerator {verdicts['fieldnull']['numbers']['host_over_traced_accelerator_warm']}; "
            f"{verdicts['fieldnull']['numbers']['traced_production_importers']} traced importers"
        ),
        "null": (
            f"CPU traced/host {verdicts['null']['numbers']['traced_over_host_cpu_categorize']}; "
            f"CPU host/accelerator {verdicts['null']['numbers']['host_over_traced_accelerator_categorize']}; "
            f"{verdicts['null']['numbers']['autodiff_gradient_nonzero_entries']} nonzero gradient entries"
        ),
        "target": (
            f"CPU traced/NumPy {verdicts['target']['numbers']['traced_over_numpy_cpu_warm']}; "
            f"NumPy/accelerator {verdicts['target']['numbers']['numpy_over_traced_accelerator_warm']}; "
            f"max error {verdicts['target']['numbers']['max_traced_scaled_error']:.2e}"
        ),
        "topology": (
            f"CPU traced/host {verdicts['topology']['numbers']['traced_over_host_cpu_single']}; "
            f"CPU host/accelerator {verdicts['topology']['numbers']['host_over_traced_accelerator_single']}; "
            f"batch 5 {verdicts['topology']['numbers']['host_over_traced_accelerator_batch5']}"
        ),
    }
    for index, name in enumerate(("fieldnull", "null", "target", "topology")):
        item = verdicts[name]
        y = summary_y + 60 + index * 35
        verdict = item["verdict"]
        evidence = summary_evidence[name]
        parts.append(
            f'<text x="65" y="{y}" font-size="13" font-weight="650">{name}</text>'
        )
        parts.append(
            f'<rect x="165" y="{y - 17}" width="110" height="23" rx="3" fill="{verdict_colors[verdict]}"/>'
        )
        parts.append(
            f'<text x="220" y="{y}" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">{verdict}</text>'
        )
        parts.append(
            f'<text x="295" y="{y}" font-size="11" fill="#333">{evidence}</text>'
        )
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def merge(cpu_path: Path, gpu_path: Path | None) -> dict[str, Any]:
    """Return a combined report with verdict synthesis."""
    cpu = json.loads(cpu_path.read_text())
    gpu = json.loads(gpu_path.read_text()) if gpu_path else None
    if cpu["environment"]["jax_platform"] != "cpu":
        raise ValueError("--cpu report was not measured on the CPU platform")
    if gpu and gpu["environment"]["jax_platform"] == "cpu":
        raise ValueError("--gpu report did not use an accelerator platform")
    if gpu and cpu["source"]["git_commit"] != gpu["source"]["git_commit"]:
        raise ValueError("CPU and accelerator reports measured different revisions")
    report = {
        "schema_version": 1,
        "benchmark": "null_stack_route",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": cpu["source"],
        "method": {
            "repeats": REPEATS,
            "warm_statistic": "minimum synchronized wall time",
            "transfer_policy": "inputs resident on device before warm timing",
            "accuracy_references": {
                "fieldnull": "analytic quartic with two minima and one saddle",
                "target": "extended-precision matrix products",
                "topology": "analytic wall-limited Gaussian anchors",
            },
        },
        "reachability": cpu["reachability"],
        "api_contracts": _api_contracts(),
        "runs": {"cpu": cpu} | ({"gpu": gpu} if gpu else {}),
    }
    report["verdicts"] = _synthesis(cpu, gpu)
    return report


def main() -> None:
    """Measure one platform or merge platform reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--profile", choices=("quick", "full"), default="full")
    measure_parser.add_argument("--output", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--cpu", type=Path, required=True)
    merge_parser.add_argument("--gpu", type=Path)
    merge_parser.add_argument("--output-json", type=Path, required=True)
    merge_parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "measure":
        report = measure(args.profile)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    report = merge(args.cpu, args.gpu)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_svg.write_text(_svg_document(report))
    print(json.dumps(report["verdicts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
