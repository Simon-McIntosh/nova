# ruff: noqa: E501
"""Connectivity accuracy, topology semantics, and execution-route cost.

The boundary module exposes traced kernels together with host-facing adapters;
the surface module exposes only its traced kernel after the unused adapter was
removed.  Boundary adapters are not independent numerical implementations: they
prepare grid objects and ordinary arrays, invoke the traced kernels, and
materialise results on the host.  This benchmark therefore answers two separate
questions without manufacturing a false arithmetic comparison:

* do the fixed-iteration connectivity kernels agree with an independent
  connected-component labeller and with analytic limited/diverted fields?;
* what do compilation, device-resident steady execution, batching, and the
  host adapters cost on the grids reached by the reconstruction callers?

The flux-surface metrics are additionally arbitrated against the exact volume
derivatives of nested elliptical surfaces.  Run one backend at a time, then
combine the captured reports into the committed JSON and SVG::

    python benchmarks/connectivity_route.py measure --label cpu --output /tmp/cpu.json
    python benchmarks/connectivity_route.py measure --label gpu --output /tmp/gpu.json
    python benchmarks/connectivity_route.py combine --input /tmp/cpu.json \
        --input /tmp/gpu.json --output connectivity_route.json \
        --figure connectivity_route.svg
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy import integrate, ndimage, optimize

from nova.equilibrium.labels import LCFS_ANGLES
from nova.equilibrium.wall_mask import inside_polygon
from nova.equilibrium import connectivity_boundary as boundary
from nova.equilibrium import flux_surface_connectivity as surface

REPEATS = 5
GRID_SHAPES = ((49, 65), (65, 97), (101, 141))
PROFILE_LEVELS = 48
PROFILE_BISECTIONS = 12
PROFILE_RAYS = 128
SURFACE_BINS = 28
BATCH_SIZE = 4


@dataclass
class Grid:
    """Minimal equilibrium-grid interface consumed by the host adapters."""

    rg: np.ndarray
    zg: np.ndarray
    inside_limiter: np.ndarray
    limiter_r: np.ndarray
    limiter_z: np.ndarray


def _block(value: Any) -> Any:
    """Wait for every device leaf and return ``value`` unchanged."""
    for leaf in jax.tree_util.tree_leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()
    return value


def _seconds(call) -> tuple[float, Any]:
    """Return synchronous wall time and result for one call."""
    start = time.perf_counter()
    value = _block(call())
    return time.perf_counter() - start, value


def _steady(call, repeats: int = REPEATS) -> tuple[float, list[float], Any]:
    """Return the minimum synchronous time, all repeats, and the last result."""
    samples = []
    value = None
    for _ in range(repeats):
        elapsed, value = _seconds(call)
        samples.append(elapsed)
    return min(samples), samples, value


def _dense_rectangle(lr: np.ndarray, lz: np.ndarray, count: int = 720):
    """Uniformly sample one closed rectangular wall in arc length."""
    rr = np.append(lr, lr[0])
    zz = np.append(lz, lz[0])
    segment = np.hypot(np.diff(rr), np.diff(zz))
    distance = np.concatenate([[0.0], np.cumsum(segment)])
    query = np.linspace(0.0, distance[-1], count, endpoint=False)
    return np.interp(query, distance, rr), np.interp(query, distance, zz)


def _limited_field(nr: int, nz: int):
    """Analytic circular flux nested inside a rectangular limiter."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.0, 1.0, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = np.exp(-(((rr - 1.0) ** 2 + zz**2) / 0.3**2))
    lr = np.array([0.55, 1.45, 1.45, 0.55, 0.55])
    lz = np.array([-0.55, -0.55, 0.55, 0.55, -0.55])
    inside = inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    wall_r, wall_z = _dense_rectangle(lr, lz)
    wall_psi = np.exp(-(((wall_r - 1.0) ** 2 + wall_z**2) / 0.3**2))
    return psi, Grid(rg, zg, inside, lr, lz), wall_r, wall_z, wall_psi


def _diverted_field(nr: int, nz: int):
    """Analytic double-blob field with one separatrix saddle."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.2, 1.2, nz)
    rr, zz = np.meshgrid(rg, zg)
    width = 0.28

    def value(r, z):
        upper = np.exp(-(((r - 1.0) ** 2 + (z - 0.25) ** 2) / width**2))
        lower = 0.9 * np.exp(-(((r - 1.0) ** 2 + (z + 0.75) ** 2) / width**2))
        return upper + lower

    psi = value(rr, zz)
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-1.1, -1.1, 1.1, 1.1, -1.1])
    inside = inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    wall_r, wall_z = _dense_rectangle(lr, lz)
    wall_psi = value(wall_r, wall_z)

    def vertical_gradient(z):
        upper = np.exp(-((z - 0.25) ** 2 / width**2))
        lower = 0.9 * np.exp(-((z + 0.75) ** 2 / width**2))
        return -2.0 * ((z - 0.25) * upper + (z + 0.75) * lower) / width**2

    saddle_z = optimize.brentq(vertical_gradient, -0.6, 0.1, xtol=1e-14)
    saddle_psi = float(value(1.0, saddle_z))
    return (
        psi,
        Grid(rg, zg, inside, lr, lz),
        wall_r,
        wall_z,
        wall_psi,
        saddle_z,
        saddle_psi,
    )


def _surface_field(nr: int, nz: int):
    """Solov'ev-like nested ellipses with closed-form volume derivatives."""
    major_radius = 0.9
    minor_radius = 0.55
    elongation = 1.6
    rg = np.linspace(0.2, 1.6, nr)
    zg = np.linspace(-1.1, 1.1, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi_n = ((rr - major_radius) / minor_radius) ** 2 + (
        zz / (minor_radius * elongation)
    ) ** 2
    psi = -psi_n
    inside = np.ones((nz, nr), dtype=bool)
    inside[rr < 0.25] = False
    lr = np.array([0.25, 1.6, 1.6, 0.25, 0.25])
    lz = np.array([-1.1, -1.1, 1.1, 1.1, -1.1])
    return psi, Grid(rg, zg, inside, lr, lz), major_radius, minor_radius, elongation


def _labelled_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Independent four-neighbour component containing ``seed``."""
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labels, _ = ndimage.label(mask, structure=structure)
    seed_label = labels[seed]
    return (labels == seed_label) if seed_label else np.zeros_like(mask, dtype=bool)


def _boundary_arguments(
    nr: int,
    nz: int,
    *,
    n_levels: int = PROFILE_LEVELS,
    n_bisect: int = PROFILE_BISECTIONS,
    n_ray: int = PROFILE_RAYS,
):
    """Device-resident arguments shared by direct and adapter timing."""
    psi, grid, wall_r, wall_z, wall_psi = _limited_field(nr, nz)
    args = (
        jnp.asarray(psi),
        jnp.asarray(grid.rg),
        jnp.asarray(grid.zg),
        jnp.asarray(grid.inside_limiter),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        n_levels,
        n_bisect,
        n_ray,
        jnp.asarray(np.asarray(LCFS_ANGLES)),
        jnp.asarray(0.999),
        jnp.asarray(wall_r),
        jnp.asarray(wall_z),
        jnp.asarray(wall_psi),
    )
    _block(args)
    return psi, grid, wall_psi, args


def _surface_arguments(nr: int, nz: int):
    """Device-resident arguments for the flux-surface metric kernel."""
    psi, grid, major_radius, minor_radius, elongation = _surface_field(nr, nz)
    args = (
        jnp.asarray(psi),
        jnp.asarray(grid.rg),
        jnp.asarray(grid.zg),
        jnp.asarray(grid.inside_limiter),
        jnp.asarray(0.0),
        jnp.asarray(-1.0),
        jnp.asarray(0.04),
        jnp.asarray(0.985),
        SURFACE_BINS,
        jnp.asarray(1.25),
    )
    _block(args)
    return psi, grid, major_radius, minor_radius, elongation, args


def _flood_cost(nr: int, nz: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the device flood and compare its mask with independent labelling."""
    psi, grid, *_ = _surface_field(nr, nz)
    psi_n = -psi
    confined = (psi_n < 0.72) & grid.inside_limiter
    confined[3:7, 3:7] = True
    seed_index = (
        int(np.argmin(np.abs(grid.zg))),
        int(np.argmin(np.abs(grid.rg - 0.9))),
    )
    seed = np.zeros_like(confined)
    seed[seed_index] = True
    reference = _labelled_component(confined, seed_index)
    confined_device = jnp.asarray(confined)
    seed_device = jnp.asarray(seed)
    _block((confined_device, seed_device))

    def call():
        return surface.flood_fill_core(confined_device, seed_device, nr + nz)

    compile_seconds, result = _seconds(call)
    steady_seconds, samples, result = _steady(call)

    def label_call():
        return _labelled_component(confined, seed_index)

    label_seconds, label_samples, _ = _steady(label_call)
    found = np.asarray(result).astype(bool)
    accuracy = {
        "grid": f"{nr}x{nz}",
        "cells": nr * nz,
        "mask_equal": bool(np.array_equal(found, reference)),
        "mismatched_cells": int(np.count_nonzero(found != reference)),
        "disconnected_pocket_excluded": bool(not np.any(found[3:7, 3:7])),
        "reference_core_cells": int(reference.sum()),
        "kernel_core_cells": int(found.sum()),
    }
    cost = {
        "grid": f"{nr}x{nz}",
        "cells": nr * nz,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_ms": 1e3 * steady_seconds,
        "steady_samples_ms": [1e3 * sample for sample in samples],
        "ndimage_label_ms": 1e3 * label_seconds,
        "ndimage_samples_ms": [1e3 * sample for sample in label_samples],
    }
    return cost, accuracy


def _boundary_cost(nr: int, nz: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure direct hard read and its thin host adapter at one grid size."""
    psi, grid, wall_psi, args = _boundary_arguments(nr, nz)

    def direct_call():
        return boundary.traced_boundary_read(*args)

    compile_seconds, direct = _seconds(direct_call)
    steady_seconds, samples, direct = _steady(direct_call)

    def host_call():
        return boundary.host_boundary_read(
            psi,
            grid,
            (1.0, 0.0),
            n_levels=PROFILE_LEVELS,
            n_bisect=PROFILE_BISECTIONS,
            n_ray=PROFILE_RAYS,
            wall_psi=wall_psi,
        )

    host_seconds, host_samples, host = _steady(host_call)

    direct_radii = np.asarray(direct["radii"])
    parity = {
        "grid": f"{nr}x{nz}",
        "host_direct_psi_bnd_abs": abs(host.psi_bnd - float(direct["psi_bnd"])),
        "host_direct_radii_max_abs_m": float(np.max(np.abs(host.radii - direct_radii))),
        "host_direct_core_cell_difference": int(
            host.n_core_cells - int(direct["n_core_cells"])
        ),
        "same_topology_class": bool(host.is_diverted == bool(direct["is_diverted"])),
    }
    cost = {
        "grid": f"{nr}x{nz}",
        "cells": nr * nz,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_ms": 1e3 * steady_seconds,
        "steady_samples_ms": [1e3 * sample for sample in samples],
        "host_adapter_ms": 1e3 * host_seconds,
        "host_adapter_samples_ms": [1e3 * sample for sample in host_samples],
        "host_over_direct": host_seconds / steady_seconds,
    }
    return cost, parity


def _moment_boundary_cost() -> dict[str, Any]:
    """Measure the hard adapter with the current-moment caller's live defaults."""
    nr, nz = 49, 65
    n_levels, n_bisect, n_ray = 96, 18, 512
    psi, grid, wall_psi, args = _boundary_arguments(
        nr,
        nz,
        n_levels=n_levels,
        n_bisect=n_bisect,
        n_ray=n_ray,
    )

    def direct_call():
        return boundary.traced_boundary_read(*args)

    compile_seconds, _ = _seconds(direct_call)
    steady_seconds, samples, _ = _steady(direct_call)

    def host_call():
        return boundary.host_boundary_read(
            psi,
            grid,
            (1.0, 0.0),
            n_levels=n_levels,
            n_bisect=n_bisect,
            n_ray=n_ray,
            wall_psi=wall_psi,
        )

    host_seconds, host_samples, _ = _steady(host_call)
    return {
        "grid": f"{nr}x{nz}",
        "cells": nr * nz,
        "boundary_levels": n_levels,
        "boundary_bisections": n_bisect,
        "boundary_rays": n_ray,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_ms": 1e3 * steady_seconds,
        "steady_samples_ms": [1e3 * sample for sample in samples],
        "host_adapter_ms": 1e3 * host_seconds,
        "host_adapter_samples_ms": [1e3 * sample for sample in host_samples],
        "host_over_direct": host_seconds / steady_seconds,
    }


def _smooth_cost(nr: int = 65, nz: int = 97) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure direct smooth read and the stencil-axis host adapter."""
    psi, grid, wall_psi, args = _boundary_arguments(nr, nz)
    smooth_args = (*args, jnp.asarray(1.0e-3))

    def direct_call():
        return boundary.traced_smooth_boundary_read(*smooth_args)

    compile_seconds, direct = _seconds(direct_call)
    steady_seconds, samples, direct = _steady(direct_call)

    def host_call():
        return boundary.host_boundary_read_smooth(
            psi,
            grid,
            (1.0, 0.0),
            temperature=1.0e-3,
            n_levels=PROFILE_LEVELS,
            n_bisect=PROFILE_BISECTIONS,
            n_ray=PROFILE_RAYS,
            wall_psi=wall_psi,
        )

    host_compile_seconds, host = _seconds(host_call)
    host_seconds, host_samples, host = _steady(host_call)
    difference = {
        "grid": f"{nr}x{nz}",
        "host_direct_psi_bnd_abs": abs(
            float(host["psi_bnd"]) - float(direct["psi_bnd"])
        ),
        "host_direct_axis_shift_m": float(
            np.hypot(float(host["axis_r"]) - 1.0, float(host["axis_z"]))
        ),
        "host_direct_core_weight_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(host["core_weight"]) - np.asarray(direct["core_weight"])
                )
            )
        ),
        "semantic_reason": (
            "the host adapter first refines the stencil magnetic axis; the direct "
            "kernel uses the supplied axis"
        ),
    }
    cost = {
        "grid": f"{nr}x{nz}",
        "cells": nr * nz,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_ms": 1e3 * steady_seconds,
        "steady_samples_ms": [1e3 * sample for sample in samples],
        "host_compile_execute_ms": 1e3 * host_compile_seconds,
        "host_adapter_ms": 1e3 * host_seconds,
        "host_adapter_samples_ms": [1e3 * sample for sample in host_samples],
        "host_over_direct": host_seconds / steady_seconds,
    }
    return cost, difference


def _boundary_batch_cost(nr: int = 65, nz: int = 97) -> dict[str, Any]:
    """Measure one compiled four-slice device batch."""
    _psi, _grid, _wall_psi, args = _boundary_arguments(nr, nz)
    psi_stack = jnp.stack([args[0] * scale for scale in (1.0, 1.01, 0.99, 1.02)])

    def one(psi):
        return boundary.traced_boundary_read(psi, *args[1:])

    batched = jax.jit(jax.vmap(one))

    def call():
        return batched(psi_stack)

    compile_seconds, _ = _seconds(call)
    steady_seconds, samples, _ = _steady(call)
    return {
        "grid": f"{nr}x{nz}",
        "batch_size": BATCH_SIZE,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_batch_ms": 1e3 * steady_seconds,
        "steady_per_slice_ms": 1e3 * steady_seconds / BATCH_SIZE,
        "steady_samples_ms": [1e3 * sample for sample in samples],
    }


def _boundary_accuracy(nr: int = 101, nz: int = 141) -> dict[str, Any]:
    """Arbitrate limited and diverted termination against analytic truth."""
    psi, grid, wall_r, wall_z, wall_psi = _limited_field(nr, nz)
    args = (
        jnp.asarray(psi),
        jnp.asarray(grid.rg),
        jnp.asarray(grid.zg),
        jnp.asarray(grid.inside_limiter),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        PROFILE_LEVELS,
        PROFILE_BISECTIONS,
        PROFILE_RAYS,
        jnp.asarray(np.asarray(LCFS_ANGLES)),
        jnp.asarray(0.999),
        jnp.asarray(wall_r),
        jnp.asarray(wall_z),
        jnp.asarray(wall_psi),
    )
    limited = _block(boundary.traced_boundary_read(*args))
    exact_wall_flux = float(np.max(wall_psi))
    axis_flux = float(limited["psi_axis"])
    span = abs(axis_flux - exact_wall_flux)
    exact_lcfs_flux = axis_flux + 0.999 * (exact_wall_flux - axis_flux)
    exact_radius = math.sqrt(-(0.3**2) * math.log(exact_lcfs_flux))
    radii = np.asarray(limited["radii"])

    psi_out = float(limited["psi_out"])
    normalised = (psi - axis_flux) / (psi_out - axis_flux)
    confined = (normalised <= float(limited["s_star"])) & grid.inside_limiter
    seed = (
        int(np.argmin(np.abs(grid.zg))),
        int(np.argmin(np.abs(grid.rg - 1.0))),
    )
    reference_core = _labelled_component(confined, seed)

    (
        psi_d,
        grid_d,
        wall_r_d,
        wall_z_d,
        wall_psi_d,
        saddle_z,
        saddle_psi,
    ) = _diverted_field(nr, nz)
    diverted = _block(
        boundary.traced_boundary_read(
            jnp.asarray(psi_d),
            jnp.asarray(grid_d.rg),
            jnp.asarray(grid_d.zg),
            jnp.asarray(grid_d.inside_limiter),
            jnp.asarray(1.0),
            jnp.asarray(0.25),
            PROFILE_LEVELS,
            PROFILE_BISECTIONS,
            PROFILE_RAYS,
            jnp.asarray(np.asarray(LCFS_ANGLES)),
            jnp.asarray(0.999),
            jnp.asarray(wall_r_d),
            jnp.asarray(wall_z_d),
            jnp.asarray(wall_psi_d),
        )
    )
    diverted_span = abs(float(diverted["psi_axis"]) - saddle_psi)
    xset = np.asarray(diverted["xset"])
    finite_x = xset[np.isfinite(xset).all(axis=1)]
    saddle_position_error = (
        float(np.min(np.hypot(finite_x[:, 0] - 1.0, finite_x[:, 1] - saddle_z)))
        if finite_x.size
        else float("inf")
    )
    return {
        "grid": f"{nr}x{nz}",
        "limited": {
            "expected_topology": "limited",
            "reported_topology": (
                "diverted" if bool(limited["is_diverted"]) else "limited"
            ),
            "boundary_flux_relative_error": abs(
                float(limited["psi_bnd"]) - exact_wall_flux
            )
            / span,
            "radius_rmse_m": float(np.sqrt(np.mean((radii - exact_radius) ** 2))),
            "radius_max_abs_m": float(np.max(np.abs(radii - exact_radius))),
            "independent_core_cells": int(reference_core.sum()),
            "kernel_core_cells": int(limited["n_core_cells"]),
            "core_cell_difference": int(limited["n_core_cells"])
            - int(reference_core.sum()),
        },
        "diverted": {
            "expected_topology": "diverted",
            "reported_topology": (
                "diverted" if bool(diverted["is_diverted"]) else "limited"
            ),
            "boundary_flux_relative_error": abs(float(diverted["psi_bnd"]) - saddle_psi)
            / diverted_span,
            "saddle_r": 1.0,
            "saddle_z": saddle_z,
            "saddle_position_error_m": saddle_position_error,
            "reported_xpoint_count": int(finite_x.shape[0]),
        },
    }


def _surface_cost(nr: int, nz: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the surface kernel and benchmark-local host materialisation."""
    _psi, _grid, *_truth, args = _surface_arguments(nr, nz)

    def direct_call():
        return surface.traced_flux_surface_bins(*args)

    compile_seconds, direct = _seconds(direct_call)
    steady_seconds, samples, direct = _steady(direct_call)

    def host_call():
        output = surface.traced_flux_surface_bins(*args)
        return {key: np.asarray(value) for key, value in output.items()}

    host_seconds, host_samples, host = _steady(host_call)
    keys = ("pn_s", "dv_dpn", "inv_r2", "inv_r", "grad2_r2", "v_cum")
    parity = {
        "grid": f"{nr}x{nz}",
        "max_abs_by_output": {
            key: float(np.max(np.abs(np.asarray(direct[key]) - host[key])))
            for key in keys
        },
        "same_core_cells": bool(
            int(direct["n_core_cells"]) == int(host["n_core_cells"])
        ),
        "materialisation_semantics": (
            "the benchmark-local host route materialises every fixed-shape output"
        ),
    }
    cost = {
        "grid": f"{nr}x{nz}",
        "cells": nr * nz,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_ms": 1e3 * steady_seconds,
        "steady_samples_ms": [1e3 * sample for sample in samples],
        "host_adapter_ms": 1e3 * host_seconds,
        "host_adapter_samples_ms": [1e3 * sample for sample in host_samples],
        "host_over_direct": host_seconds / steady_seconds,
    }
    return cost, parity


def _surface_batch_cost(nr: int = 65, nz: int = 97) -> dict[str, Any]:
    """Measure a fixed-shape four-slice surface-metric batch."""
    _psi, _grid, *_truth, args = _surface_arguments(nr, nz)
    psi_stack = jnp.stack([args[0] * scale for scale in (1.0, 1.01, 0.99, 1.02)])

    def one(psi):
        return surface.traced_flux_surface_bins(psi, *args[1:])

    batched = jax.jit(jax.vmap(one))

    def call():
        return batched(psi_stack)

    compile_seconds, _ = _seconds(call)
    steady_seconds, samples, _ = _steady(call)
    return {
        "grid": f"{nr}x{nz}",
        "batch_size": BATCH_SIZE,
        "compile_execute_ms": 1e3 * compile_seconds,
        "steady_batch_ms": 1e3 * steady_seconds,
        "steady_per_slice_ms": 1e3 * steady_seconds / BATCH_SIZE,
        "steady_samples_ms": [1e3 * sample for sample in samples],
    }


def _exact_surface_metrics(
    pn_s: np.ndarray, major_radius: float, minor_radius: float, elongation: float
) -> dict[str, np.ndarray]:
    """Exact elliptical-shell metrics, using quadrature only for gradient weight."""
    dv_dpn = np.full_like(
        pn_s, 2.0 * np.pi**2 * major_radius * elongation * minor_radius**2
    )
    inv_r = np.full_like(pn_s, 1.0 / major_radius)
    inv_r2 = 1.0 / (major_radius * np.sqrt(major_radius**2 - minor_radius**2 * pn_s))
    grad2_r2 = []
    for level in pn_s:
        amplitude = minor_radius * np.sqrt(level)

        def integrand(angle):
            radius = major_radius + amplitude * np.cos(angle)
            gradient_squared = (
                4.0
                * level
                / minor_radius**2
                * (np.cos(angle) ** 2 + np.sin(angle) ** 2 / elongation**2)
            )
            return gradient_squared / radius

        numerator = integrate.quad(
            integrand, 0.0, 2.0 * np.pi, epsabs=1e-13, epsrel=1e-13, limit=200
        )[0]
        grad2_r2.append(numerator / (2.0 * np.pi * major_radius))
    return {
        "dv_dpn": dv_dpn,
        "inv_r": inv_r,
        "inv_r2": inv_r2,
        "grad2_r2": np.asarray(grad2_r2),
    }


def _relative_error(found: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Elementwise relative error with a floating-point denominator floor."""
    return np.abs(found - expected) / np.maximum(np.abs(expected), 1e-30)


def _surface_accuracy(nr: int = 101, nz: int = 141) -> dict[str, Any]:
    """Arbitrate surface outputs against exact elliptical-shell geometry."""
    _psi, _grid, major_radius, minor_radius, elongation, args = _surface_arguments(
        nr, nz
    )
    result = _block(surface.traced_flux_surface_bins(*args))
    pn_s = np.asarray(result["pn_s"])
    expected = _exact_surface_metrics(pn_s, major_radius, minor_radius, elongation)
    interior = slice(2, -2)
    errors = {}
    for key, exact in expected.items():
        relative = _relative_error(np.asarray(result[key]), exact)
        errors[key] = {
            "median_relative": float(np.median(relative[interior])),
            "max_relative": float(np.max(relative[interior])),
            "all_levels_max_relative": float(np.max(relative)),
        }

    psi_n = -np.asarray(args[0])
    confined = (psi_n < 1.0) & np.asarray(args[3])
    seed = (
        int(np.argmin(np.abs(np.asarray(args[2])))),
        int(np.argmin(np.abs(np.asarray(args[1]) - major_radius))),
    )
    reference_core = _labelled_component(confined, seed)
    return {
        "grid": f"{nr}x{nz}",
        "analytic_geometry": {
            "major_radius_m": major_radius,
            "minor_radius_m": minor_radius,
            "elongation": elongation,
            "level_range": [float(pn_s[0]), float(pn_s[-1])],
            "interior_levels_compared": int(pn_s[interior].size),
            "relative_errors": errors,
        },
        "connectivity": {
            "independent_core_cells": int(reference_core.sum()),
            "kernel_core_cells": int(result["n_core_cells"]),
            "core_cell_difference": int(result["n_core_cells"])
            - int(reference_core.sum()),
            "core_fraction": float(result["core_fraction"]),
            "well_posed": bool(result["well_posed"]),
        },
    }


def _source_occurrences(symbols: tuple[str, ...]) -> dict[str, list[str]]:
    """Return exact production and test occurrence locations for selected symbols."""
    root = Path(os.environ.get("NOVA_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
    paths = [*root.glob("nova/**/*.py"), *root.glob("tests/**/*.py")]
    result: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    patterns = {symbol: re.compile(rf"\b{re.escape(symbol)}\b") for symbol in symbols}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for symbol, pattern in patterns.items():
                if pattern.search(line):
                    result[symbol].append(f"{relative}:{number}")
    return result


def _environment(label: str) -> dict[str, Any]:
    """Record backend, software, source revision, and invocation."""
    device = jax.devices()[0]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "label": label,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "device_platform": device.platform,
        "device_kind": device.device_kind,
        "device": str(device),
        "git_revision": revision,
        "argv": sys.argv,
    }


def measure(label: str) -> dict[str, Any]:
    """Run one captured backend campaign."""
    flood_cost, flood_accuracy = [], []
    boundary_cost, boundary_parity = [], []
    surface_cost, surface_parity = [], []
    for nr, nz in GRID_SHAPES:
        cost, accuracy = _flood_cost(nr, nz)
        flood_cost.append(cost)
        flood_accuracy.append(accuracy)

        cost, parity = _boundary_cost(nr, nz)
        boundary_cost.append(cost)
        boundary_parity.append(parity)

        cost, parity = _surface_cost(nr, nz)
        surface_cost.append(cost)
        surface_parity.append(parity)

    smooth_cost, smooth_difference = _smooth_cost()
    occurrences = _source_occurrences(
        (
            "traced_boundary_read",
            "traced_smooth_boundary_read",
            "host_boundary_read",
            "host_boundary_read_smooth",
            "host_boundary_read_batch",
            "traced_flux_surface_bins",
            "traced_assemble_flux_surface_geometry",
            "traced_flux_surface_geometry",
            "_traced_profile_shapes",
            "boundary_read_jax",
            "boundary_read_smooth_jax",
            "boundary_read",
            "boundary_read_smooth",
            "boundary_read_batch",
            "flood_fill_core",
            "flux_surface_bins_jax",
            "flux_surface_bins",
            "assemble_flux_surface_geometry_jax",
            "flux_surface_geometry_jax",
            "_profile_shapes_jax",
        )
    )
    return {
        "schema": "nova.connectivity-route-benchmark.1",
        "environment": _environment(label),
        "configuration": {
            "grid_shapes": [f"{nr}x{nz}" for nr, nz in GRID_SHAPES],
            "grid_shape_order": "radial x vertical",
            "boundary_levels": PROFILE_LEVELS,
            "boundary_bisections": PROFILE_BISECTIONS,
            "boundary_rays": PROFILE_RAYS,
            "surface_bins": SURFACE_BINS,
            "batch_size": BATCH_SIZE,
            "repeats": REPEATS,
            "timing_statistic": "minimum synchronous wall time",
            "route_boundary": (
                "direct timings start with device-resident arrays; adapter timings "
                "include conversion, launch, blocking transfer, and result materialisation"
            ),
        },
        "measurements": {
            "flood_fill_core": {"cost": flood_cost, "accuracy": flood_accuracy},
            "connectivity_boundary": {
                "cost": boundary_cost,
                "host_direct_parity": boundary_parity,
                "batch_cost": _boundary_batch_cost(),
                "smooth_cost": smooth_cost,
                "smooth_host_direct_difference": smooth_difference,
                "independent_accuracy": _boundary_accuracy(),
            },
            "flux_surface_connectivity": {
                "cost": surface_cost,
                "host_direct_parity": surface_parity,
                "batch_cost": _surface_batch_cost(),
                "independent_accuracy": _surface_accuracy(),
            },
        },
        "reachability": occurrences,
        "symbol_inventory": [
            {
                "current": "traced_boundary_read",
                "mechanism_name": "traced_boundary_read",
                "reason": "device-resident hard connectivity kernel",
            },
            {
                "current": "traced_smooth_boundary_read",
                "mechanism_name": "traced_smooth_boundary_read",
                "reason": "device-resident differentiable connectivity kernel",
            },
            {
                "current": "host_boundary_read",
                "mechanism_name": "host_boundary_read",
                "reason": "grid preparation, seed validation, and host materialisation",
            },
            {
                "current": "host_boundary_read_smooth",
                "mechanism_name": "host_boundary_read_smooth",
                "reason": "host launch that first refines the stencil axis",
            },
            {
                "current": "host_boundary_read_batch",
                "mechanism_name": "host_boundary_read_batch",
                "reason": "host-prepared vmap launch over ordinary arrays",
            },
            {
                "current": "traced_flux_surface_bins",
                "mechanism_name": "traced_flux_surface_bins",
                "reason": "device-resident fixed-shape surface metric kernel",
            },
            {
                "deleted": "flux_surface_bins",
                "mechanism_name": None,
                "disposition": "delete",
                "reason": (
                    "zero callers outside its defining module; exact parity with the "
                    "traced kernel plus host materialisation overhead"
                ),
            },
            {
                "current": "traced_assemble_flux_surface_geometry",
                "mechanism_name": "traced_assemble_flux_surface_geometry",
                "reason": "public device assembly caller",
            },
            {
                "current": "traced_flux_surface_geometry",
                "mechanism_name": "traced_flux_surface_geometry",
                "reason": "public device composition caller",
            },
            {
                "current": "_traced_profile_shapes",
                "mechanism_name": "_traced_profile_shapes",
                "reason": "internal device profile basis helper",
            },
        ],
        "verdicts": {
            "connectivity_boundary": {
                "verdict": "SINGLE IMPLEMENTATION RELOCATION",
                "basis": (
                    "the host entries invoke the traced implementation; they are adapters, "
                    "not arithmetic peers. Keep hard and smooth kernels because exact "
                    "topology and differentiability are intentionally different semantics"
                ),
                "target": "nova/equilibrium/connectivity_boundary.py",
            },
            "flux_surface_connectivity": {
                "verdict": "DELETE ONE",
                "basis": (
                    "delete the unused flux_surface_bins host adapter: it has zero "
                    "callers outside its defining module, invokes the traced kernel "
                    "exactly, and adds materialisation overhead. Relocate the remaining "
                    "single implementation"
                ),
                "delete": "flux_surface_bins",
                "relocate": ["flood_fill_core", "traced_flux_surface_bins"],
                "target": "nova/equilibrium/flux_surface_connectivity.py",
            },
        },
    }


def measure_moment(label: str) -> dict[str, Any]:
    """Run only the current-moment hard-boundary production regime."""
    return {
        "schema": "nova.connectivity-route-benchmark.1",
        "environment": _environment(label),
        "caller": "nova/equilibrium/moment.py:625",
        "measurement": _moment_boundary_cost(),
    }


def _nice(value: float) -> str:
    """Compact millisecond label for the SVG."""
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    if value >= 10:
        return f"{value:.0f}ms"
    return f"{value:.2f}ms"


def _polyline(
    rows: list[dict[str, Any]],
    key: str,
    left: float,
    top: float,
    width: float,
    height: float,
    colour: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[str, list[tuple[float, float]]]:
    """Return an SVG polyline and its points on logarithmic axes."""
    x_values = np.asarray([row["cells"] for row in rows], dtype=float)
    y_values = np.asarray([row[key] for row in rows], dtype=float)
    x_log = np.log10(x_values)
    y_log = np.log10(y_values)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x_span = max(x_max - x_min, 1.0e-12)
    y_span = max(y_max - y_min, 0.35)
    points = [
        (
            left + width * (float(x) - x_min) / x_span,
            top + height * (1.0 - (float(y) - y_min) / y_span),
        )
        for x, y in zip(x_log, y_log)
    ]
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{colour}" '
        'stroke-width="2.5"/>',
        points,
    )


def _svg_panel(
    report: dict[str, Any],
    module: str,
    left: float,
    top: float,
    title: str,
) -> str:
    """One direct-versus-adapter cost panel."""
    width, height = 350.0, 145.0
    rows = report["measurements"][module]["cost"]
    x_log = np.log10(np.asarray([row["cells"] for row in rows], dtype=float))
    y_log = np.log10(
        np.asarray(
            [
                value
                for row in rows
                for value in (row["steady_ms"], row["host_adapter_ms"])
            ],
            dtype=float,
        )
    )
    x_bounds = (float(np.min(x_log)), float(np.max(x_log)))
    y_bounds = (float(np.min(y_log)), float(np.max(y_log)))
    direct, direct_points = _polyline(
        rows,
        "steady_ms",
        left,
        top + 24,
        width,
        height,
        "#1f77b4",
        x_bounds,
        y_bounds,
    )
    host, host_points = _polyline(
        rows,
        "host_adapter_ms",
        left,
        top + 24,
        width,
        height,
        "#d95f02",
        x_bounds,
        y_bounds,
    )
    elements = [
        f'<text x="{left:.0f}" y="{top:.0f}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{left:.0f}" y="{top + 24:.0f}" width="{width:.0f}" height="{height:.0f}" class="axis"/>',
        direct,
        host,
    ]
    for row, (x, y) in zip(rows, direct_points):
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#1f77b4"/>')
        label_y = min(y + 13.0, top + 24.0 + height - 5.0)
        elements.append(
            f'<text x="{x - 5:.1f}" y="{label_y:.1f}" text-anchor="end" class="tick">'
            f"{_nice(row['steady_ms'])}</text>"
        )
        elements.append(
            f'<text x="{x:.1f}" y="{top + 186:.1f}" text-anchor="middle" class="tick">{row["grid"]}</text>'
        )
    for row, (x, y) in zip(rows, host_points):
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#d95f02"/>')
        elements.append(
            f'<text x="{x + 5:.1f}" y="{y - 5:.1f}" class="value">{_nice(row["host_adapter_ms"])}</text>'
        )
    ratio = [row["compile_execute_ms"] / row["steady_ms"] for row in rows]
    elements.append(
        f'<text x="{left:.0f}" y="{top + 208:.0f}" class="note">compile/steady {min(ratio):.0f}–{max(ratio):.0f}× · '
        '<tspan fill="#1f77b4">direct</tspan> · <tspan fill="#d95f02">host adapter</tspan></text>'
    )
    return "\n".join(elements)


def render_svg(reports: list[dict[str, Any]]) -> str:
    """Render cost curves and decisive module verdicts as standalone SVG."""
    width = 1040
    row_height = 250
    height = 170 + row_height * len(reports) + 170
    panels = []
    for index, report in enumerate(reports):
        y = 105 + index * row_height
        environment = report["environment"]
        device = f"{environment['label']}: {environment['device_kind']}"
        panels.append(
            f'<text x="40" y="{y - 24}" class="backend">{html.escape(device)}</text>'
        )
        panels.append(
            _svg_panel(report, "connectivity_boundary", 70, y, "Boundary read")
        )
        panels.append(
            _svg_panel(
                report,
                "flux_surface_connectivity",
                600,
                y,
                "Flux-surface metrics",
            )
        )
    summary_y = 120 + row_height * len(reports)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
<title id="title">Connectivity route cost and disposition</title>
<desc id="description">Direct device and host adapter cost across three grid sizes on CPU and GPU, followed by module verdicts.</desc>
<style>
  text {{ font-family: system-ui, sans-serif; fill: #17202a; }}
  .title {{ font-size: 23px; font-weight: 700; }}
  .subtitle {{ font-size: 13px; fill: #52606d; }}
  .backend {{ font-size: 15px; font-weight: 650; }}
  .panel-title {{ font-size: 14px; font-weight: 650; }}
  .axis {{ fill: #fbfcfd; stroke: #aab4be; }}
  .tick, .note {{ font-size: 11px; fill: #52606d; }}
  .value {{ font-size: 10px; fill: #7d3c0c; }}
  .verdict {{ font-size: 15px; font-weight: 700; fill: #176b4d; }}
  .summary {{ font-size: 12px; fill: #34495e; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="40" y="38" class="title">Connectivity execution routes</text>
<text x="40" y="62" class="subtitle">Minimum synchronous time; direct inputs are device-resident, adapters include host conversion and materialisation.</text>
{"".join(panels)}
<rect x="40" y="{summary_y}" width="960" height="120" rx="8" fill="#eef8f4" stroke="#7ab89f"/>
<text x="62" y="{summary_y + 30}" class="verdict">connectivity_boundary — SINGLE IMPLEMENTATION RELOCATION</text>
<text x="62" y="{summary_y + 52}" class="summary">Host entries prepare and materialise the traced kernels; hard exact topology and smooth differentiability remain intentional semantic routes.</text>
<text x="62" y="{summary_y + 82}" class="verdict">flux_surface_connectivity — DELETE ONE, THEN RELOCATE</text>
<text x="62" y="{summary_y + 104}" class="summary">Delete unused flux_surface_bins; retain and relocate the fixed-shape kernel and its shared flood primitive.</text>
</svg>'''


def combine(
    inputs: list[Path],
    moment_inputs: list[Path],
    output: Path,
    figure: Path,
) -> None:
    """Combine immutable backend captures and write the review artifacts."""
    reports = [json.loads(path.read_text()) for path in inputs]
    moment_reports = [json.loads(path.read_text()) for path in moment_inputs]
    verdicts = dict(reports[0]["verdicts"])
    verdicts["flux_surface_connectivity"] = {
        "verdict": "DELETE ONE",
        "basis": (
            "delete the unused flux_surface_bins host adapter: it has zero callers "
            "outside its defining module, invokes the traced kernel exactly, and adds "
            "materialisation overhead. Relocate the remaining single implementation"
        ),
        "delete": "flux_surface_bins",
        "relocate": ["flood_fill_core", "traced_flux_surface_bins"],
        "target": "nova/equilibrium/flux_surface_connectivity.py",
    }
    symbol_inventory = [dict(item) for item in reports[0]["symbol_inventory"]]
    for item in symbol_inventory:
        if item.get("deleted") == "flux_surface_bins":
            item.update(
                mechanism_name=None,
                disposition="delete",
                reason=(
                    "zero callers outside its defining module; exact parity with the "
                    "traced kernel plus host materialisation overhead"
                ),
            )
    for report in reports:
        report["verdicts"] = verdicts
        report["symbol_inventory"] = symbol_inventory
    combined = {
        "schema": "nova.connectivity-route-benchmark.1",
        "campaigns": reports,
        "verdicts": verdicts,
        "symbol_inventory": symbol_inventory,
        "reachability": reports[0]["reachability"],
        "reachability_summary": {
            "host_boundary_read": ["nova/equilibrium/moment.py:625"],
            "traced_smooth_boundary_read": ["nova/equilibrium/profile.py:403"],
            "flood_fill_core": ["nova/transport/current_diffusion.py:253"],
            "traced_flux_surface_bins": ["nova/transport/current_diffusion.py:611"],
            "flux_surface_bins": [],
        },
        "regime_provenance": {
            "49x65": "current-moment boundary reconstruction grid",
            "65x97": "flux-surface connectivity reference grid",
            "101x141": "diverted connectivity boundary reference grid",
            "boundary_profile_statics": (
                "48 levels, 12 bisections, and 128 rays from ReconstructProfile"
            ),
            "boundary_moment_statics": (
                "96 levels, 18 bisections, and 512 rays from the hard adapter defaults"
            ),
            "surface_statics": "28 bins from traced_flux_surface_geometry",
        },
        "moment_boundary_campaigns": moment_reports,
        "interpretation": {
            "pair_status": (
                "Neither module is an implementation pair. The boundary route is one "
                "traced implementation plus host adapters; the surface route is one "
                "traced implementation."
            ),
            "boundary_semantics": (
                "Keep the hard kernel as the exact topology reference and the smooth "
                "kernel as the differentiable solve route; they are not candidates for "
                "arithmetic collapse."
            ),
            "adapter_semantics": (
                "Host timings include array conversion, device launch, blocking transfer, "
                "and result materialisation; direct timings start device-resident."
            ),
            "flux_surface_disposition": (
                "Delete the zero-caller host adapter, then relocate the remaining "
                "single traced implementation and shared flood primitive."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    figure.write_text(render_svg(reports))


def main() -> None:
    """Run one backend campaign or combine previously captured campaigns."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument("--output", required=True, type=Path)
    moment_parser = subparsers.add_parser("moment")
    moment_parser.add_argument("--label", required=True)
    moment_parser.add_argument("--output", required=True, type=Path)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--input", action="append", required=True, type=Path)
    combine_parser.add_argument(
        "--moment-input", action="append", required=True, type=Path
    )
    combine_parser.add_argument("--output", required=True, type=Path)
    combine_parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "measure":
        result = measure(args.label)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "moment":
        result = measure_moment(args.label)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        combine(args.input, args.moment_input, args.output, args.figure)


if __name__ == "__main__":
    main()
