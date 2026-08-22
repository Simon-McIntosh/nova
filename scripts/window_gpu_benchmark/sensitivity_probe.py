"""Isolate discrete CPU--GPU choices in one coupled-window equilibrium sweep."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORMS"] = "cuda,cpu"

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import evaluate_forward_equilibrium
from nova.equilibrium.flux_surface_extraction import (
    extract_flux_surface_geometry,
)
from nova.transport.coupled_window import Waveform, equilibrium_sweep, transport_sweep
from nova.transport.evolved_state import forward_source_from_receipt


TSV_FIELDS = (
    "layer",
    "sample",
    "level",
    "cell",
    "selection",
    "quantity",
    "cpu",
    "gpu",
    "absolute_difference",
    "relative_difference",
    "ulp_scale",
    "status",
)
EXTREMUM_NAMES = ("r_in", "r_out", "z_lower", "z_upper")
WINDOW_SECONDS = 2.5e-3
SOURCE_MULTIPLIER = 0.5
ITERATION_CAP = 10
WINDOW_TOLERANCE = 5.0e-3
DAMPING = 0.5
CPU_CONTRACTION = 0.53710396334179378
GPU_CONTRACTION = 0.64062864260291186


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return str(int(array))
    return f"{float(array):.17g}"


def _append(
    rows: list[dict[str, str]],
    layer: str,
    quantity: str,
    cpu: Any,
    gpu: Any,
    *,
    sample: int | str = "",
    level: Any = "",
    cell: int | str = "",
    selection: str = "",
    absolute_difference: Any = "",
    relative_difference: Any = "",
    ulp_scale: Any = "",
    status: str = "RECORDED",
) -> None:
    rows.append(
        {
            "layer": layer,
            "sample": str(sample),
            "level": _format(level),
            "cell": str(cell),
            "selection": selection,
            "quantity": quantity,
            "cpu": _format(cpu),
            "gpu": _format(gpu),
            "absolute_difference": _format(absolute_difference),
            "relative_difference": _format(relative_difference),
            "ulp_scale": _format(ulp_scale),
            "status": status,
        }
    )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _host_tree(tree: Any) -> Any:
    ready = jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )
    return jax.tree.map(
        lambda value: np.asarray(value) if hasattr(value, "dtype") else value,
        ready,
    )


def _fingerprint(named_arrays: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    inventory = []
    for name in sorted(named_arrays):
        value = np.ascontiguousarray(np.asarray(named_arrays[name]))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.view(np.uint8))
        inventory.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": str(value.shape),
                "sha256": hashlib.sha256(value.view(np.uint8)).hexdigest(),
            }
        )
    return digest.hexdigest(), inventory


def _array_comparison(cpu: Any, gpu: Any) -> dict[str, Any]:
    left = np.asarray(cpu)
    right = np.asarray(gpu)
    if left.shape != right.shape:
        return {
            "bitwise": False,
            "maximum_absolute": np.inf,
            "maximum_relative": np.inf,
            "maximum_ulp": np.inf,
            "index": (),
        }
    if np.issubdtype(left.dtype, np.bool_) or np.issubdtype(left.dtype, np.integer):
        different = left != right
        location = tuple(np.argwhere(different)[0]) if np.any(different) else ()
        return {
            "bitwise": bool(np.array_equal(left, right)),
            "maximum_absolute": int(np.max(np.abs(left.astype(np.int64) - right))),
            "maximum_relative": 0.0,
            "maximum_ulp": 0.0,
            "index": location,
        }
    difference = np.abs(left - right)
    flat_index = int(np.argmax(difference)) if difference.size else 0
    location = np.unravel_index(flat_index, difference.shape) if difference.size else ()
    scale = np.maximum(
        np.maximum(np.abs(left), np.abs(right)), np.finfo(left.dtype).tiny
    )
    relative = difference / scale
    spacing = np.maximum(np.abs(np.spacing(left)), np.abs(np.spacing(right)))
    ulp = difference / np.maximum(spacing, np.finfo(left.dtype).tiny)
    return {
        "bitwise": bool(np.array_equal(left, right)),
        "maximum_absolute": float(np.max(difference, initial=0.0)),
        "maximum_relative": float(np.max(relative, initial=0.0)),
        "maximum_ulp": float(np.max(ulp, initial=0.0)),
        "index": tuple(int(index) for index in location),
    }


def _source_arrays(waveform: Waveform, initial_flux: Any) -> dict[str, Any]:
    return {
        "initial_flux": initial_flux,
        "source.time": waveform.time,
        "source.radial_grid": waveform.radial_grid,
        "source.phi_boundary": waveform.phi_boundary,
        "source.axis_reference": waveform.axis_reference,
        "source.boundary_reference": waveform.boundary_reference,
        **{f"source.{name}": value for name, value in waveform.values.items()},
    }


def _iteration_input(demonstration, cpu_device):
    with jax.default_device(cpu_device):
        profile, seed, _vacuum = demonstration._fixture_machine()
        baseline_equilibrium = profile.solve(
            seed, route="anderson", evaluations=demonstration.EVALUATIONS
        )
        jax.block_until_ready(baseline_equilibrium)
        lattice = demonstration._extraction_lattice(profile)
        sources = demonstration._fixture_sources(profile)
        baseline_geometry, extraction = demonstration._geometry_from_equilibrium(
            baseline_equilibrium, profile.source, lattice, sources
        )
        time_grid = np.asarray((0.0, WINDOW_SECONDS), dtype=np.float64)
        initial_geometry = Waveform.from_geometries(
            time_grid, (baseline_geometry, baseline_geometry)
        )
        initial_state = demonstration._initial_state(baseline_geometry, profile.source)
        plasma_current = np.full(
            time_grid.shape, float(baseline_equilibrium.moments.plasma_current)
        )
        transport = transport_sweep(
            initial_geometry,
            initial_state,
            time_grid,
            plasma_current,
            demonstration._torax_model(WINDOW_SECONDS),
        )
        evolved_sources = [profile.source]
        for interval, receipt in enumerate(transport.receipts):
            geometry_time = float(transport.geometry_time[interval])
            evolved_sources.append(
                forward_source_from_receipt(
                    receipt,
                    initial_geometry.sample(geometry_time).geometry(),
                    ion_density_per_electron=demonstration.ION_DENSITY_PER_ELECTRON,
                )
            )
        source_waveform = demonstration._scaled_source_waveform(
            initial_geometry,
            time_grid,
            profile.source,
            evolved_sources,
            SOURCE_MULTIPLIER,
        )
    initial_flux = np.asarray(baseline_equilibrium.flux)
    snapshot = _source_arrays(source_waveform, initial_flux)
    snapshot.update(
        {
            "transport.rho": initial_state.rho,
            "transport.psi": initial_state.psi,
            "transport.ion_temperature": initial_state.ion_temperature,
            "transport.electron_temperature": initial_state.electron_temperature,
            "transport.electron_density": initial_state.electron_density,
        }
    )
    return source_waveform, initial_flux, snapshot, extraction


def _extraction_input(demonstration, equilibrium, source, profile):
    lattice = demonstration._extraction_lattice(profile)
    sources = demonstration._fixture_sources(profile)
    psi_map = evaluate_forward_equilibrium(equilibrium, lattice, sources)
    mesh_radius, mesh_height = np.meshgrid(
        lattice.radius, lattice.height, indexing="xy"
    )
    _wall, wall_flux = demonstration.forward_fixture._wall_loop()
    inside_limiter = jnp.asarray(
        demonstration.forward_fixture._solovev(mesh_radius, mesh_height) >= wall_flux
    )
    axis_psi = float(equilibrium.topology.axis_flux)
    boundary_psi = float(equilibrium.topology.boundary_flux)
    flux_span = boundary_psi - axis_psi
    field_psi_n, field_function = demonstration._field_function(source, flux_span)
    major_radius = float(equilibrium.topology.axis[0])
    record, diagnostics = extract_flux_surface_geometry(
        psi_map,
        jnp.asarray(lattice.radius),
        jnp.asarray(lattice.height),
        inside_limiter,
        axis_psi=jnp.asarray(axis_psi),
        boundary_psi=jnp.asarray(boundary_psi),
        profile_coefficients=jnp.zeros(2, dtype=jnp.float64),
        coefficient_scale=jnp.ones(2, dtype=jnp.float64),
        ip_amperes=jnp.asarray(equilibrium.moments.plasma_current),
        major_radius=jnp.asarray(major_radius),
        boundary_toroidal_field=jnp.asarray(
            source.boundary_field_function / major_radius
        ),
        field_function_psi_n=jnp.asarray(field_psi_n),
        field_function=jnp.asarray(field_function),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=demonstration.RADIAL_CELLS,
        n_surface_bins=demonstration.SURFACE_BINS,
        return_diagnostics=True,
    )
    return psi_map, record, diagnostics


def _backend_sweep(demonstration, device, source_waveform, initial_flux):
    started = time.perf_counter()
    with jax.default_device(device):
        profile, _seed, _vacuum = demonstration._fixture_machine()
        receipt = equilibrium_sweep(
            profile,
            jnp.asarray(initial_flux),
            source_waveform,
            source_waveform.time,
            demonstration._source_from_sample,
            route="anderson",
            solve_options={
                "evaluations": demonstration.EVALUATIONS,
                "tolerance": demonstration.EQUILIBRIUM_SOLVE_TOLERANCE,
            },
        )
        samples = []
        for source_sample, equilibrium in zip(
            receipt.source_samples, receipt.equilibria, strict=True
        ):
            source = demonstration._source_from_sample(source_sample)
            psi_map, record, diagnostics = _extraction_input(
                demonstration, equilibrium, source, profile
            )
            samples.append(
                {
                    "equilibrium_flux": np.asarray(equilibrium.flux),
                    "evaluated_flux": psi_map,
                    "fixed_point_residual": equilibrium.fixed_point.residual,
                    "axis_flux": equilibrium.topology.axis_flux,
                    "boundary_flux": equilibrium.topology.boundary_flux,
                    "axis_radius": equilibrium.topology.axis[0],
                    "axis_height": equilibrium.topology.axis[1],
                    "core_cells": jnp.sum(equilibrium.domains.core),
                    "record": record,
                    "diagnostics": diagnostics,
                }
            )
        ready = _host_tree(samples)
    branches = [
        demonstration._branch_measurement(branch, 1)
        for branch in receipt.branch_receipts
    ]
    return ready, branches, time.perf_counter() - started


def _comparison_row(
    rows: list[dict[str, str]],
    layer: str,
    quantity: str,
    cpu: Any,
    gpu: Any,
    *,
    sample: int,
) -> dict[str, Any]:
    comparison = _array_comparison(cpu, gpu)
    _append(
        rows,
        layer,
        quantity,
        comparison["maximum_absolute"],
        comparison["maximum_absolute"],
        sample=sample,
        selection=str(comparison["index"]),
        absolute_difference=comparison["maximum_absolute"],
        relative_difference=comparison["maximum_relative"],
        ulp_scale=comparison["maximum_ulp"],
        status="BITWISE_EQUAL" if comparison["bitwise"] else "DIFF",
    )
    return comparison


def _band_membership(diagnostics: Mapping[str, Any], prefix: str, level_index: int):
    indices = diagnostics[f"{prefix}_band_indices"][level_index]
    valid = diagnostics[f"{prefix}_band_valid"][level_index]
    return {int(cell) for cell in indices[valid]}


def _cell_label(flat_cell: int, grid_shape: Any) -> str:
    _rows, columns = (int(value) for value in np.asarray(grid_shape))
    row, column = divmod(flat_cell, columns)
    return f"{flat_cell} ({row},{column})"


def _discrete_rows(
    rows: list[dict[str, str]],
    sample: int,
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    cpu_branch: Mapping[str, Any],
    gpu_branch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for field in (
        "limited_core_cells",
        "diverted_core_cells",
        "selected_class",
        "previous_class",
        "reason",
        "limited_available",
        "diverted_available",
    ):
        left = cpu_branch[field]
        right = gpu_branch[field]
        differs = left != right
        _append(
            rows,
            "branch_selection",
            field,
            left,
            right,
            sample=sample,
            status="DIFF" if differs else "MATCH",
        )
        if differs:
            differences.append(
                {
                    "layer": "branch_selection",
                    "quantity": field,
                    "cpu": left,
                    "gpu": right,
                }
            )

    for prefix in ("cumulative", "surface"):
        levels = cpu[f"{prefix}_level"]
        for index, level in enumerate(levels):
            left_population = int(cpu[f"{prefix}_topology_population"][index])
            right_population = int(gpu[f"{prefix}_topology_population"][index])
            _append(
                rows,
                "topology_population",
                prefix,
                left_population,
                right_population,
                sample=sample,
                level=level,
                status="DIFF" if left_population != right_population else "MATCH",
            )
            if left_population != right_population:
                differences.append(
                    {
                        "layer": "topology_population",
                        "quantity": prefix,
                        "level": float(level),
                        "cpu": left_population,
                        "gpu": right_population,
                    }
                )
            left_topology = np.asarray(cpu[f"{prefix}_participation"][index])
            right_topology = np.asarray(gpu[f"{prefix}_participation"][index])
            for cell in np.flatnonzero(left_topology != right_topology):
                cell = int(cell)
                cell_label = _cell_label(cell, cpu["cell_grid_shape"])
                _append(
                    rows,
                    "topology_membership",
                    prefix,
                    bool(left_topology[cell]),
                    bool(right_topology[cell]),
                    sample=sample,
                    level=level,
                    cell=cell_label,
                    status="DISCRETE_FLIP",
                )
                differences.append(
                    {
                        "layer": "topology_membership",
                        "quantity": prefix,
                        "level": float(level),
                        "cell": cell_label,
                        "cpu": bool(left_topology[cell]),
                        "gpu": bool(right_topology[cell]),
                    }
                )
            left_count = int(cpu[f"{prefix}_band_count"][index])
            right_count = int(gpu[f"{prefix}_band_count"][index])
            _append(
                rows,
                "band_count",
                prefix,
                left_count,
                right_count,
                sample=sample,
                level=level,
                status="DIFF" if left_count != right_count else "MATCH",
            )
            left_members = _band_membership(cpu, prefix, index)
            right_members = _band_membership(gpu, prefix, index)
            for cell in sorted(left_members ^ right_members):
                cell_label = _cell_label(cell, cpu["cell_grid_shape"])
                corners_cpu = np.asarray(cpu["cell_corner_flux"])[cell]
                corners_gpu = np.asarray(gpu["cell_corner_flux"])[cell]
                margin_cpu = min(
                    float(level - np.min(corners_cpu)),
                    float(np.max(corners_cpu) - level),
                )
                margin_gpu = min(
                    float(level - np.min(corners_gpu)),
                    float(np.max(corners_gpu) - level),
                )
                _append(
                    rows,
                    "band_membership",
                    prefix,
                    cell in left_members,
                    cell in right_members,
                    sample=sample,
                    level=level,
                    cell=cell_label,
                    absolute_difference=abs(margin_gpu - margin_cpu),
                    selection=f"bracket_margin_cpu={margin_cpu:.17g};gpu={margin_gpu:.17g}",
                    status="DISCRETE_FLIP",
                )
                differences.append(
                    {
                        "layer": "band_membership",
                        "quantity": prefix,
                        "level": float(level),
                        "cell": cell_label,
                        "cpu": cell in left_members,
                        "gpu": cell in right_members,
                        "margin_cpu": margin_cpu,
                        "margin_gpu": margin_gpu,
                    }
                )

    levels = cpu["surface_level"]
    for level_index, level in enumerate(levels):
        for field in ("clip_vertex_count", "clip_boundary"):
            left = {
                int(cell): value
                for cell, value, valid in zip(
                    cpu["surface_band_indices"][level_index],
                    cpu[f"surface_{field}"][level_index],
                    cpu["surface_band_valid"][level_index],
                    strict=True,
                )
                if valid
            }
            right = {
                int(cell): value
                for cell, value, valid in zip(
                    gpu["surface_band_indices"][level_index],
                    gpu[f"surface_{field}"][level_index],
                    gpu["surface_band_valid"][level_index],
                    strict=True,
                )
                if valid
            }
            for cell in sorted(left.keys() & right.keys()):
                if left[cell] != right[cell]:
                    _append(
                        rows,
                        field,
                        "surface",
                        left[cell],
                        right[cell],
                        sample=sample,
                        level=level,
                        cell=cell,
                        status="DISCRETE_FLIP",
                    )
                    differences.append(
                        {
                            "layer": field,
                            "quantity": "surface",
                            "level": float(level),
                            "cell": cell,
                            "cpu": int(left[cell]),
                            "gpu": int(right[cell]),
                        }
                    )
        for selection_index, selection in enumerate(EXTREMUM_NAMES):
            left_cell = int(cpu["surface_extremum_cell"][level_index, selection_index])
            right_cell = int(gpu["surface_extremum_cell"][level_index, selection_index])
            left_choice = int(
                cpu["surface_extremum_selection"][level_index, selection_index]
            )
            right_choice = int(
                gpu["surface_extremum_selection"][level_index, selection_index]
            )
            differs = (left_cell, left_choice) != (right_cell, right_choice)
            _append(
                rows,
                "extremum_selection",
                selection,
                left_choice,
                right_choice,
                sample=sample,
                level=level,
                cell=f"{left_cell}/{right_cell}",
                status="DISCRETE_FLIP" if differs else "MATCH",
            )
            if differs:
                differences.append(
                    {
                        "layer": "extremum_selection",
                        "quantity": selection,
                        "level": float(level),
                        "cell": f"{left_cell}/{right_cell}",
                        "cpu": left_choice,
                        "gpu": right_choice,
                    }
                )
    return differences


def _report(
    *,
    job_id: str,
    snapshot_hash: str,
    cpu_samples: Sequence[Mapping[str, Any]],
    gpu_samples: Sequence[Mapping[str, Any]],
    discrete: Sequence[Mapping[str, Any]],
    solve_comparisons: Sequence[Mapping[str, Any]],
    phi_comparisons: Sequence[Mapping[str, Any]],
    cpu_seconds: float,
    gpu_seconds: float,
) -> str:
    first_discrete = discrete[0] if discrete else None
    lines = [
        "# CPU--H200 single-sweep sensitivity isolation",
        "",
        (
            f"SLURM job `{job_id}` replayed one host-serialized iteration-1 input "
            f"(`sha256:{snapshot_hash}`) through one equilibrium sweep and exact "
            "dense-lattice extraction on CPU and NVIDIA H200 in float64."
        ),
        "",
        "## Solved flux maps first",
        "",
        "| sample | map | bitwise | max absolute | max relative | "
        "max ulp scale | location |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for comparison in solve_comparisons:
        lines.append(
            f"| {comparison['sample']} | `{comparison['name']}` | "
            f"`{comparison['bitwise']}` | "
            f"`{_format(comparison['maximum_absolute'])}` | "
            f"`{_format(comparison['maximum_relative'])}` | "
            f"`{_format(comparison['maximum_ulp'])}` | `{comparison['index']}` |"
        )
    lines.extend(["", "## First discrete decision", ""])
    if first_discrete is None:
        lines.append(
            "No topology, band, clip-vertex, branch, or extremum selection changed. "
            "The amplification is continuous and no tie-breaking repair is indicated."
        )
    else:
        lines.append(
            f"The first discrete difference is `{first_discrete['layer']}` / "
            f"`{first_discrete['quantity']}` at level "
            f"`{_format(first_discrete.get('level'))}`, cell "
            f"`{first_discrete.get('cell', 'n/a')}`: CPU "
            f"`{first_discrete['cpu']}`, H200 `{first_discrete['gpu']}`."
        )
        if "margin_cpu" in first_discrete:
            lines.append(
                "The cell's bracket margins are CPU "
                f"`{_format(first_discrete['margin_cpu'])}` and H200 "
                f"`{_format(first_discrete['margin_gpu'])}`. A sign change at "
                "code-generation noise scale is a genuine tie: the repair is an "
                "order-independent tie rule, not a wider tolerance."
            )
    lines.extend(
        [
            "",
            "## Downstream toroidal-flux amplification",
            "",
            "| sample | CPU Phi_b | H200 Phi_b | absolute difference | "
            "relative difference |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for sample, (cpu, gpu) in enumerate(zip(cpu_samples, gpu_samples, strict=True)):
        left = float(cpu["record"]["phi_b"])
        right = float(gpu["record"]["phi_b"])
        lines.append(
            f"| {sample} | `{_format(left)}` | `{_format(right)}` | "
            f"`{_format(abs(right - left))}` | "
            f"`{_format(abs(right - left) / max(abs(left), abs(right)))}` |"
        )
    lines.extend(
        [
            "",
            "| sample | Phi path layer | bitwise | max absolute | max relative | "
            "location |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for comparison in phi_comparisons:
        lines.append(
            f"| {comparison['sample']} | `{comparison['name']}` | "
            f"`{comparison['bitwise']}` | "
            f"`{_format(comparison['maximum_absolute'])}` | "
            f"`{_format(comparison['maximum_relative'])}` | "
            f"`{comparison['index']}` |"
        )
    changed_phi = next(
        (comparison for comparison in phi_comparisons if not comparison["bitwise"]),
        None,
    )
    if first_discrete is not None and changed_phi is not None:
        lines.extend(
            [
                "",
                (
                    f"The discrete `{first_discrete['layer']}` choice changes the "
                    f"first non-bitwise Phi-path quantity "
                    f"`{changed_phi['name']}` by up to "
                    f"`{_format(changed_phi['maximum_absolute'])}`; the tabulated "
                    "successive surface weights and integrand values show its "
                    "amplification into the final Phi_b difference."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "The machine-readable TSV carries every per-level topology population, "
                "band count, membership flip, clip-vertex flip, extremum selection, "
                "surface integrand comparison and the cell/level payload needed to "
                "trace the first changed decision into Phi_b."
            ),
            "",
            "## Runtime and window-policy context",
            "",
            f"CPU sweep plus extraction: `{_format(cpu_seconds)}` s; H200: "
            f"`{_format(gpu_seconds)}` s.",
            (
                f"The landed contraction evidence remains CPU `{CPU_CONTRACTION}` "
                f"versus H200 `{GPU_CONTRACTION}` at cap `{ITERATION_CAP}`, tolerance "
                f"`{WINDOW_TOLERANCE}` and damping `{DAMPING}`. Whether the cap should "
                "be contraction-aware is a design question; this probe does not change "
                "the cap or tolerance."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()

    gpu_devices = jax.devices("gpu")
    cpu_devices = jax.devices("cpu")
    if not gpu_devices or not cpu_devices or jax.default_backend() != "gpu":
        raise RuntimeError("the sensitivity probe requires one CUDA and one CPU device")
    gpu_device = gpu_devices[0]
    cpu_device = cpu_devices[0]

    from scripts.window_demonstration import run_window as demonstration

    source_waveform, initial_flux, snapshot, baseline_extraction = _iteration_input(
        demonstration, cpu_device
    )
    snapshot_hash, inventory = _fingerprint(snapshot)
    cpu_samples, cpu_branches, cpu_seconds = _backend_sweep(
        demonstration, cpu_device, source_waveform, initial_flux
    )
    gpu_samples, gpu_branches, gpu_seconds = _backend_sweep(
        demonstration, gpu_device, source_waveform, initial_flux
    )

    rows: list[dict[str, str]] = []
    _append(rows, "metadata", "slurm_job_id", os.environ.get("SLURM_JOB_ID", ""), "")
    _append(rows, "metadata", "snapshot_sha256", snapshot_hash, snapshot_hash)
    _append(
        rows,
        "metadata",
        "baseline_extraction_valid",
        baseline_extraction["record_valid"],
        baseline_extraction["record_valid"],
    )
    _append(rows, "metadata", "cpu_seconds", cpu_seconds, cpu_seconds)
    _append(rows, "metadata", "gpu_seconds", gpu_seconds, gpu_seconds)
    _append(rows, "metadata", "cpu_contraction", CPU_CONTRACTION, CPU_CONTRACTION)
    _append(rows, "metadata", "gpu_contraction", GPU_CONTRACTION, GPU_CONTRACTION)
    for item in inventory:
        _append(
            rows,
            "input_snapshot",
            item["name"],
            item["sha256"],
            item["sha256"],
            selection=f"dtype={item['dtype']};shape={item['shape']}",
            status="IDENTICAL_INPUT",
        )

    solve_comparisons = []
    phi_comparisons = []
    discrete = []
    for sample, (cpu, gpu, cpu_branch, gpu_branch) in enumerate(
        zip(cpu_samples, gpu_samples, cpu_branches, gpu_branches, strict=True)
    ):
        for name in (
            "equilibrium_flux",
            "evaluated_flux",
            "fixed_point_residual",
            "axis_flux",
            "boundary_flux",
            "axis_radius",
            "axis_height",
            "core_cells",
        ):
            comparison = _comparison_row(
                rows,
                "solved_flux_map",
                name,
                cpu[name],
                gpu[name],
                sample=sample,
            )
            solve_comparisons.append({"sample": sample, "name": name, **comparison})
        cpu_diagnostics = cpu["diagnostics"]
        gpu_diagnostics = gpu["diagnostics"]
        discrete.extend(
            _discrete_rows(
                rows,
                sample,
                cpu_diagnostics,
                gpu_diagnostics,
                cpu_branch,
                gpu_branch,
            )
        )
        for field in (
            "volume_derivative",
            "inverse_radius_squared",
            "field_function_surface",
            "phi_integrand",
            "phi_integrand_edge",
            "phi_boundary",
        ):
            comparison = _comparison_row(
                rows,
                "phi_path",
                field,
                cpu_diagnostics[field],
                gpu_diagnostics[field],
                sample=sample,
            )
            phi_comparisons.append({"sample": sample, "name": field, **comparison})
        for level_index, level in enumerate(cpu_diagnostics["surface_level"]):
            for field in (
                "volume_derivative",
                "inverse_radius_squared",
                "field_function_surface",
                "phi_integrand",
            ):
                left = float(cpu_diagnostics[field][level_index])
                right = float(gpu_diagnostics[field][level_index])
                _append(
                    rows,
                    "phi_integrand_level",
                    field,
                    left,
                    right,
                    sample=sample,
                    level=level,
                    absolute_difference=abs(right - left),
                    relative_difference=abs(right - left)
                    / max(abs(left), abs(right), np.finfo(np.float64).tiny),
                    status="MATCH" if left == right else "DIFF",
                )

    _write_tsv(arguments.results, rows)
    arguments.report.write_text(
        _report(
            job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
            snapshot_hash=snapshot_hash,
            cpu_samples=cpu_samples,
            gpu_samples=gpu_samples,
            discrete=discrete,
            solve_comparisons=solve_comparisons,
            phi_comparisons=phi_comparisons,
            cpu_seconds=cpu_seconds,
            gpu_seconds=gpu_seconds,
        ),
        encoding="utf-8",
    )
    first = discrete[0] if discrete else None
    print(f"snapshot_sha256={snapshot_hash}")
    print(f"cpu_seconds={cpu_seconds:.17g}")
    print(f"gpu_seconds={gpu_seconds:.17g}")
    print(f"discrete_difference_count={len(discrete)}")
    print(f"first_discrete={first}")
    for sample, (cpu, gpu) in enumerate(zip(cpu_samples, gpu_samples, strict=True)):
        print(
            f"phi_boundary[{sample}]=cpu:{float(cpu['record']['phi_b']):.17g},"
            f"gpu:{float(gpu['record']['phi_b']):.17g}"
        )
    print(f"report={arguments.report}")
    print(f"results={arguments.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
