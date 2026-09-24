"""Trace the production current-booking stages at the analytic diverted state.

The ``chord`` route first assigns full atomic-cell carriers, then
``_flux_selected_current_moments`` re-clips those carriers with each cell's
local quadratic normalized-flux level before fitting and integrating the
current density.  This evaluation records that sequence without changing it.
"""

from __future__ import annotations

# Precision must be configured before fixture modules build JAX arrays.
# ruff: noqa: E402

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from benchmarks.diverted_chord_response_attribution import physical_moments
from benchmarks.plasma_cell_map_fidelity import norms
from nova.equilibrium.clip_quadrature import (
    _density_coefficients,
    _density_sample_field,
    _quadratic_support,
    _quadratic_support_vertices,
    _sampled_arc_polynomial_moments,
    _flux_selected_current_moments,
)
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.forward_operator import (
    flux_field_polynomial,
    set_support_clip_mode,
)
from nova.equilibrium.source import _FluxSelectedProfile
from scripts.analytic_oracle_fixtures import measure as fixture

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/figures/cut-cell-current-attribution/diverted-chord-response"
MAP_INPUT = ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/chord-booking-trace"
REQUESTED = (110, 300, 500)
CASE = "diverted-single-null"
RELATIVE_TOLERANCE = 1.0e-9
EXTERIOR_REMOVAL_FRACTION = 1.0e-6
NEGATIVE_CONTROL = (
    "apply the analytic in-plasma condition at the named stage and show the "
    "exterior current removed"
)


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_reference(function: object) -> str:
    path = Path(inspect.getsourcefile(function) or "").resolve()
    line = inspect.getsourcelines(function)[1]
    return f"{path.relative_to(ROOT)}:{line}"


def crossing_record(values: np.ndarray) -> list[dict[str, float | list[int]]]:
    """Return cyclic edge crossings of a signed inside-positive level."""
    result = []
    scale = max(float(np.max(np.abs(values))), 1.0)
    tolerance = 256.0 * np.finfo(values.dtype).eps * scale
    for start in range(len(values)):
        stop = (start + 1) % len(values)
        first, second = float(values[start]), float(values[stop])
        if abs(first) <= tolerance:
            fraction = 0.0
        elif first * second < 0.0:
            fraction = abs(first) / (abs(first) + abs(second))
        else:
            continue
        result.append({"edge": [start, stop], "linear_fraction": fraction})
    return result


def trace_indices(base: dict[str, object]) -> tuple[list[int], list[int]]:
    cells = base["per_cell"]
    false_positive = [
        int(row["cell"])
        for row in cells
        if float(row["booked_current_a"]) > 0.0
        and float(row["analytic_current_a"]) == 0.0
    ]
    deficits = sorted(
        (
            (
                float(row["analytic_current_a"]) - float(row["booked_current_a"]),
                int(row["cell"]),
            )
            for row in cells
            if row["class"] == "separatrix-cut"
        ),
        reverse=True,
    )
    return false_positive, [cell for deficit, cell in deficits if deficit > 0.0][:10]


def analytic_condition_moments(
    operator, source, exact, field, confined_support, selected
):
    """Replace the production local-level density condition at the same points."""
    cell_count = len(operator.node)
    cell = jnp.arange(cell_count, dtype=jnp.int32)
    points, psi_norm, _, _, polynomial_centre, coordinate_scale = _density_sample_field(
        field, cell
    )
    density = source.core.current_density(points[..., 0], psi_norm)
    host_points = np.asarray(points)
    boundary_flux = float(exact.flux(np.asarray(exact.x_point)[None, :])[0])
    axis_flux = float(exact.axis_flux)
    polarity = np.sign(axis_flux - boundary_flux)
    analytic_signed_level = polarity * (
        exact.flux(host_points.reshape(-1, 2)).reshape(host_points.shape[:-1])
        - boundary_flux
    )
    analytic_inside = analytic_signed_level >= 0.0
    conditioned_density = jnp.where(jnp.asarray(analytic_inside), density, 0.0)
    coefficients = _density_coefficients(conditioned_density)
    raw = _sampled_arc_polynomial_moments(
        confined_support.support_vertices,
        confined_support.vertex_count,
        polynomial_centre,
        coordinate_scale,
        coefficients,
        confined_support.centroids,
    )
    live = jnp.asarray(selected) & (jnp.asarray(confined_support.vertex_count) > 0)
    moments = type(raw)(*(jnp.where(live, value, 0.0) for value in raw))
    return moments, points, psi_norm, density, analytic_inside, analytic_signed_level


def production_confined_support(profile_support, field, selected):
    coefficient = -jnp.asarray(field.coefficient)
    coefficient = coefficient.at[:, 0].add(1.0)
    clipped = _quadratic_support(
        profile_support.support_vertices,
        profile_support.vertex_count,
        profile_support.centroids,
        coefficient,
        field.centre,
        field.scale,
        selected,
    )
    moving = _quadratic_support_vertices(
        coefficient,
        profile_support.support_vertices,
        profile_support.vertex_count,
        profile_support.centroids,
        field.centre,
        field.scale,
        selected,
    )
    return clipped._replace(support_vertices=moving)


def cell_trace(
    cell: int,
    kind: str,
    base_cell: dict[str, object],
    machine,
    masks,
    field,
    profile_support,
    confined_support,
    sample_points,
    sample_psi_norm,
    production_density,
    analytic_inside,
    analytic_signed_level,
    unscaled_physical,
    booked_physical,
    counter_unscaled,
    counter_booked,
) -> dict[str, object]:
    polygon = np.asarray(machine.cell_polygons[cell])
    cell_index = jnp.asarray([cell], dtype=jnp.int32)
    vertex_value = np.asarray(
        field.sample(jnp.asarray(polygon)[None, ...], cell_index)[0][0]
    )
    production_signed = 1.0 - vertex_value
    exact = np.asarray(analytic_signed_level[cell])
    density = np.asarray(production_density[cell])
    inside = np.asarray(analytic_inside[cell])
    production_inside = np.asarray(sample_psi_norm[cell]) <= 1.0
    label = PlasmaDomain(int(np.asarray(masks.label)[cell])).name.lower()
    full_area = float(np.asarray(profile_support.full_area)[cell])
    clipped_area = float(np.asarray(confined_support.area)[cell])
    stages = [
        {
            "stage": "analytic_reference",
            "current_a": float(base_cell["analytic_current_a"]),
            "moment_written": True,
        },
        {
            "stage": "whole_cell_profile_carrier",
            "current_a": None,
            "moment_written": False,
        },
        {
            "stage": "local_quadratic_pointwise_confinement",
            "current_a": float(unscaled_physical[0, cell]),
            "radial_moment_am": float(unscaled_physical[1, cell]),
            "vertical_moment_am": float(unscaled_physical[2, cell]),
            "moment_written": True,
        },
        {
            "stage": "declared_current_normalisation",
            "current_a": float(booked_physical[0, cell]),
            "radial_moment_am": float(booked_physical[1, cell]),
            "vertical_moment_am": float(booked_physical[2, cell]),
            "moment_written": True,
        },
        {
            "stage": "frozen_coupling_basis_conversion",
            "current_a": float(booked_physical[0, cell]),
            "moment_written": True,
        },
        {
            "stage": "analytic_in_plasma_condition_counterfactual",
            "current_a": float(counter_booked[0, cell]),
            "unscaled_current_a": float(counter_unscaled[0, cell]),
            "moment_written": True,
        },
    ]
    return {
        "cell": cell,
        "trace_kind": kind,
        "base_class": base_cell["class"],
        "carrier": {
            "domain_label": label,
            "profile_participation": bool(
                np.asarray(masks.profile_participation)[cell]
            ),
            "support_included": bool(np.asarray(profile_support.included)[cell]),
            "support_boundary": bool(np.asarray(profile_support.boundary)[cell]),
            "support_vertex_count": int(np.asarray(profile_support.vertex_count)[cell]),
        },
        "production_level": {
            "cell_vertex_psi_norm": vertex_value.tolist(),
            "inside_positive_level": production_signed.tolist(),
            "crossings": crossing_record(production_signed),
            "clipped_vertex_count": int(
                np.asarray(confined_support.vertex_count)[cell]
            ),
            "full_area_m2": full_area,
            "clipped_area_m2": clipped_area,
            "clipped_area_fraction": clipped_area / full_area,
        },
        "analytic_level_at_density_points": {
            "inside_count": int(np.count_nonzero(inside)),
            "point_count": int(inside.size),
            "signed_level_min_wb": float(np.min(exact)),
            "signed_level_max_wb": float(np.max(exact)),
            "production_inside_count": int(np.count_nonzero(production_inside)),
        },
        "density_evaluation": {
            "points_rz_m": np.asarray(sample_points[cell]).tolist(),
            "production_psi_norm": np.asarray(sample_psi_norm[cell]).tolist(),
            "production_density_a_per_m2": density.tolist(),
            "production_nonzero_count": int(np.count_nonzero(density)),
            "conditioned_density_a_per_m2": np.where(inside, density, 0.0).tolist(),
            "conditioned_nonzero_count": int(np.count_nonzero(inside & (density != 0))),
        },
        "base_booked_current_a": float(base_cell["booked_current_a"]),
        "base_analytic_current_a": float(base_cell["analytic_current_a"]),
        "base_deficit_a": float(base_cell["analytic_current_a"])
        - float(base_cell["booked_current_a"]),
        "first_nonzero_production_stage": (
            "local_quadratic_pointwise_confinement"
            if unscaled_physical[0, cell] != 0.0
            else None
        ),
        "deficit_first_visible_stage": "local_quadratic_pointwise_confinement",
        "stages": stages,
    }


def measure(requested: int, output: Path) -> dict[str, object]:
    base_path = INPUT / f"{CASE}-cells-{requested}-chord.json"
    base = json.loads(base_path.read_text())
    archive_path = MAP_INPUT / f"{CASE}-cells-{requested}-chord.npz"
    with np.load(archive_path) as archive:
        archived = dict(archive)

    carrier, source, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier, exact, base["requested_cells"])
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(CASE, exact, coordinates)
    np.testing.assert_array_equal(analytic, archived["analytic"])
    empty = fixture.forward_operator(source, machine)
    _, fixture_external, cache = fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = fixture.forward_operator(source, machine, fixture_external)
    target, _, target_receipt = certificate._closed_form_current_target(
        CASE,
        source,
        operator,
        fixture.exact_current_moments(exact, operator, analytic, analytic=exact),
    )
    set_support_clip_mode("chord")
    partition = operator._support_partition(jnp.asarray(analytic), None)
    masks, _, sample_psi_norm, profile_support = partition
    field = flux_field_polynomial(
        operator._support_moment_stencils, masks.psi_norm, sample_psi_norm
    )
    selected = jnp.asarray(field.active) & (
        jnp.asarray(profile_support.vertex_count) >= 3
    )
    profile = _FluxSelectedProfile(source.core, source.common_sol)
    production_physical = _flux_selected_current_moments(
        profile_support,
        selected,
        field,
        profile,
        cut_cell_capacity=operator._cut_cell_bank_capacity,
    )
    production_coupled = operator.coupling_current_moments(production_physical)
    production_amplitude = operator.current_normalisation_amplitude(
        target, jnp.sum(production_coupled.cell_current)
    )
    production_booked = operator.scaled_current_moments(
        production_coupled, production_amplitude
    )
    confined_support = production_confined_support(profile_support, field, selected)
    (
        counter_physical,
        sample_points,
        density_psi_norm,
        production_density,
        analytic_inside,
        analytic_signed_level,
    ) = analytic_condition_moments(
        operator, source, exact, field, confined_support, selected
    )
    counter_coupled = operator.coupling_current_moments(counter_physical)
    counter_amplitude = operator.current_normalisation_amplitude(
        target, jnp.sum(counter_coupled.cell_current)
    )
    counter_booked = operator.scaled_current_moments(counter_coupled, counter_amplitude)

    production_booked_array = np.asarray(production_booked)
    counter_booked_array = np.asarray(counter_booked)
    production_physical_array = np.asarray(production_physical)
    counter_physical_array = np.asarray(counter_physical)
    archived_booked = float(base["lambda"]) * np.asarray(archived["moments"])
    nonzero = archived_booked[0] != 0.0
    relative = np.abs(
        (production_booked_array[0, nonzero] - archived_booked[0, nonzero])
        / archived_booked[0, nonzero]
    )
    max_relative = float(np.max(relative)) if relative.size else 0.0
    zero_absolute = (
        float(np.max(np.abs(production_booked_array[0, ~nonzero])))
        if np.any(~nonzero)
        else 0.0
    )
    np.testing.assert_allclose(
        production_booked_array[0],
        archived_booked[0],
        rtol=RELATIVE_TOLERANCE,
        atol=1e-12,
    )

    counter_plasma = np.asarray(operator.current_moment_image(counter_booked))
    counter_raw = np.asarray(archived["external"]) + counter_plasma
    counter_map = np.where(np.asarray(archived["shadow"]), analytic, counter_raw)
    counter_mismatch = norms(counter_map - analytic, analytic)
    base_map = norms(np.asarray(archived["mapped"]) - analytic, analytic)
    np.testing.assert_allclose(
        base_map["sup_relative"],
        base["map_mismatch"]["sup_relative"],
        rtol=1e-9,
        atol=0,
    )

    false_positive, deficits = trace_indices(base)
    by_cell = {int(row["cell"]): row for row in base["per_cell"]}
    traces = []
    for kind, cells in (
        ("booked-current-with-zero-analytic-current", false_positive),
        ("largest-separatrix-cut-deficit", deficits),
    ):
        for cell in cells:
            traces.append(
                cell_trace(
                    cell,
                    kind,
                    by_cell[cell],
                    machine,
                    masks,
                    field,
                    profile_support,
                    confined_support,
                    sample_points,
                    density_psi_norm,
                    production_density,
                    analytic_inside,
                    analytic_signed_level,
                    production_physical_array,
                    physical_moments(
                        production_booked_array,
                        np.asarray(machine.moment_geometry.second_moment),
                    ),
                    counter_physical_array,
                    physical_moments(
                        counter_booked_array,
                        np.asarray(machine.moment_geometry.second_moment),
                    ),
                )
            )

    exterior_ratios = {
        str(cell): float(
            abs(counter_booked_array[0, cell]) / abs(archived_booked[0, cell])
        )
        for cell in false_positive
    }
    for ratio in exterior_ratios.values():
        assert ratio < EXTERIOR_REMOVAL_FRACTION
    result = {
        "requested_cells": requested,
        "realised_cells": int(base["cells"]),
        "base_report": str(base_path.relative_to(ROOT)),
        "base_archive": str(archive_path.relative_to(ROOT)),
        "base_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "fixture_cache": cache,
        "current_target_a": float(target),
        "target_receipt": target_receipt,
        "production_amplitude": float(production_amplitude),
        "counterfactual_amplitude": float(counter_amplitude),
        "positive_control": {
            "relative_tolerance": RELATIVE_TOLERANCE,
            "maximum_nonzero_cell_relative_error": max_relative,
            "maximum_archived_zero_cell_absolute_error_a": zero_absolute,
            "passed": True,
        },
        "base_map_mismatch": base_map,
        "analytic_condition_map_mismatch": counter_mismatch,
        "false_positive_cells": false_positive,
        "largest_deficit_cells": deficits,
        "analytic_condition_exterior_current_fraction": exterior_ratios,
        "traces": traces,
    }
    write(output / f"row-{base['cells']}.json", result)
    print(
        f"ROW cells={base['cells']} positive={max_relative:.3e} "
        f"base_map={base_map['sup_relative']:.12g} "
        f"conditioned_map={counter_mismatch['sup_relative']:.12g}",
        flush=True,
    )
    return result


def summarize(report: dict[str, object], output: Path) -> None:
    rows = report["rows"]
    row_550 = next(row for row in rows if row["realised_cells"] == 550)
    cell_64 = next(trace for trace in row_550["traces"] if trace["cell"] == 64)
    report["named_operation"] = (
        "local-quadratic pointwise confinement in _flux_selected_current_moments"
    )
    report["first_current_stage"] = "local_quadratic_pointwise_confinement"
    report["deficit_stage"] = "local_quadratic_pointwise_confinement"
    report["source_reference"] = source_reference(_flux_selected_current_moments)
    report["conclusion"] = (
        "The whole-cell profile carrier is not itself the current-booking error. "
        "The first current moment is written when _flux_selected_current_moments "
        "clips that carrier with the cell-local quadratic psi_N=1 level, samples "
        "the profile density on the same local polynomial, and integrates the "
        "fitted density over that support. The production-local level admits "
        "analytic-exterior cells and under-covers analytic separatrix-cut cells."
    )
    report["cell_64_booked_current_a"] = cell_64["base_booked_current_a"]
    report["cell_64_conditioned_current_a"] = cell_64["stages"][-1]["current_a"]
    report["cell_64_conditioned_fraction"] = row_550[
        "analytic_condition_exterior_current_fraction"
    ]["64"]
    report["completed"] = True
    write(output / "report.json", report)

    lines = [
        "# Chord booking operation trace",
        "",
        report["conclusion"],
        "",
        f"Production source: `{report['source_reference']}`.",
        "",
        "Despite the route name, `chord` supplies a whole-cell carrier. The traced "
        "admission is the subsequent local quadratic level test; no straight chord "
        "segment decides these moments.",
        "",
        "| Realised cells | False-positive cells | Max booked-current reproduction "
        "error | Base map sup | Analytic-condition map sup |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['realised_cells']} | {', '.join(map(str, row['false_positive_cells']))} "
            f"| {row['positive_control']['maximum_nonzero_cell_relative_error']:.3e} "
            f"| {row['base_map_mismatch']['sup_relative']:.9g} "
            f"| {row['analytic_condition_map_mismatch']['sup_relative']:.9g} |"
        )
    lines += [
        "",
        "## Traced moments",
        "",
        "| Cells | Cell | Selection | Carrier | Production crossings | "
        "Production density points inside | Analytic points inside | Analytic A | "
        "Booked A | Conditioned A |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for trace in row["traces"]:
            lines.append(
                f"| {row['realised_cells']} | {trace['cell']} | {trace['trace_kind']} "
                f"| {trace['carrier']['domain_label']} "
                f"| {len(trace['production_level']['crossings'])} "
                f"| {trace['analytic_level_at_density_points']['production_inside_count']} "
                f"| {trace['analytic_level_at_density_points']['inside_count']} "
                f"| {trace['base_analytic_current_a']:.9g} "
                f"| {trace['base_booked_current_a']:.9g} "
                f"| {trace['stages'][-1]['current_a']:.9g} |"
            )
    lines += [
        "",
        "The full 25-point density values, edge-crossing records, support areas, "
        "first moments, and every stage current are retained in `report.json` and "
        "the three row receipts.",
        "",
        "## Declared negative control",
        "",
        NEGATIVE_CONTROL,
        "",
        f"At 550 cells, cell 64 changes from {report['cell_64_booked_current_a']:.12g} A "
        f"to {report['cell_64_conditioned_current_a']:.12g} A, a fraction "
        f"{report['cell_64_conditioned_fraction']:.3e}. The resulting map mismatch "
        "sup is reported above rather than inferred from the removed current.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(json.loads((args.output / "report.json").read_text()), args.output)
        return

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"revision={revision} tree={ROOT} command={sys.argv!r}", flush=True)
    assert jax.default_backend() == "gpu"
    print(f"device={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
    report = {
        "revision": revision,
        "worktree": str(ROOT),
        "job": os.environ.get("SLURM_JOB_ID"),
        "negative_control": NEGATIVE_CONTROL,
        "completed": False,
        "rows": [],
    }
    write(args.output / "report.json", report)
    started = time.monotonic()
    for requested in REQUESTED:
        report["rows"].append(measure(requested, args.output))
        write(args.output / "report.json", report)
    report["wall_seconds"] = time.monotonic() - started
    summarize(report, args.output)
    print("BOOKING_TRACE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
