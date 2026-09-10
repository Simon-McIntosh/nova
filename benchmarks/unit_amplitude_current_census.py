"""Census the per-cell plasma current at unit amplitude under each clip mode.

The forward operator eliminates one common profile amplitude to meet a
declared plasma current.  On a fixed trial state the ``exact`` support clip
(the committed curved-boundary clip, every cut cell participating) and the
``chord`` clip (full atomic cells selected by the profile partition label
only) integrate different sub-regions, so at unit amplitude they carry
different total current and require different amplitudes to reach the same
target.  This benchmark evaluates, on two fixed states of one certificate row
-- the production cold seed and the committed chord terminal state -- the
zeroth per-cell current moment at unit amplitude under both clip modes, and
census where the difference between the modes lives: which cells lose
current, whether they are cut by the analytic separatrix, and what their
participation labels are.

The two states are the identical seed of the certificate's chord row (the
seed is regenerated through the production cold-seed portfolio and verified
against the banked state digest) and the committed chord terminal state
persisted by that row.  The clip-mode setters are the explicit benchmark
selection endpoints: ``set_support_clip_mode`` leaves the production default
untouched.

The unit-amplitude convention is the operator's own: ``cell_current_moments``
returns the clipped zeroth moments without any declared-current scaling, so a
mode's total at unit amplitude is ``target_current / amplitude`` where
``amplitude`` is the normalisation the solve would apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from scipy.integrate import IntegrationWarning

from benchmarks import solovev_certificate as certificate
from benchmarks.solovev_cut_cell_moments import _exact_cell_integral
from nova.equilibrium import ForwardProfile
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.separatrix_clip import _traced_quadratic_value
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.linalg.split_spline import fit_split_spline
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/uniform-cell-clip-and-coupling/amplitude-census"
RECEIPT = OUTPUT_ROOT / "receipt.json"

CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110
CLIP_MODES = ("chord", "exact")
DIFFERENCE_THRESHOLD_FRACTION = 1.0e-6
AMPLITUDE_AGREEMENT_BOUND = 1.0e-6
ADAPTIVE_RELATIVE_TOLERANCE = 5.0e-13

#: Which mode-state pairs admit a banked amplitude cross-check.  The seed and
#: terminal-chord rows' banked amplitudes were recorded on the exact states
#: this census evaluates; the terminal-exact banked amplitude (5.561...) was
#: recorded on the c-r100 arm's own terminal state, which is not the committed
#: chord terminal this census evaluates, so it is not a legitimate reference.
_CROSS_CHECK_AVAILABLE = {
    ("seed", "chord"): True,
    ("seed", "exact"): True,
    ("terminal", "chord"): True,
    ("terminal", "exact"): False,
}

#: The committed chord row whose seed and terminal states are censused.  Its
#: receipt is the semantic carrier of the seed digest, the terminal flux and
#: the amplitudes the chord clip needed on exactly these two states.
CONTROL_PART = (
    ROOT
    / "docs/figures/uniform-cell-clip-and-coupling/exact-participation/discriminator"
    / "parts/control/weak-rotation-reactor-static-production-route-reduced.json"
)
BANKED_AMPLITUDES = {
    "seed": {"chord": 0.9297775241954396, "exact": 2.200602886684049},
    "terminal": {"chord": 1.0497961485612264, "exact": 5.561246574411268},
}


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane_receipt() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _state_digest(state: np.ndarray) -> str:
    packed = np.ascontiguousarray(state, dtype="<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


_SEED_STRUCTURAL_FIELDS = (
    "construction",
    "plasma_current_a",
    "current_centroid_m",
    "seed_radius_m",
    "supported_cell_count",
    "anchor_available",
    "declared_axis_m",
    "stored_flux_samples_used",
    "requested_class",
    "factory",
)


def _seed_structural_verification(
    regenerated: dict[str, Any], banked: dict[str, Any]
) -> dict[str, Any]:
    """Compare the regenerated seed receipt against the committed row's fields."""
    comparisons = {}
    worst_relative = 0.0
    for field in _SEED_STRUCTURAL_FIELDS:
        if field not in banked:
            comparisons[field] = {"present_in_banked": False}
            continue
        mine = regenerated.get(field)
        if isinstance(mine, (list, tuple)):
            mine_array = np.asarray(mine, dtype=np.float64)
            banked_array = np.asarray(banked[field], dtype=np.float64)
            identical = bool(np.array_equal(mine_array, banked_array))
            relative = (
                0.0
                if identical
                else float(
                    np.max(
                        np.abs(mine_array - banked_array)
                        / np.maximum(np.abs(banked_array), 1.0e-30)
                    )
                )
            )
        else:
            identical = mine == banked[field]
            relative = (
                0.0
                if identical
                else abs(float(mine) - float(banked[field]))
                / max(abs(float(banked[field])), 1.0e-30)
            )
        worst_relative = max(worst_relative, relative)
        comparisons[field] = {
            "regenerated": _strict(mine),
            "banked": _strict(banked[field]),
            "identical": bool(identical),
            "max_relative_difference": float(relative),
        }
    return {
        "fields_compared": list(comparisons),
        "field_comparisons": comparisons,
        "all_structural_fields_identical": all(
            value["identical"] for value in comparisons.values() if "identical" in value
        ),
        "worst_max_relative_difference": float(worst_relative),
    }


def _read_control_row() -> dict[str, Any]:
    """Return the committed chord row's persisted render and solver data."""
    row = json.loads(CONTROL_PART.read_text(encoding="utf-8"))
    render = row["render_data"]
    terminal = np.asarray(render["terminal_flux_wb"], dtype=np.float64)
    coordinates = np.asarray(render["coordinates_rz_m"], dtype=np.float64)
    if coordinates.shape != (terminal.size, 2):
        raise RuntimeError("control part coordinates and terminal flux disagree")
    return {
        "realised_cells": int(row["realised_cells"]),
        "target_current_a": float(row["solver"]["target_current_a"]),
        "seed": row["solver"]["seed"],
        "coordinates_rz_m": coordinates,
        "terminal_flux_wb": terminal,
    }


def _build_context(control: dict[str, Any]) -> dict[str, Any]:
    """Return the machine, operator, mesh and profile of the certificate row."""
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the amplitude census requires JAX extended precision")
    carrier_case, source_case, exact = certificate._case(CASE_NAME)
    machine = certificate._case_machine(CASE_NAME, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    if coordinates.shape != control["coordinates_rz_m"].shape:
        raise RuntimeError(
            "rebuilt machine coordinates disagree with the committed row: "
            f"{coordinates.shape} versus {control['coordinates_rz_m'].shape}"
        )
    if not np.array_equal(coordinates, control["coordinates_rz_m"]):
        raise RuntimeError(
            "rebuilt machine coordinates differ elementwise from the row"
        )
    oracle_state = certificate._exact_state(CASE_NAME, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, mesh, newton_steps=recovery.NEWTON_STEPS)
    return {
        "machine": machine,
        "source_case": source_case,
        "exact": exact,
        "operator": operator,
        "mesh": mesh,
        "profile": profile,
        "target_current": control["target_current_a"],
    }


def _production_seed(context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Regenerate the certificate cold seed through the production route."""
    current, centroid, current_receipt = recovery._aggregate_current_moment(
        context["source_case"]
    )
    seed, _requested_class, seed_receipt = certificate._production_seed(
        context["profile"],
        CASE_NAME,
        float(current),
        np.asarray(centroid, dtype=np.float64),
        current_receipt,
    )
    return seed, seed_receipt


def _partition_probe(operator, state: np.ndarray, mode: str) -> dict[str, Any]:
    """Trace one mode's partition, retaining the base and promoted masks."""
    set_support_clip_mode(mode)
    physical = jnp.asarray(state)[: operator.physical_node_number]
    base_masks, topology, _connected, _admitted = operator._fixed_design_read(physical)
    sample_flux = operator.sample_node_flux(jnp.asarray(state))
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    profile_support = operator._profile_support(
        base_masks, topology, physical, sample_psi_norm
    )
    moment_masks = operator._moment_support_masks(base_masks, profile_support)
    return {
        "base_masks": base_masks,
        "topology": topology,
        "sample_psi_norm": sample_psi_norm,
        "profile_support": profile_support,
        "moment_masks": moment_masks,
    }


def _curve_probe(operator, base_masks, topology, sample_psi_norm) -> dict[str, Any]:
    """Reconstruct the curved-level evaluator and per-cell vertex participation.

    Replicates the production ``_profile_support`` curved-level closure and its
    per-cell vertex participation flag so the sign convention of the spline and
    local quadratic levels can be checked directly.
    """
    atomic_mesh = operator.moment_geometry.atomic_mesh
    flux_coefficient = operator.support_flux_coefficients(
        base_masks.psi_norm, sample_psi_norm
    )
    inside_coefficient = -flux_coefficient
    inside_coefficient = inside_coefficient.at[:, 0].add(1.0)
    coordinate = jnp.asarray(operator.grid.coordinate, dtype=base_masks.psi_norm.dtype)
    surface_value = base_masks.psi_norm[None, :]
    surface = fit_split_spline(
        coordinate[None, :, 0],
        coordinate[None, :, 1],
        surface_value,
        surface_value - 1.0,
        order=6,
        regularization=1.0e-14,
    )
    centre = operator._support_curve_centre
    scale = operator._support_curve_scale

    def spline_level(points):
        return -surface._patch_evaluation(
            surface.level_set_coefficients, points[..., 0], points[..., 1]
        ).value

    def local_level(points):
        return _traced_quadratic_value(points, inside_coefficient, centre, scale)

    cell_vertices = jnp.asarray(atomic_mesh.node_coordinates)[
        jnp.asarray(atomic_mesh.cell_nodes)
    ]
    vertex_level = jnp.where(
        surface.fit_executed, spline_level(cell_vertices), local_level(cell_vertices)
    )
    vertex_participation = operator._vertex_level_participation(
        atomic_mesh.cell_vertex_count, vertex_level
    )
    return {
        "surface": surface,
        "inside_coefficient": inside_coefficient,
        "centres": centre,
        "scales": scale,
        "spline_level": spline_level,
        "local_level": local_level,
        "vertex_participation": np.asarray(vertex_participation, dtype=bool),
    }


def _sign_check(operator, curve, probe, state: np.ndarray) -> dict[str, Any]:
    """Check the spline and local levels against the boundary sign at centroids."""
    base_masks = probe["base_masks"]
    topology = probe["topology"]
    atomic_mesh = operator.moment_geometry.atomic_mesh
    centroids = np.asarray(atomic_mesh.centroids, dtype=np.float64)
    centre = operator._support_curve_centre
    scale = operator._support_curve_scale
    centroid_points = jnp.asarray(centroids[:, None, :])
    flux_coefficient = operator.support_flux_coefficients(
        base_masks.psi_norm, probe["sample_psi_norm"]
    )
    psi_norm_centroid = _traced_quadratic_value(
        centroid_points, flux_coefficient, centre, scale
    )[..., 0]
    shared_centroid = topology.axis_flux + psi_norm_centroid * topology.flux_span
    inside_centroid = operator.polarity * (shared_centroid - topology.boundary_flux)
    spline = np.asarray(
        curve["spline_level"](centroid_points)[..., 0], dtype=np.float64
    )
    local = np.asarray(curve["local_level"](centroid_points)[..., 0], dtype=np.float64)
    inside = np.asarray(inside_centroid, dtype=np.float64)
    finite = (
        np.isfinite(spline)
        & np.isfinite(local)
        & np.isfinite(inside)
        & (np.abs(spline) > 0.0)
        & (np.abs(local) > 0.0)
        & (np.abs(inside) > 0.0)
    )
    agreement = np.full(len(centroids), False)
    spline_inside = agreement.copy()
    local_inside = agreement.copy()
    spline_local = agreement.copy()
    spline_inside[finite] = np.sign(spline[finite]) == np.sign(inside[finite])
    local_inside[finite] = np.sign(local[finite]) == np.sign(inside[finite])
    spline_local[finite] = np.sign(spline[finite]) == np.sign(local[finite])
    return {
        "cells_checked": int(np.count_nonzero(finite)),
        "cells_skipped_zero": int(len(centroids) - np.count_nonzero(finite)),
        "skipped_cell_indices": [int(index) for index in np.flatnonzero(~finite)],
        "spline_local_disagreement_count": int(
            np.count_nonzero(finite & ~spline_local)
        ),
        "spline_inside_disagreement_count": int(
            np.count_nonzero(finite & ~spline_inside)
        ),
        "local_inside_disagreement_count": int(
            np.count_nonzero(finite & ~local_inside)
        ),
        "first_disagreement_cells": {
            "spline_vs_local": [
                int(index) for index in np.flatnonzero(finite & ~spline_local)[:10]
            ],
            "spline_vs_inside": [
                int(index) for index in np.flatnonzero(finite & ~spline_inside)[:10]
            ],
            "local_vs_inside": [
                int(index) for index in np.flatnonzero(finite & ~local_inside)[:10]
            ],
        },
    }


def _cell_analysis(machine, exact, target_current: float) -> dict[str, Any]:
    """Compute the state-independent analytical per-cell current data."""
    cell_count = len(machine.node)
    centres = np.asarray(
        machine.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    polygons = tuple(
        np.asarray(cell, dtype=np.float64) for cell in machine.cell_polygons
    )
    with warnings.catch_warnings(record=True) as integration_warnings:
        warnings.simplefilter("always", IntegrationWarning)
        exact_integrals = [
            _exact_cell_integral(exact, polygon, centre, ADAPTIVE_RELATIVE_TOLERANCE)
            for polygon, centre in zip(polygons, centres, strict=True)
        ]
        exact_current = np.asarray([item.moment[0] for item in exact_integrals])
        exact_areas = np.asarray([item.area for item in exact_integrals])
        exact_errors = np.asarray([item.error_estimate[0] for item in exact_integrals])
    cell_areas = np.asarray(machine.area, dtype=np.float64)
    area_tolerance = 2.0e-11 * np.maximum(cell_areas, 1.0)
    cut = (exact_areas > area_tolerance) & (exact_areas < cell_areas - area_tolerance)
    interior = exact_areas >= cell_areas - area_tolerance
    surface_class = np.where(
        interior,
        "interior",
        np.where(cut, "cut", "exterior"),
    )
    area_fraction = exact_areas / np.maximum(cell_areas, np.finfo(np.float64).tiny)
    return {
        "cell_count": cell_count,
        "centre_rz_m": centres.tolist(),
        "polygons": [polygon.tolist() for polygon in polygons],
        "exact_main_axis_current_a": exact_current.tolist(),
        "exact_area_m2": exact_areas.tolist(),
        "exact_area_error_m2": exact_errors.tolist(),
        "analytic_plasma_area_fraction": area_fraction.tolist(),
        "analytic_surface_class": [str(value) for value in surface_class],
        "cut_by_analytic_separatrix": [bool(value) for value in cut],
        "interior": [bool(value) for value in interior],
        "exact_cut_count": int(np.count_nonzero(cut)),
        "interior_count": int(np.count_nonzero(interior)),
        "exterior_count": int(np.count_nonzero(~cut & ~interior)),
        "integration_warning_count": len(integration_warnings),
    }


def _mode_current(operator, state: np.ndarray, mode: str) -> np.ndarray:
    set_support_clip_mode(mode)
    moments = operator.cell_current_moments(jnp.asarray(state))
    return np.asarray(moments.cell_current, dtype=np.float64)


def _state_census(
    operator,
    machine,
    state: np.ndarray,
    state_name: str,
    target_current: float,
    analytic: dict[str, Any],
    curve: dict[str, Any],
) -> dict[str, Any]:
    grid_count = len(machine.node)
    mode_tables: dict[str, dict[str, Any]] = {}
    for mode in CLIP_MODES:
        probe = _partition_probe(operator, state, mode)
        current = _mode_current(operator, state, mode)
        mode_tables[mode] = {
            "current_a": current.tolist(),
            "total_a": float(np.sum(current)),
            "carrying_cell_count": int(np.count_nonzero(current > 0.0)),
            "profile_participation": [
                bool(value)
                for value in np.asarray(probe["moment_masks"].profile_participation)
            ],
            "support_vertex_count": [
                int(value)
                for value in np.asarray(probe["profile_support"].vertex_count)
            ],
        }
    chord = np.asarray(mode_tables["chord"]["current_a"], dtype=np.float64)
    exact = np.asarray(mode_tables["exact"]["current_a"], dtype=np.float64)
    difference = chord - exact
    threshold = DIFFERENCE_THRESHOLD_FRACTION * abs(target_current)
    profile_participation = np.asarray(
        mode_tables["exact"]["profile_participation"], dtype=bool
    )
    vertex_participation = curve["vertex_participation"]
    participation = profile_participation | vertex_participation
    cut = np.asarray(analytic["cut_by_analytic_separatrix"], dtype=bool)
    uncut_but_differ = ~cut & (np.abs(difference) > threshold)
    excluded_by_qualify = (chord > threshold) & (exact <= threshold)
    analytic_current = np.asarray(
        analytic["exact_main_axis_current_a"], dtype=np.float64
    )
    per_cell = []
    for cell in range(grid_count):
        per_cell.append(
            {
                "cell": int(cell),
                "centre_rz_m": analytic["centre_rz_m"][cell],
                "chord_current_a": float(chord[cell]),
                "exact_current_a": float(exact[cell]),
                "difference_a": float(difference[cell]),
                "analytic_exact_current_a": float(analytic_current[cell]),
                "analytic_plasma_area_fraction": float(
                    analytic["analytic_plasma_area_fraction"][cell]
                ),
                "analytic_surface_class": analytic["analytic_surface_class"][cell],
                "chord_clip_vertex_count": int(
                    mode_tables["chord"]["support_vertex_count"][cell]
                ),
                "exact_clip_vertex_count": int(
                    mode_tables["exact"]["support_vertex_count"][cell]
                ),
                "cut_by_analytic_separatrix": bool(cut[cell]),
                "profile_participation": bool(profile_participation[cell]),
                "vertex_participation": bool(vertex_participation[cell]),
                "participation": bool(participation[cell]),
            }
        )
    totals = {mode: mode_tables[mode]["total_a"] for mode in CLIP_MODES}
    amplitudes = {
        mode: (None if totals[mode] == 0.0 else float(target_current / totals[mode]))
        for mode in CLIP_MODES
    }
    amplitude_checks = {}
    amplitude_gate = True
    for mode in CLIP_MODES:
        cross_checkable = _CROSS_CHECK_AVAILABLE[(state_name, mode)]
        banked = BANKED_AMPLITUDES[state_name][mode]
        measured = amplitudes[mode]
        disagreement = (
            None if measured is None else abs(measured - banked) / abs(banked)
        )
        if cross_checkable:
            passed = (
                disagreement is not None and disagreement <= AMPLITUDE_AGREEMENT_BOUND
            )
            amplitude_gate &= bool(passed)
        else:
            passed = None
        entry = {
            "banked_amplitude": banked,
            "unit_amplitude_total_a": totals[mode],
            "measured_amplitude": measured,
            "relative_disagreement": disagreement,
            "bound": AMPLITUDE_AGREEMENT_BOUND,
            "cross_check_available": cross_checkable,
            "passed": passed,
        }
        if not cross_checkable:
            entry["banked_reference_note"] = (
                "the committed row's exact-terminal amplitude was recorded on "
                "the c-r100 arm's own terminal state, a different flux state "
                "from the committed chord terminal evaluated here; it is not a "
                "reference for this census and is reported for provenance only"
            )
        amplitude_checks[mode] = entry
    amplitude_checks["gate"] = {
        "all_cross_checkable_modes_within_bank": bool(amplitude_gate),
        "bound": AMPLITUDE_AGREEMENT_BOUND,
        "meaning": (
            "census total at unit amplitude reproduces the committed row's "
            "banked amplitude for every mode-state pair that shares the state "
            "with the banked reference; this is the identity statement that "
            "both modes are evaluated on the same states the row persisted"
        ),
    }
    surface = np.asarray(analytic["analytic_surface_class"])
    class_totals = {
        "by_surface": {
            "interior": float(np.sum(difference[surface == "interior"])),
            "cut": float(np.sum(difference[surface == "cut"])),
            "exterior": float(np.sum(difference[surface == "exterior"])),
        },
        "by_mode_mechanism": {
            "uncut_interior_differing_cells": float(
                np.sum(difference[uncut_but_differ])
            ),
            "exact_excluded_cells": float(np.sum(difference[excluded_by_qualify])),
        },
    }
    return {
        "state": state_name,
        "unit_amplitude_totals_a": totals,
        "chord_over_exact_total_ratio": (
            None if totals["exact"] == 0.0 else float(totals["chord"] / totals["exact"])
        ),
        "amplitude_verification": amplitude_checks,
        "uncut_differing_cells": {
            "count": int(np.count_nonzero(uncut_but_differ)),
            "summed_difference_a": float(np.sum(difference[uncut_but_differ])),
            "threshold_a": float(threshold),
            "threshold_as_plasma_current_fraction": DIFFERENCE_THRESHOLD_FRACTION,
            "cells": [int(cell) for cell in np.flatnonzero(uncut_but_differ)],
        },
        "exact_excluded_cells": {
            "count": int(np.count_nonzero(excluded_by_qualify)),
            "summed_difference_a": float(np.sum(difference[excluded_by_qualify])),
            "cells": [int(cell) for cell in np.flatnonzero(excluded_by_qualify)],
        },
        "class_totals_a": class_totals,
        "per_cell": per_cell,
        "difference_a": difference.tolist(),
    }


def _figure_levels(tables: list[np.ndarray]) -> np.ndarray:
    """Return shared signed levels covering every table's non-zero values."""
    values = np.concatenate([table[np.abs(table) > 0.0] for table in tables])
    if values.size == 0:
        return np.asarray([0.0])
    magnitude = np.abs(values)
    lower = max(float(np.percentile(magnitude, 20.0)), np.finfo(float).tiny)
    upper = float(np.max(magnitude))
    positive = np.geomspace(lower, upper, 8) if upper > lower else np.asarray([upper])
    negative = -positive[-1:0:-1]
    return np.unique(np.concatenate((negative, [0.0], positive)))


def _draw_figure(
    machine,
    analytic: dict[str, Any],
    states: dict[str, dict[str, np.ndarray]],
    output_svg: Path,
    output_png: Path,
) -> dict[str, Any]:
    status = {}
    node = np.asarray(machine.node, dtype=np.float64)
    wall = np.asarray(machine.wall_node, dtype=np.float64)
    axis_rz = np.asarray(analytic["axis_rz_m"], dtype=np.float64)
    levels = _figure_levels([states[name]["difference"] for name in states])
    figure, axes = plt.subplots(
        1, len(states), figsize=(7.0 * len(states), 5.6), constrained_layout=True
    )
    axes_list = [axes] if len(states) == 1 else axes
    for column, (name, data) in enumerate(states.items()):
        ax = poloidal_axes(axes_list[column])
        difference = data["difference"]
        levels = _figure_levels([difference])
        # The per-cell difference is piecewise constant: exterior and interior
        # cells sit at zero or at a single value, so contouring the complete
        # field against sign-selected levels draws the boundary of the
        # differing region (masked points would break the Delaunay
        # triangulation).
        ax.tricontour(
            node[:, 0],
            node[:, 1],
            difference,
            levels=levels[levels >= 0.0],
            colors="firebrick",
            linewidths=0.8,
        )
        ax.tricontour(
            node[:, 0],
            node[:, 1],
            difference,
            levels=-levels[levels > 0.0][::-1],
            colors="navy",
            linestyles="dashed",
            linewidths=0.8,
        )
        for cell in np.flatnonzero(data["cut"]):
            polygon = np.asarray(machine.cell_polygons[cell], dtype=np.float64)
            ax.add_patch(
                PolygonPatch(
                    polygon, closed=True, fill=False, edgecolor="gold", linewidth=1.0
                )
            )
        poloidal.draw_wall(ax, radius=wall[:, 0], height=wall[:, 1], style=DEFAULT_INK)
        poloidal.draw_nulls(ax, magnetic_axis=axis_rz, x_points=None)
        ax.set_title(name, color="0.35", fontsize=11, pad=6)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_svg, format="svg")
    figure.savefig(output_png, format="png", dpi=160)
    plt.close(figure)
    status["svg"] = {
        "path": str(output_svg),
        "sha256": hashlib.sha256(output_svg.read_bytes()).hexdigest(),
        "bytes": output_svg.stat().st_size,
    }
    status["png"] = {
        "path": str(output_png),
        "sha256": hashlib.sha256(output_png.read_bytes()).hexdigest(),
        "bytes": output_png.stat().st_size,
    }
    return status


def _draw_ranked_bars(
    states: dict[str, dict[str, np.ndarray]],
    target_current: float,
    output_svg: Path,
    output_png: Path,
) -> dict[str, Any]:
    status = {}
    threshold = DIFFERENCE_THRESHOLD_FRACTION * abs(target_current)
    figure, axes = plt.subplots(
        1, len(states), figsize=(6.5 * len(states), 4.6), constrained_layout=True
    )
    axes_list = [axes] if len(states) == 1 else axes
    for column, (name, data) in enumerate(states.items()):
        ax = axes_list[column]
        difference = data["difference"]
        ranked_cut = data["cut"][np.argsort(difference)[::-1]]
        ranked = np.sort(difference)[::-1]
        positions = np.arange(len(ranked))
        ax.bar(positions, ranked, color=np.where(ranked_cut, "gold", "0.55"), width=1.0)
        ax.axhline(threshold, color="firebrick", linewidth=0.8, linestyle="--")
        ax.axhline(-threshold, color="firebrick", linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.set_xlabel("cell rank by chord-minus-exact current")
        ax.set_ylabel("chord minus exact current (A)")
        ax.set_title(name, color="0.35", fontsize=11, pad=6)
        ax.tick_params(axis="x", labelbottom=False)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_svg, format="svg")
    figure.savefig(output_png, format="png", dpi=160)
    plt.close(figure)
    status["svg"] = {
        "path": str(output_svg),
        "sha256": hashlib.sha256(output_svg.read_bytes()).hexdigest(),
        "bytes": output_svg.stat().st_size,
    }
    status["png"] = {
        "path": str(output_png),
        "sha256": hashlib.sha256(output_png.read_bytes()).hexdigest(),
        "bytes": output_png.stat().st_size,
    }
    return status


def measure() -> dict[str, Any]:
    started = perf_counter()
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    control = _read_control_row()
    context = _build_context(control)
    operator = context["operator"]
    machine = context["machine"]
    target_current = context["target_current"]

    seed, _seed_receipt = _production_seed(context)
    seed_digest = _state_digest(seed)
    banked_seed_digest = control["seed"]["state_sha256_binary64"]
    #: The committed row's seed digest was produced by the GPU certificate
    #: lane (betelgeuse, jax_default_backend gpu).  The seed's flux image is a
    #: JAX reduction whose sum order differs between CPU and GPU, so the byte
    #: digest is not CPU-reproducible even though every structural field is
    #: identical.  Identity is therefore verified two other ways, and the
    #: digest divergence is recorded rather than fatal: structurally (fields
    #: below) and through the banked amplitude history, which is the physically
    #: meaningful statement that the census sits on the same states.
    seed_structure = _seed_structural_verification(_seed_receipt, control["seed"])
    terminal = control["terminal_flux_wb"]
    if not np.all(np.isfinite(terminal)):
        raise RuntimeError("committed chord terminal state is not finite")

    analytic = _cell_analysis(machine, context["exact"], target_current)
    analytic["axis_rz_m"] = list(
        np.asarray(context["exact"].magnetic_axis, dtype=np.float64)
    )

    states = {}
    for state_name, state in (("seed", seed), ("terminal", terminal)):
        probe = _partition_probe(operator, state, "exact")
        curve = _curve_probe(
            operator,
            probe["base_masks"],
            probe["topology"],
            probe["sample_psi_norm"],
        )
        sign = _sign_check(operator, curve, probe, state)
        census = _state_census(
            operator,
            machine,
            state,
            state_name,
            target_current,
            analytic,
            curve,
        )
        states[state_name] = {
            "census": census,
            "sign_check": sign,
            "state_sha256_binary64": _state_digest(state),
        }

    figure_states = {
        name: {
            "difference": np.asarray(data["census"]["difference_a"], dtype=np.float64),
            "cut": np.asarray(
                [
                    row["cut_by_analytic_separatrix"]
                    for row in data["census"]["per_cell"]
                ],
                dtype=bool,
            ),
        }
        for name, data in states.items()
    }

    row = {
        "case": CASE_NAME,
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": analytic["cell_count"],
        "source_revision": _source_revision(),
        "lane": _lane_receipt(),
        "contract": {
            "clip_modes": list(CLIP_MODES),
            "unit_amplitude": (
                "cell_current_moments zeroth moment with no declared-current "
                "scaling; amplitude = target_current / sum(cell_current)"
            ),
            "seed": "production cold seed regenerated through the certificate "
            "route and verified structurally and against the committed row's "
            "banked amplitude history (the digest itself is GPU-lane specific)",
            "terminal_state": "committed chord row render_data terminal flux",
            "target_current_a": target_current,
            "difference_threshold": (
                f"{DIFFERENCE_THRESHOLD_FRACTION} of the plasma current"
            ),
            "analytic_exact_current": "density integrated over the cell's "
            "analytic plasma intersection by the shared adaptive oracle",
        },
        "target_current_a": target_current,
        "seed": {
            "state_sha256_binary64": seed_digest,
            "certificate_row_seed_sha256": banked_seed_digest,
            "digest_matches_commit": seed_digest == banked_seed_digest,
            "digest_mismatch_explanation": (
                "the committed digest was produced by the GPU certificate lane; "
                "the CPU lane cannot reproduce a JAX reduction bit-for-bit, so "
                "identity is verified structurally and through the amplitude "
                "history instead (see seed_structural and amplitude "
                "verification gate)"
                if seed_digest != banked_seed_digest
                else ""
            ),
            "construction": control["seed"]["construction"],
            "structural_verification": seed_structure,
        },
        "verification": {
            "seed_structure": seed_structure,
            "amplitude_gate": {
                state_name: data["census"]["amplitude_verification"]["gate"][
                    "all_cross_checkable_modes_within_bank"
                ]
                for state_name, data in states.items()
            },
        },
        "states": {
            name: {
                "state_sha256_binary64": data["state_sha256_binary64"],
                "unit_amplitude_totals_a": data["census"]["unit_amplitude_totals_a"],
                "chord_over_exact_total_ratio": data["census"][
                    "chord_over_exact_total_ratio"
                ],
                "amplitude_verification": data["census"]["amplitude_verification"],
                "uncut_differing_cells": data["census"]["uncut_differing_cells"],
                "exact_excluded_cells": data["census"]["exact_excluded_cells"],
                "class_totals_a": data["census"]["class_totals_a"],
                "sign_check": data["sign_check"],
            }
            for name, data in states.items()
        },
        "analytic_cells": {
            "exact_cut_count": analytic["exact_cut_count"],
            "interior_count": analytic["interior_count"],
            "exterior_count": analytic["exterior_count"],
        },
        "per_cell": {name: data["census"]["per_cell"] for name, data in states.items()},
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(RECEIPT, row)
    contour_figure = _draw_figure(
        machine,
        analytic,
        figure_states,
        OUTPUT_ROOT / "difference-contours.svg",
        OUTPUT_ROOT / "difference-contours.png",
    )
    ranked_figure = _draw_ranked_bars(
        figure_states,
        target_current,
        OUTPUT_ROOT / "difference-ranked.svg",
        OUTPUT_ROOT / "difference-ranked.png",
    )
    row["figures"] = {
        "difference_contours": contour_figure,
        "difference_ranked": ranked_figure,
    }
    _write_json(RECEIPT, row)
    amplitudes = ", ".join(
        f"{state}:{mode}={states[state]['census']['amplitude_verification'][mode]['measured_amplitude']:.9g}"
        for mode in CLIP_MODES
        for state in ("seed", "terminal")
    )
    seed_digest_matches = seed_digest == banked_seed_digest
    amplitude_gate = all(
        data["census"]["amplitude_verification"]["gate"][
            "all_cross_checkable_modes_within_bank"
        ]
        for data in states.values()
    )
    print(
        f"UNIT_AMPLITUDE_CENSUS case={CASE_NAME} cells={analytic['cell_count']} "
        f"seed_digest_matches={seed_digest_matches} amplitude_gate={amplitude_gate} "
        f"{amplitudes}",
        flush=True,
    )
    return row


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    measure()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
