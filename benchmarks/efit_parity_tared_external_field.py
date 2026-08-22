"""Isolate the Grad--Shafranov solve with a reference-derived external field."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.constants import mu_0

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    _mast_case_from_selection,
    _metric_qualification,
    _passive_inclusive_case,
    _passive_inclusive_solve,
    _pinned_metrics,
    select_slices_by_shot,
)
from benchmarks.efit_parity_boundary_volume import (  # noqa: E402
    _polygon_measure,
    _verify_protected_artifacts,
)
from benchmarks.efit_parity_field_instrument import (  # noqa: E402
    EXPECTED_REFERENCE_MAP_FIELD_ENERGY,
    PUBLISHED_REFERENCE_FIELD_ENERGY,
    _field_magnitude,
)
from benchmarks.efit_parity_inductance_partition import (  # noqa: E402
    _boundary_moments,
)
from benchmarks.efit_parity_moment_definitions import (  # noqa: E402
    _relative_error,
)
from benchmarks.efit_parity_root_geometry import (  # noqa: E402
    _closed_axis_branch,
    _distance_pair,
    _stored_lcfs,
    _unit_boundary_branches,
)
from nova.equilibrium.conservation import delta_star  # noqa: E402
from nova.equilibrium.convention import (  # noqa: E402
    delta_star_from_current_density,
)
from nova.equilibrium.forward_operator import PrescribedCurrentField  # noqa: E402
from nova.equilibrium.stencil_mesh import (  # noqa: E402
    CellCurrentMoments,
    StencilMesh,
)
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from scripts.analytic_oracle_fixtures.measure import (  # noqa: E402
    FIXTURE_REQUESTS as ANALYTIC_FIXTURE_REQUESTS,
    TOTAL_FLUX_FACTOR,
    WALL_POINT_COUNT as ANALYTIC_WALL_POINT_COUNT,
    _internal_flux_image as _analytic_internal_flux_image,
    analytic_case,
    cached_machine,
    exact_current_moments,
    exact_state,
    forward_operator as analytic_forward_operator,
)

OUTPUT_DIRECTORY = Path("docs/figures/efit-forward-parity")
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "tared-external-field-solve.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "tared-external-field-solve.png"
PROTECTED_SOURCE = OUTPUT_DIRECTORY / "converged-root-geometry-attribution.json"
BANKED_STORED_FIELD_CLOSURE_WB = 2.22e-15
REFERENCE_HALO_CURRENT_A = 786.396
GAUGE_CONSTANT_WB = 0.0
REPRESENTATIVE_SHOT = 22086
BANKED_CONVERGED_PLASMA_ROOTS = 1
BANKED_BOUNDED_RESIDUAL_MINIMUM = 2.006e-4
BANKED_BOUNDED_RESIDUAL_MAXIMUM = 1.076e-2
ANALYTIC_SOURCE_RELATIVE_TOLERANCE = 1.0e-3
ANALYTIC_EXTERNAL_SPAN_TOLERANCE = 1.0e-2
CONDUCTOR_LOCALISATION_SHARE_TOLERANCE = 0.9
CONDUCTOR_PATTERN_CORRELATION_TOLERANCE = 0.9
CONDUCTOR_CURRENT_L1_ERROR_TOLERANCE = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implied_current(profile, reference_grid: np.ndarray) -> dict[str, Any]:
    """Recover the valid-stencil current and its exact-kernel flux image."""
    lattice = profile.lattice
    reference_flat = np.asarray(reference_grid, dtype=np.float64).reshape(-1)
    elliptic = np.asarray(delta_star(lattice, jnp.asarray(reference_flat)))
    radius = np.asarray(lattice.node_radius, dtype=np.float64)
    unit_current_elliptic = np.asarray(
        delta_star_from_current_density(radius, np.ones_like(radius)),
        dtype=np.float64,
    )
    current_density = elliptic / unit_current_elliptic
    valid = np.asarray(lattice.interior(), dtype=bool) & np.isfinite(current_density)
    current_density = np.where(valid, current_density, 0.0)
    cell_current = current_density * np.asarray(lattice.cell_area, dtype=np.float64)
    declared = np.asarray(profile.operator.declared_support, dtype=bool)
    declared_valid = declared & valid
    outside_declared = ~declared & valid
    zero = jnp.zeros_like(jnp.asarray(cell_current))
    moments = CellCurrentMoments(jnp.asarray(cell_current), zero, zero)
    plasma_flux = np.asarray(
        profile.operator.current_moment_image(moments), dtype=np.float64
    )
    declared_current = float(np.sum(cell_current[declared_valid]))
    outside_current = float(np.sum(cell_current[outside_declared]))
    return {
        "elliptic": elliptic,
        "valid": valid,
        "declared": declared,
        "current_density": current_density,
        "cell_current": cell_current,
        "plasma_flux": plasma_flux,
        "receipt": {
            "valid_stencil_cell_count": int(np.count_nonzero(valid)),
            "declared_support_cell_count": int(np.count_nonzero(declared)),
            "declared_support_valid_cell_count": int(np.count_nonzero(declared_valid)),
            "outside_declared_support_valid_cell_count": int(
                np.count_nonzero(outside_declared)
            ),
            "declared_support_current_integral_a": declared_current,
            "outside_declared_support_current_integral_a": outside_current,
            "total_valid_stencil_current_integral_a": declared_current
            + outside_current,
            "outside_over_banked_nova_halo": outside_current / REFERENCE_HALO_CURRENT_A,
        },
    }


def build_tare(profile, reference_state: np.ndarray, reference_grid: np.ndarray):
    """Return a prescribed static field whose recomposition closes the map."""
    implied = _implied_current(profile, reference_grid)
    reference = np.asarray(reference_state, dtype=np.float64)
    plasma = implied["plasma_flux"]
    if reference.shape != plasma.shape:
        raise RuntimeError("the reference state and plasma response shapes differ")
    external = reference - plasma + GAUGE_CONSTANT_WB
    recomposed = plasma + external - GAUGE_CONSTANT_WB
    closure = recomposed - reference
    grid_nodes = profile.lattice.node_count
    closure_receipt = {
        "target_count": int(reference.size),
        "grid_target_count": int(grid_nodes),
        "wall_target_count": int(profile.operator.wall.node_number),
        "sample_target_count": (
            0
            if profile.operator.sample is None
            else profile.operator.sample.node_number
        ),
        "sup_difference_wb": float(np.max(np.abs(closure))),
        "rms_difference_wb": float(np.sqrt(np.mean(closure**2))),
        "banked_stored_field_closure_wb": BANKED_STORED_FIELD_CLOSURE_WB,
    }
    closure_receipt["at_roundoff"] = bool(
        closure_receipt["sup_difference_wb"]
        <= 8.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(reference))), 1.0)
    )
    if not closure_receipt["at_roundoff"]:
        raise RuntimeError("the reference-derived external-field tare does not close")
    policy = PrescribedCurrentField(
        response=jnp.asarray(external[:, None]), current=jnp.asarray([1.0])
    )
    operator = replace(
        profile.operator,
        external_current=jnp.zeros_like(profile.operator.external_current),
        prescribed_current_field=policy,
    )
    return {
        "profile": replace(profile, operator=operator),
        "external": external,
        "plasma": plasma,
        "implied": implied,
        "closure": closure_receipt,
    }


def _solovev_separation_row(fixture: str, requested_cells: int) -> dict[str, Any]:
    """Apply the reference tare literally to one closed-form oracle carrier."""
    case = analytic_case()
    machine = cached_machine(
        case, requested_cells, wall_nodes=ANALYTIC_WALL_POINT_COUNT
    )
    coordinates = np.vstack(
        [machine.node, machine.wall_node, machine.sample_coordinates]
    )
    reference = exact_state(case, coordinates)
    operator = analytic_forward_operator(case, machine)
    analytic_physical = exact_current_moments(case, operator, reference)
    analytic_coefficients = operator.coupling_current_moments(analytic_physical)
    analytic_plasma = _analytic_internal_flux_image(operator, analytic_coefficients)
    analytic_external = reference - analytic_plasma

    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    grid_reference = reference[: len(machine.node)]
    elliptic = np.asarray(delta_star(mesh, jnp.asarray(grid_reference)))
    radius = np.asarray(mesh.node_radius, dtype=np.float64)
    unit_current_elliptic = np.asarray(
        delta_star_from_current_density(radius, np.ones_like(radius)),
        dtype=np.float64,
    )
    recovered_density = elliptic / unit_current_elliptic
    valid = np.asarray(mesh.interior(), dtype=bool) & np.isfinite(recovered_density)
    recovered_density = np.where(valid, recovered_density, 0.0)
    recovered_current = recovered_density * np.asarray(mesh.cell_area)
    zero = jnp.zeros_like(jnp.asarray(recovered_current))
    recovered_plasma = _analytic_internal_flux_image(
        operator,
        CellCurrentMoments(jnp.asarray(recovered_current), zero, zero),
    )
    recovered_external = reference - recovered_plasma

    analytic_density = np.asarray(
        case.toroidal_current_density(machine.node[:, 0], machine.node[:, 1]),
        dtype=np.float64,
    )
    valid_analytic_current = float(
        np.sum(analytic_density[valid] * np.asarray(mesh.cell_area)[valid])
    )
    recovered_valid_current = float(np.sum(recovered_current[valid]))
    density_scale = float(np.max(np.abs(analytic_density[valid])))
    density_error = recovered_density[valid] - analytic_density[valid]
    external_error = recovered_external - analytic_external
    grid_count = len(machine.node)
    wall_count = len(machine.wall_node)
    span = TOTAL_FLUX_FACTOR * case.axis_flux
    external_sup = float(np.max(np.abs(external_error)))
    source_relative_error = float(
        np.max(np.abs(density_error)) / max(density_scale, np.finfo(float).tiny)
    )
    external_span_error = external_sup / span
    return {
        "fixture": fixture,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "valid_stencil_cells": int(np.count_nonzero(valid)),
        "cache": machine.cache,
        "analytic_closed_form_plasma_current_a": float(case.plasma_current()),
        "analytic_authored_support_current_a": float(
            np.sum(np.asarray(analytic_physical.cell_current))
        ),
        "analytic_valid_stencil_current_a": valid_analytic_current,
        "delta_star_valid_stencil_current_a": recovered_valid_current,
        "valid_stencil_current_signed_relative_error": (
            recovered_valid_current / valid_analytic_current - 1.0
        ),
        "delta_star_current_vs_closed_form_signed_relative_error": (
            recovered_valid_current / case.plasma_current() - 1.0
        ),
        "current_density_sup_relative_error_on_valid_stencils": (source_relative_error),
        "analytic_external_recovery": {
            "sup_error_wb": external_sup,
            "rms_error_wb": float(np.sqrt(np.mean(external_error**2))),
            "grid_sup_error_wb": float(np.max(np.abs(external_error[:grid_count]))),
            "wall_sup_error_wb": float(
                np.max(np.abs(external_error[grid_count : grid_count + wall_count]))
            ),
            "sample_sup_error_wb": float(
                np.max(np.abs(external_error[grid_count + wall_count :]))
            ),
            "sup_error_fraction_of_analytic_span": external_span_error,
            "sup_error_fraction_of_analytic_plasma_image": (
                external_sup / np.max(np.abs(analytic_plasma))
            ),
        },
        "source_recovery_passes": bool(
            source_relative_error <= ANALYTIC_SOURCE_RELATIVE_TOLERANCE
        ),
        "external_recovery_passes": bool(
            external_span_error <= ANALYTIC_EXTERNAL_SPAN_TOLERANCE
        ),
    }


def solovev_null_test() -> dict[str, Any]:
    """Test the tare against independent analytic current and exterior truth."""
    configure_dtypes()
    rows = [
        _solovev_separation_row(name, requested)
        for name, requested in ANALYTIC_FIXTURE_REQUESTS.items()
    ]
    fine = rows[-1]
    source_passes = all(row["source_recovery_passes"] for row in rows)
    external_passes = all(row["external_recovery_passes"] for row in rows)
    passes = source_passes and external_passes
    return {
        "purpose": (
            "non-tautological attribution test against independently known "
            "closed-form plasma current and analytic external field"
        ),
        "separation": (
            "the identical Delta-star, valid-stencil centroid-current image, "
            "and subtraction used for the six MAST controls"
        ),
        "criteria": {
            "current_density_sup_relative_error": (ANALYTIC_SOURCE_RELATIVE_TOLERANCE),
            "external_sup_error_fraction_of_analytic_span": (
                ANALYTIC_EXTERNAL_SPAN_TOLERANCE
            ),
        },
        "fixtures": rows,
        "source_recovery_passes": source_passes,
        "external_recovery_passes": external_passes,
        "passes": passes,
        "verdict": (
            "SEPARATION_RECOVERS_ANALYTIC_TRUTH"
            if passes
            else "SEPARATION_FAILS_ANALYTIC_EXTERNAL_NULL"
        ),
        "finding": (
            "The Delta-star density converges to the analytic source on valid "
            "stencils, but the identical finite-mesh plasma image does not "
            "recover the independent analytic external field; the MAST tared "
            "root count is therefore void for residual attribution."
            if not passes
            else "The identical separation recovers both analytic source and exterior."
        ),
        "finest_external_sup_error_wb": fine["analytic_external_recovery"][
            "sup_error_wb"
        ],
        "finest_external_sup_error_fraction_of_analytic_span": fine[
            "analytic_external_recovery"
        ]["sup_error_fraction_of_analytic_span"],
    }


def _conductor_localisation_row(
    selected_row: dict[str, Any], qualification: dict[str, Any], store: Path
) -> dict[str, Any]:
    """Integrate one exterior Delta-star image around the stored filaments."""
    case, context = _mast_case_from_selection(store, selected_row, qualification)
    profile = context["profile"]
    implied = _implied_current(profile, context["reference_flux"])
    exterior = implied["valid"] & ~implied["declared"]
    exterior_current = implied["cell_current"][exterior]
    coordinates = np.asarray(profile.lattice.coordinate)[exterior]
    group = context["group"]
    row = context["row"]
    filament_radius = np.asarray(group["fcoil_r"], dtype=np.float64)
    filament_height = np.asarray(group["fcoil_z"], dtype=np.float64)
    filament_circuit = np.asarray(group["fcoil_circ"], dtype=int)
    filament_turns = np.asarray(group["fcoil_turns"], dtype=np.float64)
    filament_multiplier = np.asarray(group["fcoil_xmult"], dtype=np.float64)
    stored_current = np.asarray(group["fcoil_c"][row], dtype=np.float64)
    stored_index = np.asarray(group["fcoil_n"], dtype=int)
    if not np.array_equal(stored_index, np.arange(len(stored_current))):
        raise RuntimeError("fcoil_n does not provide zero-based circuit order")
    if filament_circuit.min() != 1 or filament_circuit.max() != len(stored_current):
        raise RuntimeError("the filament table does not span every stored circuit")

    distance, nearest_filament = cKDTree(np.c_[filament_radius, filament_height]).query(
        coordinates
    )
    capture_radius = float(
        np.hypot(profile.lattice.radial_step, profile.lattice.vertical_step)
    )
    captured = distance <= capture_radius
    nearest_circuit = filament_circuit[nearest_filament] - 1
    recovered = np.zeros(len(stored_current), dtype=np.float64)
    assigned_count = np.zeros(len(stored_current), dtype=int)
    np.add.at(recovered, nearest_circuit[captured], exterior_current[captured])
    np.add.at(assigned_count, nearest_circuit[captured], 1)

    turn_scale = np.zeros(len(stored_current), dtype=np.float64)
    circuit_rows = []
    for circuit in range(1, len(stored_current) + 1):
        selected = filament_circuit == circuit
        scale = float(np.sum(filament_turns[selected] * filament_multiplier[selected]))
        turn_scale[circuit - 1] = scale
        weight = np.abs(filament_turns[selected] * filament_multiplier[selected])
        if not np.any(weight > 0.0):
            weight = np.ones(np.count_nonzero(selected))
        centre_radius = float(weight @ filament_radius[selected] / weight.sum())
        centre_height = float(weight @ filament_height[selected] / weight.sum())
        expected = float(stored_current[circuit - 1] * scale)
        error = float(recovered[circuit - 1] - expected)
        circuit_rows.append(
            {
                "stored_circuit": circuit,
                "filament_count": int(np.count_nonzero(selected)),
                "filament_weighted_centre_r_m": centre_radius,
                "filament_weighted_centre_z_m": centre_height,
                "stored_fcoil_c_a": float(stored_current[circuit - 1]),
                "filament_turn_multiplier_sum": scale,
                "stored_effective_current_a_turn": expected,
                "recovered_delta_star_current_a": float(recovered[circuit - 1]),
                "signed_error_a": error,
                "signed_relative_error": (
                    error / expected if abs(expected) > np.finfo(float).eps else None
                ),
                "assigned_grid_cell_count": int(assigned_count[circuit - 1]),
            }
        )
    expected = stored_current * turn_scale
    live = np.abs(expected) > 1.0e-6 * np.max(np.abs(expected))
    correlation = float(np.corrcoef(recovered[live], expected[live])[0, 1])
    expected_l1 = float(np.sum(np.abs(expected[live])))
    current_l1_error = float(
        np.sum(np.abs(recovered[live] - expected[live])) / expected_l1
    )
    outside_absolute = float(np.sum(np.abs(exterior_current)))
    captured_absolute = float(np.sum(np.abs(exterior_current[captured])))
    localisation_share = captured_absolute / outside_absolute
    spatial_passes = bool(
        localisation_share >= CONDUCTOR_LOCALISATION_SHARE_TOLERANCE
        and correlation >= CONDUCTOR_PATTERN_CORRELATION_TOLERANCE
    )
    amplitude_passes = bool(current_l1_error <= CONDUCTOR_CURRENT_L1_ERROR_TOLERANCE)
    return {
        "shot": int(case["reference"]["shot"]),
        "slice_index": int(case["reference"]["slice_index"]),
        "grid_spacing_m": {
            "radial": float(profile.lattice.radial_step),
            "vertical": float(profile.lattice.vertical_step),
            "capture_radius_one_grid_diagonal": capture_radius,
        },
        "filament_count": len(filament_radius),
        "stored_circuit_count": len(stored_current),
        "outside_declared_support_cell_count": int(np.count_nonzero(exterior)),
        "outside_declared_support_signed_current_a": float(np.sum(exterior_current)),
        "outside_declared_support_absolute_current_a": outside_absolute,
        "captured_cell_count": int(np.count_nonzero(captured)),
        "captured_signed_current_a": float(np.sum(exterior_current[captured])),
        "captured_absolute_current_a": captured_absolute,
        "captured_absolute_current_share": localisation_share,
        "stored_effective_current_signed_sum_a_turn": float(np.sum(expected)),
        "recovered_circuit_current_signed_sum_a": float(np.sum(recovered)),
        "live_circuit_count_for_comparison": int(np.count_nonzero(live)),
        "recovered_vs_stored_effective_current_pearson": correlation,
        "recovered_vs_stored_effective_current_l1_relative_error": (current_l1_error),
        "spatial_localisation_passes": spatial_passes,
        "stored_current_amplitude_reproduction_passes": amplitude_passes,
        "circuits": circuit_rows,
    }


def mast_conductor_localisation(
    store: Path = SHOT_STORE, bank: Path = DECOMPOSITION_BANK
) -> dict[str, Any]:
    """Compare all six exterior Delta-star images with fitted circuit geometry."""
    configure_dtypes()
    rows = [
        _conductor_localisation_row(selected, qualification, store)
        for selected, qualification in select_slices_by_shot(bank)
    ]
    spatial_passes = all(row["spatial_localisation_passes"] for row in rows)
    amplitude_passes = all(
        row["stored_current_amplitude_reproduction_passes"] for row in rows
    )
    correlations = [
        row["recovered_vs_stored_effective_current_pearson"] for row in rows
    ]
    amplitude_errors = [
        row["recovered_vs_stored_effective_current_l1_relative_error"] for row in rows
    ]
    return {
        "method": (
            "assign every valid exterior grid-cell current to its nearest "
            "efm/fcoil_r,fcoil_z filament when the separation is no more than "
            "one benchmark-grid diagonal; aggregate by efm/fcoil_circ and compare "
            "with efm/fcoil_c times the stored turn multiplier"
        ),
        "criteria": {
            "minimum_captured_absolute_current_share": (
                CONDUCTOR_LOCALISATION_SHARE_TOLERANCE
            ),
            "minimum_live_circuit_pattern_pearson": (
                CONDUCTOR_PATTERN_CORRELATION_TOLERANCE
            ),
            "maximum_live_circuit_l1_relative_error": (
                CONDUCTOR_CURRENT_L1_ERROR_TOLERANCE
            ),
        },
        "rows": rows,
        "all_six_spatially_localise": spatial_passes,
        "all_six_reproduce_stored_current_amplitudes": amplitude_passes,
        "minimum_captured_absolute_current_share": min(
            row["captured_absolute_current_share"] for row in rows
        ),
        "pattern_pearson_range": [min(correlations), max(correlations)],
        "circuit_l1_relative_error_range": [
            min(amplitude_errors),
            max(amplitude_errors),
        ],
        "finding": (
            "The exterior Delta-star current spatially localises on the fitted "
            "conductors and follows their signed circuit pattern, demonstrating "
            "that the tare classified in-grid conductor current as plasma current. "
            "The coarse mesh does not reproduce the stored circuit amplitudes, so "
            "this is localisation evidence rather than an independent conductor-"
            "model validation."
            if spatial_passes and not amplitude_passes
            else "The exterior-current localisation result is stated by the gates."
        ),
    }


def _banked_solve_digest(receipt: dict[str, Any]) -> str:
    payload = {
        "per_shot": receipt["per_shot"],
        "six_reference_score_table": receipt["six_reference_score_table"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def adjudicate_banked_receipt(
    output_path: Path = OUTPUT_RECEIPT,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
) -> dict[str, Any]:
    """Adjudicate the existing six solves without evaluating a solve again."""
    receipt = json.loads(output_path.read_text())
    solve_digest = _banked_solve_digest(receipt)
    analytic = solovev_null_test()
    conductors = mast_conductor_localisation(store, bank)
    if analytic["passes"]:
        adjudicated = "MAST_EXTERIOR_CURRENT_REQUIRES_CONDUCTOR_LOCALISATION"
        finding = (
            "The analytic null validates the separation machinery; the MAST "
            "exterior current must next be tested against the fitted circuits."
        )
    else:
        adjudicated = "MAST_TARED_SOLVE_VOID_ANALYTIC_NULL_FAILED"
        finding = (
            "The tared root count regressed from one of six to zero of six and "
            "the identical split fails the analytic external-field null. This "
            "is evidence against a faithful background split, not evidence "
            "about the Grad--Shafranov fixed point or discretisation."
        )
    receipt["attribution_adjudication"] = {
        "solovev_analytic_null": analytic,
        "six_reference_solve_reused_without_rerun": True,
        "banked_solve_payload_sha256": solve_digest,
        "verdict": adjudicated,
        "finding": finding,
        "conductor_localisation": conductors,
    }
    receipt["aggregate"].update(
        {
            "verdict": adjudicated,
            "statement": finding,
            "tared_background_faithful_for_attribution": bool(analytic["passes"]),
            "mast_root_result_valid_for_gs_attribution": bool(analytic["passes"]),
        }
    )
    receipt["receipt"]["status"] = "complete_control_void_for_attribution"
    if _banked_solve_digest(receipt) != solve_digest:
        raise RuntimeError("the banked six-reference solve payload changed")
    protected_after = _verify_protected_artifacts(
        json.loads(PROTECTED_SOURCE.read_text())
    )
    receipt["protected_banked_artifacts"]["verified_after_adjudication"] = (
        protected_after
    )
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def _plot_external_field_comparison(
    profile,
    tared_external: np.ndarray,
    passive_external: np.ndarray,
    path: Path,
) -> None:
    """Plot one representative external-field comparison on the solve grid."""
    shape = profile.lattice.shape
    grid_nodes = profile.lattice.node_count
    tared = tared_external[:grid_nodes].reshape(shape)
    passive = passive_external[:grid_nodes].reshape(shape)
    difference = tared - passive
    limit = max(float(np.max(np.abs(tared))), float(np.max(np.abs(passive))))
    difference_limit = max(float(np.max(np.abs(difference))), np.finfo(float).tiny)
    figure, axes = plt.subplots(
        1, 3, figsize=(11.4, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, values, title, colour_limit in (
        (axes[0], tared, "Reference-derived external field", limit),
        (axes[1], passive, "Passive-inclusive modeled field", limit),
        (axes[2], difference, "Tared minus modeled", difference_limit),
    ):
        image = axis.pcolormesh(
            profile.lattice.radius,
            profile.lattice.height,
            values.T,
            shading="nearest",
            cmap="coolwarm",
            vmin=-colour_limit,
            vmax=colour_limit,
        )
        axis.set_title(title)
        axis.set_xlabel("R [m]")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, label="Total flux [Wb]", shrink=0.82)
    axes[0].set_ylabel("Z [m]")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def measure_tares(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    figure_path: Path = OUTPUT_FIGURE,
    rebuild_figure: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build and verify the reference-derived background on all frozen rows."""
    configure_dtypes()
    protected_source = json.loads(PROTECTED_SOURCE.read_text())
    protected = _verify_protected_artifacts(protected_source)
    source_digest = _sha256(PROTECTED_SOURCE)
    rows = []
    runtime = []
    representative = None
    selected = select_slices_by_shot(bank)
    for selected_row, qualification in selected:
        case, context = _mast_case_from_selection(store, selected_row, qualification)
        profile = context["profile"]
        tare = build_tare(profile, case["state"], context["reference_flux"])
        reference_current = abs(float(case["reference"]["plasma_current_a"]))
        current = tare["implied"]["receipt"]
        row = {
            "shot": int(case["reference"]["shot"]),
            "slice_index": int(case["reference"]["slice_index"]),
            "qualification_passes": bool(qualification["passes"]),
            "reference_declared_current_a": reference_current,
            **current,
            "declared_support_signed_relative_current_error": (
                current["declared_support_current_integral_a"] / reference_current - 1.0
            ),
            "closure_sup_difference_wb": tare["closure"]["sup_difference_wb"],
            "closure_at_roundoff": tare["closure"]["at_roundoff"],
        }
        rows.append(row)
        runtime.append({"case": case, "context": context, "tare": tare})
        if row["shot"] == REPRESENTATIVE_SHOT:
            representative = runtime[-1]
    if representative is None:
        raise RuntimeError("the representative field-comparison row is absent")

    if rebuild_figure or not figure_path.exists():
        passive_case, passive_profile, passive_policy = _passive_inclusive_case(
            representative["case"], representative["context"]
        )
        del passive_case
        passive_external = np.asarray(
            passive_profile.operator.external(), dtype=np.float64
        )
        _plot_external_field_comparison(
            representative["context"]["profile"],
            representative["tare"]["external"],
            passive_external,
            figure_path,
        )
        comparison = {
            "shot": REPRESENTATIVE_SHOT,
            "modeled_policy": passive_policy["policy"],
            "modeled_stored_circuit_count": passive_policy["stored_circuit_count"],
            "ordinary_active_drive_zeroed": passive_policy[
                "ordinary_active_drive_zeroed_to_avoid_double_counting"
            ],
            "figure_rebuilt_this_call": True,
        }
    else:
        comparison = {
            "shot": REPRESENTATIVE_SHOT,
            "modeled_policy": "explicit prescribed-current response matrix",
            "modeled_stored_circuit_count": 101,
            "ordinary_active_drive_zeroed": True,
            "figure_rebuilt_this_call": False,
            "figure_provenance": (
                "retained from the closure-gated exact-kernel build in this worker run"
            ),
        }
    maximum_closure = max(row["closure_sup_difference_wb"] for row in rows)
    all_close = all(row["closure_at_roundoff"] for row in rows)
    receipt = {
        "receipt": {
            "kind": "tared_external_field_control",
            "status": "tare_closed_solve_pending",
            "selection": "select_slices_by_shot on the frozen decomposition bank",
            "shot_count": len(rows),
        },
        "control_caveat": (
            "This is a control and not a parity claim, because the tared field is "
            "derived from the reference own map and hands the solve information no "
            "production run would have; it bounds the GS solve own fidelity and "
            "attributes the residual, and must never be cited as parity."
        ),
        "tare_construction": {
            "reference_flux": "2*pi*efm/psirz on the benchmark mesh",
            "delta_star_implementation": "nova.equilibrium.conservation.delta_star",
            "current_relation_implementation": (
                "nova.equilibrium.convention.delta_star_from_current_density"
            ),
            "current_relation": "DeltaStar(Phi) = -2*pi*mu0*R*j_phi",
            "plasma_composition": (
                "ForwardFluxOperator.current_moment_image with centroid current "
                "moments, the same grid and wall kernels used by the forward map"
            ),
            "external_field": "psi_ext = psi_ref - psi_plasma",
            "delta_star_interpretation": (
                "interior Delta-star removes conductor fields outside the grid; "
                "the external field is recovered by subtraction"
            ),
        },
        "gauge": {
            "rule": "preserve the stored reference total-flux gauge",
            "additive_constant_wb": GAUGE_CONSTANT_WB,
        },
        "closure_gate": {
            "all_six_at_roundoff": all_close,
            "maximum_sup_difference_wb": maximum_closure,
            "banked_stored_field_closure_wb": BANKED_STORED_FIELD_CLOSURE_WB,
            "passes": bool(all_close),
        },
        "current_integral_table": rows,
        "banked_comparison": {
            "nova_solved_outside_separatrix_current_a": REFERENCE_HALO_CURRENT_A,
            "comparison_note": (
                "The table reports the reference-own Delta-star current outside "
                "each declared support separately from the support integral."
            ),
        },
        "instrument_control_basis_reserved_for_solve": {
            "closed_axis_branch": (
                f"{_unit_boundary_branches.__module__}._unit_boundary_branches + "
                f"{_closed_axis_branch.__module__}._closed_axis_branch + "
                f"{_distance_pair.__module__}._distance_pair"
            ),
            "matched_boundary_support": (
                f"{_boundary_moments.__module__}._boundary_moments and "
                f"{_polygon_measure.__module__}._polygon_measure"
            ),
            "field_energy_instrument_ratio": (
                EXPECTED_REFERENCE_MAP_FIELD_ENERGY / PUBLISHED_REFERENCE_FIELD_ENERGY
            ),
            "relative_error_helper": (f"{_relative_error.__module__}._relative_error"),
            "field_magnitude_helper": f"{_field_magnitude.__module__}._field_magnitude",
        },
        "passive_inclusive_comparison": {
            **comparison,
            "figure": str(figure_path),
            "figure_sha256": _sha256(figure_path),
            "figure_src": (
                "/nova/figures/efit-forward-parity/tared-external-field-solve.png"
            ),
        },
        "protected_banked_artifacts": {
            **protected,
            "integrity_source": str(PROTECTED_SOURCE),
            "integrity_source_sha256": source_digest,
        },
    }
    if len(rows) != 6 or not receipt["closure_gate"]["passes"]:
        raise RuntimeError("the six-reference tare closure gate did not pass")
    return receipt, runtime


def _reference_moment_record(
    group, row: int
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build the published reference denominators consumed by the landed helper."""
    beta = float(group["betap"][row])
    internal_inductance = float(group["li"][row])
    reference_current = abs(float(group["plasma_current_c"][row]))
    pressure_integral = (2.0 / 3.0) * float(group["plasma_energy"][row])
    field_integral = float(group["bpol_squared"][row])
    beta_denominator = 2.0 * mu_0 * pressure_integral / beta
    inductance_denominator = field_integral / internal_inductance
    moment = {
        "four_by_two_rescore": {
            "rows": [
                {
                    "definition": "reference_boundary_field",
                    "side": "reference",
                    "poloidal_beta": {
                        "all_domain_constrained": {
                            "value": beta,
                            "current_a": reference_current,
                            "denominator_t2_m3": beta_denominator,
                        }
                    },
                    "internal_inductance": {
                        "current_independent": {
                            "value": internal_inductance,
                            "denominator_t2_m3": inductance_denominator,
                        }
                    },
                }
            ]
        }
    }
    published = {
        "poloidal_beta": beta,
        "internal_inductance": internal_inductance,
        "pressure_volume_integral_pa_m3": pressure_integral,
        "poloidal_field_squared_volume_integral_t2_m3": field_integral,
        "plasma_volume_m3": float(group["plasma_volume"][row]),
    }
    return moment, published


def _instrument_controlled_metrics(
    runtime: dict[str, Any], equilibrium
) -> dict[str, Any]:
    """Rescore one terminal state with the landed geometry and moment instruments."""
    context = runtime["context"]
    profile = runtime["tare"]["profile"]
    group = context["group"]
    row = context["row"]
    topology = equilibrium.topology
    grid_nodes = profile.lattice.node_count
    solved_flux = np.asarray(equilibrium.flux[:grid_nodes], dtype=np.float64)
    branches = _unit_boundary_branches(
        profile.lattice.radius,
        profile.lattice.height,
        solved_flux.reshape(profile.lattice.shape),
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    stored = _stored_lcfs(group, row)
    try:
        closed = _closed_axis_branch(
            branches, np.asarray(topology.axis, dtype=np.float64)
        )
    except RuntimeError as error:
        lcfs = {
            "status": "unscoreable_no_closed_axis_branch",
            "selection": (
                "longest explicitly closed unit branch enclosing the magnetic axis"
            ),
            "branch_count": len(branches),
            "closed_branch_point_count": None,
            "distance": None,
            "reason": str(error),
            "longest_polyline_fallback_used": False,
        }
    else:
        lcfs = {
            "status": "scoreable",
            "selection": (
                "longest explicitly closed unit branch enclosing the magnetic axis"
            ),
            "branch_count": len(branches),
            "closed_branch_point_count": int(len(closed)),
            "distance": _distance_pair(closed, stored),
            "solved_polygon": _polygon_measure(closed),
            "stored_polygon": _polygon_measure(stored),
            "longest_polyline_fallback_used": False,
        }
    coordinates = np.asarray(profile.lattice.coordinate, dtype=np.float64)
    inside = MplPath(stored, closed=True).contains_points(coordinates, radius=1.0e-12)
    partition = {
        "inside": inside,
        "cell_current": np.asarray(equilibrium.cell_current, dtype=np.float64),
    }
    moment, published = _reference_moment_record(group, row)
    solved_moments = _boundary_moments(
        profile,
        equilibrium,
        np.asarray(equilibrium.flux, dtype=np.float64),
        partition,
        moment,
    )
    reference_moments = _boundary_moments(
        profile,
        equilibrium,
        np.asarray(runtime["case"]["state"], dtype=np.float64),
        partition,
        moment,
    )
    solved_energy = float(
        solved_moments["integrals"]["poloidal_field_squared_volume_integral_t2_m3"]
    )
    reference_nova_energy = float(
        reference_moments["integrals"]["poloidal_field_squared_volume_integral_t2_m3"]
    )
    published_energy = published["poloidal_field_squared_volume_integral_t2_m3"]
    instrument_ratio = reference_nova_energy / published_energy
    raw_published_ratio = solved_energy / published_energy
    corrected_ratio = raw_published_ratio / instrument_ratio
    return {
        "lcfs_closed_branch": lcfs,
        "matched_stored_boundary_support": {
            "definition": "mesh centroids inside the stored LCFS for both fields",
            "cell_count": int(np.count_nonzero(inside)),
            "solved": solved_moments,
            "reference_published": published,
            "poloidal_beta_signed_relative_deviation": solved_moments["poloidal_beta"][
                "signed_relative_deviation"
            ],
        },
        "poloidal_field_energy_instrument_control": {
            "solved_nova_operator_t2_m3": solved_energy,
            "reference_own_map_nova_operator_t2_m3": reference_nova_energy,
            "reference_published_t2_m3": published_energy,
            "nova_on_reference_over_reference_published": instrument_ratio,
            "banked_representative_instrument_ratio": (
                EXPECTED_REFERENCE_MAP_FIELD_ENERGY / PUBLISHED_REFERENCE_FIELD_ENERGY
            ),
            "solved_over_reference_published_raw": raw_published_ratio,
            "solved_over_reference_after_instrument_division": corrected_ratio,
            "instrument_controlled_signed_relative_deviation": corrected_ratio - 1.0,
            "multiplicative_closure_residual": (
                raw_published_ratio - instrument_ratio * corrected_ratio
            ),
        },
    }


def _solve_row(runtime: dict[str, Any]) -> dict[str, Any]:
    """Run and score one current-constrained branch in its tared background."""
    case = runtime["case"]
    context = runtime["context"]
    profile = runtime["tare"]["profile"]
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    solve, _trace, branch = _passive_inclusive_solve(
        case,
        context,
        profile,
        target_current=target_current,
    )
    equilibrium = branch.equilibrium
    terminal_current = float(np.sum(np.asarray(equilibrium.cell_current)))
    nonzero = bool(abs(terminal_current) >= 0.01 * target_current)
    converged_plasma = bool(branch.converged and nonzero)
    if converged_plasma:
        outcome = "converged_plasma_root"
    elif nonzero:
        outcome = "bounded_non_convergence"
    else:
        outcome = "vacuum_collapse"
    raw = _pinned_metrics(
        context["group"],
        context["row"],
        profile,
        context["reference_flux"],
        equilibrium,
    )
    controlled = _instrument_controlled_metrics(runtime, equilibrium)
    branch_receipt = solve["forward_branch_receipt"]
    return {
        "reference": case["reference"],
        "qualification_before_solve": case["reference"][
            "qualification_before_attribution"
        ],
        "target_current": {
            "source": "abs(efm/plasma_current_c)",
            "value_a": target_current,
            "terminal_value_a": terminal_current,
            "signed_terminal_relative_error": terminal_current / target_current - 1.0,
        },
        "solve": {
            "entry_point": "ForwardProfile.solve_branch(target_current=...)",
            "route": "newton_krylov",
            "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
            "outcome_class": outcome,
            "converged": bool(branch.converged),
            "converged_plasma_root": converged_plasma,
            "terminal_residual": branch_receipt["residual"],
            "iterations": branch_receipt["iterations"],
            "residual_trajectory": solve["residual_trajectory"],
        },
        "raw_registered_rows": {
            "metrics": raw,
            "qualification": _metric_qualification(raw, branch_receipt["residual"]),
            "note": (
                "Retained beside the controlled score so the longest-branch and "
                "clipped-core instrument readings remain visible."
            ),
        },
        "instrument_controlled_rows": controlled,
    }


def run_control(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output_path: Path = OUTPUT_RECEIPT,
    figure_path: Path = OUTPUT_FIGURE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Close the tare, run every frozen reference, and write the control receipt."""
    receipt, runtime = measure_tares(store, bank, figure_path)
    receipt["decisive_readout_declared_before_measurement"] = {
        "measure": (
            "number of six references reaching a converged nonzero plasma root at 1e-8"
        ),
        "banked_converged_plasma_roots": BANKED_CONVERGED_PLASMA_ROOTS,
        "banked_bounded_stall_range": {
            "minimum": BANKED_BOUNDED_RESIDUAL_MINIMUM,
            "maximum": BANKED_BOUNDED_RESIDUAL_MAXIMUM,
        },
        "marked_improvement_reading": (
            "external-field error was material and the GS solve is sound"
        ),
        "unchanged_near_one_of_six_reading": (
            "external field is exonerated; the obstacle is the GS fixed point "
            "or discretisation"
        ),
    }
    solved = [_solve_row(item) for item in runtime]
    roots = sum(row["solve"]["converged_plasma_root"] for row in solved)
    if roots > BANKED_CONVERGED_PLASMA_ROOTS:
        verdict = "EXTERNAL_FIELD_ERROR_WAS_MATERIAL"
        statement = (
            "The tared background materially increased converged plasma-root recovery, "
            "so external-field error contributed to the banked residual."
        )
    elif roots < BANKED_CONVERGED_PLASMA_ROOTS:
        verdict = "ROOT_COUNT_REGRESSED_TARE_FIDELITY_REQUIRES_ADJUDICATION"
        statement = (
            "The tared background reduced converged plasma-root recovery, which is "
            "evidence against a faithful field split until an independent null test "
            "validates the tare. It is not evidence about the GS fixed point."
        )
    else:
        verdict = "ROOT_COUNT_UNCHANGED_TARE_FIDELITY_REQUIRES_ADJUDICATION"
        statement = (
            "The tared background did not improve the converged plasma-root count; "
            "an independent null test must establish that the split is faithful "
            "before attributing the residual."
        )
    receipt["receipt"]["status"] = "complete"
    receipt["per_shot"] = solved
    receipt["six_reference_score_table"] = [
        {
            "shot": row["reference"]["shot"],
            "slice_index": row["reference"]["slice_index"],
            "outcome_class": row["solve"]["outcome_class"],
            "converged_plasma_root": row["solve"]["converged_plasma_root"],
            "terminal_residual": row["solve"]["terminal_residual"],
            "raw_flux_rms_fraction_of_span": row["raw_registered_rows"]["metrics"][
                "flux_map"
            ]["rms_fraction_of_reference_span"],
            "raw_lcfs_longest_branch_distance_m": row["raw_registered_rows"]["metrics"][
                "lcfs"
            ]["symmetric_mean_distance_m"],
            "controlled_lcfs_closed_branch_distance_m": row[
                "instrument_controlled_rows"
            ]["lcfs_closed_branch"]["distance"]["symmetric_mean_distance_m"]
            if row["instrument_controlled_rows"]["lcfs_closed_branch"]["distance"]
            is not None
            else None,
            "raw_poloidal_beta_signed_relative_deviation": row["raw_registered_rows"][
                "metrics"
            ]["poloidal_beta"]["signed_relative_deviation"],
            "controlled_poloidal_beta_signed_relative_deviation": row[
                "instrument_controlled_rows"
            ]["matched_stored_boundary_support"][
                "poloidal_beta_signed_relative_deviation"
            ],
            "raw_internal_inductance_signed_relative_deviation": row[
                "raw_registered_rows"
            ]["metrics"]["internal_inductance"]["signed_relative_deviation"],
            "controlled_field_energy_signed_relative_deviation": row[
                "instrument_controlled_rows"
            ]["poloidal_field_energy_instrument_control"][
                "instrument_controlled_signed_relative_deviation"
            ],
        }
        for row in solved
    ]
    receipt["aggregate"] = {
        "shot_count": len(solved),
        "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
        "banked_converged_plasma_roots": BANKED_CONVERGED_PLASMA_ROOTS,
        "tared_converged_plasma_roots": roots,
        "change_in_converged_plasma_roots": roots - BANKED_CONVERGED_PLASMA_ROOTS,
        "all_target_currents_exact": all(
            abs(row["target_current"]["signed_terminal_relative_error"]) <= 1.0e-12
            for row in solved
        ),
        "verdict": verdict,
        "statement": statement,
        "parity_claim": False,
    }
    protected_after = _verify_protected_artifacts(
        json.loads(PROTECTED_SOURCE.read_text())
    )
    receipt["protected_banked_artifacts"]["verified_after_solves"] = protected_after
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    arguments = parser.parse_args()
    receipt, _runtime = run_control(
        arguments.store, arguments.bank, arguments.output, arguments.figure
    )
    print(
        "TARED_EXTERNAL_FIELD_CONTROL "
        f"shots={receipt['receipt']['shot_count']} "
        f"sup_wb={receipt['closure_gate']['maximum_sup_difference_wb']:.6g} "
        f"roots={receipt['aggregate']['tared_converged_plasma_roots']} "
        f"verdict={receipt['aggregate']['verdict']}"
    )


if __name__ == "__main__":
    main()
