"""Isolate the Grad--Shafranov solve with a reference-derived external field."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.constants import mu_0

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import benchmarks.efit_forward_parity_slice as parity_slice  # noqa: E402
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
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "tared-plasma-support-solve.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "tared-plasma-support-solve.png"
BANKED_VOID_RECEIPT = OUTPUT_DIRECTORY / "tared-external-field-solve.json"
PROTECTED_SOURCE = OUTPUT_DIRECTORY / "converged-root-geometry-attribution.json"
MESH_SENSITIVITY_DIRECTORY = Path("docs/figures/moment-conditioned-basin-entry")
MESH_SENSITIVITY_RECEIPT = MESH_SENSITIVITY_DIRECTORY / "stall-mesh-sensitivity.json"
MESH_SENSITIVITY_FIGURE = MESH_SENSITIVITY_DIRECTORY / "stall-mesh-sensitivity.png"
BANKED_STORED_FIELD_CLOSURE_WB = 2.22e-15
REFERENCE_HALO_CURRENT_A = 786.396
GAUGE_CONSTANT_WB = 0.0
REPRESENTATIVE_SHOT = 22086
BANKED_CONVERGED_PLASMA_ROOTS = 1
BANKED_UNCORRECTED_TARE_ROOTS = 0
BANKED_MODELLED_BACKGROUND_ROOTS = 1
BANKED_BOUNDED_RESIDUAL_MINIMUM = 2.006e-4
BANKED_BOUNDED_RESIDUAL_MAXIMUM = 1.076e-2
ANALYTIC_SOURCE_RELATIVE_TOLERANCE = 1.0e-3
ANALYTIC_EXTERNAL_SPAN_TOLERANCE = 1.0e-2
CONDUCTOR_LOCALISATION_SHARE_TOLERANCE = 0.9
CONDUCTOR_PATTERN_CORRELATION_TOLERANCE = 0.9
CONDUCTOR_CURRENT_L1_ERROR_TOLERANCE = 0.5
MESH_STRIDES = {"coarse": 2, "fine": 1}
MINIMUM_MESH_SCALING_ORDER = 0.5
MAXIMUM_MESH_INVARIANT_ORDER_MAGNITUDE = 0.25
CONTROL_CAVEAT = (
    "This is a control and not a parity claim, because the tared field is "
    "derived from the reference own map and hands the solve information no "
    "production run would have; it bounds the GS solve own fidelity and "
    "attributes the residual, and must never be cited as parity."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_stamp(checkout: Path = Path(".")) -> dict[str, str]:
    """Return the clean commit and tree identities used for a measurement."""
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("mesh sensitivity measurement requires a clean checkout")

    def revision(name: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {"commit": revision("HEAD"), "tree": revision("HEAD^{tree}")}


@contextmanager
def _profile_grid_stride(stride: int):
    """Select one stored-grid stride while reusing the landed profile builder."""
    original = parity_slice.GRID_STRIDE
    parity_slice.GRID_STRIDE = stride
    try:
        yield
    finally:
        parity_slice.GRID_STRIDE = original


def _mast_case_at_grid_stride(
    store: Path,
    selected: dict[str, Any],
    qualification: dict[str, Any],
    stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one MAST case on a declared stride of its stored 65-point axes."""
    if stride not in MESH_STRIDES.values():
        raise ValueError(f"unsupported stored-grid stride {stride}")
    with _profile_grid_stride(stride):
        case, context = parity_slice._mast_case_from_selection(
            store, selected, qualification
        )
    profile = context["profile"]
    case["mesh"] = {
        **case["mesh"],
        "kind": (
            f"{len(profile.lattice.radius)} by {len(profile.lattice.height)} "
            "rectangular benchmark lattice"
        ),
        "stored_axis_stride": stride,
        "realised_cells": profile.lattice.node_count,
        "radial_step_m": float(profile.lattice.radial_step),
        "vertical_step_m": float(profile.lattice.vertical_step),
    }
    return case, context


def _classify_mesh_floor(
    coarse_residual: float,
    fine_residual: float,
    coarse_spacing: float,
    fine_spacing: float,
) -> dict[str, Any]:
    """Classify one budget-terminal floor from its observed mesh order."""
    if min(coarse_residual, fine_residual, coarse_spacing, fine_spacing) <= 0.0:
        raise ValueError("residuals and mesh spacings must be positive")
    spacing_ratio = coarse_spacing / fine_spacing
    if spacing_ratio <= 1.0:
        raise ValueError("the fine mesh spacing must be smaller than coarse")
    residual_ratio = fine_residual / coarse_residual
    observed_order = float(
        np.log(coarse_residual / fine_residual) / np.log(spacing_ratio)
    )
    if observed_order >= MINIMUM_MESH_SCALING_ORDER:
        verdict = "floor-scales-with-mesh"
    elif abs(observed_order) <= MAXIMUM_MESH_INVARIANT_ORDER_MAGNITUDE:
        verdict = "mesh-invariant"
    else:
        verdict = "ambiguous"
    return {
        "fine_over_coarse_terminal_residual": residual_ratio,
        "observed_mesh_order": observed_order,
        "verdict": verdict,
    }


def _implied_current(profile, reference_grid: np.ndarray) -> dict[str, Any]:
    """Recover current, composing only declared plasma support into its image."""
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
    plasma_cell_current = np.where(declared_valid, cell_current, 0.0)
    zero = jnp.zeros_like(jnp.asarray(plasma_cell_current))
    moments = CellCurrentMoments(jnp.asarray(plasma_cell_current), zero, zero)
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
        "plasma_cell_current": plasma_cell_current,
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
            "plasma_image_current_integral_a": float(np.sum(plasma_cell_current)),
            "plasma_image_uses_declared_support_only": True,
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
    masks, _topology, _sample, _core_support, _common_support = (
        operator._support_partition(jnp.asarray(reference))
    )
    declared = np.asarray(operator.source.declared_support(masks), dtype=bool)
    declared_valid = declared & valid
    analytic_density = np.asarray(
        case.toroidal_current_density(machine.node[:, 0], machine.node[:, 1]),
        dtype=np.float64,
    )
    analytic_plasma_current = np.where(
        declared_valid,
        analytic_density * np.asarray(mesh.cell_area),
        0.0,
    )
    recovered_plasma_current = np.where(declared_valid, recovered_current, 0.0)
    zero = jnp.zeros_like(jnp.asarray(recovered_plasma_current))
    analytic_plasma = _analytic_internal_flux_image(
        operator,
        CellCurrentMoments(jnp.asarray(analytic_plasma_current), zero, zero),
    )
    analytic_external = reference - analytic_plasma
    recovered_plasma = _analytic_internal_flux_image(
        operator,
        CellCurrentMoments(jnp.asarray(recovered_plasma_current), zero, zero),
    )
    recovered_external = reference - recovered_plasma

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
        "declared_support_cells": int(np.count_nonzero(declared)),
        "declared_support_valid_stencil_cells": int(np.count_nonzero(declared_valid)),
        "cache": machine.cache,
        "analytic_closed_form_plasma_current_a": float(case.plasma_current()),
        "analytic_authored_support_current_a": float(
            np.sum(np.asarray(analytic_physical.cell_current))
        ),
        "analytic_valid_stencil_current_a": valid_analytic_current,
        "analytic_declared_support_centroid_current_a": float(
            np.sum(analytic_plasma_current)
        ),
        "delta_star_valid_stencil_current_a": recovered_valid_current,
        "delta_star_declared_support_current_a": float(
            np.sum(recovered_plasma_current)
        ),
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
            "the identical Delta-star recovery followed by a declared-plasma-"
            "support centroid-current image and subtraction used for the six "
            "MAST controls"
        ),
        "plasma_composition_support": (
            "the source-declared boundary support intersected with valid stencils; "
            "every other cell remains in the external field by subtraction"
        ),
        "criteria": {
            "current_density_sup_relative_error": (ANALYTIC_SOURCE_RELATIVE_TOLERANCE),
            "external_sup_error_fraction_of_analytic_span": (
                ANALYTIC_EXTERNAL_SPAN_TOLERANCE
            ),
        },
        "banked_uncorrected_support_failures": {
            "coarse": {
                "sup_error_wb": 1.1361563949563815,
                "sup_error_fraction_of_analytic_span": 0.8225776910715236,
            },
            "fine": {
                "sup_error_wb": 0.7791504291587774,
                "sup_error_fraction_of_analytic_span": 0.5641052269387783,
            },
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
            "stencils, but the declared-support plasma image does not recover "
            "the independent analytic external field; no MAST solve may run "
            "and residual attribution remains unavailable."
            if not passes
            else (
                "The declared-support separation recovers both the analytic "
                "source and exterior, so the six-reference control may run."
            )
        ),
        "finest_external_sup_error_wb": fine["analytic_external_recovery"][
            "sup_error_wb"
        ],
        "finest_external_sup_error_fraction_of_analytic_span": fine[
            "analytic_external_recovery"
        ]["sup_error_fraction_of_analytic_span"],
    }


def analytic_gate_receipt() -> dict[str, Any]:
    """Return the independently checked stop/go receipt without a MAST solve."""
    analytic = solovev_null_test()
    protected_source = json.loads(PROTECTED_SOURCE.read_text())
    protected = _verify_protected_artifacts(protected_source)
    status = (
        "analytic_null_passed_solve_authorised"
        if analytic["passes"]
        else "analytic_null_failed_solve_prohibited"
    )
    return {
        "receipt": {
            "kind": "tared_plasma_support_control",
            "status": status,
            "selection": "six frozen references remain unopened for solving",
            "shot_count": 0,
        },
        "control_caveat": CONTROL_CAVEAT,
        "execution_order": [
            "Solovev analytic null",
            "six-reference current-constrained solve only after a passing fine mesh",
        ],
        "analytic_null_gate": analytic,
        "attribution": {
            "available": False,
            "finding": (
                "The analytic null passed; attribution remains pending the "
                "six-reference solve."
                if analytic["passes"]
                else (
                    "The analytic null failed, so the six-reference solve is "
                    "prohibited and attribution remains unavailable."
                )
            ),
        },
        "gauge": {
            "rule": "preserve the stored reference total-flux gauge",
            "additive_constant_wb": GAUGE_CONSTANT_WB,
        },
        "protected_banked_artifacts": {
            **protected,
            "integrity_source": str(PROTECTED_SOURCE),
            "integrity_source_sha256": _sha256(PROTECTED_SOURCE),
        },
    }


def run_analytic_gate(output_path: Path = OUTPUT_RECEIPT) -> dict[str, Any]:
    """Measure and persist the analytic stop/go result before any MAST solve."""
    receipt = analytic_gate_receipt()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


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
    output_path: Path = BANKED_VOID_RECEIPT,
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
            "kind": "tared_plasma_support_control",
            "status": "tare_closed_solve_pending",
            "selection": "select_slices_by_shot on the frozen decomposition bank",
            "shot_count": len(rows),
        },
        "control_caveat": CONTROL_CAVEAT,
        "tare_construction": {
            "reference_flux": "2*pi*efm/psirz on the benchmark mesh",
            "delta_star_implementation": "nova.equilibrium.conservation.delta_star",
            "current_relation_implementation": (
                "nova.equilibrium.convention.delta_star_from_current_density"
            ),
            "current_relation": "DeltaStar(Phi) = -2*pi*mu0*R*j_phi",
            "plasma_composition": (
                "ForwardFluxOperator.current_moment_image with centroid current "
                "moments only on the reference declared-boundary plasma support; "
                "all other valid cells, including in-grid conductor cells, remain "
                "in psi_ext by subtraction"
            ),
            "external_field": "psi_ext = psi_ref - psi_plasma",
            "delta_star_interpretation": (
                "Delta-star recovers all valid-cell current, but only the declared "
                "plasma support is composed as plasma; the external field is the "
                "reference remainder"
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
                "/nova/figures/efit-forward-parity/tared-plasma-support-solve.png"
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
    """Pass the analytic null, then solve and score every frozen reference."""
    gate_receipt = run_analytic_gate(output_path)
    analytic = gate_receipt["analytic_null_gate"]
    if not analytic["passes"]:
        return gate_receipt, []
    receipt, runtime = measure_tares(store, bank, figure_path)
    receipt["execution_order"] = gate_receipt["execution_order"]
    receipt["analytic_null_gate"] = analytic
    receipt["decisive_readout_declared_before_measurement"] = {
        "measure": (
            "number of six references reaching a converged nonzero plasma root at 1e-8"
        ),
        "banked_converged_plasma_roots": BANKED_CONVERGED_PLASMA_ROOTS,
        "banked_uncorrected_tare_converged_plasma_roots": (
            BANKED_UNCORRECTED_TARE_ROOTS
        ),
        "banked_modelled_background_converged_plasma_roots": (
            BANKED_MODELLED_BACKGROUND_ROOTS
        ),
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
            "The ground-truth-qualified tared background materially increased "
            "converged plasma-root recovery above one of six, so the earlier "
            "residual was external-field error and the GS solve is sound."
        )
    else:
        verdict = "EXTERNAL_FIELD_EXONERATED_FIXED_POINT_OR_DISCRETISATION"
        statement = (
            "The ground-truth-qualified tared background recovered at most one of "
            "six plasma roots, so the external field is exonerated and the obstacle "
            "is the GS fixed point or its discretisation."
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
        "banked_uncorrected_tare_converged_plasma_roots": (
            BANKED_UNCORRECTED_TARE_ROOTS
        ),
        "banked_modelled_background_converged_plasma_roots": (
            BANKED_MODELLED_BACKGROUND_ROOTS
        ),
        "tared_converged_plasma_roots": roots,
        "change_in_converged_plasma_roots": roots - BANKED_CONVERGED_PLASMA_ROOTS,
        "all_target_currents_exact": all(
            abs(row["target_current"]["signed_terminal_relative_error"]) <= 1.0e-12
            for row in solved
        ),
        "verdict": verdict,
        "statement": statement,
        "analytic_null_passed_before_solves": True,
        "attribution_available": True,
        "parity_claim": False,
    }
    receipt["attribution"] = {
        "available": True,
        "basis": (
            "the corrected declared-support tare passed the independent Solovev "
            "external-field recovery null before any MAST solve"
        ),
        "finding": statement,
    }
    protected_after = _verify_protected_artifacts(
        json.loads(PROTECTED_SOURCE.read_text())
    )
    receipt["protected_banked_artifacts"]["verified_after_solves"] = protected_after
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt, runtime


def _mesh_terminal_row(runtime: dict[str, Any]) -> dict[str, Any]:
    """Run one banked-budget terminal measurement without parity rescoring."""
    case = runtime["case"]
    context = runtime["context"]
    profile = runtime["tare"]["profile"]
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    solve, _trace, branch = _passive_inclusive_solve(
        case,
        context,
        profile,
        newton_budget=parity_slice.NEWTON_STEPS,
        target_current=target_current,
    )
    equilibrium = branch.equilibrium
    terminal_current = float(np.sum(np.asarray(equilibrium.cell_current)))
    branch_receipt = solve["forward_branch_receipt"]
    return {
        "mesh": case["mesh"],
        "tare_closure": runtime["tare"]["closure"],
        "solve": {
            "entry_point": "ForwardProfile.solve_branch(target_current=...)",
            "route": "newton_krylov",
            "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
            "newton_promotion_budget": parity_slice.NEWTON_STEPS,
            "gmres_iterations_per_promotion": parity_slice.GMRES_ITERATIONS,
            "warmup_sweeps": parity_slice.WARMUP_SWEEPS,
            "relaxation": parity_slice.RELAXATION,
            "step_cap": parity_slice.STEP_CAP,
            "converged": bool(branch.converged),
            "terminal_residual": branch_receipt["residual"],
            "iterations": branch_receipt["iterations"],
            "residual_trajectory": solve["residual_trajectory"],
            "target_current_a": target_current,
            "terminal_current_a": terminal_current,
            "signed_terminal_current_relative_error": (
                terminal_current / target_current - 1.0
            ),
        },
    }


def _split_verdict(rows: list[dict[str, Any]]) -> str:
    """Return a unanimous per-reference verdict without pooling residuals."""
    verdicts = {row["verdict"] for row in rows}
    return verdicts.pop() if len(verdicts) == 1 else "ambiguous"


def _plot_mesh_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot each residual floor directly, separated by topology stratum."""
    figure, axes = plt.subplots(
        1, 2, figsize=(9.4, 4.2), sharey=True, constrained_layout=True
    )
    strata = (
        ("closed-axis", "Closed axis-enclosing branch (2/2 stalled)"),
        (
            "confinement-construction",
            "Confinement construction (3/4 stalled; 1 root excluded)",
        ),
    )
    colours = plt.get_cmap("tab10")
    for axis, (stratum, title) in zip(axes, strata, strict=True):
        selected = [row for row in rows if row["stratum"] == stratum]
        for index, row in enumerate(selected):
            values = [
                row["mesh_levels"]["coarse"]["terminal_residual"],
                row["mesh_levels"]["fine"]["terminal_residual"],
            ]
            colour = colours(index)
            axis.plot([0, 1], values, marker="o", lw=1.5, color=colour)
            label = f"{row['shot']}/{row['slice_index']}  {row['verdict']}"
            axis.annotate(
                label,
                (1, values[1]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=colour,
            )
        axis.axhline(
            FIXED_POINT_CRITERION,
            color="black",
            lw=0.8,
            ls="--",
        )
        axis.set_xticks([0, 1], ["33×33", "65×65"])
        axis.set_xlim(-0.12, 1.8)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Stored-grid resolution")
        axis.grid(axis="y", which="both", alpha=0.18)
    axes[0].set_ylabel("Budget-terminal fixed-point residual")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_mesh_sensitivity(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    banked_control: Path = OUTPUT_RECEIPT,
    output_path: Path = MESH_SENSITIVITY_RECEIPT,
    figure_path: Path = MESH_SENSITIVITY_FIGURE,
) -> dict[str, Any]:
    """Measure the five banked tared stalls on the stored fine grid."""
    configure_dtypes()
    source_stamp = _source_stamp()
    protected_source = json.loads(PROTECTED_SOURCE.read_text())
    protected_before = _verify_protected_artifacts(protected_source)
    banked = json.loads(banked_control.read_text())
    analytic = solovev_null_test()
    if not analytic["passes"]:
        raise RuntimeError("the analytic external-field null no longer passes")
    if banked["aggregate"]["registered_fixed_point_criterion"] != 1.0e-8:
        raise RuntimeError("the banked fixed-point criterion changed")
    if not banked["protected_banked_artifacts"]["all_digests_match"]:
        raise RuntimeError("the banked control did not verify protected artifacts")

    banked_rows = {
        (int(row["reference"]["shot"]), int(row["reference"]["slice_index"])): row
        for row in banked["per_shot"]
    }
    stalled = {
        key: row
        for key, row in banked_rows.items()
        if not row["solve"]["converged_plasma_root"]
    }
    converged = [
        key for key, row in banked_rows.items() if row["solve"]["converged_plasma_root"]
    ]
    if len(banked_rows) != 6 or len(stalled) != 5 or len(converged) != 1:
        raise RuntimeError("the banked tared-control cohort changed")
    if any(
        row["solve"]["iterations"] != parity_slice.NEWTON_STEPS
        for row in stalled.values()
    ):
        raise RuntimeError("a banked stall did not consume the registered budget")

    selection = {
        (int(selected["shot"]), int(selected["slice_index"])): (
            selected,
            qualification,
        )
        for selected, qualification in select_slices_by_shot(bank)
    }
    if set(selection) != set(banked_rows):
        raise RuntimeError("the frozen selection and banked tared cohort differ")

    rows = []
    for key, coarse in stalled.items():
        selected, qualification = selection[key]
        if not qualification["passes"]:
            raise RuntimeError(f"reference {key} lost its input qualification")
        case, context = _mast_case_at_grid_stride(
            store, selected, qualification, MESH_STRIDES["fine"]
        )
        tare = build_tare(
            profile=context["profile"],
            reference_state=case["state"],
            reference_grid=context["reference_flux"],
        )
        fine = _mesh_terminal_row({"case": case, "context": context, "tare": tare})
        fine_spacing = max(
            fine["mesh"]["radial_step_m"], fine["mesh"]["vertical_step_m"]
        )
        coarse_spacing = MESH_STRIDES["coarse"] * fine_spacing
        classification = _classify_mesh_floor(
            float(coarse["solve"]["terminal_residual"]),
            float(fine["solve"]["terminal_residual"]),
            coarse_spacing,
            fine_spacing,
        )
        status = coarse["instrument_controlled_rows"]["lcfs_closed_branch"]["status"]
        stratum = "closed-axis" if status == "scoreable" else "confinement-construction"
        rows.append(
            {
                "shot": key[0],
                "slice_index": key[1],
                "stratum": stratum,
                "registered_split_denominator": (2 if stratum == "closed-axis" else 4),
                "mesh_levels": {
                    "coarse": {
                        "source": str(banked_control),
                        "stored_axis_stride": MESH_STRIDES["coarse"],
                        "realised_cells": 33 * 33,
                        "mesh_spacing_m": coarse_spacing,
                        "registered_fixed_point_criterion": (
                            coarse["solve"]["registered_fixed_point_criterion"]
                        ),
                        "newton_promotion_budget": coarse["solve"]["iterations"],
                        "gmres_iterations_per_promotion": (
                            parity_slice.GMRES_ITERATIONS
                        ),
                        "iterations": coarse["solve"]["iterations"],
                        "terminal_residual": coarse["solve"]["terminal_residual"],
                        "residual_trajectory": coarse["solve"]["residual_trajectory"],
                        "reused_without_rerun": True,
                    },
                    "fine": {
                        **fine["solve"],
                        **fine["mesh"],
                        "mesh_spacing_m": fine_spacing,
                        "closure_sup_difference_wb": fine["tare_closure"][
                            "sup_difference_wb"
                        ],
                        "closure_at_roundoff": fine["tare_closure"]["at_roundoff"],
                        "measured_this_run": True,
                    },
                },
                **classification,
            }
        )

    rows.sort(key=lambda row: (row["stratum"], row["shot"], row["slice_index"]))
    split_rows = {
        stratum: [row for row in rows if row["stratum"] == stratum]
        for stratum in ("closed-axis", "confinement-construction")
    }
    split_verdicts = {
        stratum: _split_verdict(items) for stratum, items in split_rows.items()
    }
    if set(split_verdicts.values()) == {"floor-scales-with-mesh"}:
        branch_verdict = "discretisation-limited"
        branch_statement = (
            "Every topology stratum shows a unanimous residual floor that scales "
            "down with refinement; operator discretisation owns the frontier."
        )
    elif set(split_verdicts.values()) == {"mesh-invariant"}:
        branch_verdict = "map/basin property"
        branch_statement = (
            "Every topology stratum shows a unanimous mesh-invariant floor; the "
            "constraint-by-construction path regains its premise."
        )
    else:
        branch_verdict = "ambiguous"
        branch_statement = (
            "The per-reference mesh verdicts are not unanimous in both topology "
            "strata; the held long-budget classifier is required."
        )

    _plot_mesh_sensitivity(rows, figure_path)
    protected_after = _verify_protected_artifacts(
        json.loads(PROTECTED_SOURCE.read_text())
    )
    receipt = {
        "receipt": {
            "kind": "tared_stall_mesh_sensitivity",
            "status": "complete",
            "source": source_stamp,
            "banked_control": str(banked_control),
            "banked_control_sha256": _sha256(banked_control),
        },
        "control_basis": {
            "ground_truth_validation": analytic,
            "control_caveat": CONTROL_CAVEAT,
            "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
            "banked_solver_budget": {
                "newton_promotions": parity_slice.NEWTON_STEPS,
                "gmres_iterations_per_promotion": parity_slice.GMRES_ITERATIONS,
                "warmup_sweeps": parity_slice.WARMUP_SWEEPS,
                "relaxation": parity_slice.RELAXATION,
                "step_cap": parity_slice.STEP_CAP,
            },
            "mesh_ladder": {
                "coarse": {"stored_axis_stride": MESH_STRIDES["coarse"]},
                "fine": {"stored_axis_stride": MESH_STRIDES["fine"]},
            },
            "per_reference_classifier": {
                "observed_order": (
                    "log(coarse residual / fine residual) / "
                    "log(coarse spacing / fine spacing)"
                ),
                "floor_scales_with_mesh": (
                    f"observed order >= {MINIMUM_MESH_SCALING_ORDER}"
                ),
                "mesh_invariant": (
                    "absolute observed order <= "
                    f"{MAXIMUM_MESH_INVARIANT_ORDER_MAGNITUDE}"
                ),
                "ambiguous": "every other finite result, including worsening",
            },
        },
        "cohort": {
            "banked_reference_count": 6,
            "stalled_reference_count": 5,
            "converged_reference_excluded": {
                "shot": converged[0][0],
                "slice_index": converged[0][1],
            },
            "preregistered_split": {
                "closed-axis": {
                    "banked_reference_count": 2,
                    "stalled_reference_count": len(split_rows["closed-axis"]),
                },
                "confinement-construction": {
                    "banked_reference_count": 4,
                    "stalled_reference_count": len(
                        split_rows["confinement-construction"]
                    ),
                },
            },
            "residuals_pooled": False,
        },
        "per_reference": rows,
        "strata": {
            stratum: {
                "per_reference_verdicts": [
                    {
                        "shot": row["shot"],
                        "slice_index": row["slice_index"],
                        "verdict": row["verdict"],
                    }
                    for row in items
                ],
                "unanimous_verdict": split_verdicts[stratum],
            }
            for stratum, items in split_rows.items()
        },
        "aggregate": {
            "method": "branch only when both non-pooled topology strata are unanimous",
            "branch_verdict": branch_verdict,
            "statement": branch_statement,
            "per_reference_counts": {
                verdict: sum(row["verdict"] == verdict for row in rows)
                for verdict in (
                    "floor-scales-with-mesh",
                    "mesh-invariant",
                    "ambiguous",
                )
            },
        },
        "figure": {
            "path": str(figure_path),
            "src": (
                "/nova/figures/moment-conditioned-basin-entry/"
                "stall-mesh-sensitivity.png"
            ),
            "sha256": _sha256(figure_path),
        },
        "protected_banked_artifacts": {
            "before": protected_before,
            "after": protected_after,
            "all_digests_unchanged": bool(
                protected_before["all_digests_match"]
                and protected_after["all_digests_match"]
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument("--analytic-gate-only", action="store_true")
    parser.add_argument("--mesh-sensitivity", action="store_true")
    arguments = parser.parse_args()
    if arguments.mesh_sensitivity:
        mesh_output = (
            MESH_SENSITIVITY_RECEIPT
            if arguments.output == OUTPUT_RECEIPT
            else arguments.output
        )
        mesh_figure = (
            MESH_SENSITIVITY_FIGURE
            if arguments.figure == OUTPUT_FIGURE
            else arguments.figure
        )
        receipt = run_mesh_sensitivity(
            arguments.store,
            arguments.bank,
            output_path=mesh_output,
            figure_path=mesh_figure,
        )
        counts = receipt["aggregate"]["per_reference_counts"]
        print(
            "TARED_STALL_MESH_SENSITIVITY "
            f"scaling={counts['floor-scales-with-mesh']} "
            f"invariant={counts['mesh-invariant']} "
            f"ambiguous={counts['ambiguous']} "
            f"verdict={receipt['aggregate']['branch_verdict']}"
        )
        return
    if arguments.analytic_gate_only:
        receipt = run_analytic_gate(arguments.output)
        analytic = receipt["analytic_null_gate"]
        print(
            "TARED_PLASMA_SUPPORT_ANALYTIC_NULL "
            f"passes={analytic['passes']} "
            f"fine_sup_wb={analytic['finest_external_sup_error_wb']:.6g} "
            "fine_span_fraction="
            f"{analytic['finest_external_sup_error_fraction_of_analytic_span']:.6g}"
        )
        return
    receipt, _runtime = run_control(
        arguments.store, arguments.bank, arguments.output, arguments.figure
    )
    if not receipt["analytic_null_gate"]["passes"]:
        analytic = receipt["analytic_null_gate"]
        print(
            "TARED_PLASMA_SUPPORT_CONTROL_STOPPED "
            f"fine_sup_wb={analytic['finest_external_sup_error_wb']:.6g} "
            "fine_span_fraction="
            f"{analytic['finest_external_sup_error_fraction_of_analytic_span']:.6g}"
        )
        return
    print(
        "TARED_PLASMA_SUPPORT_CONTROL "
        f"shots={receipt['receipt']['shot_count']} "
        f"sup_wb={receipt['closure_gate']['maximum_sup_difference_wb']:.6g} "
        f"roots={receipt['aggregate']['tared_converged_plasma_roots']} "
        f"verdict={receipt['aggregate']['verdict']}"
    )


if __name__ == "__main__":
    main()
