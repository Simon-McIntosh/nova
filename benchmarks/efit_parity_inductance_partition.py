"""Partition constrained current and rescore moments on separatrix support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.constants import mu_0

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    select_slices_by_shot,
)
from benchmarks.efit_parity_boundary_volume import (  # noqa: E402
    _polygon_measure,
    _verify_protected_artifacts,
)
from benchmarks.efit_parity_moment_definitions import (  # noqa: E402
    _relative_error as _relative_deviation,
)
from benchmarks.efit_parity_root_geometry import (  # noqa: E402
    _assert_reference,
    _boundary_branch_status,
    _sha256_array,
    _unit_boundary_branches,
)
from nova.equilibrium import fixed_point  # noqa: E402
from nova.equilibrium.conservation import poloidal_field  # noqa: E402
from nova.equilibrium.observation import declared_pressure  # noqa: E402
from nova.equilibrium.topology import TopologyClass  # noqa: E402
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

OUTPUT_DIRECTORY = Path("docs/figures/efit-forward-parity")
SOURCE_RECEIPT = OUTPUT_DIRECTORY / "converged-root-geometry-attribution.json"
MOMENT_RECEIPT = OUTPUT_DIRECTORY / "moment-definition-rescore.json"
BOUNDARY_RECEIPT = OUTPUT_DIRECTORY / "boundary-enclosed-volume-reconciliation.json"
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "inductance-deficit-partition.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "inductance-deficit-partition.png"
TARGET_SHOT = 22086
EXPECTED_SLICE_INDEX = 43
EXPECTED_TOTAL_CURRENT_A = 933034.875
EXPECTED_OUT_OF_CORE_CURRENT_A = 516076.6208740445
BANKED_CORE_CELL_COUNT = 95
EXPECTED_SOLVED_BOUNDARY_AREA_M2 = 1.7718309749521555
EXPECTED_SOLVED_BOUNDARY_VOLUME_M3 = 8.518524547051108
EXPECTED_MATCHED_SUPPORT_LI_DEVIATION = -0.35695307611867966
EXPECTED_MATCHED_SUPPORT_BETA_DEVIATION = 0.05818098840723218


def _source_digests(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _terminal_observation(source: dict[str, Any], store: Path, bank: Path):
    """Reconstruct the single banked terminal equilibrium without solving."""
    selected = next(
        item
        for item in select_slices_by_shot(bank)
        if int(item[0]["shot"]) == TARGET_SHOT
    )
    case, context = _mast_case_from_selection(store, *selected)
    _assert_reference(case["reference"])
    profile = context["profile"]
    terminal = source["constrained_root"]["terminal_state"]
    state_array = np.asarray(terminal["values"], dtype=np.float64)
    if _sha256_array(state_array) != terminal["sha256"]:
        raise RuntimeError("serialized terminal-state digest does not match")
    state = jnp.asarray(state_array)
    history = fixed_point.FixedPointResult(
        state=state,
        residual=jnp.asarray(source["constrained_root"]["terminal_residual"]),
        trace=jnp.asarray([], dtype=state.dtype),
    )
    target_current = float(source["constrained_root"]["target_current_a"])
    equilibrium = profile._receipt(
        state,
        history,
        TopologyClass.DIVERTED,
        target_current,
    )
    topology = equilibrium.topology
    solved_flux = state_array[: profile.lattice.node_count].reshape(
        profile.lattice.shape
    )
    branches = _unit_boundary_branches(
        profile.lattice.radius,
        profile.lattice.height,
        solved_flux,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    contour, display, boundary_status = _boundary_branch_status(
        branches, np.asarray(topology.axis, dtype=np.float64)
    )
    return profile, equilibrium, contour, state_array, boundary_status, display


def _cell_membership(profile, contour: np.ndarray) -> np.ndarray:
    """Return the cell centroids enclosed by one contour; the sole membership test."""
    coordinates = np.asarray(profile.lattice.coordinate, dtype=np.float64)
    return np.asarray(
        MplPath(contour, closed=True).contains_points(coordinates, radius=1.0e-12),
        dtype=bool,
    )


def _cell_partition(
    profile,
    equilibrium,
    contour: np.ndarray,
    *,
    solved_branch: bool = True,
) -> dict[str, Any]:
    """Partition every mesh cell against one contour.

    ``solved_branch`` adds the invariants that only a solved closed branch can
    carry: every confined-core centroid lies inside it, and the currents close
    on the bank. A stored contour is a different region, so those claims are
    not made of it.
    """
    coordinates = np.asarray(profile.lattice.coordinate, dtype=np.float64)
    inside = _cell_membership(profile, contour)
    core = np.asarray(equilibrium.domains.core, dtype=bool)
    core_cell_count = int(np.count_nonzero(core))
    if core_cell_count == 0:
        raise RuntimeError("the confined-core mask is empty")
    if solved_branch and np.any(core & ~inside):
        raise RuntimeError("a confined-core centroid lies outside the closed branch")
    interior_noncore = inside & ~core
    exterior = ~(core | inside)
    if not np.all(core | interior_noncore | exterior):
        raise RuntimeError("the three support sets do not cover the mesh")

    cell_current = np.asarray(equilibrium.cell_current, dtype=np.float64)
    records: dict[str, dict[str, float | int]] = {}
    for name, mask in (
        ("confined_core", core),
        ("inside_closed_branch_outside_core", interior_noncore),
        ("outside_closed_branch", exterior),
    ):
        records[name] = {
            "cell_count": int(np.count_nonzero(mask)),
            "cell_current_a": float(np.sum(cell_current[mask])),
        }
    total = float(np.sum(cell_current))
    closure = sum(float(record["cell_current_a"]) for record in records.values())
    out_of_core = float(
        records["inside_closed_branch_outside_core"]["cell_current_a"]
    ) + float(records["outside_closed_branch"]["cell_current_a"])
    if solved_branch:
        if not np.isclose(total, EXPECTED_TOTAL_CURRENT_A, rtol=0.0, atol=2.0e-9):
            raise RuntimeError("the reconstructed total current differs from the pin")
        if not np.isclose(
            out_of_core, EXPECTED_OUT_OF_CORE_CURRENT_A, rtol=0.0, atol=2.0e-9
        ):
            raise RuntimeError("the out-of-core current differs from the bank")
    return {
        "coordinates": coordinates,
        "inside": inside,
        "core": core,
        "interior_noncore": interior_noncore,
        "exterior": exterior,
        "cell_current": cell_current,
        "records": records,
        "summed_current_a": closure,
        "pinned_current_a": EXPECTED_TOTAL_CURRENT_A,
        "closure_residual_a": (closure - EXPECTED_TOTAL_CURRENT_A)
        if solved_branch
        else None,
        "out_of_core_current_a": out_of_core,
        "core_cell_count": core_cell_count,
        "banked_core_cell_count": BANKED_CORE_CELL_COUNT,
        "core_cell_count_matches_bank": core_cell_count == BANKED_CORE_CELL_COUNT,
    }


def _core_support_divergence(
    core_cell_count: int, core_current: float, moment: dict[str, Any]
) -> dict[str, Any]:
    """Report the live-versus-banked core support difference against the bank."""
    banked = moment["current_supports"]
    banked_current = float(banked["confined_core"])
    return {
        "live_core_cell_count": int(core_cell_count),
        "banked_core_cell_count": BANKED_CORE_CELL_COUNT,
        "cell_counts_differ": int(core_cell_count) != BANKED_CORE_CELL_COUNT,
        "live_core_current_a": float(core_current),
        "banked_core_current_a": banked_current,
        "current_agreement_atol_a": 2.0e-9,
        "current_agrees": bool(
            np.isclose(core_current, banked_current, rtol=0.0, atol=2.0e-9)
        ),
    }


def _boundary_moments(
    profile,
    equilibrium,
    state_array: np.ndarray,
    partition: dict[str, Any],
    moment: dict[str, Any],
) -> dict[str, Any]:
    inside = partition["inside"]
    radius = np.asarray(profile.lattice.node_radius, dtype=np.float64)
    area = np.asarray(profile.operator.area, dtype=np.float64)
    volume = 2.0 * np.pi * radius * area
    radial, vertical = poloidal_field(
        profile.lattice, jnp.asarray(state_array[: profile.lattice.node_count])
    )
    field_squared = (
        np.asarray(radial, dtype=np.float64) ** 2
        + np.asarray(vertical, dtype=np.float64) ** 2
    )
    pressure = np.asarray(
        declared_pressure(
            profile.operator.source,
            equilibrium.domains,
            jnp.asarray(radius),
            equilibrium.topology.flux_span,
        ),
        dtype=np.float64,
    )
    enclosed_volume = float(np.sum(volume[inside]))
    enclosed_major_radius = float(
        np.sum(radius[inside] * volume[inside]) / enclosed_volume
    )
    enclosed_current = float(np.sum(partition["cell_current"][inside]))
    pressure_integral = float(np.sum(pressure[inside] * volume[inside]))
    field_integral = float(np.sum(field_squared[inside] * volume[inside]))

    reference_row = next(
        row
        for row in moment["four_by_two_rescore"]["rows"]
        if row["definition"] == "reference_boundary_field"
        and row["side"] == "reference"
    )
    reference_beta = reference_row["poloidal_beta"]["all_domain_constrained"]
    reference_li = reference_row["internal_inductance"]["current_independent"]
    reference_current = float(reference_beta["current_a"])
    reference_beta_denominator = (
        float(reference_beta["denominator_t2_m3"])
        * (enclosed_current / reference_current) ** 2
    )
    reference_li_denominator = float(reference_li["denominator_t2_m3"])
    solved_beta = 2.0 * mu_0 * pressure_integral / reference_beta_denominator
    solved_li = field_integral / reference_li_denominator
    beta_deviation = _relative_deviation(solved_beta, float(reference_beta["value"]))
    li_deviation = _relative_deviation(solved_li, float(reference_li["value"]))
    return {
        "support": {
            "cell_count": int(np.count_nonzero(inside)),
            "toroidal_volume_m3": enclosed_volume,
            "volume_weighted_major_radius_m": enclosed_major_radius,
            "current_integral_a": enclosed_current,
        },
        "integrals": {
            "pressure_volume_integral_pa_m3": pressure_integral,
            "poloidal_field_squared_volume_integral_t2_m3": field_integral,
        },
        "reference_boundary_field_definition": {
            "poloidal_beta": (
                "2*mu0*integral(p dV) divided by the reference squared-mean "
                "boundary-field denominator, rescaled to the enclosed current"
            ),
            "internal_inductance": (
                "integral(Bp^2 dV) divided by the published mean-squared "
                "boundary-field denominator; current is not used in this definition"
            ),
        },
        "poloidal_beta": {
            "solved": solved_beta,
            "published_reference": float(reference_beta["value"]),
            "denominator_t2_m3": reference_beta_denominator,
            "signed_relative_deviation": beta_deviation,
            "matched_support_estimate_signed_relative_deviation": (
                EXPECTED_MATCHED_SUPPORT_BETA_DEVIATION
            ),
            "change_from_matched_support_estimate": (
                beta_deviation - EXPECTED_MATCHED_SUPPORT_BETA_DEVIATION
            ),
        },
        "internal_inductance": {
            "solved": solved_li,
            "published_reference": float(reference_li["value"]),
            "denominator_t2_m3": reference_li_denominator,
            "signed_relative_deviation": li_deviation,
            "matched_support_estimate_signed_relative_deviation": (
                EXPECTED_MATCHED_SUPPORT_LI_DEVIATION
            ),
            "change_from_matched_support_estimate": (
                li_deviation - EXPECTED_MATCHED_SUPPORT_LI_DEVIATION
            ),
        },
    }


def _plot_partition(
    contour: np.ndarray, partition: dict[str, Any] | None, path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 6.2), constrained_layout=True)
    if partition is None:
        axis.plot(
            contour[:, 0],
            contour[:, 1],
            color="#b55d00",
            lw=1.5,
            label="open unit contour",
        )
    else:
        coordinates = partition["coordinates"]
        for mask, label, color, size in (
            (partition["exterior"], "outside closed branch", "#b8b8b8", 8),
            (
                partition["interior_noncore"],
                "inside branch, outside core",
                "#e68613",
                18,
            ),
            (partition["core"], "confined core", "#2f6f9f", 22),
        ):
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=size,
                color=color,
                linewidths=0.0,
                label=label,
            )
        closed = np.vstack((contour, contour[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="#202020", lw=1.5)
    axis.set_title("MAST 22086/43 current-support partition")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_aspect("equal")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _open_partition_receipt(
    source_path: Path,
    moment_path: Path,
    boundary_path: Path,
    integrity: dict[str, Any],
    moment: dict[str, Any],
    equilibrium,
    boundary_status: dict[str, Any],
) -> dict[str, Any]:
    """Report current facts independent of a closed boundary."""
    core = np.asarray(equilibrium.domains.core, dtype=bool)
    current = np.asarray(equilibrium.cell_current, dtype=np.float64)
    total = float(np.sum(current))
    core_current = float(np.sum(current[core]))
    if not np.isclose(total, EXPECTED_TOTAL_CURRENT_A, rtol=0.0, atol=2.0e-9):
        raise RuntimeError("the reconstructed total current differs from the pin")
    reason = boundary_status["reason"]
    undefined = {"status": "undefined_open_boundary", "reason": reason}
    return {
        "receipt": {
            "kind": "inductance_deficit_partition",
            "shot": TARGET_SHOT,
            "slice_index": EXPECTED_SLICE_INDEX,
            "source_receipts": [str(source_path), str(moment_path), str(boundary_path)],
            "source_receipt_sha256": _source_digests(
                (source_path, moment_path, boundary_path)
            ),
        },
        "execution_contract": {
            "nonlinear_solve_calls": 0,
            "equilibrium_observations_from_serialised_state": 1,
            "cell_classification": "undefined: no closed axis-enclosing boundary",
        },
        "boundary_status": boundary_status,
        "separatrix": {
            "status": "open",
            "reason": reason,
            "poloidal_area_m2": None,
            "exact_solid_of_revolution_m3": None,
        },
        "current_partition_table": {
            "confined_core": {
                "cell_count": int(np.count_nonzero(core)),
                "cell_current_a": core_current,
            },
            "banked_confined_core": {
                "cell_count": BANKED_CORE_CELL_COUNT,
                "cell_current_a": moment["current_supports"]["confined_core"],
            },
            "inside_closed_branch_outside_core": {
                **undefined,
                "cell_count": None,
                "cell_current_a": None,
            },
            "outside_closed_branch": {
                **undefined,
                "cell_count": None,
                "cell_current_a": None,
            },
        },
        "core_support_divergence": _core_support_divergence(
            int(np.count_nonzero(core)), core_current, moment
        ),
        "current_closure": {
            "direct_total_current_a": total,
            "direct_out_of_core_current_a": total - core_current,
            "pinned_total_current_a": EXPECTED_TOTAL_CURRENT_A,
            "banked_out_of_core_current_a": EXPECTED_OUT_OF_CORE_CURRENT_A,
            "summed_partition_current_a": None,
            "partitioned_out_of_core_current_a": None,
            "closure_residual_a": None,
        },
        "boundary_enclosed_moment_rescore": undefined,
        "discriminator_result": {
            **undefined,
            "current_location_supported_reading": None,
            "inside_share_of_out_of_core_current": None,
            "internal_inductance_mechanism_verdict": None,
        },
        "protected_banked_artifacts": integrity,
        "figure": str(OUTPUT_FIGURE),
    }


def measure(
    source_path: Path = SOURCE_RECEIPT,
    moment_path: Path = MOMENT_RECEIPT,
    boundary_path: Path = BOUNDARY_RECEIPT,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the zero-solve current partition and spatial moment rescore."""
    configure_dtypes()
    source = json.loads(source_path.read_text())
    moment = json.loads(moment_path.read_text())
    boundary = json.loads(boundary_path.read_text())
    integrity = _verify_protected_artifacts(source)
    profile, equilibrium, contour, state_array, boundary_status, display = (
        _terminal_observation(source, store, bank)
    )
    if contour is None:
        return (
            _open_partition_receipt(
                source_path,
                moment_path,
                boundary_path,
                integrity,
                moment,
                equilibrium,
                boundary_status,
            ),
            {
                "contour": display,
                "partition": None,
                "core": np.asarray(equilibrium.domains.core, dtype=bool),
                "cell_current": np.asarray(equilibrium.cell_current, dtype=np.float64),
            },
        )
    contour_measure = _polygon_measure(contour)
    if not np.isclose(
        contour_measure["poloidal_area_m2"],
        EXPECTED_SOLVED_BOUNDARY_AREA_M2,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise RuntimeError("the reconstructed separatrix area differs from the bank")
    if not np.isclose(
        contour_measure["exact_solid_of_revolution_m3"],
        EXPECTED_SOLVED_BOUNDARY_VOLUME_M3,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise RuntimeError("the reconstructed separatrix volume differs from the bank")
    banked_solved = boundary["contour_comparison_table"][0]
    if banked_solved["contour"] != "solved_closed_axis_enclosing_branch":
        raise RuntimeError("the banked solved-contour row drifted")

    partition = _cell_partition(profile, equilibrium, contour)
    moments = _boundary_moments(profile, equilibrium, state_array, partition, moment)
    inside_out_of_core = float(
        partition["records"]["inside_closed_branch_outside_core"]["cell_current_a"]
    )
    outside = float(partition["records"]["outside_closed_branch"]["cell_current_a"])
    supported_reading = (
        "MASKING_ARTIFACT" if inside_out_of_core > outside else "PHYSICS_EDGE_CURRENT"
    )
    deciding_fraction = inside_out_of_core / partition["out_of_core_current_a"]
    receipt = {
        "receipt": {
            "kind": "inductance_deficit_partition",
            "shot": TARGET_SHOT,
            "slice_index": EXPECTED_SLICE_INDEX,
            "source_receipts": [str(source_path), str(moment_path), str(boundary_path)],
            "source_receipt_sha256": _source_digests(
                (source_path, moment_path, boundary_path)
            ),
        },
        "discriminator_declared_before_measurement": {
            "inside_closed_branch_outside_core": "MASKING_ARTIFACT",
            "outside_closed_branch": "PHYSICS_EDGE_CURRENT",
            "decision_rule": (
                "The larger share of the banked out-of-core current determines "
                "which reading the partition supports."
            ),
        },
        "execution_contract": {
            "nonlinear_solve_calls": 0,
            "equilibrium_observations_from_serialised_state": 1,
            "cell_classification": (
                "point-in-polygon test of every mesh cell centroid against the "
                "closed axis-enclosing separatrix branch"
            ),
        },
        "separatrix": contour_measure,
        "current_partition_table": partition["records"],
        "core_support_divergence": _core_support_divergence(
            partition["core_cell_count"],
            partition["records"]["confined_core"]["cell_current_a"],
            moment,
        ),
        "current_closure": {
            "summed_partition_current_a": partition["summed_current_a"],
            "pinned_total_current_a": partition["pinned_current_a"],
            "closure_residual_a": partition["closure_residual_a"],
            "banked_out_of_core_current_a": EXPECTED_OUT_OF_CORE_CURRENT_A,
            "partitioned_out_of_core_current_a": partition["out_of_core_current_a"],
        },
        "boundary_enclosed_moment_rescore": moments,
        "discriminator_result": {
            "current_location_supported_reading": supported_reading,
            "inside_share_of_out_of_core_current": deciding_fraction,
            "outside_share_of_out_of_core_current": 1.0 - deciding_fraction,
            "deciding_figure_a": max(inside_out_of_core, outside),
            "internal_inductance_mechanism_verdict": (
                "MASKING_CURRENT_CONFIRMED_BUT_FIELD_DEFICIT_SURVIVES"
                if supported_reading == "MASKING_ARTIFACT"
                and moments["internal_inductance"]["signed_relative_deviation"] < 0.0
                else supported_reading
            ),
            "reading": (
                "The current omitted by the 95-cell core mask lies predominantly "
                "inside the closed separatrix, so the current partition supports "
                "the masking reading. The spatial field integral nevertheless "
                "leaves the inductance deficit standing, so masking does not by "
                "itself explain that field deficit."
                if supported_reading == "MASKING_ARTIFACT"
                else "The current omitted by the 95-cell core mask lies predominantly "
                "outside the closed separatrix, so the surviving deficit reflects "
                "a real edge and scrape-off-layer current distribution."
            ),
        },
        "protected_banked_artifacts": integrity,
        "figure": str(OUTPUT_FIGURE),
    }
    return receipt, {"contour": contour, "partition": partition}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    arguments = parser.parse_args()
    receipt, fields = measure()
    _plot_partition(fields["contour"], fields["partition"], arguments.figure)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
