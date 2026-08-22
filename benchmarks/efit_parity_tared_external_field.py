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

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
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
    _unit_boundary_branches,
)
from nova.equilibrium.conservation import delta_star  # noqa: E402
from nova.equilibrium.convention import (  # noqa: E402
    delta_star_from_current_density,
)
from nova.equilibrium.forward_operator import PrescribedCurrentField  # noqa: E402
from nova.equilibrium.stencil_mesh import CellCurrentMoments  # noqa: E402
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

OUTPUT_DIRECTORY = Path("docs/figures/efit-forward-parity")
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "tared-external-field-solve.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "tared-external-field-solve.png"
PROTECTED_SOURCE = OUTPUT_DIRECTORY / "converged-root-geometry-attribution.json"
BANKED_STORED_FIELD_CLOSURE_WB = 2.22e-15
REFERENCE_HALO_CURRENT_A = 786.396
GAUGE_CONSTANT_WB = 0.0
REPRESENTATIVE_SHOT = 22086


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

    passive_case, passive_profile, passive_policy = _passive_inclusive_case(
        representative["case"], representative["context"]
    )
    del passive_case
    passive_external = np.asarray(passive_profile.operator.external(), dtype=np.float64)
    _plot_external_field_comparison(
        representative["context"]["profile"],
        representative["tare"]["external"],
        passive_external,
        figure_path,
    )
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
            "shot": REPRESENTATIVE_SHOT,
            "modeled_policy": passive_policy["policy"],
            "modeled_stored_circuit_count": passive_policy["stored_circuit_count"],
            "ordinary_active_drive_zeroed": passive_policy[
                "ordinary_active_drive_zeroed_to_avoid_double_counting"
            ],
            "figure": str(figure_path),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    arguments = parser.parse_args()
    receipt, _runtime = measure_tares(arguments.store, arguments.bank, arguments.figure)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        "TARED_EXTERNAL_FIELD_CLOSURE "
        f"shots={receipt['receipt']['shot_count']} "
        f"sup_wb={receipt['closure_gate']['maximum_sup_difference_wb']:.6g} "
        f"passes={receipt['closure_gate']['passes']}"
    )


if __name__ == "__main__":
    main()
