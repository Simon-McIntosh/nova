"""Compare circuit-complete and shipped-only DIII-D forward solves.

The cohort is selected without consulting a solve score.  It contains the first
five lexicographic, polarity-screened shots with a finite diverted median label
and excludes every shot used by the fixed-wiring calibration.  Label flux is
used for the prescribed source functions, branch seed, and scoring only.  The
conductor-current path receives competition magnetics channels and geometry,
so no label-derived current can enter either arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    REGISTERED_RESIDUAL_TOLERANCE,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _eligible_frame,
    _plasma_mask,
    _read,
    _separatrix,
    _solve_registered,
    build_profile,
    canonical_axes,
    contour_separation,
    gauge_metrics,
)
from benchmarks.diiid_state_of_play_figures import boundary_gradient_minimum
from nova.imas.diiid_current import (
    complete_profile_current_adapter,
    shipped_current_at,
)
from nova.imas.diiid_description import (
    PF_ACTIVE_CIRCUIT,
    POLOIDAL_CONDUCTORS,
    dataset_machine_description,
    geometry_digest,
)
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path(
    "docs/figures/coil-circuit-discovery/circuit-driven-forward-validation"
)
CALIBRATION_RECEIPT = Path(
    "docs/figures/coil-circuit-discovery/grid_residual_current_regression_receipt.json"
)
POLARITY_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/current-polarity/"
    "current_polarity_audit_receipt.json"
)
RECEIPT_NAME = "circuit_driven_forward_validation_receipt.json"
FIGURE_NAME = "circuit_driven_forward_validation_overlay.png"
FRAME_COUNT = 5
LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS = 0.0429


@dataclass(frozen=True)
class SelectedFrame:
    """One score-independent out-of-cohort diverted frame."""

    path: Path
    frame: int
    time_ms: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_population(
    path: Path = CALIBRATION_RECEIPT,
) -> tuple[set[tuple[str, int]], set[str], dict[str, Any]]:
    """Return the exact fixed-wiring calibration population and its proof."""

    receipt = json.loads(path.read_text())
    records = receipt["records"]
    exact = {(str(item["shot"]), int(item["frame"])) for item in records}
    shots = {shot for shot, _frame in exact}
    selection = receipt["selection"]
    if len(records) != 60 or selection["frames"] != 60:
        raise RuntimeError("the fixed-wiring calibration bank no longer has 60 frames")
    if len(shots) != 20 or selection["shots"] != 20:
        raise RuntimeError("the fixed-wiring calibration bank no longer has 20 shots")
    return exact, shots, receipt


def polarity_population(path: Path = POLARITY_RECEIPT) -> set[str]:
    """Return the complete banked current-polarity exclusion population."""

    receipt = json.loads(path.read_text())
    census = receipt["full_corpus_census"]
    affected = {str(name) for name in census["affected_shots"]}
    if census["shot_count"] != 7_041 or len(affected) != 603:
        raise RuntimeError("the polarity census no longer carries 7,041/603 shots")
    return affected


def select_frames(
    paths: list[Path],
    calibration_shots: set[str],
    polarity_affected: set[str],
    count: int = FRAME_COUNT,
) -> list[SelectedFrame]:
    """Take one median diverted frame from each first admissible shot."""

    selected: list[SelectedFrame] = []
    for path in sorted(paths):
        if path.name in calibration_shots or path.name in polarity_affected:
            continue
        row = _read(path, _LABEL_COLUMNS)
        frame = _eligible_frame(row)
        if frame is None:
            continue
        selected.append(SelectedFrame(path, frame, float(row["efit_times"][frame])))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} admissible frames were found")
    return selected


def _strict_float(value: Any) -> float | None:
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _distribution(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if finite.size == 0:
        return {"minimum": None, "median": None, "maximum": None, "mean": None}
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _current_receipt(adapter, current_a: np.ndarray) -> dict[str, Any]:
    rows = []
    for declaration, value, uncertainty in zip(
        adapter.resolution.declarations,
        current_a,
        adapter.resolution.prescribed_standard_deviation_a,
        strict=True,
    ):
        relation = declaration.relation
        rows.append(
            {
                "name": declaration.name,
                "value_a_turn": float(value),
                "tier": declaration.tier.value,
                "authority": declaration.provenance,
                "relation_source": None if relation is None else relation.source,
                "relation_scale": None if relation is None else relation.scale,
                "uncertainty_a_turn": float(uncertainty),
                "relation_provenance": (
                    None if relation is None else relation.provenance
                ),
            }
        )
    return {
        "response_order": list(adapter.resolution.names),
        "complete_count": len(current_a),
        "unknown_parameter_count": len(adapter.resolution.unknown_indices),
        "all_finite": bool(np.all(np.isfinite(current_a))),
        "conductors": rows,
        "response": adapter.response_receipt,
    }


def _label_boundary(row: dict[str, Any], frame: int) -> np.ndarray:
    count = int(row["efit_lcfs_n"][frame])
    return np.column_stack(
        (
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        )
    )


def solve_arm(
    profile,
    seed: np.ndarray,
    label: np.ndarray,
    row: dict[str, Any],
    frame: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run and score one prescribed-current arm through the landed solver."""

    equilibrium, termination, _current = _solve_registered(profile, seed)
    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    predicted = np.asarray(
        equilibrium.flux[: profile.lattice.node_count], dtype=float
    ).reshape(profile.lattice.shape)
    topology = equilibrium.topology
    residual = float(equilibrium.fixed_point.residual)
    finite = bool(equilibrium.finite.passed) and bool(np.all(np.isfinite(predicted)))
    diverted = bool(topology.diverted)
    converged = bool(
        finite
        and diverted
        and np.isfinite(residual)
        and residual <= REGISTERED_RESIDUAL_TOLERANCE
    )
    trace = [
        float(value)
        for value in np.asarray(equilibrium.fixed_point.trace, dtype=float)
        if np.isfinite(value)
    ]
    interior = _plasma_mask(row, frame, radius, height)
    r_squared, fractional_rms, gauge, aligned = gauge_metrics(
        label, predicted, interior
    )
    label_boundary = _label_boundary(row, frame)
    predicted_boundary = _separatrix(
        radius,
        height,
        predicted,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    boundary_mean, boundary_maximum = contour_separation(
        predicted_boundary, label_boundary
    )
    full_radius, full_height = canonical_axes(row)
    label_x = boundary_gradient_minimum(
        full_radius,
        full_height,
        np.asarray(row["efit_psirz"][frame], dtype=float),
        label_boundary,
    )
    solved_x = np.asarray(topology.x_point, dtype=float)
    x_separation = float(np.linalg.norm(solved_x - label_x))
    metrics = {
        "interior_r_squared": _strict_float(r_squared),
        "fractional_flux_rms": _strict_float(fractional_rms),
        "additive_gauge_wb": _strict_float(gauge),
        "boundary_mean_separation_m": (
            _strict_float(boundary_mean / 1000.0) if converged else None
        ),
        "boundary_maximum_separation_m": (
            _strict_float(boundary_maximum / 1000.0) if converged else None
        ),
        "x_point_separation_m": _strict_float(x_separation) if converged else None,
        "within_label_representability_ceiling": bool(
            converged
            and np.isfinite(fractional_rms)
            and fractional_rms <= LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS
        ),
    }
    record = {
        "converged": converged,
        "finite": finite,
        "diverted": diverted,
        "fixed_point_relative_residual": _strict_float(residual),
        "residual_tolerance": REGISTERED_RESIDUAL_TOLERANCE,
        "convergence_criterion": (
            "finite and diverted with fixed-point relative residual no greater "
            f"than {REGISTERED_RESIDUAL_TOLERANCE:g}"
        ),
        "residual_trajectory": trace,
        "solver_termination": termination,
        "topology": {
            "axis_rz_m": [float(value) for value in np.asarray(topology.axis)],
            "x_point_rz_m": [float(value) for value in solved_x],
            "label_x_point_rz_m": [float(value) for value in label_x],
        },
        "metrics": metrics,
    }
    fields = {
        "radius": radius,
        "height": height,
        "aligned": aligned,
        "boundary": predicted_boundary,
        "x_point": solved_x,
    }
    return record, fields


def solve_frame(
    selected: SelectedFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the shared labelled source and solve both current arms."""

    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_GEOMETRY_COLUMNS, *_CURRENT_COLUMNS))
    )
    row = _read(selected.path, columns)
    row["_source_path"] = str(selected.path)
    profile, seed, label, wall, reliable, wall_statement = build_profile(
        row, selected.frame, REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )

    current_row = {
        name: row[name]
        for name in (
            *_GEOMETRY_COLUMNS,
            *_CURRENT_COLUMNS,
            "efit_grid_R",
            "efit_grid_Z",
        )
    }
    current_row["_source_path"] = str(selected.path)
    description = dataset_machine_description(
        current_row, source_row=str(selected.path)
    ).physical
    shipped = shipped_current_at(
        current_row,
        description,
        POLOIDAL_CONDUCTORS,
        selected.time_ms,
    )
    shipped_vector = np.asarray([shipped[name] for name in POLOIDAL_CONDUCTORS])
    np.testing.assert_allclose(
        np.asarray(profile.operator.external_current),
        shipped_vector,
        rtol=0.0,
        atol=1.0e-9,
    )
    adapter = complete_profile_current_adapter(
        profile,
        shipped_names=POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped,
        use_circuit=True,
    )
    circuit_vector = adapter.resolution.current(())
    if len(circuit_vector) != 24 or adapter.resolution.unknown_names:
        raise RuntimeError("the circuit did not prescribe all 24 conductor currents")

    shipped_arm, shipped_fields = solve_arm(profile, seed, label, row, selected.frame)
    circuit_arm, circuit_fields = solve_arm(
        adapter.profile, seed, label, row, selected.frame
    )
    record = {
        "shot": selected.path.name,
        "frame": selected.frame,
        "time_ms": selected.time_ms,
        "source_parquet": str(selected.path),
        "source_parquet_sha256": _sha256(selected.path),
        "geometry_digest": geometry_digest(row),
        "qualification": {
            "finite_diverted_label": True,
            "polarity_screened": True,
            "calibration_frame_member": False,
            "calibration_shot_member": False,
            "reliable_flux_function_surfaces": reliable,
        },
        "source_and_seed": {
            "profile_functions": "extracted from the EFIT label",
            "branch_seed": "EFIT label map in Nova convention",
            "pseudo_wall": wall_statement,
            "pseudo_wall_expansion": REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        },
        "circuit_driven": {
            "current_receipt": _current_receipt(adapter, circuit_vector),
            "solve": circuit_arm,
        },
        "shipped_only": {
            "current_receipt": {
                "response_order": list(POLOIDAL_CONDUCTORS),
                "complete_count": len(shipped_vector),
                "unknown_parameter_count": 0,
                "all_finite": bool(np.all(np.isfinite(shipped_vector))),
                "conductors": [
                    {
                        "name": name,
                        "value_a_turn": float(shipped[name]),
                        "authority": f"same-frame shipped magnetics_{name} channel",
                    }
                    for name in POLOIDAL_CONDUCTORS
                ],
            },
            "solve": shipped_arm,
        },
    }
    fields = {
        "label": label,
        "label_boundary": _label_boundary(row, selected.frame),
        "wall": wall,
        "shipped": shipped_fields,
        "circuit": circuit_fields,
    }
    return record, fields


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the paired aggregate without hiding non-converged frames."""

    def arm(name: str) -> dict[str, Any]:
        solves = [record[name]["solve"] for record in records]
        converged = [solve for solve in solves if solve["converged"]]
        rms = [solve["metrics"]["fractional_flux_rms"] for solve in converged]
        return {
            "converged_frames": len(converged),
            "nonconverged_frames": len(solves) - len(converged),
            "fixed_point_relative_residual": _distribution(
                [solve["fixed_point_relative_residual"] for solve in solves]
            ),
            "fractional_flux_rms_on_converged_frames": _distribution(rms),
            "boundary_mean_separation_m_on_converged_frames": _distribution(
                [solve["metrics"]["boundary_mean_separation_m"] for solve in converged]
            ),
            "x_point_separation_m_on_converged_frames": _distribution(
                [solve["metrics"]["x_point_separation_m"] for solve in converged]
            ),
            "frames_within_label_representability_ceiling": sum(
                solve["metrics"]["within_label_representability_ceiling"]
                for solve in solves
            ),
        }

    circuit = arm("circuit_driven")
    shipped = arm("shipped_only")
    paired_rms = []
    for record in records:
        circuit_solve = record["circuit_driven"]["solve"]
        shipped_solve = record["shipped_only"]["solve"]
        if circuit_solve["converged"] and shipped_solve["converged"]:
            paired_rms.append(
                {
                    "shot": record["shot"],
                    "frame": record["frame"],
                    "circuit_minus_shipped_fractional_flux_rms": (
                        circuit_solve["metrics"]["fractional_flux_rms"]
                        - shipped_solve["metrics"]["fractional_flux_rms"]
                    ),
                }
            )
    return {
        "frame_count": len(records),
        "shot_count": len({record["shot"] for record in records}),
        "circuit_driven": circuit,
        "shipped_only": shipped,
        "paired_converged_fractional_flux_rms_deltas": paired_rms,
        "circuit_improves_fractional_flux_rms_on_paired_converged_frames": sum(
            item["circuit_minus_shipped_fractional_flux_rms"] < 0.0
            for item in paired_rms
        ),
    }


def render_overlay(
    records: list[dict[str, Any]], fields: list[dict[str, Any]], path: Path
) -> None:
    """Plot the label and both terminal forward maps for every frame."""

    figure, axes = plt.subplots(
        len(records), 3, figsize=(11.5, 2.6 * len(records)), constrained_layout=True
    )
    for row_axes, record, frame_fields in zip(axes, records, fields, strict=True):
        label = frame_fields["label"]
        radius = frame_fields["shipped"]["radius"]
        height = frame_fields["shipped"]["height"]
        finite = np.concatenate(
            (
                label[np.isfinite(label)],
                frame_fields["shipped"]["aligned"][
                    np.isfinite(frame_fields["shipped"]["aligned"])
                ],
                frame_fields["circuit"]["aligned"][
                    np.isfinite(frame_fields["circuit"]["aligned"])
                ],
            )
        )
        low, high = np.quantile(finite, [0.01, 0.99])
        panels = (
            (label, "EFIT label", None, None),
            (
                frame_fields["shipped"]["aligned"],
                "Shipped 19 only",
                frame_fields["shipped"]["boundary"],
                frame_fields["shipped"]["x_point"],
            ),
            (
                frame_fields["circuit"]["aligned"],
                "Circuit-complete 24",
                frame_fields["circuit"]["boundary"],
                frame_fields["circuit"]["x_point"],
            ),
        )
        for axis, (flux, title, boundary, x_point) in zip(
            row_axes, panels, strict=True
        ):
            image = axis.pcolormesh(
                radius,
                height,
                flux.T,
                shading="auto",
                cmap="viridis",
                vmin=low,
                vmax=high,
            )
            label_boundary = frame_fields["label_boundary"]
            axis.plot(
                label_boundary[:, 0],
                label_boundary[:, 1],
                color="white",
                linewidth=1.0,
                label="label LCFS",
            )
            if boundary is not None and len(boundary):
                axis.plot(
                    boundary[:, 0],
                    boundary[:, 1],
                    color="tab:red",
                    linestyle="--",
                    linewidth=0.9,
                    label="solve separatrix",
                )
            if x_point is not None and np.all(np.isfinite(x_point)):
                axis.plot(*x_point, marker="x", color="tab:red", markersize=5)
            axis.set_aspect("equal")
            axis.set_xlabel("R [m]")
            axis.set_ylabel("Z [m]")
            axis.set_title(title, fontsize=9)
            figure.colorbar(image, ax=axis, label="total poloidal flux [Wb]")
        row_axes[0].text(
            0.02,
            0.98,
            f"{Path(record['shot']).stem[-8:]} frame {record['frame']}",
            transform=row_axes[0].transAxes,
            va="top",
            fontsize=7,
            color="white",
        )
        for axis, name in zip(
            row_axes[1:], ("shipped_only", "circuit_driven"), strict=True
        ):
            solve = record[name]["solve"]
            axis.text(
                0.02,
                0.02,
                f"res={solve['fixed_point_relative_residual']:.3e}\n"
                f"converged={solve['converged']}",
                transform=axis.transAxes,
                fontsize=7,
                color="white",
                va="bottom",
            )
    axes[0, 0].legend(loc="lower left", fontsize=6)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path, frame_count: int = FRAME_COUNT) -> dict[str, Any]:
    """Execute the out-of-cohort paired forward validation."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    calibration_frames, calibration_shots, calibration = calibration_population()
    affected = polarity_population()
    selected = select_frames(
        list(data.glob("*.parquet")), calibration_shots, affected, frame_count
    )
    selected_pairs = {(item.path.name, item.frame) for item in selected}
    if selected_pairs & calibration_frames:
        raise RuntimeError("selected frame overlaps the fixed-wiring calibration bank")
    if {item.path.name for item in selected} & calibration_shots:
        raise RuntimeError("selected shot overlaps the fixed-wiring calibration bank")
    if {item.path.name for item in selected} & affected:
        raise RuntimeError("selected shot overlaps the polarity exclusion population")

    records = []
    fields = []
    for selected_frame in selected:
        record, frame_fields = solve_frame(selected_frame)
        records.append(record)
        fields.append(frame_fields)
    figure_path = output / FIGURE_NAME
    render_overlay(records, fields, figure_path)
    aggregate = summarize(records)
    circuit_summary = aggregate["circuit_driven"]
    shipped_summary = aggregate["shipped_only"]
    receipt = {
        "measurement": (
            "out-of-cohort DIII-D GS forward validation of the fixed-wiring "
            "pf_active circuit"
        ),
        "selection": {
            "rule": (
                "one median eligible diverted frame from each of the first five "
                "lexicographic shots absent from both the polarity population "
                "and every fixed-wiring calibration shot; no solve score consulted"
            ),
            "selected_frames": [
                {
                    "shot": item.path.name,
                    "frame": item.frame,
                    "time_ms": item.time_ms,
                }
                for item in selected
            ],
            "selected_frame_count": len(selected),
            "selected_shot_count": len({item.path.name for item in selected}),
            "all_finite_diverted": True,
            "all_polarity_screened": True,
            "calibration_bank": {
                "receipt": str(CALIBRATION_RECEIPT),
                "sha256": _sha256(CALIBRATION_RECEIPT),
                "frame_count": len(calibration_frames),
                "shot_count": len(calibration_shots),
                "selection_receipt_frames": calibration["selection"]["frames"],
                "exact_selected_pair_intersection": [],
                "selected_shot_intersection": [],
                "strictly_outside": True,
            },
            "polarity_bank": {
                "receipt": str(POLARITY_RECEIPT),
                "sha256": _sha256(POLARITY_RECEIPT),
                "affected_shot_count": len(affected),
                "selected_intersection": [],
            },
        },
        "arms": {
            "circuit_driven": (
                "24 response columns and currents: 19 shipped competition "
                "channels plus 5 fixed-wiring pf_active circuit drives"
            ),
            "shipped_only": "the original 19 shipped response columns and currents",
            "shared": (
                "same prescribed EFIT-derived profile functions, label branch seed, "
                "grid, pseudo-wall and landed Newton-Krylov solve policy"
            ),
        },
        "current_path_audit": {
            "competition_current_channels": [
                f"magnetics_{name}" for name in POLOIDAL_CONDUCTORS
            ],
            "circuit_source_channel": "magnetics_ECOILA",
            "label_derived_current_reads": 0,
            "per_frame_current_fits": 0,
            "least_squares_updates": 0,
            "unknown_current_parameters": 0,
            "label_use": (
                "prescribed source functions, branch seed and scoring only; current "
                "extraction receives a row containing geometry and magnetics fields"
            ),
        },
        "solver": {
            "entry_point": "nova.equilibrium.forward.ForwardProfile.solve",
            "route": "newton_krylov",
            "relative_residual_tolerance": REGISTERED_RESIDUAL_TOLERANCE,
        },
        "comparison": {
            "flux_gauge": "one additive constant over the labelled LCFS interior",
            "label_representability_ceiling_fractional_rms": (
                LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS
            ),
            "ceiling_scope": (
                "a measured label representability floor, not a guarantee that the "
                "free-boundary solver or circuit arm reaches it"
            ),
            "boundary_and_x_point_metrics": "reported only for converged solves",
        },
        "aggregate": aggregate,
        "verdict": {
            "circuit_converged_frames": circuit_summary["converged_frames"],
            "shipped_only_converged_frames": shipped_summary["converged_frames"],
            "circuit_frames_within_representability_ceiling": circuit_summary[
                "frames_within_label_representability_ceiling"
            ],
            "recovery_demonstrated": bool(
                circuit_summary["converged_frames"] == len(records)
                and circuit_summary["frames_within_label_representability_ceiling"]
                == len(records)
            ),
            "statement": (
                "Recovery is demonstrated only if every selected circuit-driven "
                "solve converges and falls within the 4.29% fractional-RMS label "
                "representability ceiling; the measured counts above are authoritative."
            ),
        },
        "caveats": {
            "label_representability": (
                "The EFIT label has a 4.29% fractional-RMS representability ceiling "
                "under the landed comparison, so smaller discrepancies cannot be "
                "attributed uniquely to conductor currents."
            ),
            "e89_systematic": {
                "name": "end_loop_bundle_normalisation",
                "E89UP_effective_gain_minus_integer_wiring": 0.04569475694961733,
                "E89DN_effective_gain_minus_integer_wiring": 0.04562407643237165,
                "statement": (
                    "The E89 drives retain the measured shared normalisation "
                    "systematic; this study does not reinterpret it as exact wiring."
                ),
            },
            "circuit_closure": (
                "The calibration receipt found only 1 of 60 frames passed its "
                "post-fit closure rule, so label flux retains non-conductor content."
            ),
        },
        "pf_active_circuit": PF_ACTIVE_CIRCUIT.as_record(),
        "frames": records,
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "overlay_figure": str(figure_path),
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output, arguments.frames)
    print(json.dumps(receipt["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
