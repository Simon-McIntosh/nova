"""Measure solve-free tensor-spline compression of banked flux fields.

The receipt covers both protected equilibrium banks named by Nova's equilibrium
guidance.  Each serialized root and its same-carrier reference field is fitted
on the banked plasma-cell coordinates.  Errors are always separated into the
closed-flux region, a fixed normalized-flux band about the separatrix, and the
scrape-off layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import hstack
from scipy.sparse.linalg import lsqr


OUTPUT = Path("docs/figures/coefficient-space-newton/representation-degree.json")
STORED_ROOT_DIRECTORY = Path("scripts/root_gate_attribution")
ORACLE_ROOT_DIRECTORY = Path("scripts/oracle_rebaseline")
PRODUCTION_RECEIPT = Path(
    "docs/figures/topology-preserving-continuation/conditioning-repair.json"
)
REFERENCE_MODULE = Path("tests/test_equilibrium_forward_reference.py")
KNOT_COUNT_LADDER = (4, 6, 8, 10, 12, 14, 16, 18, 20)
SPLINE_DEGREE = 3
SEPARATRIX_HALF_WIDTH_PSI_N = 0.05
REGION_NAMES = ("closed_flux", "separatrix_band", "scrape_off_layer")


@dataclass(frozen=True)
class FieldTarget:
    """One banked field sampled at its plasma-cell coordinates."""

    identity: str
    kind: str
    carrier: str
    coordinate: np.ndarray
    flux: np.ndarray
    psi_norm: np.ndarray
    source: dict[str, Any]
    banked_terminal_residual: float | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_reference_targets() -> list[FieldTarget]:
    reference = _load_module(REFERENCE_MODULE, "flux_representation_reference")
    reference.configure_dtypes()
    case = reference.require_reference()
    scorecard_path = STORED_ROOT_DIRECTORY / "results.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    targets: list[FieldTarget] = []
    for carrier in ("coarse", "fine"):
        bank_path = STORED_ROOT_DIRECTORY / f"{carrier}-terminal-root.npz"
        with np.load(bank_path, allow_pickle=False) as bank:
            coordinate = np.asarray(bank["support_centroids_m"], dtype=np.float64)
            cell_count = coordinate.shape[0]
            root_flux = np.asarray(bank["state"][:cell_count], dtype=np.float64)
            root_psi_norm = (root_flux - float(bank["axis_flux_wb"])) / float(
                bank["flux_span_wb"]
            )
            reference_flux = np.asarray(
                case.flux(coordinate[:, 0], coordinate[:, 1]), dtype=np.float64
            )
            reference_psi_norm = (reference_flux - float(case.flux_axis)) / float(
                case.flux_span
            )
        shared_source = {
            "bank": str(bank_path),
            "bank_sha256": _sha256(bank_path),
            "scorecard": str(scorecard_path),
            "scorecard_sha256": _sha256(scorecard_path),
            "plasma_cell_coordinates": "support_centroids_m",
        }
        targets.extend(
            (
                FieldTarget(
                    identity=f"stored-dina-reference-{carrier}",
                    kind="banked_flux_map",
                    carrier=carrier,
                    coordinate=coordinate,
                    flux=reference_flux,
                    psi_norm=reference_psi_norm,
                    source={
                        **shared_source,
                        "map_authority": (
                            "DINA pulse 135011 run 7 data-dictionary 3.39.0 "
                            "time-slice 353 sampled through ReferenceCase.flux"
                        ),
                    },
                    banked_terminal_residual=None,
                ),
                FieldTarget(
                    identity=f"stored-dina-root-{carrier}",
                    kind="banked_converged_root",
                    carrier=carrier,
                    coordinate=coordinate,
                    flux=root_flux,
                    psi_norm=root_psi_norm,
                    source={**shared_source, "field_array": "state[:cell_count]"},
                    banked_terminal_residual=float(
                        scorecard["fixtures"][carrier]["reproduction"]["banked"][
                            "terminal_residual"
                        ]
                    ),
                ),
            )
        )
    return targets


def _oracle_targets() -> list[FieldTarget]:
    fixture = _load_module(
        Path("scripts/analytic_oracle_fixtures/measure.py"),
        "flux_representation_oracle_fixture",
    )
    case = fixture.analytic_case()
    scorecard_path = ORACLE_ROOT_DIRECTORY / "results.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    targets: list[FieldTarget] = []
    for carrier in ("coarse", "fine"):
        bank_path = ORACLE_ROOT_DIRECTORY / f"root-{carrier}.npz"
        machine = fixture.cached_machine(
            case,
            fixture.FIXTURE_REQUESTS[carrier],
            wall_nodes=fixture.WALL_POINT_COUNT,
        )
        coordinate = np.asarray(machine.node, dtype=np.float64)
        cell_count = coordinate.shape[0]
        with np.load(bank_path, allow_pickle=False) as bank:
            oracle_flux = np.asarray(
                bank["oracle_state"][:cell_count], dtype=np.float64
            )
            root_flux = np.asarray(bank["root_state"][:cell_count], dtype=np.float64)
            oracle_psi_norm = np.asarray(bank["oracle_grid_psi_norm"], dtype=np.float64)
            root_psi_norm = np.asarray(bank["root_grid_psi_norm"], dtype=np.float64)
            terminal_residual = float(
                scorecard["fixtures"][carrier]["metric"]["fixed_point_residual"][
                    "recovery_value"
                ]
            )
        shared_source = {
            "bank": str(bank_path),
            "bank_sha256": _sha256(bank_path),
            "scorecard": str(scorecard_path),
            "scorecard_sha256": _sha256(scorecard_path),
            "plasma_cell_coordinates": (
                f"warm semantic machine cache {machine.cache['semantic_key']}"
            ),
            "requested_cells": int(fixture.FIXTURE_REQUESTS[carrier]),
        }
        targets.extend(
            (
                FieldTarget(
                    identity=f"closed-form-oracle-map-{carrier}",
                    kind="banked_flux_map",
                    carrier=carrier,
                    coordinate=coordinate,
                    flux=oracle_flux,
                    psi_norm=oracle_psi_norm,
                    source={
                        **shared_source,
                        "field_array": "oracle_state[:cell_count]",
                    },
                    banked_terminal_residual=None,
                ),
                FieldTarget(
                    identity=f"closed-form-oracle-root-{carrier}",
                    kind="banked_converged_root",
                    carrier=carrier,
                    coordinate=coordinate,
                    flux=root_flux,
                    psi_norm=root_psi_norm,
                    source={**shared_source, "field_array": "root_state[:cell_count]"},
                    banked_terminal_residual=terminal_residual,
                ),
            )
        )
    return targets


def _open_uniform_knots(
    lower: float, upper: float, coefficient_count: int
) -> np.ndarray:
    interior_count = coefficient_count - SPLINE_DEGREE - 1
    interior = np.linspace(lower, upper, interior_count + 2)[1:-1]
    return np.concatenate(
        (
            np.full(SPLINE_DEGREE + 1, lower),
            interior,
            np.full(SPLINE_DEGREE + 1, upper),
        )
    )


def _tensor_design(coordinate: np.ndarray, knot_count: int):
    radial = coordinate[:, 0]
    vertical = coordinate[:, 1]
    radial_knots = _open_uniform_knots(
        float(np.min(radial)), float(np.max(radial)), knot_count
    )
    vertical_knots = _open_uniform_knots(
        float(np.min(vertical)), float(np.max(vertical)), knot_count
    )
    radial_basis = BSpline.design_matrix(
        radial, radial_knots, SPLINE_DEGREE, extrapolate=False
    )
    vertical_basis = BSpline.design_matrix(
        vertical, vertical_knots, SPLINE_DEGREE, extrapolate=False
    )
    return hstack(
        [
            radial_basis.multiply(vertical_basis[:, index : index + 1])
            for index in range(knot_count)
        ],
        format="csr",
    )


def _region_masks(psi_norm: np.ndarray) -> dict[str, np.ndarray]:
    lower = 1.0 - SEPARATRIX_HALF_WIDTH_PSI_N
    upper = 1.0 + SEPARATRIX_HALF_WIDTH_PSI_N
    return {
        "closed_flux": psi_norm < lower,
        "separatrix_band": (psi_norm >= lower) & (psi_norm <= upper),
        "scrape_off_layer": psi_norm > upper,
    }


def _regional_errors(
    target: np.ndarray, fitted: np.ndarray, psi_norm: np.ndarray
) -> dict[str, dict[str, float | int]]:
    span = float(np.max(target) - np.min(target))
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("a banked field has no finite nonzero flux span")
    absolute = np.abs(fitted - target)
    result: dict[str, dict[str, float | int]] = {}
    for region, mask in _region_masks(psi_norm).items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise RuntimeError(f"{region} has no cells")
        result[region] = {
            "cell_count": count,
            "relative_sup_error": float(np.max(absolute[mask]) / span),
            "relative_rms_error": float(
                np.sqrt(np.mean(np.square(absolute[mask]))) / span
            ),
        }
    return result


def _production_residual() -> dict[str, Any]:
    receipt = json.loads(PRODUCTION_RECEIPT.read_text(encoding="utf-8"))
    values = [
        float(row["repaired_conditioning_enabled"]["terminal_relative_residual"])
        for row in receipt["frame_records"]
    ]
    return {
        "source": str(PRODUCTION_RECEIPT),
        "source_sha256": _sha256(PRODUCTION_RECEIPT),
        "frame_count": len(values),
        "range": [min(values), max(values)],
        "comparison_threshold": min(values),
        "threshold_policy": (
            "minimum terminal relative residual in the five-frame repaired "
            "production receipt, making the representation comparison use the "
            "hardest reached production floor"
        ),
    }


def _measure_target(target: FieldTarget, threshold: float) -> dict[str, Any]:
    rungs = []
    first_below: int | None = None
    for knot_count in KNOT_COUNT_LADDER:
        design = _tensor_design(target.coordinate, knot_count)
        solution = lsqr(
            design,
            target.flux,
            atol=1.0e-13,
            btol=1.0e-13,
            iter_lim=max(2000, 8 * knot_count**2),
        )
        fitted = np.asarray(design @ solution[0], dtype=np.float64)
        errors = _regional_errors(target.flux, fitted, target.psi_norm)
        binding_region = max(
            REGION_NAMES,
            key=lambda region: float(errors[region]["relative_sup_error"]),
        )
        closed_error = float(errors["closed_flux"]["relative_sup_error"])
        if first_below is None and closed_error < threshold:
            first_below = knot_count
        rungs.append(
            {
                "knot_count_per_axis": knot_count,
                "coefficient_lattice": [knot_count, knot_count],
                "coefficient_count": knot_count**2,
                "cell_count_replaced": int(target.flux.size),
                "cells_per_coefficient": float(target.flux.size / knot_count**2),
                "least_squares_iterations": int(solution[2]),
                "least_squares_stop_code": int(solution[1]),
                "regional_relative_fit_error": errors,
                "binding_region_by_relative_sup_error": binding_region,
            }
        )
    terminal = rungs[-1]
    separatrix_binding = (
        terminal["binding_region_by_relative_sup_error"] == "separatrix_band"
    )
    return {
        "identity": target.identity,
        "kind": target.kind,
        "carrier": target.carrier,
        "cell_count": int(target.flux.size),
        "coordinate_sha256": _array_sha256(target.coordinate),
        "flux_sha256": _array_sha256(target.flux),
        "psi_norm_sha256": _array_sha256(target.psi_norm),
        "source": target.source,
        "banked_terminal_residual": target.banked_terminal_residual,
        "first_knot_count_closed_flux_below_production_terminal_residual": (
            first_below
        ),
        "separatrix_band_binding_at_terminal_rung": separatrix_binding,
        "separatrix_binding_statement": (
            "The separatrix-band relative sup error is the binding regional term "
            "at the terminal rung."
            if separatrix_binding
            else "The separatrix-band relative sup error is not the binding "
            "regional term at the terminal rung."
        ),
        "rungs": rungs,
    }


def measure() -> dict[str, Any]:
    targets = _stored_reference_targets() + _oracle_targets()
    production = _production_residual()
    measured = [
        _measure_target(target, float(production["comparison_threshold"]))
        for target in targets
    ]
    binding_count = sum(
        bool(target["separatrix_band_binding_at_terminal_rung"]) for target in measured
    )
    return {
        "schema": "nova.flux-representation-degree",
        "artifact": str(OUTPUT),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "driver": str(Path(__file__)),
        "driver_sha256": _sha256(Path(__file__)),
        "execution": {
            "mode": "banked-field fitting only",
            "equilibrium_solves_run": 0,
            "forward_map_evaluations_run": 0,
            "declaration": (
                "No equilibrium solve was run. The driver loads serialized root "
                "arrays, stored reference interpolation, and warm coordinate-only "
                "machine caches, then performs only sparse linear least-squares fits."
            ),
        },
        "fit_contract": {
            "basis": "tensor-product open-uniform cubic B-spline",
            "spline_degree": SPLINE_DEGREE,
            "knot_count_ladder_per_axis": list(KNOT_COUNT_LADDER),
            "knot_count_convention": (
                "per-axis coefficient count; a 12 by 12 rung has 144 coefficients; "
                "the complete clamped knot vector has knot_count + degree + 1 entries"
            ),
            "fit": "unweighted sparse least squares on all banked plasma cells",
            "relative_error_denominator": "full target-map flux span",
            "degree_threshold_metric": "closed-flux relative sup error",
            "regional_metrics": ["relative_sup_error", "relative_rms_error"],
            "separatrix_band": {
                "coordinate": "target-local normalized flux psi_N",
                "lower_inclusive": 1.0 - SEPARATRIX_HALF_WIDTH_PSI_N,
                "upper_inclusive": 1.0 + SEPARATRIX_HALF_WIDTH_PSI_N,
                "half_width": SEPARATRIX_HALF_WIDTH_PSI_N,
            },
            "region_partition": {
                "closed_flux": "psi_N < 0.95",
                "separatrix_band": "0.95 <= psi_N <= 1.05",
                "scrape_off_layer": "psi_N > 1.05",
            },
        },
        "production_terminal_residual": production,
        "coverage": {
            "banked_flux_maps": sum(
                target["kind"] == "banked_flux_map" for target in measured
            ),
            "banked_converged_roots": sum(
                target["kind"] == "banked_converged_root" for target in measured
            ),
            "targets": len(measured),
            "rungs_per_target": len(KNOT_COUNT_LADDER),
        },
        "targets": measured,
        "conclusion": {
            "separatrix_band_binding_targets_at_terminal_rung": binding_count,
            "target_count": len(measured),
            "separatrix_band_is_universally_binding": binding_count == len(measured),
            "statement": (
                f"The separatrix band is the binding terminal-rung relative-sup "
                f"term for {binding_count} of {len(measured)} banked targets."
            ),
        },
    }


def validate(receipt: dict[str, Any]) -> None:
    if receipt["execution"]["equilibrium_solves_run"] != 0:
        raise AssertionError("the receipt does not prove the solve-free contract")
    if receipt["execution"]["forward_map_evaluations_run"] != 0:
        raise AssertionError("a forward map evaluation entered the receipt")
    if receipt["coverage"] != {
        "banked_flux_maps": 4,
        "banked_converged_roots": 4,
        "targets": 8,
        "rungs_per_target": len(KNOT_COUNT_LADDER),
    }:
        raise AssertionError("the protected bank coverage is incomplete")
    for target in receipt["targets"]:
        if len(target["rungs"]) != len(KNOT_COUNT_LADDER):
            raise AssertionError(f"{target['identity']} has an incomplete ladder")
        if target[
            "first_knot_count_closed_flux_below_production_terminal_residual"
        ] not in (*KNOT_COUNT_LADDER, None):
            raise AssertionError(f"{target['identity']} has an invalid crossing")
        for rung, expected_knot_count in zip(
            target["rungs"], KNOT_COUNT_LADDER, strict=True
        ):
            if rung["knot_count_per_axis"] != expected_knot_count:
                raise AssertionError(f"{target['identity']} ladder changed")
            if rung["coefficient_count"] != expected_knot_count**2:
                raise AssertionError(f"{target['identity']} coefficient count is wrong")
            errors = rung["regional_relative_fit_error"]
            if set(errors) != set(REGION_NAMES):
                raise AssertionError(f"{target['identity']} region split is incomplete")
            for region in REGION_NAMES:
                values = errors[region]
                if values["cell_count"] <= 0:
                    raise AssertionError(f"{target['identity']} has an empty region")
                if not np.isfinite(values["relative_sup_error"]):
                    raise AssertionError(f"{target['identity']} has a non-finite error")
                if not np.isfinite(values["relative_rms_error"]):
                    raise AssertionError(f"{target['identity']} has a non-finite error")


def _write(receipt: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate the existing receipt only"
    )
    arguments = parser.parse_args()
    if arguments.check:
        receipt = json.loads(OUTPUT.read_text(encoding="utf-8"))
        validate(receipt)
        print(
            "representation-degree check passed: "
            f"{receipt['coverage']['targets']} targets, "
            f"{receipt['coverage']['rungs_per_target']} rungs each, zero solves"
        )
        return
    receipt = measure()
    validate(receipt)
    _write(receipt)
    print(
        f"wrote {OUTPUT}: {receipt['coverage']['targets']} targets, "
        f"{receipt['coverage']['rungs_per_target']} rungs each, zero solves"
    )


if __name__ == "__main__":
    main()
