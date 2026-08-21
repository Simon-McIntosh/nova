"""Validate the measured closed-form equilibrium recovery gate registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OUTPUT.parents[1]
EXPECTED_GATE_NAMES = {
    "standing_forcing_sup_wb",
    "fixed_point_residual",
    "axis_position_m",
    "flux_sup_fraction_of_span",
    "flux_rms_fraction_of_span",
    "plasma_current_fraction",
    "poloidal_beta_fraction",
    "internal_inductance_fraction",
    "field_integral_fraction",
    "grad_shafranov_relative",
    "divergence_b_relative",
    "divergence_j_relative",
    "topology_class",
    "x_point_absence",
}


def _digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_report(path: Path | None = None) -> dict[str, object]:
    """Load the merged recovery report."""
    source = OUTPUT / "results.json" if path is None else Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def validate_registry(report: dict[str, object]) -> dict[str, object]:
    """Re-evaluate every proposed bound and cross-grid convergence clause."""
    registry = report.get("gate_registry", {})
    missing = sorted(EXPECTED_GATE_NAMES - set(registry))
    unexpected = sorted(set(registry) - EXPECTED_GATE_NAMES)
    failed_bounds: list[str] = []
    failed_convergence: list[str] = []
    malformed: list[str] = []
    for name, gate in registry.items():
        required = {
            "status",
            "measured_floor",
            "proposed_bound",
            "headroom",
            "fixture_pass",
            "convergence_clause",
        }
        if not required.issubset(gate):
            malformed.append(name)
            continue
        if gate["status"] != "proposed" or gate["proposed_bound"] is None:
            malformed.append(name)
        if not all(bool(value) for value in gate["fixture_pass"].values()):
            failed_bounds.append(name)
        if not bool(gate["convergence_clause"].get("passed")):
            failed_convergence.append(name)
    passed = not any((missing, unexpected, malformed))
    return {
        "passed": passed,
        "all_bounds_pass": not failed_bounds,
        "all_convergence_clauses_pass": not failed_convergence,
        "missing": missing,
        "unexpected": unexpected,
        "failed_bounds": failed_bounds,
        "failed_convergence": failed_convergence,
        "malformed": malformed,
    }


def validate_gauge_discipline(report: dict[str, object]) -> dict[str, object]:
    """Prove that raw amplitudes and normalised flux use local field anchors."""
    mixed: list[str] = []
    foreign: list[str] = []
    for fixture_name, fixture in report.get("fixtures", {}).items():
        receipt = fixture.get("gauge_receipt", {})
        if receipt.get("raw_flux_comparison_gauge") != "shared_exact_exterior":
            mixed.append(fixture_name)
        if receipt.get("psi_norm_root_anchors_from") != "root_field":
            foreign.append(f"{fixture_name}:root")
        if receipt.get("psi_norm_oracle_anchors_from") != "closed_form_field":
            foreign.append(f"{fixture_name}:oracle")
        if receipt.get("reference_gauge_constant_used"):
            mixed.append(f"{fixture_name}:reference-constant")
    registry = report.get("gate_registry", {})
    mixed_gauge_gates = sorted(
        name for name, gate in registry.items() if gate.get("gauge") == "mixed"
    )
    foreign_anchor_gates = sorted(
        name
        for name, gate in registry.items()
        if gate.get("psi_norm_anchor") == "foreign"
    )
    mixed_gauge_gates.extend(sorted(mixed))
    foreign_anchor_gates.extend(sorted(foreign))
    return {
        "passed": not mixed_gauge_gates and not foreign_anchor_gates,
        "mixed_gauge_gates": mixed_gauge_gates,
        "foreign_anchor_gates": foreign_anchor_gates,
    }


def validate_artifacts(report: dict[str, object]) -> dict[str, object]:
    """Match every serialized state array to the receipt digest and shape."""
    failures: list[str] = []
    arrays_checked = 0
    fixtures_checked = 0
    for fixture_name, fixture in report.get("fixtures", {}).items():
        artifact = fixture.get("root_artifact", {})
        relative = artifact.get("path")
        if relative is None:
            failures.append(f"{fixture_name}:missing-path")
            continue
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            failures.append(f"{fixture_name}:missing-file")
            continue
        with np.load(path, allow_pickle=False) as stored:
            expected = artifact.get("arrays", {})
            if set(stored.files) != set(expected):
                failures.append(f"{fixture_name}:array-set")
                continue
            fixtures_checked += 1
            for name, identity in expected.items():
                values = np.asarray(stored[name])
                arrays_checked += 1
                if list(values.shape) != identity["shape"]:
                    failures.append(f"{fixture_name}:{name}:shape")
                if values.dtype.str != identity["dtype"]:
                    failures.append(f"{fixture_name}:{name}:dtype")
                if _digest(values) != identity["sha256"]:
                    failures.append(f"{fixture_name}:{name}:digest")
    return {
        "passed": not failures,
        "fixtures_checked": fixtures_checked,
        "root_arrays_checked": arrays_checked,
        "failures": failures,
    }
