"""Size plasma-boundary curvature jumps without running an equilibrium solve.

The protected stored-reference and closed-form-oracle banks are read on their
coarse and fine plasma carriers.  Their declared source profiles are evaluated
at normalized flux one, and Nova's total-poloidal-flux convention maps those
values directly to toroidal current density and the Grad-Shafranov operator.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path as PolygonPath

from nova.equilibrium.convention import (
    grad_shafranov_source,
    toroidal_current_density,
)


OUTPUT = Path("docs/figures/coefficient-space-newton/edge-current-jump.json")
STORED_ROOT_DIRECTORY = Path("scripts/root_gate_attribution")
ORACLE_ROOT_DIRECTORY = Path("scripts/oracle_rebaseline")
REFERENCE_MODULE = Path("tests/test_equilibrium_forward_reference.py")
ORACLE_MODULE = Path("scripts/analytic_oracle_fixtures/measure.py")
NEGLIGIBLE_JUMP_FRACTION_LIMIT = 4096.0 * np.finfo(np.float64).eps
BOUNDARY_SAMPLE_COUNT = 4096
FORBIDDEN_SOLVE_CALLS = frozenset(
    {"solve", "solve_branch", "newton_krylov", "fixed_point"}
)


@dataclass(frozen=True)
class SourceCase:
    """One physical source evaluated on one banked plasma carrier."""

    identity: str
    family: str
    carrier: str
    coordinate: np.ndarray
    plasma_mask: np.ndarray
    psi_norm: np.ndarray
    boundary: np.ndarray
    declared_pressure_gradient: float
    effective_pressure_gradient: np.ndarray
    diamagnetic_gradient: np.ndarray
    plasma_pressure_gradient: np.ndarray
    plasma_diamagnetic_gradient: np.ndarray
    inputs: dict[str, Any]


def _sha256(path: Path) -> str:
    """Return the hexadecimal digest of one evidence input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str):
    """Load an evidence fixture without importing it as a test module."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _statistics(values: np.ndarray) -> dict[str, float]:
    """Return finite minimum, median and maximum statistics."""
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0 or not np.all(np.isfinite(finite)):
        raise RuntimeError("a reported distribution is empty or non-finite")
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def _stored_reference_cases() -> list[SourceCase]:
    """Return the stored DINA source on both protected carriers."""
    reference = _load_module(REFERENCE_MODULE, "edge_current_stored_reference")
    reference.configure_dtypes()
    case = reference.require_reference()
    boundary = np.asarray(case.boundary, dtype=np.float64)
    edge_psi_norm = np.ones(len(boundary), dtype=np.float64)
    edge_pressure = np.interp(edge_psi_norm, case.psi_norm, case.p_prime)
    edge_diamagnetic = np.interp(edge_psi_norm, case.psi_norm, case.ff_prime)
    cases = []
    for carrier in ("coarse", "fine"):
        bank_path = STORED_ROOT_DIRECTORY / f"{carrier}-terminal-root.npz"
        with np.load(bank_path, allow_pickle=False) as bank:
            coordinate = np.asarray(bank["support_centroids_m"], dtype=np.float64)
        psi_norm = (
            np.asarray(case.flux(coordinate[:, 0], coordinate[:, 1]))
            - float(case.flux_axis)
        ) / float(case.flux_span)
        plasma_mask = PolygonPath(boundary).contains_points(coordinate, radius=1.0e-12)
        clipped = np.clip(psi_norm, 0.0, 1.0)
        cases.append(
            SourceCase(
                identity=f"stored-dina-{carrier}",
                family="stored_dina_reference",
                carrier=carrier,
                coordinate=coordinate,
                plasma_mask=plasma_mask,
                psi_norm=psi_norm,
                boundary=boundary,
                declared_pressure_gradient=float(case.p_prime[-1]),
                effective_pressure_gradient=edge_pressure,
                diamagnetic_gradient=edge_diamagnetic,
                plasma_pressure_gradient=np.interp(
                    clipped, case.psi_norm, case.p_prime
                ),
                plasma_diamagnetic_gradient=np.interp(
                    clipped, case.psi_norm, case.ff_prime
                ),
                inputs={
                    "source": (
                        "DINA pulse 135011 run 7 data-dictionary 3.39.0 time-slice 353"
                    ),
                    "source_module": str(REFERENCE_MODULE),
                    "carrier_bank": str(bank_path),
                    "carrier_bank_sha256": _sha256(bank_path),
                    "pressure_profile": "ReferenceCase.p_prime",
                    "diamagnetic_profile": "ReferenceCase.ff_prime",
                    "boundary": "ReferenceCase.boundary",
                },
            )
        )
    return cases


def _oracle_boundary(case) -> np.ndarray:
    """Return a symmetric sampling of the exact analytic boundary."""
    half_count = BOUNDARY_SAMPLE_COUNT // 2
    radius, upper_height, _weight, _offset = case._surface_nodes(0.0, half_count)
    upper = np.column_stack([radius, upper_height])
    lower = np.column_stack([radius[::-1], -upper_height[::-1]])
    return np.concatenate([upper, lower])


def _oracle_cases() -> list[SourceCase]:
    """Return the rotating closed-form source on both protected carriers."""
    oracle = _load_module(ORACLE_MODULE, "edge_current_closed_form_oracle")
    oracle.configure_dtypes()
    case = oracle.analytic_case()
    profile = oracle.analytic_profile(case)
    boundary = _oracle_boundary(case)
    boundary_radius = jnp.asarray(boundary[:, 0], dtype=jnp.float64)
    edge_psi_norm = jnp.ones(len(boundary), dtype=jnp.float64)
    declared_pressure = float(profile.p_prime(jnp.asarray(1.0, dtype=jnp.float64)))
    edge_pressure = np.asarray(
        profile.pressure_gradient(boundary_radius, edge_psi_norm), dtype=np.float64
    )
    edge_diamagnetic = np.asarray(profile.ff_prime(edge_psi_norm), dtype=np.float64)
    cases = []
    for carrier in ("coarse", "fine"):
        bank_path = ORACLE_ROOT_DIRECTORY / f"root-{carrier}.npz"
        machine = oracle.cached_machine(
            case,
            oracle.FIXTURE_REQUESTS[carrier],
            wall_nodes=oracle.WALL_POINT_COUNT,
        )
        coordinate = np.asarray(machine.node, dtype=np.float64)
        with np.load(bank_path, allow_pickle=False) as bank:
            psi_norm = np.asarray(bank["oracle_grid_psi_norm"], dtype=np.float64)
        if len(psi_norm) != len(coordinate):
            raise RuntimeError(f"{bank_path} does not match its plasma carrier")
        plasma_mask = np.asarray(
            case.contains(coordinate[:, 0], coordinate[:, 1]), dtype=bool
        )
        plasma_radius = jnp.asarray(coordinate[:, 0], dtype=jnp.float64)
        plasma_psi_norm = jnp.asarray(psi_norm, dtype=jnp.float64)
        cases.append(
            SourceCase(
                identity=f"closed-form-oracle-{carrier}",
                family="closed_form_rotating_oracle",
                carrier=carrier,
                coordinate=coordinate,
                plasma_mask=plasma_mask,
                psi_norm=psi_norm,
                boundary=boundary,
                declared_pressure_gradient=declared_pressure,
                effective_pressure_gradient=np.asarray(edge_pressure, dtype=np.float64),
                diamagnetic_gradient=np.asarray(edge_diamagnetic, dtype=np.float64),
                plasma_pressure_gradient=np.asarray(
                    profile.pressure_gradient(plasma_radius, plasma_psi_norm),
                    dtype=np.float64,
                ),
                plasma_diamagnetic_gradient=np.asarray(
                    profile.ff_prime(plasma_psi_norm), dtype=np.float64
                ),
                inputs={
                    "source": case.name,
                    "source_module": str(ORACLE_MODULE),
                    "carrier_bank": str(bank_path),
                    "carrier_bank_sha256": _sha256(bank_path),
                    "carrier_cache_key": machine.cache["semantic_key"],
                    "pressure_profile": "analytic_profile(case).p_prime",
                    "effective_pressure_gradient": (
                        "RotatingDomainProfile.pressure_gradient at fixed radius"
                    ),
                    "diamagnetic_profile": "analytic_profile(case).ff_prime",
                    "boundary": "RotatingEquilibrium._surface_nodes at zero flux",
                },
            )
        )
    return cases


def _measure(case: SourceCase) -> dict[str, Any]:
    """Measure one carrier's edge source and operator jump."""
    if not np.any(case.plasma_mask):
        raise RuntimeError(f"{case.identity} has no plasma samples")
    boundary_radius = case.boundary[:, 0]
    boundary_current = np.asarray(
        toroidal_current_density(
            boundary_radius,
            case.effective_pressure_gradient,
            case.diamagnetic_gradient,
        ),
        dtype=np.float64,
    )
    boundary_operator = np.asarray(
        grad_shafranov_source(
            boundary_radius,
            case.effective_pressure_gradient,
            case.diamagnetic_gradient,
        ),
        dtype=np.float64,
    )
    plasma_operator = np.asarray(
        grad_shafranov_source(
            case.coordinate[:, 0],
            case.plasma_pressure_gradient,
            case.plasma_diamagnetic_gradient,
        ),
        dtype=np.float64,
    )
    carrier_peak = float(np.max(np.abs(plasma_operator[case.plasma_mask])))
    boundary_peak = float(np.max(np.abs(boundary_operator)))
    peak_operator = max(carrier_peak, boundary_peak)
    if not np.isfinite(peak_operator) or peak_operator <= 0.0:
        raise RuntimeError(f"{case.identity} has no finite nonzero operator scale")
    absolute_jump = np.abs(boundary_operator)
    fraction = absolute_jump / peak_operator
    maximum_fraction = float(np.max(fraction))
    smooth_basis_suffices = bool(maximum_fraction <= NEGLIGIBLE_JUMP_FRACTION_LIMIT)
    return {
        "identity": case.identity,
        "family": case.family,
        "carrier": case.carrier,
        "inputs": case.inputs,
        "equilibrium_solve_run": False,
        "plasma_operator_normalisation": {
            "sample_count": int(np.count_nonzero(case.plasma_mask))
            + len(case.boundary),
            "carrier_centroid_count": int(np.count_nonzero(case.plasma_mask)),
            "boundary_limit_count": len(case.boundary),
            "peak_absolute_delta_star_total_flux_wb_per_m2": peak_operator,
            "sampling": (
                "banked plasma-carrier centroids inside the case boundary plus "
                "the source limit approached from inside the boundary"
            ),
        },
        "boundary_evaluation": {
            "normalised_flux": 1.0,
            "sample_count": len(case.boundary),
            "major_radius_m": _statistics(boundary_radius),
            "declared_pressure_gradient_at_reference_radius_pa_per_wb": (
                case.declared_pressure_gradient
            ),
            "effective_pressure_gradient_at_fixed_radius_pa_per_wb": _statistics(
                case.effective_pressure_gradient
            ),
            "diamagnetic_gradient_t2_m2_per_wb": _statistics(case.diamagnetic_gradient),
        },
        "toroidal_current_density_on_boundary_a_per_m2": {
            "signed": _statistics(boundary_current),
            "absolute": _statistics(np.abs(boundary_current)),
        },
        "delta_star_total_flux_jump_on_boundary_wb_per_m2": {
            "signed_inside_minus_vacuum": _statistics(boundary_operator),
            "absolute": _statistics(absolute_jump),
            "absolute_fraction_of_peak_plasma_operator": _statistics(fraction),
        },
        "smooth_global_basis_verdict": {
            "suffices": smooth_basis_suffices,
            "verdict": (
                "SUFFICIENT_EDGE_SOURCE_NUMERICALLY_ZERO"
                if smooth_basis_suffices
                else "INSUFFICIENT_MATERIAL_CURVATURE_JUMP"
            ),
            "maximum_jump_fraction": maximum_fraction,
            "negligible_fraction_limit": NEGLIGIBLE_JUMP_FRACTION_LIMIT,
            "statement": (
                "The boundary source is zero to the declared binary64 "
                "numerical-zero criterion, so this case does not require a "
                "curvature-jump support."
                if smooth_basis_suffices
                else "The boundary jump exceeds the declared binary64 "
                "numerical-zero criterion, so a globally smooth basis alone "
                "is not sufficient for this case."
            ),
        },
    }


def _solve_call_audit() -> dict[str, Any]:
    """Prove this receipt generator contains no equilibrium-solve call."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else None
        if isinstance(function, ast.Name):
            name = function.id
        if name in FORBIDDEN_SOLVE_CALLS:
            calls.append({"name": name, "line": node.lineno})
    return {
        "passed": not calls,
        "forbidden_call_names": sorted(FORBIDDEN_SOLVE_CALLS),
        "calls_found": calls,
    }


def run(output: Path) -> dict[str, Any]:
    """Measure both source families and write the JSON receipt."""
    audit = _solve_call_audit()
    if not audit["passed"]:
        raise RuntimeError(f"equilibrium-solve calls found: {audit['calls_found']}")
    measurements = [
        _measure(case) for case in (*_stored_reference_cases(), *_oracle_cases())
    ]
    if len(measurements) != 4:
        raise RuntimeError("the receipt requires two families on two carriers")
    sufficient = sum(
        row["smooth_global_basis_verdict"]["suffices"] for row in measurements
    )
    maximum = max(
        row["smooth_global_basis_verdict"]["maximum_jump_fraction"]
        for row in measurements
    )
    receipt = {
        "receipt": "solve-free plasma-edge current and curvature-jump sizing",
        "execution_contract": {
            "equilibrium_solve_run": False,
            "solve_call_audit": audit,
            "source_evaluation_only": True,
            "case_count": len(measurements),
        },
        "convention": {
            "flux": "total poloidal flux Phi in Wb",
            "profile_coordinate": (
                "normalized flux psi_N with the source evaluated at psi_N = 1 "
                "from inside the plasma"
            ),
            "toroidal_current_density": (
                "j_phi = -2*pi*(R*p_prime + FF_prime/(mu_0*R))"
            ),
            "operator": (
                "Delta_star(Phi) = -2*pi*mu_0*R*j_phi = "
                "4*pi^2*(mu_0*R^2*p_prime + FF_prime)"
            ),
            "vacuum_operator": 0.0,
        },
        "negligibility_criterion": {
            "quantity": (
                "maximum absolute boundary jump divided by peak absolute "
                "plasma operator"
            ),
            "limit": NEGLIGIBLE_JUMP_FRACTION_LIMIT,
            "basis": (
                "4096 binary64 epsilon; a numerical-zero test fixed by "
                "arithmetic precision, not by an achieved source value"
            ),
        },
        "cases": measurements,
        "aggregate": {
            "case_count": len(measurements),
            "smooth_basis_sufficient_count": sufficient,
            "smooth_basis_insufficient_count": len(measurements) - sufficient,
            "maximum_jump_fraction_across_cases": maximum,
            "globally_smooth_basis_suffices_for_all_banked_cases": (
                sufficient == len(measurements)
            ),
            "verdict": (
                "SMOOTH_BASIS_SUFFICIENT_FOR_ALL_BANKED_CASES"
                if sufficient == len(measurements)
                else "CURVATURE_JUMP_SUPPORT_REQUIRED_FOR_CASE_SET"
            ),
            "statement": (
                "Every banked case has a numerically zero edge source."
                if sufficient == len(measurements)
                else "At least one banked case carries a material edge source, "
                "so a globally smooth basis alone does not cover the full case "
                "set."
            ),
            "equilibrium_solve_run": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Parse the output path, write the receipt and print its headline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.output)
    aggregate = receipt["aggregate"]
    print(
        "EDGE_CURRENT_JUMP "
        f"cases={aggregate['case_count']} "
        f"smooth_sufficient={aggregate['smooth_basis_sufficient_count']} "
        f"max_fraction={aggregate['maximum_jump_fraction_across_cases']:.12g} "
        f"verdict={aggregate['verdict']} "
        "equilibrium_solve_run=false"
    )


if __name__ == "__main__":
    main()
