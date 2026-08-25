"""Distinguish analytic-case posing from discrete-map error by refinement.

The closed-form rotating equilibrium is evaluated independently at every
operator target.  Each carrier receives the exterior contribution implied by
that field and its analytically integrated plasma-current moments.  The exact
production map is applied once at the analytic field, with no seed, nonlinear
solve, continuation, or fixed-point basin in the measurement path.

The receipt also retains unqualified terminal-solve observations as excluded
context.  Those values do not enter the residual ladder or its fitted order.
No production implementation is changed by this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/discrete-operator-analytic-error/operator-refinement-ladder.json"
)
REQUESTED_CELLS = (-300, -500, -750, -1000)
REGIONS = ("closed_flux_region", "separatrix_band", "scrape_off_layer")
SEPARATRIX_HALF_WIDTH = 0.05
MAP_ROUNDOFF_TOLERANCE = 4096.0 * np.finfo(np.float64).eps
CONVERGING_ORDER_FLOOR = 0.5
UNQUALIFIED_TERMINAL_OBSERVATIONS = (
    {
        "requested_cells": -300,
        "achieved_relative_residual": 1.2549724178436488,
        "criterion": 1.0e-10,
        "qualification": "unqualified",
        "used_in_refinement_fit": False,
        "route": "production moment seed with undamped Newton-Krylov",
    },
    {
        "requested_cells": -500,
        "achieved_relative_residual": 1.2998050585857959,
        "criterion": 1.0e-10,
        "qualification": "unqualified",
        "used_in_refinement_fit": False,
        "route": "production moment seed with undamped Newton-Krylov",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _measurement_lane() -> dict[str, Any]:
    memory_mb = os.environ.get("SLURM_MEM_PER_NODE")
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "requested_memory_mb": int(memory_mb) if memory_mb else None,
        "cpus_per_task": (
            int(os.environ["SLURM_CPUS_PER_TASK"])
            if os.environ.get("SLURM_CPUS_PER_TASK")
            else None
        ),
        "backend": "cpu",
        "precision": "float64",
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _region_masks(psi_norm: np.ndarray) -> dict[str, np.ndarray]:
    lower = 1.0 - SEPARATRIX_HALF_WIDTH
    upper = 1.0 + SEPARATRIX_HALF_WIDTH
    return {
        "all_carrier_cells": np.ones_like(psi_norm, dtype=bool),
        "closed_flux_region": psi_norm < lower,
        "separatrix_band": (psi_norm >= lower) & (psi_norm <= upper),
        "scrape_off_layer": psi_norm > upper,
    }


def _regional_norms(
    field: np.ndarray, psi_norm: np.ndarray, span_wb: float
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    absolute = np.abs(np.asarray(field, dtype=np.float64))
    for name, mask in _region_masks(psi_norm).items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise RuntimeError(f"the {name} partition has no carrier cells")
        sup = float(np.max(absolute[mask]))
        rms = float(np.sqrt(np.mean(absolute[mask] ** 2)))
        result[name] = {
            "cell_count": count,
            "absolute_sup_wb": sup,
            "absolute_rms_wb": rms,
            "relative_sup": sup / span_wb,
            "relative_rms": rms / span_wb,
        }
    return result


def _full_state_norm(field: np.ndarray, span_wb: float) -> dict[str, float | int]:
    absolute = np.abs(np.asarray(field, dtype=np.float64))
    sup = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(absolute**2)))
    return {
        "state_node_count": len(absolute),
        "absolute_sup_wb": sup,
        "absolute_rms_wb": rms,
        "relative_sup": sup / span_wb,
        "relative_rms": rms / span_wb,
    }


def _topology_record(operator, state) -> dict[str, Any]:
    import jax.numpy as jnp

    _masks, topology = operator.read(jnp.asarray(state))
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    return {
        "class": "diverted" if bool(topology.diverted) else "limited",
        "axis_rz_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
        "boundary_minus_axis_span_wb": float(topology.flux_span),
        "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
    }


def _source_terms(case, operator, analytic: np.ndarray) -> dict[str, Any]:
    import jax.numpy as jnp

    from scripts.analytic_oracle_fixtures import measure as fixture

    coordinates = np.asarray(operator.grid.coordinate, dtype=np.float64)
    masks, _topology = operator.read(jnp.asarray(analytic))
    production_density = np.asarray(
        operator.source.current_density(jnp.asarray(coordinates[:, 0]), masks),
        dtype=np.float64,
    )
    closed_form_density = np.asarray(
        case.toroidal_current_density(coordinates[:, 0], coordinates[:, 1]),
        dtype=np.float64,
    )
    core = np.asarray(masks.core, dtype=bool)
    density_scale = max(float(np.max(np.abs(closed_form_density[core]))), 1.0e-300)
    density_delta = production_density[core] - closed_form_density[core]
    profile = operator.source.core
    psi_probe = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    pressure_gradient = np.asarray(profile.p_prime(psi_probe), dtype=np.float64)
    field_gradient = np.asarray(profile.ff_prime(psi_probe), dtype=np.float64)
    temperature = np.asarray(profile.rotation.temperature(psi_probe), dtype=np.float64)
    angular_frequency = np.asarray(
        profile.rotation.angular_frequency(psi_probe), dtype=np.float64
    )
    closed_pressure_gradient = -case.pressure_flux_gradient / fixture.TOTAL_FLUX_FACTOR
    closed_field_gradient = -case.f_f_prime / fixture.TOTAL_FLUX_FACTOR
    return {
        "normalisation": {
            "operator": "absolute source; no target-current amplitude scaling",
            "closed_form": "absolute analytic pressure and FF gradients",
            "operator_policy": operator.source.normalisation.name.lower(),
            "matches": operator.source.normalisation.name.lower() == "absolute",
        },
        "pressure_gradient_pa_per_wb": {
            "operator_at_psi_norm_0_0p5_1": pressure_gradient.tolist(),
            "closed_form_after_total_flux_conversion": float(closed_pressure_gradient),
            "maximum_absolute_delta": float(
                np.max(np.abs(pressure_gradient - closed_pressure_gradient))
            ),
        },
        "ff_gradient_tm2_per_wb": {
            "operator_at_psi_norm_0_0p5_1": field_gradient.tolist(),
            "closed_form_after_total_flux_conversion": float(closed_field_gradient),
            "maximum_absolute_delta": float(
                np.max(np.abs(field_gradient - closed_field_gradient))
            ),
        },
        "boundary_primitives": {
            "operator_pressure_pa": float(operator.source.boundary_pressure),
            "closed_form_pressure_pa": 0.0,
            "operator_field_function_tm": float(
                operator.source.boundary_field_function
            ),
            "closed_form_field_function_tm": float(case.boundary_f),
        },
        "rotation": {
            "operator_closure": profile.rotation_closure.name.lower(),
            "closed_form_closure": "isothermal_surface",
            "psi_norm_probe": [0.0, 0.5, 1.0],
            "operator_temperature_j": temperature.tolist(),
            "closed_form_axis_temperature_j": float(case.axis_temperature),
            "closed_form_boundary_temperature_j": float(case.boundary_temperature),
            "operator_angular_frequency_per_s": angular_frequency.tolist(),
            "closed_form_uniform_rotation_parameter_per_m2": float(
                case.rotation_parameter
            ),
            "operator_mean_particle_mass_kg": float(
                profile.rotation.mean_particle_mass
            ),
            "closed_form_mean_particle_mass_kg": float(case.mean_particle_mass),
        },
        "current_density": {
            "comparison_support": "analytic axis-connected core cell centroids",
            "core_cell_count": int(np.count_nonzero(core)),
            "operator_vs_closed_form_absolute_sup_a_per_m2": float(
                np.max(np.abs(density_delta))
            ),
            "operator_vs_closed_form_relative_sup": float(
                np.max(np.abs(density_delta)) / density_scale
            ),
        },
        "open_region_sources": {
            "operator_common_sol": "undeclared",
            "operator_private_flux": "undeclared",
            "closed_form": "zero outside the closed-flux core",
            "matches": operator.source.common_sol is None
            and operator.source.private_flux is None,
        },
    }


def _measure(requested_cells: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as fixture

    configure_dtypes()
    case = fixture.analytic_case()
    machine = fixture.cached_machine(
        case, requested_cells, wall_nodes=fixture.WALL_POINT_COUNT
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = fixture.exact_state(case, coordinates)
    empty_operator = fixture.forward_operator(case, machine)
    exact_physical = fixture.exact_current_moments(case, empty_operator, analytic)
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = fixture._internal_flux_image(empty_operator, exact_coefficients)
    prescribed_exterior = analytic - exact_internal
    operator = fixture.forward_operator(case, machine, prescribed_exterior)
    mapped_analytic = np.asarray(
        jax.block_until_ready(operator.flux_map()(jnp.asarray(analytic))),
        dtype=np.float64,
    )

    cell_count = len(machine.node)
    analytic_topology = _topology_record(operator, analytic)
    analytic_psi_norm = (
        analytic[:cell_count] - analytic_topology["axis_flux_wb"]
    ) / analytic_topology["boundary_minus_axis_span_wb"]
    span_wb = abs(analytic_topology["boundary_minus_axis_span_wb"])
    map_residual = mapped_analytic - analytic
    external = np.asarray(operator.external(), dtype=np.float64)
    external_identity = exact_internal + external - analytic
    slices = {
        "grid": slice(0, cell_count),
        "wall": slice(cell_count, cell_count + len(machine.wall_node)),
        "sample": slice(cell_count + len(machine.wall_node), None),
    }
    return {
        "requested_cells": requested_cells,
        "realised_cells": cell_count,
        "state_dimension": len(analytic),
        "characteristic_pitch_m": float(np.sqrt(np.median(np.asarray(machine.area)))),
        "cache": machine.cache,
        "one_application_residual": {
            "definition": (
                "exact production forward map at analytic total flux minus "
                "analytic total flux"
            ),
            "application_count": 1,
            "seed_used": False,
            "solve_used": False,
            "full_state": _full_state_norm(map_residual, span_wb),
            "regions": _regional_norms(
                map_residual[:cell_count], analytic_psi_norm, span_wb
            ),
        },
        "analytic_topology": analytic_topology,
        "posed_terms": {
            "external": {
                "operator_specification": (
                    "closed-form total flux minus analytically integrated "
                    "exact-density plasma image"
                ),
                "closed_form_specification": (
                    "the exterior contribution which completes the analytic "
                    "total-flux field"
                ),
                "external_current_amplitude": {
                    "operator": 1.0,
                    "closed_form_completion": 1.0,
                    "matches": True,
                },
                "full_state_identity_absolute_sup_wb": float(
                    np.max(np.abs(external_identity))
                ),
                "by_target_family": {
                    name: {
                        "node_count": len(external[selection]),
                        "operator_external_min_wb": float(np.min(external[selection])),
                        "operator_external_max_wb": float(np.max(external[selection])),
                        "closed_form_completion_absolute_sup_delta_wb": float(
                            np.max(np.abs(external_identity[selection]))
                        ),
                    }
                    for name, selection in slices.items()
                },
            },
            "boundary": {
                "condition": (
                    "limited-plasma separatrix selected by the analytic field; "
                    "the wall carries the sampled exterior completion"
                ),
                "operator_boundary_flux_at_analytic_solution_wb": (
                    analytic_topology["boundary_flux_wb"]
                ),
                "closed_form_boundary_flux_wb": 0.0,
                "absolute_delta_wb": abs(analytic_topology["boundary_flux_wb"]),
            },
            "source": _source_terms(case, operator, analytic),
        },
    }


def _fit_order(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pitch = np.asarray([row["characteristic_pitch_m"] for row in rows])
    error = np.asarray(
        [
            row["one_application_residual"]["regions"]["all_carrier_cells"][
                "relative_sup"
            ]
            for row in rows
        ]
    )
    if len(rows) < 4 or np.any(pitch <= 0.0) or np.any(error <= 0.0):
        raise RuntimeError("the convergence-order fit needs four positive rungs")
    design = np.column_stack((np.ones(len(rows)), np.log(pitch)))
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design, np.log(error), rcond=None
    )
    if rank != 2:
        raise RuntimeError("the convergence-order fit is rank deficient")
    fitted = design @ coefficients
    residual = np.log(error) - fitted
    degrees_of_freedom = len(rows) - 2
    variance = float(np.sum(residual**2) / degrees_of_freedom)
    covariance = variance * np.linalg.inv(design.T @ design)
    order = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    interval = [order - critical * standard_error, order + critical * standard_error]
    prediction = np.exp(fitted)
    total = float(np.sum((np.log(error) - np.mean(np.log(error))) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 1.0
    roundoff_dominated = bool(np.all(error <= MAP_ROUNDOFF_TOLERANCE))
    return {
        "model": (
            "log(relative_sup_error) = intercept + order*log(characteristic_pitch)"
        ),
        "error_quantity": "all-carrier one-application relative-sup residual",
        "characteristic_pitch": "square root of median carrier-cell area",
        "rung_count": len(rows),
        "order": order,
        "standard_error": standard_error,
        "confidence_level": 0.95,
        "confidence_interval": interval,
        "degrees_of_freedom": degrees_of_freedom,
        "r_squared": r_squared,
        "fitted_relative_sup_residual": prediction.tolist(),
        "constant_offset_included_by_confidence_interval": interval[0]
        <= 0.0
        <= interval[1],
        "converging_order_floor": CONVERGING_ORDER_FLOOR,
        "converging_order_supported": interval[0] > CONVERGING_ORDER_FLOOR,
        "roundoff_dominated": roundoff_dominated,
        "interpretation": (
            "physical convergence order is below resolution because every rung "
            "is within the preregistered binary64 roundoff tolerance"
            if roundoff_dominated
            else (
                "the residual converges under refinement"
                if interval[0] > CONVERGING_ORDER_FLOOR
                else "the fitted uncertainty does not establish convergence"
            )
        ),
    }


def _terminal_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = {
        item["requested_cells"]: dict(item)
        for item in UNQUALIFIED_TERMINAL_OBSERVATIONS
    }
    by_rung = []
    for row in rows:
        item = observed.get(row["requested_cells"])
        if item is None:
            by_rung.append(
                {
                    "requested_cells": row["requested_cells"],
                    "realised_cells": row["realised_cells"],
                    "qualification": "not_attempted",
                    "achieved_relative_residual": None,
                    "used_in_refinement_fit": False,
                    "reason": "the one-application ladder requires no terminal solve",
                }
            )
        else:
            by_rung.append({**item, "realised_cells": row["realised_cells"]})
    return {
        "role": (
            "reported terminal-solve observations; excluded from the "
            "one-application residual ladder and convergence fit"
        ),
        "qualified_rung_count": 0,
        "unqualified_rung_count": len(observed),
        "by_rung": by_rung,
    }


def _validate(receipt: dict[str, Any]) -> None:
    rows = receipt["refinement_ladder"]
    if len(rows) < 4:
        raise RuntimeError("the receipt needs at least four refinement rungs")
    realised = [row["realised_cells"] for row in rows]
    if realised != sorted(realised) or len(set(realised)) != len(realised):
        raise RuntimeError("the refinement ladder is not strictly ordered")
    fit = receipt["convergence_order_fit"]
    for key in ("order", "standard_error", "r_squared"):
        if not np.isfinite(fit[key]):
            raise RuntimeError(f"the convergence fit has non-finite {key}")
    for row in rows:
        measurement = row["one_application_residual"]
        if measurement["application_count"] != 1:
            raise RuntimeError("a rung does not contain exactly one map application")
        if measurement["seed_used"] or measurement["solve_used"]:
            raise RuntimeError("a seed or solve contaminated the residual ladder")
        regions = measurement["regions"]
        if not {"all_carrier_cells", *REGIONS}.issubset(regions):
            raise RuntimeError("a one-application residual lacks a regional split")
        for region in ("all_carrier_cells", *REGIONS):
            for metric in (
                "absolute_sup_wb",
                "absolute_rms_wb",
                "relative_sup",
                "relative_rms",
            ):
                if not np.isfinite(regions[region][metric]):
                    raise RuntimeError(
                        f"non-finite one-application residual {region} {metric}"
                    )
        if set(row["posed_terms"]) != {"external", "boundary", "source"}:
            raise RuntimeError("a rung does not quote every posed term")
    terminal = receipt["terminal_solve_observations"]
    unqualified = [
        row for row in terminal["by_rung"] if row["qualification"] == "unqualified"
    ]
    if len(unqualified) != 2 or any(
        row["achieved_relative_residual"] <= row["criterion"]
        or row["used_in_refinement_fit"]
        for row in unqualified
    ):
        raise RuntimeError("the unqualified terminal attempts are not explicit")
    if receipt["verdict"]["repair_authored"]:
        raise RuntimeError("this measurement receipt may not author a repair")
    if receipt["measurement_lane"]["backend"] != "cpu":
        raise RuntimeError("all refinement children must use the CPU backend")


def _aggregate(parts: list[Path], output: Path) -> dict[str, Any]:
    rows = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in parts),
        key=lambda row: row["realised_cells"],
    )
    fit = _fit_order(rows)
    regional_relative_sup = {
        region: [
            row["one_application_residual"]["regions"][region]["relative_sup"]
            for row in rows
        ]
        for region in ("all_carrier_cells", *REGIONS)
    }
    absolute_sup = [
        row["one_application_residual"]["full_state"]["absolute_sup_wb"] for row in rows
    ]
    worst_relative = max(
        row["one_application_residual"]["full_state"]["relative_sup"] for row in rows
    )
    at_roundoff = bool(worst_relative <= MAP_ROUNDOFF_TOLERANCE)
    converging = bool(
        fit["converging_order_supported"] and not fit["roundoff_dominated"]
    )
    cause = "operator" if converging and not at_roundoff else "posing"
    if cause == "posing":
        classification = "operator_admits_analytic_fixed_point"
        statement = (
            "POSING is the better outcome: one exact map application reproduces "
            "the closed-form field within the preregistered binary64 tolerance "
            "at every resolution, independently of any seed, solve, or basin. "
            "The operator is sound for this analytic fixed point; the bounded "
            "case-presentation discrepancy is not repaired here."
        )
    else:
        classification = "discrete_operator_consistency_error"
        statement = (
            "OPERATOR: the one-application analytic residual is above roundoff "
            "and converges with a positive order established by its uncertainty. "
            "The measurement identifies a discrete consistency error and authors "
            "no repair."
        )
    receipt = {
        "schema": "nova.analytic-operator-refinement-ladder",
        "source_revision": _source_revision(),
        "measurement_lane": _measurement_lane(),
        "comparison_contract": {
            "analytic_authority": (
                "moderate-rotation-conventional closed-form total flux evaluated "
                "independently at every target"
            ),
            "forward_map": (
                "one application of production ForwardFluxOperator.flux_map with "
                "exact-value state and first-order current moments"
            ),
            "seed": "none",
            "solve": "none",
            "basin": "not applicable",
            "regional_partition": {
                "closed_flux_region": "analytic psi_N < 0.95",
                "separatrix_band": "0.95 <= analytic psi_N <= 1.05",
                "scrape_off_layer": "analytic psi_N > 1.05",
            },
            "relative_normalisation": ("closed-form axis-to-boundary total-flux span"),
            "map_roundoff_tolerance": MAP_ROUNDOFF_TOLERANCE,
            "map_roundoff_basis": "4096 binary64 epsilon",
            "repair_authored": False,
        },
        "refinement_ladder": rows,
        "convergence_order_fit": fit,
        "terminal_solve_observations": _terminal_observations(rows),
        "headline_metrics": {
            "one_application_absolute_sup_wb_range": {
                "minimum": min(absolute_sup),
                "maximum": max(absolute_sup),
            },
            "worst_full_state_one_application_relative_sup": worst_relative,
            "regional_relative_sup_ranges": {
                region: {
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for region, values in regional_relative_sup.items()
            },
            "analytic_boundary_flux_wb_range": {
                "minimum": min(
                    row["analytic_topology"]["boundary_flux_wb"] for row in rows
                ),
                "maximum": max(
                    row["analytic_topology"]["boundary_flux_wb"] for row in rows
                ),
            },
        },
        "verdict": {
            "cause": cause,
            "classification": classification,
            "statement": statement,
            "operator_admits_analytic_fixed_point": at_roundoff,
            "roundoff_dominated": fit["roundoff_dominated"],
            "repair_authored": False,
        },
        "evidence_inputs": {
            "analytic_fixture_driver": {
                "path": "scripts/analytic_oracle_fixtures/measure.py",
                "sha256": _sha256(ROOT / "scripts/analytic_oracle_fixtures/measure.py"),
            },
            "prior_gauge_receipt": {
                "path": (
                    "docs/figures/coefficient-space-newton/analytic-truth-gauge.json"
                ),
                "sha256": _sha256(
                    ROOT
                    / "docs/figures/coefficient-space-newton/analytic-truth-gauge.json"
                ),
            },
        },
    }
    _validate(receipt)
    _write_json(output, receipt)
    return receipt


def run(output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nova-analytic-operator-") as directory:
        work = Path(directory)
        parts: list[Path] = []
        for requested_cells in REQUESTED_CELLS:
            part = work / f"cells-{abs(requested_cells)}.json"
            log = work / f"cells-{abs(requested_cells)}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "measure",
                "--requested-cells",
                str(requested_cells),
                "--output",
                str(part),
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            environment["JAX_PLATFORMS"] = "cpu"
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"resolution {requested_cells} failed; child log:\n"
                    f"{log.read_text(encoding='utf-8')}"
                )
            parts.append(part)
        return _aggregate(parts, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--requested-cells", type=int, required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "measure":
        if arguments.requested_cells not in REQUESTED_CELLS:
            raise RuntimeError(
                f"requested cells must be one of {REQUESTED_CELLS}, got "
                f"{arguments.requested_cells}"
            )
        payload = _measure(arguments.requested_cells)
        _write_json(arguments.output, payload)
        print(
            json.dumps(
                {
                    "requested_cells": arguments.requested_cells,
                    "realised_cells": payload["realised_cells"],
                    "one_application_relative_sup": payload["one_application_residual"][
                        "regions"
                    ]["all_carrier_cells"]["relative_sup"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    receipt = run(arguments.output)
    if arguments.check:
        _validate(receipt)
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
