"""Measure analytic map floors and residual tangents without solving.

Each row constructs the certificate's analytic exterior completion once, then
reuses that operator for the exact allocation and whole-cell control.  The
measurement applies each map at the analytic flux, linearizes the same frozen-
shadow residual used by the Newton inner iteration, and compares its tangent
with central finite differences.  It never enters a nonlinear or linear solve.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import fixed_point
from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/cut-cell-current-attribution/oracle-start"
    / "map-floor-jacobian.json"
)
DEFAULT_REPORT_DIRECTORY = DEFAULT_OUTPUT.parent
PART_DIRECTORY_NAME = "map-jacobian-parts"
FINITE_DIFFERENCE_STEPS = (1.0e-5, 1.0e-7)
RANDOM_DIRECTION_COUNT = 4
RANDOM_SEED = 271828
MODES = ("exact", "chord")
ROWS = (
    ("weak-rotation-reactor-static", -110),
    ("weak-rotation-reactor-static", -300),
    ("moderate-rotation-conventional-static", -110),
    ("moderate-rotation-conventional-static", -300),
    (certificate.DIVERTED_CASE_NAME, -300),
    (certificate.DIVERTED_CASE_NAME, -500),
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("the measurement requires one scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if cpus != 8:
        raise RuntimeError(f"expected eight CPUs, received {cpus}")
    if reservation != "gpu_0003_grpA":
        raise RuntimeError(f"unexpected reservation {reservation!r}")
    if platforms != "cuda,cpu":
        raise RuntimeError(f"expected JAX_PLATFORMS=cuda,cpu, received {platforms!r}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    ).strip()
    if "H200" not in gpu:
        raise RuntimeError(f"the measurement requires an H200, received {gpu!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": reservation,
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "1")),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "gpu": gpu,
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "jax_cuda_devices": [str(device) for device in jax.devices("gpu")],
        "jax_cpu_devices": [str(device) for device in jax.devices("cpu")],
    }


def _row_slug(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-cells-{abs(requested_cells)}"


def _part_path(output: Path, case_name: str, requested_cells: int, mode: str) -> Path:
    name = f"{_row_slug(case_name, requested_cells)}-{mode}.json"
    return output.parent / PART_DIRECTORY_NAME / name


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _pointer(
    function: Callable[..., Any], markers: tuple[str, ...] = ()
) -> dict[str, Any]:
    lines, start = inspect.getsourcelines(function)
    path = Path(inspect.getsourcefile(function) or "")
    try:
        rendered_path = str(path.relative_to(ROOT))
    except ValueError:
        rendered_path = str(path)
    located = []
    for marker in markers:
        matches = [index for index, line in enumerate(lines) if marker in line]
        if not matches:
            raise RuntimeError(
                f"source marker {marker!r} is absent from {rendered_path}"
            )
        located.append({"text": marker, "line": start + matches[0]})
    return {
        "path": rendered_path,
        "line_start": start,
        "line_end": start + len(lines) - 1,
        "markers": located,
    }


def _norms(delta: np.ndarray, span: float, grid_count: int) -> dict[str, float]:
    grid = np.asarray(delta, dtype=np.float64)[:grid_count]
    absolute_rms = float(np.sqrt(np.mean(grid**2)))
    absolute_sup = float(np.max(np.abs(grid)))
    return {
        "absolute_rms_wb": absolute_rms,
        "absolute_sup_wb": absolute_sup,
        "relative_rms_of_span": absolute_rms / span,
        "relative_sup_of_span": absolute_sup / span,
    }


def _topology(operator: Any, state: np.ndarray) -> dict[str, Any]:
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError as error:
        return {
            "read_status": "no_qualified_axis",
            "axis_flux_wb": None,
            "boundary_flux_wb": None,
            "exception_text": str(error),
        }
    return {
        "read_status": "qualified_axis",
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
        "exception_text": None,
    }


def _booked_current(
    operator: Any,
    analytic: np.ndarray,
    requested_class: int,
    target_current: float,
) -> dict[str, Any]:
    moments = operator.cell_current_moments(jnp.asarray(analytic), requested_class)
    booked = float(jnp.sum(moments.cell_current))
    amplitude = float(operator.current_normalisation_amplitude(target_current, booked))
    return {
        "booked_plasma_current_a": booked,
        "analytic_plasma_current_a": target_current,
        "booked_over_analytic": booked / target_current,
        "normalisation_amplitude": amplitude,
        "cell_current_sha256_binary64": _array_digest(moments.cell_current),
    }


def _smooth_directions(
    coordinates: np.ndarray, count: int, seed: int
) -> list[np.ndarray]:
    points = np.asarray(coordinates, dtype=np.float64)
    centre = np.mean(points, axis=0)
    scale = np.maximum(np.ptp(points, axis=0), 1.0e-12)
    normalized = (points - centre) / scale
    radius = normalized[:, 0]
    height = normalized[:, 1]
    basis = np.column_stack(
        (
            np.ones(len(points)),
            radius,
            height,
            radius * height,
            radius**2 - np.mean(radius**2),
            height**2 - np.mean(height**2),
            np.sin(np.pi * radius),
            np.cos(np.pi * height),
            np.sin(np.pi * (radius + height)),
            np.exp(-5.0 * (radius**2 + height**2)),
        )
    )
    generator = np.random.default_rng(seed)
    directions: list[np.ndarray] = []
    for _ in range(count):
        direction = basis @ generator.normal(size=basis.shape[1])
        direction -= np.mean(direction)
        norm = float(np.max(np.abs(direction)))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("the smooth-direction instrument produced no signal")
        directions.append(direction / norm)
    return directions


def _certificate_residual(candidate, shadowed_map, frozen_shadow):
    """Return the I-minus-J residual action used by the Newton inner solve."""
    return candidate - shadowed_map(candidate, frozen_shadow)


def _relative_discrepancy(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(candidate - reference))
    denominator = max(
        float(np.linalg.norm(reference)),
        float(np.linalg.norm(candidate)),
        np.finfo(np.float64).tiny,
    )
    return numerator / denominator


def _jacobian_probe(
    operator: Any,
    analytic: np.ndarray,
    coordinates: np.ndarray,
    span: float,
    requested_class: int,
    target_current: float,
) -> dict[str, Any]:
    shadowed_map = operator.flux_map_with_shadow(
        requested_class=requested_class,
        target_current=target_current,
    )
    state = jnp.asarray(analytic)
    frozen_shadow = operator.residual_shadow_mask(state, requested_class)

    def residual(candidate):
        return _certificate_residual(candidate, shadowed_map, frozen_shadow)

    residual_at_analytic, tangent = jax.linearize(residual, state)
    directions = _smooth_directions(coordinates, RANDOM_DIRECTION_COUNT, RANDOM_SEED)
    records = []
    for index, direction in enumerate(directions):
        exact_jvp = np.asarray(tangent(jnp.asarray(direction)), dtype=np.float64)
        step_records = []
        for relative_step in FINITE_DIFFERENCE_STEPS:
            absolute_step = relative_step * span
            plus = np.asarray(
                residual(state + absolute_step * jnp.asarray(direction)),
                dtype=np.float64,
            )
            minus = np.asarray(
                residual(state - absolute_step * jnp.asarray(direction)),
                dtype=np.float64,
            )
            finite_difference = (plus - minus) / (2.0 * absolute_step)
            step_records.append(
                {
                    "relative_step_of_flux_span": relative_step,
                    "absolute_step_wb": absolute_step,
                    "relative_jvp_discrepancy": _relative_discrepancy(
                        exact_jvp, finite_difference
                    ),
                    "exact_jvp_rms": float(np.sqrt(np.mean(exact_jvp**2))),
                    "finite_difference_rms": float(
                        np.sqrt(np.mean(finite_difference**2))
                    ),
                    "finite_difference_detected_nonzero_action": bool(
                        np.any(finite_difference != 0.0)
                    ),
                }
            )
        records.append(
            {
                "direction": f"smooth_random_{index + 1}",
                "unit_sup_direction_sha256": _array_digest(direction),
                "exact_jvp_detected_nonzero_action": bool(np.any(exact_jvp != 0.0)),
                "relative_steps": step_records,
            }
        )
    jax.block_until_ready(residual_at_analytic)
    return {
        "residual_definition": (
            "state minus the certificate's shadow-frozen target-normalised map; "
            "fixed_point.newton_krylov forms the same I-minus-J linear action "
            "after linearizing its frozen map"
        ),
        "nonlinear_solve_entered": False,
        "linear_solve_entered": False,
        "random_seed": RANDOM_SEED,
        "residual_at_analytic_rms_wb": float(
            np.sqrt(np.mean(np.asarray(residual_at_analytic, dtype=np.float64) ** 2))
        ),
        "directions": records,
        "source_pointers": {
            "benchmark_residual": _pointer(
                _certificate_residual,
                ("return candidate - shadowed_map(candidate, frozen_shadow)",),
            ),
            "production_inner_newton": _pointer(
                fixed_point._newton_krylov_inner,
                (
                    "mapped, tangent = jax.linearize(frozen_map, state)",
                    "residual_vector = mapped - state",
                    "return vector - tangent(vector)",
                ),
            ),
        },
    }


def _mode_measure(
    output: Path,
    case_name: str,
    requested_cells: int,
    mode: str,
    operator: Any,
    analytic: np.ndarray,
    coordinates: np.ndarray,
    grid_count: int,
    span: float,
    requested_class: int,
    target_current: float,
    exact_internal: np.ndarray,
) -> dict[str, Any]:
    started = perf_counter()
    set_support_clip_mode(mode)
    if support_clip_mode() != mode:
        raise RuntimeError(f"clip-mode setter did not select {mode}")

    external = np.asarray(operator.external(), dtype=np.float64)
    analytic_moment_map = external + exact_internal
    unscaled_map = operator.flux_map(requested_class=requested_class)
    certificate_map = operator.flux_map(
        requested_class=requested_class,
        target_current=target_current,
    )
    unscaled_mapped = np.asarray(
        jax.block_until_ready(unscaled_map(jnp.asarray(analytic))), dtype=np.float64
    )
    certificate_mapped = np.asarray(
        jax.block_until_ready(certificate_map(jnp.asarray(analytic))),
        dtype=np.float64,
    )
    part = _part_path(output, case_name, requested_cells, mode)
    measured = {
        "schema": "nova.oracle-start-map-jacobian-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "mode": mode,
        "mode_semantics": (
            "signed-flux spline-chain exact clip"
            if mode == "exact"
            else "production whole-cell booking control"
        ),
        "exterior_term": {
            "definition": (
                "analytic total flux minus the analytically integrated exact "
                "plasma-current moment image; it is constructed before selecting "
                "a clip mode and is identical for exact and control"
            ),
            "sha256_binary64": _array_digest(external),
            "source_pointers": {
                "construction": _pointer(
                    run,
                    (
                        "analytic - exact_internal",
                        "operator = oracle_fixture.forward_operator",
                    ),
                ),
                "fixture_image": _pointer(
                    oracle_fixture._internal_flux_image,
                    ("return np.asarray(",),
                ),
            },
        },
        "residual_definitions": {
            "map_floor": (
                "mapped analytic flux minus analytic flux, normalized only by "
                "the analytic grid-flux span"
            ),
            "certificate_relative_residual": (
                "max(abs(mapped-state)) / max(abs(mapped)); recorded as a pointer "
                "but not substituted for the requested span-normalized map floor"
            ),
            "jacobian": (
                "state minus the target-normalised map on the residual shadow "
                "frozen at the analytic state"
            ),
            "source_pointers": {
                "production_map": _pointer(
                    ForwardFluxOperator.flux_map,
                    (
                        "external = self.external",
                        "return self._exclude_shadow_residual",
                    ),
                ),
                "production_relative_residual": _pointer(
                    fixed_point._relative_residual,
                    ("return jnp.max(jnp.abs(mapped - state))",),
                ),
            },
        },
        "one_application": {
            "analytic_moments_anchor": {
                "definition": "external plus analytically integrated exact moments",
                **_norms(analytic_moment_map - analytic, span, grid_count),
            },
            "unscaled_production_map": {
                "definition": (
                    "production allocation and moment conversion without "
                    "target-current normalisation, requested topology class fixed"
                ),
                **_norms(unscaled_mapped - analytic, span, grid_count),
            },
            "certificate_target_normalised_map": {
                "definition": (
                    "the production certificate map with requested topology class and "
                    "analytic total-current target"
                ),
                **_norms(certificate_mapped - analytic, span, grid_count),
            },
            "booked_current": _booked_current(
                operator, analytic, requested_class, target_current
            ),
        },
        "jacobian": None,
        "completed": False,
        "wall_seconds": None,
    }
    _write_json(part, measured)
    certificate_floor = measured["one_application"]["certificate_target_normalised_map"]
    print(
        f"MAP_FLOOR case={case_name} cells={abs(requested_cells)} mode={mode} "
        f"rms={certificate_floor['relative_rms_of_span']:.8e} "
        f"sup={certificate_floor['relative_sup_of_span']:.8e}",
        flush=True,
    )
    measured["jacobian"] = _jacobian_probe(
        operator,
        analytic,
        coordinates,
        span,
        requested_class,
        target_current,
    )
    measured["wall_seconds"] = perf_counter() - started
    measured["completed"] = True
    _write_json(part, measured)
    print(
        f"JACOBIAN_DONE case={case_name} cells={abs(requested_cells)} mode={mode}",
        flush=True,
    )
    return measured


def _row_explanation(modes: dict[str, Any]) -> dict[str, Any]:
    exact = modes["exact"]["one_application"]
    control = modes["chord"]["one_application"]
    anchor = exact["analytic_moments_anchor"]["relative_sup_of_span"]
    exact_floor = exact["certificate_target_normalised_map"]["relative_sup_of_span"]
    control_floor = control["certificate_target_normalised_map"]["relative_sup_of_span"]
    same_exterior = (
        modes["exact"]["exterior_term"]["sha256_binary64"]
        == modes["chord"]["exterior_term"]["sha256_binary64"]
    )
    if anchor > 1.0e-10:
        classification = "exterior_completion_does_not_close_exact_moments"
        sentence = (
            "The analytic-moment anchor itself misses the analytic flux, so the "
            "exterior completion is the first inconsistent term."
        )
    elif exact_floor > 5.0 * max(control_floor, 1.0e-12):
        classification = "exact_allocation_or_moment_path"
        sentence = (
            "The same exterior and residual close analytic moments to roundoff, "
            "while the exact production allocation has a substantially larger floor "
            "than whole-cell booking; the floor enters through the exact allocation "
            "or its production moment conversion, not the exterior or residual sign."
        )
    elif max(exact_floor, control_floor) > 1.0e-8:
        classification = "shared_production_map_path"
        sentence = (
            "Both allocation modes miss despite a roundoff analytic-moment anchor, "
            "so a production-map term shared by both modes is responsible."
        )
    else:
        classification = "analytic_fixed_point_admitted"
        sentence = (
            "Both production allocation modes admit the analytic fixed point at the "
            "measured precision."
        )
    return {
        "classification": classification,
        "same_exterior_sha256": same_exterior,
        "analytic_moment_anchor_relative_sup": anchor,
        "exact_certificate_map_relative_sup": exact_floor,
        "whole_cell_certificate_map_relative_sup": control_floor,
        "sentence": sentence,
    }


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# Analytic map floor and residual tangent",
        "",
        "No Newton or linear solve was entered. Every row-mode part records the "
        "exterior construction and the exact source lines for the map, relative "
        "residual, benchmark residual, and production I-minus-J action.",
        "",
        "| Row | Mode | Anchor sup | Certificate map rms / sup | "
        "Booked / analytic | Worst JVP discrepancy at 1e-5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in receipt["rows"]:
        for mode in MODES:
            measured = row["modes"][mode]
            application = measured["one_application"]
            anchor = application["analytic_moments_anchor"]
            mapped = application["certificate_target_normalised_map"]
            current = application["booked_current"]
            discrepancies = [
                step["relative_jvp_discrepancy"]
                for direction in measured["jacobian"]["directions"]
                for step in direction["relative_steps"]
                if step["relative_step_of_flux_span"] == 1.0e-5
            ]
            lines.append(
                f"| {row['case']} {abs(row['requested_cells'])} | {mode} | "
                f"{anchor['relative_sup_of_span']:.3e} | "
                f"{mapped['relative_rms_of_span']:.3e} / "
                f"{mapped['relative_sup_of_span']:.3e} | "
                f"{current['booked_over_analytic']:.8f} | "
                f"{max(discrepancies):.3e} |"
            )
        lines.extend(("", row["explanation"]["sentence"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, report_directory: Path) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    started = perf_counter()
    lane = _allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    original_mode = support_clip_mode()
    rows = []
    try:
        for case_name, requested_cells in ROWS:
            carrier_case, source_case, exact = certificate._case(case_name)
            machine = certificate._case_machine(
                case_name, carrier_case, exact, requested_cells
            )
            coordinates = np.vstack(
                (machine.node, machine.wall_node, machine.sample_coordinates)
            )
            analytic = certificate._exact_state(case_name, exact, coordinates)
            empty_operator = oracle_fixture.forward_operator(source_case, machine)
            exact_physical = oracle_fixture.exact_current_moments(
                source_case, empty_operator, analytic
            )
            exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
            exact_internal = np.asarray(
                oracle_fixture._internal_flux_image(empty_operator, exact_coefficients),
                dtype=np.float64,
            )
            operator = oracle_fixture.forward_operator(
                source_case,
                machine,
                analytic - exact_internal,
            )
            target_current, _centroid, target_receipt = (
                certificate._closed_form_current_target(
                    case_name, source_case, operator, exact_physical
                )
            )
            requested_class = int(
                TopologyClass.DIVERTED
                if certificate._is_diverted_case(case_name)
                else TopologyClass.LIMITED
            )
            topology = _topology(operator, analytic)
            if topology["axis_flux_wb"] is None:
                raise RuntimeError(
                    f"the analytic topology instrument could not read {case_name}"
                )
            span = abs(
                float(topology["axis_flux_wb"]) - float(topology["boundary_flux_wb"])
            )
            if not np.isfinite(span) or span <= 0.0:
                raise RuntimeError(f"the analytic flux span is invalid for {case_name}")
            modes = {
                mode: _mode_measure(
                    output,
                    case_name,
                    requested_cells,
                    mode,
                    operator,
                    analytic,
                    coordinates,
                    len(machine.node),
                    span,
                    requested_class,
                    target_current,
                    exact_internal,
                )
                for mode in MODES
            }
            row = {
                "case": case_name,
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "state_dimension": len(analytic),
                "analytic_flux_span_wb": span,
                "analytic_current_target_a": target_current,
                "analytic_current_target_receipt": target_receipt,
                "interaction_matrix_cache": machine.cache,
                "interaction_matrix_construction_count": 1,
                "modes": modes,
                "explanation": _row_explanation(modes),
            }
            rows.append(row)
            _write_json(
                output.parent
                / PART_DIRECTORY_NAME
                / f"{_row_slug(case_name, requested_cells)}.json",
                row,
            )
    finally:
        set_support_clip_mode(original_mode)
    receipt = {
        "schema": "nova.oracle-start-map-jacobian",
        "version": 1,
        "source_revision": _source_revision(),
        "production_code_modified": False,
        "nonlinear_solve_entered": False,
        "linear_solve_entered": False,
        "lane": {
            **lane,
            "persistent_compilation_cache": cache.receipt(),
            "wall_seconds": perf_counter() - started,
            "exit_marker": "ORACLE_START_MAP_JACOBIAN_EXIT=0",
        },
        "design": {
            "rows": [
                {"case": case_name, "requested_cells": requested_cells}
                for case_name, requested_cells in ROWS
            ],
            "modes": {
                "exact": "signed-flux spline-chain exact clip",
                "chord": "production whole-cell booking control",
            },
            "finite_difference_relative_steps": FINITE_DIFFERENCE_STEPS,
            "random_jacobian_directions": RANDOM_DIRECTION_COUNT,
            "random_seed": RANDOM_SEED,
            "interaction_matrix_policy": (
                "one cached machine and operator per row, reused across both modes"
            ),
        },
        "rows": rows,
    }
    _write_json(output, receipt)
    _write_report(report_directory / "map-jacobian-report.md", receipt)
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    receipt = run(arguments.output, arguments.report_directory)
    print(
        json.dumps(
            {
                "completed_rows": len(receipt["rows"]),
                "nonlinear_solve_entered": receipt["nonlinear_solve_entered"],
                "linear_solve_entered": receipt["linear_solve_entered"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("ORACLE_START_MAP_JACOBIAN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
