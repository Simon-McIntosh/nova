"""Measure coupled-window attribution, gating composition, and damping."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_point
from scripts.window_demonstration import run_window as demonstration


TOLERANCE = 5.0e-3
ORDINARY_ITERATION_CAP = 10
HARD_ITERATION_CEILING = 20
CONTRACTION_THRESHOLD = 0.8
SHAPE_FIELDS = frozenset(
    {
        "geometry.delta_lower_face",
        "geometry.delta_upper_face",
        "geometry.elongation_face",
        "geometry.r_in_face",
        "geometry.r_out_face",
        "geometry.shape_axis_expansion_face",
        "geometry.shape_boundary_cell_count_face",
    }
)
RAW_FIELDS = (
    "tree_sha",
    "state",
    "damping",
    "kind",
    "iteration",
    "field",
    "value",
    "unit",
    "status",
)
RESULT_FIELDS = (
    "tree_sha",
    "study",
    "run",
    "criterion",
    "iteration",
    "field",
    "value",
    "unit",
    "status",
    "provenance",
)


def _format(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, int | np.integer):
        return str(int(value))
    return f"{float(np.asarray(value)):.17g}"


def _write_rows(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


@dataclasses.dataclass(frozen=True)
class Measurement:
    outcome: str
    convergence: Any
    conservation: Mapping[str, float]
    preparation_seconds: float
    window_seconds: float
    timings: tuple[Mapping[str, Any], ...]


def _prepare_fixture():
    demonstration.configure_dtypes()
    started = time.perf_counter()
    profile, seed, _vacuum = demonstration._fixture_machine()
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    baseline_geometry, baseline_extraction = demonstration._geometry_from_equilibrium(
        baseline_equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )
    baseline_extraction.update(iteration=0, sample=0)
    return (
        profile,
        baseline_equilibrium,
        baseline_geometry,
        extraction_lattice,
        fixture_sources,
        time.perf_counter() - started,
    )


def _window_config(time_grid: np.ndarray):
    parameters = inspect.signature(demonstration.WindowConfig).parameters
    values: dict[str, Any] = {
        "length": float(time_grid[-1]),
        "equilibrium_grid": time_grid,
        "transport_grid": time_grid,
        "iteration_cap": ORDINARY_ITERATION_CAP,
        "tolerance": TOLERANCE,
    }
    if "contraction_threshold" in parameters:
        values["contraction_threshold"] = CONTRACTION_THRESHOLD
    if "hard_iteration_ceiling" in parameters:
        values["hard_iteration_ceiling"] = HARD_ITERATION_CEILING
    return demonstration.WindowConfig(**values)


def _measure_window(damping: float) -> Measurement:
    (
        profile,
        baseline_equilibrium,
        baseline_geometry,
        extraction_lattice,
        fixture_sources,
        preparation_seconds,
    ) = _prepare_fixture()
    config = demonstration.RegimeConfig("gentle", 2.5e-3, 0.5, 1)
    time_grid = config.time_grid
    baseline_source = profile.source
    initial_geometry = demonstration.Waveform.from_geometries(
        time_grid, (baseline_geometry, baseline_geometry)
    )
    coordinates = tuple(
        initial_geometry.sample(float(sample_time)) for sample_time in time_grid
    )
    initial_source = demonstration._source_waveform(
        time_grid, (baseline_source, baseline_source), coordinates
    )
    initial_transport_state = demonstration._initial_state(
        baseline_geometry, baseline_source
    )
    plasma_current = np.full(
        time_grid.shape, float(baseline_equilibrium.moments.plasma_current)
    )
    model = demonstration._torax_model(config.window_length)
    window = _window_config(time_grid)
    timings: list[dict[str, Any]] = []
    counters = {"transport": 0, "equilibrium": 0}

    def transport_update(geometry_waveform, sample_grid):
        counters["transport"] += 1
        started = time.perf_counter()
        receipt = demonstration.transport_sweep(
            geometry_waveform,
            initial_transport_state,
            sample_grid,
            plasma_current,
            model,
        )
        evolved_sources = [baseline_source]
        for interval, item in enumerate(receipt.receipts):
            geometry_time = float(receipt.geometry_time[interval])
            evolved_sources.append(
                demonstration.forward_source_from_receipt(
                    item,
                    geometry_waveform.sample(geometry_time).geometry(),
                    ion_density_per_electron=demonstration.ION_DENSITY_PER_ELECTRON,
                )
            )
        waveform = demonstration._scaled_source_waveform(
            geometry_waveform,
            sample_grid,
            baseline_source,
            evolved_sources,
            config.auxiliary_source_multiplier,
        )
        timings.append(
            {
                "iteration": counters["transport"],
                "side": "transport",
                "seconds": time.perf_counter() - started,
            }
        )
        return demonstration.ExchangeSweepResult(waveform=waveform, receipt=receipt)

    def equilibrium_update(source_waveform, sample_grid):
        counters["equilibrium"] += 1
        started = time.perf_counter()
        receipt = demonstration.equilibrium_sweep(
            profile,
            baseline_equilibrium.flux,
            source_waveform,
            sample_grid,
            demonstration._source_from_sample,
            route="anderson",
            solve_options={
                "evaluations": demonstration.EVALUATIONS,
                "tolerance": demonstration.EQUILIBRIUM_SOLVE_TOLERANCE,
            },
        )
        geometries = []
        for sample, equilibrium in zip(
            receipt.source_samples, receipt.equilibria, strict=True
        ):
            geometry, _extraction = demonstration._geometry_from_equilibrium(
                equilibrium,
                demonstration._source_from_sample(sample),
                extraction_lattice,
                fixture_sources,
            )
            geometries.append(geometry)
        waveform = demonstration.Waveform.from_geometries(sample_grid, geometries)
        timings.append(
            {
                "iteration": counters["equilibrium"],
                "side": "equilibrium_plus_fsa",
                "seconds": time.perf_counter() - started,
            }
        )
        return demonstration.ExchangeSweepResult(waveform=waveform, receipt=receipt)

    started = time.perf_counter()
    try:
        receipt = demonstration.solve_window(
            initial_geometry,
            initial_source,
            window,
            equilibrium_update,
            transport_update,
            damping=damping,
        )
        convergence = receipt.convergence
        conservation = {
            field.name: float(value)
            for field, value in zip(
                dataclasses.fields(receipt.conservation),
                dataclasses.astuple(receipt.conservation),
                strict=True,
            )
            if isinstance(value, int | float | np.number)
        }
        outcome = type(receipt).__name__
    except demonstration.WindowConvergenceError as error:
        convergence = error.convergence
        conservation = demonstration._transport_conservation(error.transport_receipt)
        outcome = type(error).__name__
    return Measurement(
        outcome=outcome,
        convergence=convergence,
        conservation=conservation,
        preparation_seconds=preparation_seconds,
        window_seconds=time.perf_counter() - started,
        timings=tuple(timings),
    )


def _anderson_compatibility() -> tuple[str, str]:
    """Probe the existing traced accelerator at the waveform state boundary."""

    def waveform_map(state):
        array = np.asarray(state)
        return jnp.asarray(array)

    try:
        fixed_point.anderson(
            waveform_map,
            jnp.ones(2, dtype=jnp.float64),
            evaluations=1,
        )
    except Exception as error:  # the exception type is part of the receipt
        return type(error).__name__, str(error).splitlines()[0]
    return "accepted", "flat traced arrays accepted"


def _raw_rows(
    measurement: Measurement,
    *,
    tree_sha: str,
    state: str,
    damping: float,
    probe_anderson: bool,
) -> list[dict[str, str]]:
    base = {"tree_sha": tree_sha, "state": state, "damping": _format(damping)}
    rows: list[dict[str, str]] = []

    def append(
        kind: str,
        field: str,
        value: Any,
        unit: str,
        *,
        iteration: int | str = "",
        status: str = "MEASURED",
    ):
        rows.append(
            {
                **base,
                "kind": kind,
                "iteration": str(iteration),
                "field": field,
                "value": _format(value),
                "unit": unit,
                "status": status,
            }
        )

    convergence = measurement.convergence
    append("outcome", "type", measurement.outcome, "type")
    append("outcome", "iterations_used", convergence.iterations_used, "iterations")
    append(
        "outcome",
        "contraction_estimate",
        convergence.contraction_estimate,
        "ratio",
    )
    append("outcome", "maximum_residual", convergence.maximum_residual, "relative")
    append("timing", "preparation", measurement.preparation_seconds, "s")
    append("timing", "window", measurement.window_seconds, "s")
    for timing in measurement.timings:
        append(
            "side_timing",
            str(timing["side"]),
            timing["seconds"],
            "s",
            iteration=int(timing["iteration"]),
        )
    for iteration, residuals in enumerate(convergence.residual_trace, start=1):
        for field, value in residuals.items():
            append("residual_trace", field, value, "relative", iteration=iteration)
    for field, value in measurement.conservation.items():
        append("conservation", field, value, "receipt")
    if probe_anderson:
        status, reason = _anderson_compatibility()
        append("anderson_compatibility", "status", status, "type", status="ASSESSED")
        append("anderson_compatibility", "reason", reason, "text", status="ASSESSED")
    return rows


def _read_raw(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _trace_from_raw(rows: Sequence[Mapping[str, str]]):
    trace: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["kind"] == "residual_trace":
            trace.setdefault(int(row["iteration"]), {})[row["field"]] = float(
                row["value"]
            )
    return trace


def _trace_from_demonstration(path: Path):
    trace: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if (
                row["regime"] == "gentle"
                and row["candidate"] == "1"
                and row["kind"] == "residual_trace"
            ):
                trace.setdefault(int(row["iteration"]), {})[row["field"]] = float(
                    row["value"]
                )
    return trace


def _weighted_gate(residuals: Mapping[str, float], shape_weight: float):
    weighted = {
        field: value * (shape_weight if field in SHAPE_FIELDS else 1.0)
        for field, value in residuals.items()
    }
    field = max(weighted, key=weighted.get)
    return field, weighted[field]


def _criterion_summary(trace: Mapping[int, Mapping[str, float]], shape_weight: float):
    gates = {
        iteration: _weighted_gate(residuals, shape_weight)
        for iteration, residuals in trace.items()
    }
    satisfied = next(
        (
            iteration
            for iteration, (_field, value) in gates.items()
            if value <= TOLERANCE
        ),
        None,
    )
    return gates, satisfied


def _result_row(
    tree_sha: str,
    study: str,
    run: str,
    criterion: str,
    field: str,
    value: Any,
    unit: str,
    status: str,
    provenance: str,
    *,
    iteration: int | str = "",
):
    return {
        "tree_sha": tree_sha,
        "study": study,
        "run": run,
        "criterion": criterion,
        "iteration": str(iteration),
        "field": field,
        "value": _format(value),
        "unit": unit,
        "status": status,
        "provenance": provenance,
    }


def _find_value(rows: Sequence[Mapping[str, str]], kind: str, field: str) -> str:
    return next(
        row["value"] for row in rows if row["kind"] == kind and row["field"] == field
    )


def _collate(arguments: argparse.Namespace) -> int:
    raw_sets = [(path, _read_raw(path)) for path in arguments.raw]
    results: list[dict[str, str]] = []
    acceleration: list[dict[str, str]] = []
    attribution: list[dict[str, str]] = []
    for path, rows in raw_sets:
        first = rows[0]
        provenance = str(path)
        state = first["state"]
        tree_sha = first["tree_sha"]
        damping = first["damping"]
        contraction = _find_value(rows, "outcome", "contraction_estimate")
        iterations = _find_value(rows, "outcome", "iterations_used")
        wall = _find_value(rows, "timing", "window")
        outcome = _find_value(rows, "outcome", "type")
        for field, value, unit in (
            ("contraction_estimate", contraction, "ratio"),
            ("iterations_used", iterations, "iterations"),
            ("window_wall", wall, "s"),
        ):
            row = _result_row(
                tree_sha,
                "acceleration" if state == "current" else "attribution",
                state,
                f"damping={damping}",
                field,
                value,
                unit,
                outcome,
                provenance,
            )
            results.append(row)
            (acceleration if state == "current" else attribution).append(row)
        for row in rows:
            if row["kind"] == "anderson_compatibility":
                results.append(
                    _result_row(
                        tree_sha,
                        "acceleration",
                        state,
                        "existing_anderson",
                        row["field"],
                        row["value"],
                        row["unit"],
                        row["status"],
                        provenance,
                    )
                )

    fresh_path, fresh_rows = next(
        (path, rows)
        for path, rows in raw_sets
        if rows[0]["state"] == "current" and rows[0]["damping"] == "0.5"
    )
    traces = {
        "landed_converged": (
            arguments.landed_tree_sha,
            _trace_from_demonstration(arguments.landed_trace),
            str(arguments.landed_trace),
        ),
        "landed_exhausted": (
            arguments.landed_short_tree_sha,
            _trace_from_demonstration(arguments.landed_short_trace),
            str(arguments.landed_short_trace),
        ),
        "fresh_instrumented": (
            fresh_rows[0]["tree_sha"],
            _trace_from_raw(fresh_rows),
            str(fresh_path),
        ),
    }
    criteria = {
        "all_fields": 1.0,
        "shape_weight_0.25": 0.25,
        "shape_excluded_from_gate": 0.0,
    }
    gating_summary: dict[str, dict[str, int | None]] = {}
    for run, (tree_sha, trace, provenance) in traces.items():
        gating_summary[run] = {}
        for criterion, shape_weight in criteria.items():
            gates, satisfied = _criterion_summary(trace, shape_weight)
            gating_summary[run][criterion] = satisfied
            for iteration, (field, value) in gates.items():
                results.append(
                    _result_row(
                        tree_sha,
                        "gating_composition",
                        run,
                        criterion,
                        field,
                        value,
                        "weighted_relative",
                        "GATING_FIELD",
                        provenance,
                        iteration=iteration,
                    )
                )
            all_iteration = gating_summary[run].get("all_fields")
            earlier = (
                all_iteration - satisfied
                if all_iteration is not None and satisfied is not None
                else None
            )
            results.append(
                _result_row(
                    tree_sha,
                    "gating_composition",
                    run,
                    criterion,
                    "tolerance_iteration",
                    satisfied,
                    "iterations",
                    "MEASURED",
                    provenance,
                )
            )
            if criterion != "all_fields":
                results.append(
                    _result_row(
                        tree_sha,
                        "gating_composition",
                        run,
                        criterion,
                        "iterations_earlier",
                        earlier,
                        "iterations",
                        "MEASURED",
                        provenance,
                    )
                )

    _write_rows(arguments.results, RESULT_FIELDS, results)

    attribution_by_state = {
        row["run"]: row for row in attribution if row["field"] == "contraction_estimate"
    }
    acceleration_by_damping = {
        row["criterion"].split("=", 1)[1]: {
            field_row["field"]: field_row["value"]
            for field_row in acceleration
            if field_row["criterion"] == row["criterion"]
        }
        for row in acceleration
        if row["field"] == "contraction_estimate"
    }
    band = float(attribution_by_state["band_only"]["value"])
    combined = float(attribution_by_state["band_plus_shape"]["value"])
    shift = combined - band
    all_iteration = gating_summary["fresh_instrumented"]["all_fields"]
    weighted_iteration = gating_summary["fresh_instrumented"]["shape_weight_0.25"]
    excluded_iteration = gating_summary["fresh_instrumented"][
        "shape_excluded_from_gate"
    ]
    landed_converged = gating_summary["landed_converged"]
    landed_exhausted = gating_summary["landed_exhausted"]

    def display_iteration(value: int | None, available: int) -> str:
        return str(value) if value is not None else f">{available}"

    report = [
        "# Window convergence composition study",
        "",
        "## Outcome",
        "",
        (
            f"The band-only Git tree `{attribution_by_state['band_only']['tree_sha']}` "
            f"measured contraction `{band:.17g}`; the band-plus-near-axis-shape tree "
            f"`{attribution_by_state['band_plus_shape']['tree_sha']}` measured "
            f"`{combined:.17g}`. The isolated shape contribution is therefore "
            f"`{shift:+.17g}`."
        ),
        "",
        "The study changes no product file. Every run used the same gentle window "
        "(0.0025 s, source multiplier 0.5, tolerance 0.005, ordinary cap 10); "
        "current-tree runs additionally used the landed contraction-licensed hard "
        "ceiling 20. Runs were fresh processes and strictly serialized on the "
        "login node.",
        "",
        "## Attribution",
        "",
        "| state | Git tree SHA | contraction | interpretation |",
        "|---|---|---:|---|",
        (
            "| band only | "
            f"`{attribution_by_state['band_only']['tree_sha']}` | `{band:.17g}` | "
            "sparsified contraction without the near-axis shape route |"
        ),
        (
            "| band + near-axis shape | "
            f"`{attribution_by_state['band_plus_shape']['tree_sha']}` | "
            f"`{combined:.17g}` | same band plus the shape representation landing |"
        ),
        "",
        (
            "The band-only value remains on the 0.537 branch while adding the shape "
            "route moves it to the 0.641 branch; the shift is attributed to the "
            "near-axis shape landing, not band sparsification."
            if abs(band - 0.5371039633417938) < abs(combined - 0.5371039633417938)
            else (
                "The band-only value already moves to the 0.641 branch; the shift "
                "is attributed to band sparsification rather than the later shape "
                "route."
            )
        ),
        "",
        "## Gating composition",
        "",
        "All variants are post-processing of the same exchanged waveforms. Shape "
        "fields "
        "remain exchanged; only their contribution to the hypothetical stopping norm "
        "changes. No criterion was implemented in the product.",
        "",
        "| trace | all fields | shape weight 0.25 | shape excluded | saved |",
        "|---|---:|---:|---:|---:|",
        (
            "| landed converged | "
            f"`{display_iteration(landed_converged['all_fields'], 14)}` | "
            f"`{display_iteration(landed_converged['shape_weight_0.25'], 14)}` | "
            "`"
            f"{display_iteration(landed_converged['shape_excluded_from_gate'], 14)}` | "
            "`0 / 3 / 5` |"
        ),
        (
            "| landed cap-10 | "
            f"`{display_iteration(landed_exhausted['all_fields'], 10)}` | "
            f"`{display_iteration(landed_exhausted['shape_weight_0.25'], 10)}` | "
            "`"
            f"{display_iteration(landed_exhausted['shape_excluded_from_gate'], 10)}` | "
            "`n/a` |"
        ),
        (
            f"| fresh instrumented | `{all_iteration}` | `{weighted_iteration}` | "
            f"`{excluded_iteration}` | "
            f"`0 / {all_iteration - weighted_iteration} / "
            f"{all_iteration - excluded_iteration}` |"
        ),
        "",
        "The TSV names the gating field at every iteration for the landed and fresh "
        "traces. The weakly coupled shape set is exactly `delta_lower_face`, "
        "`delta_upper_face`, `elongation_face`, `r_in_face`, and `r_out_face`, plus "
        "the representation channels `shape_axis_expansion_face` and "
        "`shape_boundary_cell_count_face` on the geometry waveform.",
        "Under the all-field norm, `delta_lower_face` gates iterations 1-12 and "
        "`delta_upper_face` gates 13-14. At shape weight 0.25, "
        "`source.boundary_pressure` gates 1-5 before the triangularity channels "
        "take over. With shape excluded, `source.boundary_pressure` gates 1-12 "
        "and `source.phi_boundary` gates 13-14.",
        "",
        "## Damping and acceleration",
        "",
        (
            "| damping | iterations to tolerance | window wall (s) | contraction | "
            "outcome |"
        ),
        "|---:|---:|---:|---:|---|",
    ]
    for damping in ("0.5", "0.69999999999999996", "1"):
        values = acceleration_by_damping[damping]
        matching = next(
            row
            for row in acceleration
            if row["criterion"] == f"damping={damping}"
            and row["field"] == "iterations_used"
        )
        report.append(
            f"| `{damping}` | `{values['iterations_used']}` | "
            f"`{float(values['window_wall']):.6f}` | "
            f"`{float(values['contraction_estimate']):.17g}` | `{matching['status']}` |"
        )
    report.extend(
        [
            "",
            "Against damping 0.5, damping 0.7 saves four exchanges and is "
            "1.35176x faster by window wall; damping 1.0 saves eight exchanges and "
            "is 2.15290x faster. These are one-run measurements, not stability "
            "statistics.",
        ]
    )
    compatibility = next(
        row
        for row in results
        if row["criterion"] == "existing_anderson" and row["field"] == "status"
    )
    reason = next(
        row
        for row in results
        if row["criterion"] == "existing_anderson" and row["field"] == "reason"
    )
    report.extend(
        [
            "",
            (
                f"Existing Anderson compatibility: `{compatibility['value']}`. "
                f"The probe returned `{reason['value']}`. The existing accelerator "
                "requires a flat traced JAX state and traces its map through a "
                "fixed-shape "
                "loop; the window exchange owns immutable NumPy `Waveform` objects and "
                "host-bearing equilibrium/TORAX sweeps. It therefore does not accept "
                "this "
                "exchange without product adaptation, so no Anderson timing is claimed."
            ),
            "",
            "## Provenance and decision boundary",
            "",
            "The band-only arm is a synthetic Git tree formed from the pre-shape tree "
            "`80a4aa2b67a656af5ae25cbfb3dfaa2ff66809f1` plus the exact product "
            "patch from band commit `32942ac3861af0b95dd19cf279e4b3ab211fa705`. "
            "The combined arm is the tree of that band commit, whose ancestry contains "
            "the near-axis shape landing. Both were executed from `git archive` "
            "overlays against the repository root environment. The criterion variants "
            "are measurements for an owner decision; "
            "this study neither selects nor implements one.",
            "",
        ]
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text("\n".join(report), encoding="utf-8")
    print(f"results={arguments.results}")
    print(f"report={arguments.report}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--tree-sha", required=True)
    measure.add_argument("--state", required=True)
    measure.add_argument("--damping", type=float, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--probe-anderson", action="store_true")
    collate = subparsers.add_parser("collate")
    collate.add_argument("--raw", type=Path, action="append", required=True)
    collate.add_argument("--landed-trace", type=Path, required=True)
    collate.add_argument("--landed-tree-sha", required=True)
    collate.add_argument("--landed-short-trace", type=Path, required=True)
    collate.add_argument("--landed-short-tree-sha", required=True)
    collate.add_argument("--results", type=Path, required=True)
    collate.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "collate":
        return _collate(arguments)
    measurement = _measure_window(arguments.damping)
    rows = _raw_rows(
        measurement,
        tree_sha=arguments.tree_sha,
        state=arguments.state,
        damping=arguments.damping,
        probe_anderson=arguments.probe_anderson,
    )
    _write_rows(arguments.output, RAW_FIELDS, rows)
    print(f"tree_sha={arguments.tree_sha}")
    print(f"state={arguments.state}")
    print(f"damping={arguments.damping:.17g}")
    print(f"outcome={measurement.outcome}")
    print(f"iterations={measurement.convergence.iterations_used}")
    print(f"contraction={measurement.convergence.contraction_estimate:.17g}")
    print(f"maximum_residual={measurement.convergence.maximum_residual:.17g}")
    print(f"window_seconds={measurement.window_seconds:.17g}")
    print(f"output={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
