"""Build the two-source EFIT reproduction gate on one warm accelerator process.

The gate reports two dimensionless fields without conflating them: the
gauge-aligned production-forward flux error relative to the labelled flux span,
and the independently banked label Grad--Shafranov inconsistency.  The latter is
never recomputed here.  DIII-D and MAST solve construction is delegated to the
existing benchmark modules that own those machine conventions.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jsonschema import Draft202012Validator

from benchmarks import diiid_forward_gs_match as diiid
from benchmarks import efit_forward_parity_slice as mast
from benchmarks.mast_response_carrier_warm import (
    DEFAULT_CARRIER as DEFAULT_MAST_RESPONSE_CARRIER,
)
from benchmarks.mast_response_carrier_warm import load_carrier
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

matplotlib.use("Agg")

DEFAULT_OUTPUT = Path("docs/figures/gs-absolute-accuracy/efit-reproduction.json")
DEFAULT_FIGURE_DIRECTORY = Path("docs/figures/gs-absolute-accuracy/efit")
DEFAULT_ZERO_RECEIPT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260901T062947707320-sse-efit-reproduction-completion/"
    "job-output/zero_iteration_receipt.json"
)
DEFAULT_ZERO_FIELDS = DEFAULT_ZERO_RECEIPT.with_name(
    "zero_iteration_residual_fields.npz"
)
DEFAULT_BACKEND_RECEIPT = Path(
    "docs/figures/solver-convergence-regression/backend-divergence.json"
)
MAIN_BACKEND_RECEIPT = Path("/home/ITER/mcintos/Code/nova") / DEFAULT_BACKEND_RECEIPT
DEFAULT_DIIID_MACHINE_ARTIFACT_CACHE = Path(
    "/work/projects/imas_gpu/sophelio/diiid_machine_artifact_cache"
)
MAST_FLOOR = 0.0116147034
DIIID_FLOOR = 0.0832365867
MARGINAL_MAST_REFERENCES = {(21978, 35), (22086, 43)}
HEARTBEAT_SECONDS = 30.0


def _cache_monitor() -> dict[str, float | int]:
    """Count persistent-cache events without inferring hits from elapsed time."""

    import jax.monitoring as monitoring

    events: dict[str, float | int] = {"hits": 0, "saved_seconds": 0.0}

    def hit(event: str, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/cache_hits":
            events["hits"] = int(events["hits"]) + 1

    def saved(event: str, duration_secs: float, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            events["saved_seconds"] = float(events["saved_seconds"]) + duration_secs

    monitoring.register_event_listener(hit)
    monitoring.register_event_duration_secs_listener(saved)
    return events


def _cache_snapshot(events: dict[str, float | int]) -> tuple[int, float]:
    return int(events["hits"]), float(events["saved_seconds"])


def _cache_frame_report(
    events: dict[str, float | int], before: tuple[int, float]
) -> dict[str, Any]:
    """Report cache reuse observed while compiling one frame."""

    hits = int(events["hits"]) - before[0]
    saved_seconds = float(events["saved_seconds"]) - before[1]
    return {
        "status": "hit" if hits else "miss",
        "hit_count": hits,
        "compile_time_saved_seconds": saved_seconds,
        "evidence": "JAX persistent compilation-cache monitoring events",
    }


@contextmanager
def _timed_stage(
    name: str,
    timings: dict[str, float],
    *,
    frame: str,
    visible: bool,
) -> Iterator[None]:
    """Measure one stage and emit flushed progress while its first frame runs."""

    started = time.perf_counter()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(HEARTBEAT_SECONDS):
            elapsed = time.perf_counter() - started
            print(
                f"EFIT_STAGE_HEARTBEAT frame={frame} stage={name} "
                f"elapsed_seconds={elapsed:.3f}",
                flush=True,
            )

    worker = None
    if visible:
        print(f"EFIT_STAGE_BEGIN frame={frame} stage={name}", flush=True)
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        timings[name] = elapsed
        stopped.set()
        if worker is not None:
            worker.join()
            print(
                f"EFIT_STAGE_END frame={frame} stage={name} "
                f"elapsed_seconds={elapsed:.6f}",
                flush=True,
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _schema() -> dict[str, Any]:
    term = {
        "type": "object",
        "required": ["name", "value", "norm", "field_unit"],
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number", "minimum": 0},
            "norm": {"type": "string"},
            "field_unit": {"type": "string"},
        },
        "additionalProperties": True,
    }
    row = {
        "type": "object",
        "required": [
            "machine",
            "frame_identity",
            "band",
            "solve_against_label",
            "label_gs_inconsistency",
            "solver_qualification",
            "compile_warm_wall_seconds",
            "stage_timings_seconds",
            "compile_cache",
            "passes",
            "figure_src",
        ],
        "properties": {
            "machine": {"enum": ["MAST", "DIII-D"]},
            "frame_identity": {"type": "object"},
            "band": {"type": "object"},
            "solve_against_label": term,
            "label_gs_inconsistency": term,
            "solver_qualification": {"type": "object"},
            "compile_warm_wall_seconds": {"type": "number", "minimum": 0},
            "stage_timings_seconds": {"type": "object"},
            "compile_cache": {
                "type": "object",
                "required": ["status", "hit_count", "compile_time_saved_seconds"],
                "properties": {
                    "status": {"enum": ["hit", "miss"]},
                    "hit_count": {"type": "integer", "minimum": 0},
                    "compile_time_saved_seconds": {
                        "type": "number",
                        "minimum": 0,
                    },
                },
            },
            "passes": {"type": "boolean"},
            "figure_src": {"type": "string", "pattern": "^/nova/figures/"},
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["measurement", "execution", "floors", "rows", "aggregate"],
        "properties": {
            "measurement": {"type": "string"},
            "execution": {"type": "object"},
            "floors": {
                "type": "object",
                "required": ["DIII-D", "MAST"],
                "properties": {
                    "DIII-D": {"const": DIIID_FLOOR},
                    "MAST": {"const": MAST_FLOOR},
                },
            },
            "rows": {"type": "array", "minItems": 2, "items": row},
            "aggregate": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _backend_qualification(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    comparison = report["comparison"]
    return {
        "source": str(DEFAULT_BACKEND_RECEIPT),
        "source_sha256": _sha256(path),
        "paired_arm": report["execution_contract"]["paired_arm"],
        "first_backend_divergence": comparison["first_divergence"],
        "all_common_trip_differences_inside_float64_noise": bool(
            comparison["first_divergence"] is None
            and not comparison["amplifying_sub_stage"]["largest_observed_difference"][
                "beyond_float64_reduction_order_noise"
            ]
        ),
        "qualification": (
            "solver-basin marginal; paired CPU/GPU trajectories agree inside the "
            "declared float64 reduction-order floor, so backend averaging is forbidden"
        ),
    }


def _floor_inputs(
    receipt_path: Path, field_path: Path
) -> tuple[dict[str, dict[str, Any]], Any]:
    receipt = json.loads(receipt_path.read_text())
    by_machine = {row["machine"]: row for row in receipt if int(row["stride"]) == 1}
    expected = {"DIII-D": DIIID_FLOOR, "MAST": MAST_FLOOR}
    for machine, value in expected.items():
        measured = float(by_machine[machine]["relative_grad_shafranov_sup"])
        if round(measured, 10) != value:
            raise RuntimeError(f"{machine} banked floor changed: {measured:.10f}")
    return by_machine, np.load(field_path)


def _plot_pair(
    path: Path,
    title: str,
    solve_radius: np.ndarray,
    solve_height: np.ndarray,
    solve_field: np.ndarray,
    floor_radius: np.ndarray,
    floor_height: np.ndarray,
    floor_field: np.ndarray,
) -> None:
    limit = max(
        float(np.nanmax(np.abs(solve_field))),
        float(np.nanmax(np.abs(floor_field))),
        np.finfo(float).eps,
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)
    for axis, radius, height, field, label in (
        (axes[0], solve_radius, solve_height, solve_field, "Nova solve - EFIT label"),
        (
            axes[1],
            floor_radius,
            floor_height,
            floor_field,
            "EFIT label GS inconsistency",
        ),
    ):
        image = axis.pcolormesh(
            radius,
            height,
            field.T,
            shading="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title(label, fontsize=9)
    figure.suptitle(title)
    figure.colorbar(
        image,
        ax=axes,
        label="dimensionless field (own declared normalization)",
        shrink=0.86,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _band(machine: str) -> dict[str, Any]:
    if machine == "DIII-D":
        return {
            "name": "core-coherent label interior",
            "solve_mask": (
                "score-blind labelled LCFS interior on the registered stride-two "
                "solve grid"
            ),
            "floor_mask": "stride-one stencil-valid source interior",
            "interpretation": (
                "the banked floor peaks 1.97 mm from the magnetic axis and carries "
                "98.54% of classified L2 energy in the interior bulk"
            ),
        }
    return {
        "name": "edge-adjacent label interior",
        "solve_mask": (
            "complete registered MAST solve grid after additive gauge alignment"
        ),
        "floor_mask": "stride-one stencil-valid source interior",
        "interpretation": (
            "the banked floor peak is 66.5 mm from the LCFS; the "
            "stencil-eroded region labels are retained as a caveat"
        ),
    }


def _persisted_mast_response_cache(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen-six response with its complete semantic ledger."""

    response, carrier = load_carrier(path)
    with np.load(path, allow_pickle=False) as archive:
        input_digests = json.loads(
            str(np.asarray(archive["input_digests_json"]).item())
        )
        audit = json.loads(str(np.asarray(archive["audit_json"]).item()))
    audit["stored_circuit_count"] = carrier["stored_circuit_count"]
    return (
        {
            "response": response,
            "input_digests": input_digests,
            "audit": audit,
        },
        carrier,
    )


def _compile_mast_branch(
    case: dict[str, Any],
    profile: Any,
    target_current: float,
) -> tuple[Any, jax.Array]:
    """Lower and compile the production branch separately from its execution."""

    initial = jnp.asarray(case["state"])

    def solve(state: jax.Array) -> Any:
        return profile.solve_branch(
            state,
            TopologyClass.DIVERTED,
            route="newton_krylov",
            target_current=target_current,
            tolerance=mast.FIXED_POINT_CRITERION,
            newton_steps=mast.NEWTON_STEPS,
            gmres_iterations=mast.GMRES_ITERATIONS,
            warmup=mast.WARMUP_SWEEPS,
            relaxation=mast.RELAXATION,
            step_cap=mast.STEP_CAP,
        )

    return jax.jit(solve).lower(initial).compile(), initial


def _use_diiid_machine_artifact_cache(path: Path) -> None:
    """Bind physical-wall resolution to a compute-node-visible cache."""

    keyword_defaults = dict(diiid.build_profile.__kwdefaults__ or {})
    keyword_defaults["machine_artifact_cache"] = path
    diiid.build_profile.__kwdefaults__ = keyword_defaults


def _mast_rows(
    output: Path,
    floor: dict[str, Any],
    floor_fields: Any,
    backend: dict[str, Any],
    response_carrier: Path,
    cache_events: dict[str, float | int],
    *,
    frame_limit: int | None = None,
) -> list[dict[str, Any]]:
    shared_timings: dict[str, float] = {}
    with _timed_stage(
        "slice_selection", shared_timings, frame="MAST:first", visible=True
    ):
        selected = mast.select_slices_by_shot(mast.DECOMPOSITION_BANK)
    if frame_limit is not None:
        if frame_limit < 1:
            raise ValueError("frame_limit must be positive")
        selected = selected[:frame_limit]
    first_reference = selected[0][0]
    first_label = (
        f"MAST:{int(first_reference['shot'])}/{int(first_reference['slice_index'])}"
    )
    with _timed_stage(
        "persisted_response_carrier_load",
        shared_timings,
        frame=first_label,
        visible=True,
    ):
        response_cache, carrier = _persisted_mast_response_cache(response_carrier)
    rows = []
    for index, (selected_row, qualification) in enumerate(selected):
        visible = True
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        frame_label = f"MAST:{shot}/{slice_index}"
        stage_timings = dict(shared_timings) if visible else {}
        started = time.perf_counter()
        with _timed_stage(
            "mast_case_from_selection",
            stage_timings,
            frame=frame_label,
            visible=visible,
        ):
            case, context = mast._mast_case_from_selection(
                mast.SHOT_STORE,
                selected_row,
                qualification,
            )
        with _timed_stage(
            "passive_inclusive_case",
            stage_timings,
            frame=frame_label,
            visible=visible,
        ):
            passive_case, profile, policy = mast._passive_inclusive_case(
                case, context, response_cache
            )
        if not policy["response_matrix_reused"]:
            raise RuntimeError("the persisted MAST response carrier was not reused")
        reference = case["reference"]
        target_current = abs(float(reference["plasma_current_a"]))
        cache_before = _cache_snapshot(cache_events)
        with _timed_stage(
            "jit_compile", stage_timings, frame=frame_label, visible=visible
        ):
            compiled_solve, initial = _compile_mast_branch(
                passive_case, profile, target_current
            )
        compile_cache = _cache_frame_report(cache_events, cache_before)
        with _timed_stage(
            "first_solve", stage_timings, frame=frame_label, visible=visible
        ):
            branch = compiled_solve(initial)
            jax.block_until_ready(branch)
        with _timed_stage(
            "comparison", stage_timings, frame=frame_label, visible=visible
        ):
            equilibrium = branch.equilibrium
            solved = np.asarray(
                equilibrium.flux[: profile.lattice.node_count], dtype=np.float64
            ).reshape(profile.lattice.shape)
            label = np.asarray(context["reference_flux"], dtype=np.float64)
            offset = float(np.mean(label - solved))
            difference = solved + offset - label
            span = float(np.ptp(label))
            field = difference / span
            value = float(np.sqrt(np.mean(field**2)))
        shot = int(reference["shot"])
        slice_index = int(reference["slice_index"])
        name = f"mast-{shot}-row-{slice_index}.png"
        with _timed_stage("figure", stage_timings, frame=frame_label, visible=visible):
            _plot_pair(
                output / name,
                f"MAST {shot}/{slice_index}",
                np.asarray(profile.lattice.radius),
                np.asarray(profile.lattice.height),
                field,
                floor_fields["MAST_1_radius"],
                floor_fields["MAST_1_height"],
                floor_fields["MAST_1_relative"],
            )
        elapsed = time.perf_counter() - started
        marginal = (shot, slice_index) in MARGINAL_MAST_REFERENCES
        solver_qualification = {
            "converged": bool(branch.converged),
            "finite": bool(equilibrium.finite.passed),
            "requested_class": TopologyClass(int(branch.requested_class)).name.lower(),
            "achieved_class": TopologyClass(int(branch.achieved_class)).name.lower(),
            "topology_consistent": bool(branch.topology_consistent),
            "fixed_point_residual": float(branch.residual),
            "fixed_point_tolerance": mast.FIXED_POINT_CRITERION,
            "marginal_solver_basin": marginal,
            "basin_evidence": backend if marginal else None,
        }
        passes = bool(branch.converged and value <= MAST_FLOOR)
        rows.append(
            {
                "machine": "MAST",
                "frame_identity": {
                    "label": f"{shot}/{slice_index} pure",
                    "shot": shot,
                    "slice_index": slice_index,
                    "time_s": float(reference["time_s"]),
                    "source": "efm/psirz via benchmarks/efit_forward_parity_slice.py",
                },
                "band": _band("MAST"),
                "solve_against_label": {
                    "name": "gauge-aligned solve-label RMS fraction of label span",
                    "value": value,
                    "norm": "RMS over the complete registered solve grid",
                    "field_unit": "fraction of EFIT label peak-to-peak span",
                    "additive_gauge_wb": offset,
                },
                "label_gs_inconsistency": {
                    "name": "EFIT label under unchanged Nova interior operator",
                    "value": MAST_FLOOR,
                    "norm": "relative interior Grad-Shafranov sup at stride one",
                    "field_unit": "fraction of interior drive sup",
                    "banked_full_precision": float(
                        floor["relative_grad_shafranov_sup"]
                    ),
                },
                "solver_qualification": solver_qualification,
                "compile_warm_wall_seconds": elapsed,
                "stage_timings_seconds": stage_timings,
                "compile_cache": compile_cache,
                "persisted_response_carrier": carrier,
                "passes": passes,
                "pass_rule": (
                    "solver converged and solve_against_label.value <= "
                    "label_gs_inconsistency.value"
                ),
                "figure_src": f"/nova/figures/gs-absolute-accuracy/efit/{name}",
            }
        )
        print(
            f"MAST {shot}/{slice_index} wall={elapsed:.3f}s "
            f"solve={value:.10f} floor={MAST_FLOOR:.10f} pass={passes} "
            f"compile_cache={compile_cache['status']} "
            f"cache_hits={compile_cache['hit_count']}",
            flush=True,
        )
    return rows


def _diiid_rows(
    output: Path,
    floor: dict[str, Any],
    floor_fields: Any,
    machine_artifact_cache: Path,
    cache_events: dict[str, float | int],
) -> list[dict[str, Any]]:
    _use_diiid_machine_artifact_cache(machine_artifact_cache)
    shared_timings: dict[str, float] = {}
    with _timed_stage(
        "frame_selection", shared_timings, frame="DIII-D:first", visible=True
    ):
        paths = sorted(diiid.DEFAULT_DATA.glob("*.parquet"))
        selected = diiid.select_frames(
            paths, diiid.EXECUTION_FRAME_COUNT, diiid.polarity_population()
        )
    rows = []
    for index, selected_frame in enumerate(selected):
        started = time.perf_counter()
        frame_label = f"DIII-D:{Path(selected_frame.path).name}:{selected_frame.frame}"
        stage_timings = dict(shared_timings) if index == 0 else {}
        with _timed_stage("input_read", stage_timings, frame=frame_label, visible=True):
            row = diiid._read(
                selected_frame.path,
                diiid._LABEL_COLUMNS
                + diiid._GEOMETRY_COLUMNS
                + diiid._CURRENT_COLUMNS
                + diiid._PLASMA_CURRENT_COLUMNS,
            )
            row["_source_path"] = str(selected_frame.path)
        cache_before = _cache_snapshot(cache_events)
        with _timed_stage(
            "production_solve", stage_timings, frame=frame_label, visible=True
        ):
            result, fields = diiid._solve_frame_retaining_failure(
                row,
                selected_frame.frame,
                diiid.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
            )
        compile_cache = _cache_frame_report(cache_events, cache_before)
        with _timed_stage("comparison", stage_timings, frame=frame_label, visible=True):
            unavailable = fields.get("plot_unavailable_reason")
            if unavailable is not None:
                raise RuntimeError(
                    f"DIII-D {result.shot}:{result.frame} has no scoreable field: "
                    f"{unavailable}"
                )
            span = float(np.ptp(fields["labelled"]))
            field = np.asarray(fields["difference"], dtype=np.float64) / span
            value = float(result.metrics.interior_fractional_rms)
        stem = Path(result.shot).stem
        name = f"diiid-{stem}-frame-{result.frame}.png"
        with _timed_stage("figure", stage_timings, frame=frame_label, visible=True):
            _plot_pair(
                output / name,
                f"DIII-D {stem}:{result.frame}",
                fields["radius"],
                fields["height"],
                field,
                floor_fields["DIII_D_1_radius"],
                floor_fields["DIII_D_1_height"],
                floor_fields["DIII_D_1_relative"],
            )
        elapsed = time.perf_counter() - started
        passes = bool(result.converged and value <= DIIID_FLOOR)
        rows.append(
            {
                "machine": "DIII-D",
                "frame_identity": {
                    "label": f"{result.shot}:{result.frame}",
                    "shot": result.shot,
                    "frame": result.frame,
                    "time_ms": result.time_ms,
                    "source": "efit_psirz via benchmarks/diiid_forward_gs_match.py",
                },
                "band": _band("DIII-D"),
                "solve_against_label": {
                    "name": "gauge-aligned solve-label RMS over label-core variation",
                    "value": value,
                    "norm": "fractional RMS on the score-blind labelled LCFS interior",
                    "field_unit": "fraction of centred EFIT label RMS",
                    "additive_gauge_wb": result.metrics.additive_gauge_wb,
                },
                "label_gs_inconsistency": {
                    "name": "EFIT label under unchanged Nova interior operator",
                    "value": DIIID_FLOOR,
                    "norm": "relative interior Grad-Shafranov sup at stride one",
                    "field_unit": "fraction of interior drive sup",
                    "banked_full_precision": float(
                        floor["relative_grad_shafranov_sup"]
                    ),
                },
                "solver_qualification": {
                    "converged": result.converged,
                    "finite": result.finite,
                    "fixed_point_residual": result.fixed_point_relative_residual,
                    "fixed_point_tolerance": result.residual_tolerance,
                    "achieved_topology_class": result.achieved_topology_class,
                    "termination": result.solver_termination,
                    "marginal_solver_basin": False,
                },
                "compile_warm_wall_seconds": elapsed,
                "stage_timings_seconds": stage_timings,
                "compile_cache": compile_cache,
                "passes": passes,
                "pass_rule": (
                    "solver converged and solve_against_label.value <= "
                    "label_gs_inconsistency.value"
                ),
                "figure_src": f"/nova/figures/gs-absolute-accuracy/efit/{name}",
            }
        )
        print(
            f"DIII-D {result.shot}:{result.frame} wall={elapsed:.3f}s "
            f"solve={value:.10f} floor={DIIID_FLOOR:.10f} pass={passes} "
            f"compile_cache={compile_cache['status']} "
            f"cache_hits={compile_cache['hit_count']}",
            flush=True,
        )
    return rows


def run(
    output: Path,
    figure_directory: Path,
    zero_receipt: Path,
    zero_fields: Path,
    backend_receipt: Path,
    mast_response_carrier: Path,
    diiid_machine_artifact_cache: Path,
) -> dict[str, Any]:
    mast.configure_dtypes()
    cache_events = _cache_monitor()
    compilation_cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    print(
        f"EFIT_COMPILATION_CACHE directory={compilation_cache.directory} ",
        f"version={compilation_cache.version_key}",
        flush=True,
    )
    floor_rows, fields = _floor_inputs(zero_receipt, zero_fields)
    backend = _backend_qualification(backend_receipt)
    rows = _mast_rows(
        figure_directory,
        floor_rows["MAST"],
        fields,
        backend,
        mast_response_carrier,
        cache_events,
    )
    rows.extend(
        _diiid_rows(
            figure_directory,
            floor_rows["DIII-D"],
            fields,
            diiid_machine_artifact_cache,
            cache_events,
        )
    )
    allocation = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get(
            "SLURM_JOB_RESERVATION", os.environ.get("NOVA_EFIT_RESERVATION")
        ),
        "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    }
    data = {
        "measurement": (
            "production forward solve versus EFIT label with independent label-GS floor"
        ),
        "execution": {
            "allocation": allocation,
            "compile_policy": (
                "one process with the explicit versioned persistent JAX compilation "
                "cache; per-frame monitoring events distinguish hits from misses; "
                "recorded wall is per complete frame after preceding compilation state"
            ),
            "persistent_compilation_cache": {
                **compilation_cache.receipt(),
                "observed_hits": int(cache_events["hits"]),
                "observed_compile_time_saved_seconds": float(
                    cache_events["saved_seconds"]
                ),
            },
            "solver_source_modified": False,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "driver_sha256": _sha256(Path(__file__)),
            "zero_iteration_receipt": str(zero_receipt),
            "zero_iteration_receipt_sha256": _sha256(zero_receipt),
            "zero_iteration_fields_sha256": _sha256(zero_fields),
            "mast_response_carrier": str(mast_response_carrier),
            "diiid_machine_artifact_cache": str(diiid_machine_artifact_cache),
            "reuse_map_rows": [
                "docs/research/forward-accuracy-reuse-map.html#efit-reproduction "
                "rows 50-55"
            ],
        },
        "floors": {"DIII-D": DIIID_FLOOR, "MAST": MAST_FLOOR},
        "rows": rows,
        "aggregate": {
            "frame_count": len(rows),
            "pass_count": sum(row["passes"] for row in rows),
            "fail_count": sum(not row["passes"] for row in rows),
            "marginal_rows": [
                row["frame_identity"]["label"]
                for row in rows
                if row["solver_qualification"]["marginal_solver_basin"]
            ],
            "all_figures_project_absolute": all(
                row["figure_src"].startswith("/nova/figures/gs-absolute-accuracy/efit/")
                for row in rows
            ),
            "verdict": (
                "PASS_ALL_FRAMES_AT_OR_BELOW_LABEL_GS_FLOOR"
                if all(row["passes"] for row in rows)
                else "FAIL_ONE_OR_MORE_FRAMES_ABOVE_LABEL_GS_FLOOR_OR_UNCONVERGED"
            ),
        },
    }
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(data)
    receipt = {"schema": schema, "data": _strict(data)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY
    )
    parser.add_argument("--zero-receipt", type=Path, default=DEFAULT_ZERO_RECEIPT)
    parser.add_argument("--zero-fields", type=Path, default=DEFAULT_ZERO_FIELDS)
    parser.add_argument("--backend-receipt", type=Path, default=DEFAULT_BACKEND_RECEIPT)
    parser.add_argument(
        "--mast-response-carrier",
        type=Path,
        default=DEFAULT_MAST_RESPONSE_CARRIER,
    )
    parser.add_argument(
        "--diiid-machine-artifact-cache",
        type=Path,
        default=DEFAULT_DIIID_MACHINE_ARTIFACT_CACHE,
    )
    parser.add_argument(
        "--diagnostic-mast-frame",
        action="store_true",
        help="time one MAST frame without writing the full gate receipt",
    )
    arguments = parser.parse_args()
    backend = arguments.backend_receipt
    if not backend.exists() and MAIN_BACKEND_RECEIPT.exists():
        backend = MAIN_BACKEND_RECEIPT
    if arguments.diagnostic_mast_frame:
        mast.configure_dtypes()
        floor_rows, fields = _floor_inputs(
            arguments.zero_receipt, arguments.zero_fields
        )
        qualification = _backend_qualification(backend)
        rows = _mast_rows(
            arguments.figure_directory,
            floor_rows["MAST"],
            fields,
            qualification,
            arguments.mast_response_carrier,
            {"hits": 0, "saved_seconds": 0.0},
            frame_limit=1,
        )
        row = rows[0]
        print(
            "EFIT_MAST_DIAGNOSTIC "
            + json.dumps(
                {
                    "frame_identity": row["frame_identity"],
                    "stage_timings_seconds": row["stage_timings_seconds"],
                    "compile_warm_wall_seconds": row["compile_warm_wall_seconds"],
                    "response_matrix_reused": True,
                    "solve_against_label": row["solve_against_label"],
                    "solver_qualification": row["solver_qualification"],
                    "figure": str(
                        arguments.figure_directory / Path(row["figure_src"]).name
                    ),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return
    receipt = run(
        arguments.output,
        arguments.figure_directory,
        arguments.zero_receipt,
        arguments.zero_fields,
        backend,
        arguments.mast_response_carrier,
        arguments.diiid_machine_artifact_cache,
    )
    aggregate = receipt["data"]["aggregate"]
    print(
        "EFIT_REPRODUCTION_GATE "
        f"frames={aggregate['frame_count']} passes={aggregate['pass_count']} "
        f"verdict={aggregate['verdict']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
