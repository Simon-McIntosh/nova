"""Measure the once-built raster (R,Z) FluxTarget against a per-edit rebuild.

The interactive consumer conditions its decoder on a rectangular (R, Z) grid
that Nova evaluates directly.  The raster flux target carries the coil
Green's matrix and the plasma-cell Green's matrix on that grid; both are
static under coil-current edits, so a target held once costs a matrix product
per edit, while a target rebuilt each edit pays the Green's construction
again.  This driver quantifies that difference on one converging MAST arm.

Both arms evaluate the same raster flux expression:

    psi = source_target @ active_current + plasma_target @ moments + passive

``source_target`` is the active-winding response on the raster (the matrix
the forward solve's grid target holds), ``plasma_target`` the rectangular
cell response, ``moments`` the plasma cell currents of the fitted reference
flux on the raster, and ``passive`` the constant contribution of the
passive/vessel circuits for the edited active circuit.  The moments are
fixed across the ten edits so the measurement isolates the target's own
evaluation cost from the solver's plasma response.  The rebuilt arm
reconstructs the two Green's matrices from the same geometry and grid the
solve used, so the arm evaluations must be bitwise identical; the once arm
merely times the matrices being reused instead of rebuilt.

Run on one reserved H200 through the root-venv python with the persistent
compilation cache configured, exactly as the coil-edit-latency driver it
reuses is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmarks import coil_edit_latency
from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as response_carrier
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.imas.mast_vacuum_response import coil_sections
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/forward-solve-api/raster-receipt/raster-target-receipt.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/forward-solve-api/raster-receipt/per-edit-latency.png"
)
SHOT = 22086
SLICE_INDEX = 43
TEN_EDIT_FRACTIONS = coil_edit_latency.EDIT_FRACTIONS[1:11]
EDIT_COUNT = len(TEN_EDIT_FRACTIONS)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _active_slots(
    active_mapping: list[dict[str, Any]], families: tuple[str, ...]
) -> np.ndarray:
    """Return the stored 101-vector slot of each active family, in solve order."""
    order = {str(row["family"]): int(row["stored_circuit"]) for row in active_mapping}
    slots = np.asarray([order[name] for name in families], dtype=int)
    if slots.ndim != 1 or slots.shape != (len(families),):
        raise ValueError("active family slots must align one-to-one with families")
    return slots


def _build_target(
    geometry: dict[str, Any],
    coordinate: np.ndarray,
    families: tuple[str, ...],
    dr: float,
    dz: float,
) -> dict[str, np.ndarray]:
    """Assemble the raster (R,Z) coil and plasma-cell Green's matrices."""
    source = parity._source_response(geometry, coordinate, families)
    plasma = parity._plasma_response(coordinate, coordinate, dr, dz)
    if source.shape[0] != coordinate.shape[0] or plasma.shape != (
        coordinate.shape[0],
        coordinate.shape[0],
    ):
        raise ValueError("raster target matrices must match the rectangular grid")
    return {"source_target": np.asarray(source), "plasma_target": np.asarray(plasma)}


def _raster_case(carrier_path: Path) -> dict[str, Any]:
    """Assemble the passive-inclusive raster case without solving.

    Mirrors the non-solve fraction of the coil-edit-latency case preparation:
    the persisted 101-circuit response on the 1126 raster-plus-wall targets,
    the active-mapping and the P4/P5 boundary-circuit selection.  The plasma
    moments come from the fitted reference flux on the raster, so the raster
    flux keeps the solve's physics without depending on solve convergence.
    """
    response_cache, _metadata = coil_edit_latency._response_cache(carrier_path)
    selected = {"shot": SHOT, "slice_index": SLICE_INDEX}
    case, context = parity._mast_case_from_selection(
        SHOT_STORE, selected, qualification=None
    )
    _passive, profile, policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    if not policy["response_matrix_reused"]:
        raise RuntimeError("the persisted response carrier was not reused")
    if policy["stored_circuit_count"] != 101:
        raise RuntimeError("the passive-inclusive current vector is not complete")
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None or prescribed.current.shape != (101,):
        raise RuntimeError("the operator does not hold the 101-circuit vector")
    response = np.asarray(prescribed.response, dtype=np.float64)
    base_current = np.asarray(prescribed.current, dtype=np.float64)
    wall_start = profile.operator.grid.node_number
    candidates = []
    for row in policy["active_mapping"]:
        if row["family"] not in coil_edit_latency.BOUNDARY_COIL_FAMILIES:
            continue
        circuit = int(row["stored_circuit"])
        candidates.append(
            {
                "family": row["family"],
                "stored_circuit": circuit,
                "two_percent_wall_flux_sup_wb": float(
                    0.02
                    * abs(base_current[circuit])
                    * np.max(np.abs(response[wall_start:, circuit]))
                ),
            }
        )
    if len(candidates) != len(coil_edit_latency.BOUNDARY_COIL_FAMILIES):
        raise RuntimeError("the P4/P5 boundary-circuit mapping is incomplete")
    selected_coil = max(
        candidates,
        key=lambda row: row["two_percent_wall_flux_sup_wb"],
    )
    circuit_index = int(selected_coil["stored_circuit"])
    reference_flux = np.asarray(case["state"], dtype=np.float64)[
        : profile.lattice.node_count
    ]
    return {
        "profile": profile,
        "policy": policy,
        "response": response,
        "base_current": base_current,
        "circuit_index": circuit_index,
        "coil_mapping": selected_coil,
        "reference_flux": reference_flux,
        "target_current": abs(float(case["reference"]["plasma_current_a"])),
        "reference": case["reference"],
    }


def _evaluate(
    source_target: jax.Array,
    plasma_target: jax.Array,
    active_current: jax.Array,
    moments: jax.Array,
    passive: jax.Array,
) -> jax.Array:
    """Evaluate the raster flux from the same coil and cell currents."""
    return (
        jnp.dot(source_target, active_current)
        + jnp.dot(plasma_target, moments)
        + passive
    )


def _render(
    output: Path,
    radius: np.ndarray,
    height: np.ndarray,
    shape: tuple[int, int],
    psi: np.ndarray,
    once_wall_ms: np.ndarray,
    rebuild_wall_ms: np.ndarray,
    fractions: np.ndarray,
) -> None:
    """Write the raster flux image and the per-edit latency comparison."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": (1.0, 1.05)}
    )
    flux = axes[0].pcolormesh(
        radius,
        height,
        psi.reshape(shape).T,
        cmap="viridis",
        shading="auto",
    )
    figure.colorbar(flux, ax=axes[0], label=r"$\psi$ [Wb]")
    axes[0].set_title("Raster flux on the receiver grid")
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal")

    positions = np.arange(EDIT_COUNT)
    width = 0.38
    axes[1].bar(
        positions - width / 2,
        once_wall_ms,
        width,
        label="target held once",
        color="#4C72B0",
    )
    axes[1].bar(
        positions + width / 2,
        rebuild_wall_ms,
        width,
        label="target rebuilt per edit",
        color="#DD8452",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        [f"{fraction * 100.0:+.0f}%" for fraction in fractions], rotation=45
    )
    axes[1].set_ylabel("wall per edit [ms]")
    axes[1].set_title("Per-edit cost: reuse versus rebuild")
    axes[1].legend()
    axes[1].grid(axis="y", which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def run(
    output: Path,
    figure: Path,
    carrier_path: Path,
) -> dict[str, Any]:
    """Compile once and measure ten edits under both target policies."""
    total_started = time.perf_counter()
    configure_dtypes()
    coil_edit_latency._require_measurement_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    stop = threading.Event()
    reporter = threading.Thread(
        target=coil_edit_latency._heartbeat,
        args=(stop, total_started),
        daemon=True,
    )
    reporter.start()
    try:
        case_store = _raster_case(carrier_path)
        profile = case_store["profile"]
        operator = profile.operator
        geometry = MachineGeometryRegistry.default().select(SHOT).configuration.geometry
        families = tuple(sorted(coil_sections(geometry)))
        active_mapping = case_store["policy"]["active_mapping"]
        coordinate = np.asarray(profile.lattice.coordinate, dtype=np.float64)
        radius = np.asarray(profile.lattice.radius, dtype=np.float64)
        height = np.asarray(profile.lattice.height, dtype=np.float64)
        dr = float(profile.lattice.radial_step)
        dz = float(profile.lattice.vertical_step)
        slots = _active_slots(active_mapping, families)
        base_current = case_store["base_current"]
        base_active = np.asarray(base_current[slots], dtype=np.float64)
        circuit_index = case_store["circuit_index"]
        active_position = int(np.flatnonzero(slots == circuit_index)[0])
        response = case_store["response"]
        response_raster = response[: profile.lattice.node_count]

        jitted = jax.jit(_evaluate)
        # The Green's construction is warm-consistent but not reproducible
        # bitwise across the solve build itself (roundoff-scale, measured
        # ~2e-13), so warm it before either arm times or compares it, and
        # judge the solve matrices by tolerance, not by bit equality.
        _build_target(geometry, coordinate, families, dr, dz)
        built_once = _build_target(geometry, coordinate, families, dr, dz)
        source_once = jnp.asarray(built_once["source_target"])
        plasma_once = jnp.asarray(built_once["plasma_target"])
        held_source = np.asarray(operator.grid.source_target, dtype=np.float64)
        held_plasma = np.asarray(operator.grid.plasma_target, dtype=np.float64)
        source_reproduces = float(np.max(np.abs(np.asarray(source_once) - held_source)))
        plasma_reproduces = float(np.max(np.abs(np.asarray(plasma_once) - held_plasma)))
        if max(source_reproduces, plasma_reproduces) > 1.0e-9:
            raise RuntimeError("the rebuilt Green's matrices disagree with the solve")
        passive = response_raster @ base_current - held_source @ base_active

        base_moments = operator.cell_current_moments(
            jnp.asarray(case_store["reference_flux"]), TopologyClass.DIVERTED
        )
        moments = jnp.asarray(base_moments.cell_current, dtype=jnp.float64)
        if moments.shape != (profile.lattice.node_count,):
            raise ValueError("base plasma moments must carry one cell current each")

        edit_walls = []
        for fraction in TEN_EDIT_FRACTIONS:
            values = np.asarray(base_active, dtype=np.float64).copy()
            values[active_position] *= 1.0 + fraction
            edit_walls.append(jnp.asarray(values))
        stablehlo = jitted.lower(
            source_once,
            plasma_once,
            edit_walls[0],
            moments,
            jnp.asarray(passive),
        ).as_text(dialect="stablehlo")
        stablehlo_identity = _sha256_bytes(stablehlo.encode())

        started = time.perf_counter()
        once_result = jitted(
            source_once, plasma_once, edit_walls[0], moments, jnp.asarray(passive)
        )
        jax.block_until_ready(once_result)
        first_edit_wall_ms = 1.0e3 * (time.perf_counter() - started)

        once_ms: list[float] = []
        once_flux: list[np.ndarray] = []
        for active in edit_walls:
            started = time.perf_counter()
            result = jitted(
                source_once, plasma_once, active, moments, jnp.asarray(passive)
            )
            jax.block_until_ready(result)
            once_ms.append(1.0e3 * (time.perf_counter() - started))
            once_flux.append(np.asarray(result))

        rebuild_ms: list[float] = []
        rebuilt_flux: list[np.ndarray] = []
        for active in edit_walls:
            started = time.perf_counter()
            rebuilt = _build_target(geometry, coordinate, families, dr, dz)
            result = jitted(
                jnp.asarray(rebuilt["source_target"]),
                jnp.asarray(rebuilt["plasma_target"]),
                active,
                moments,
                jnp.asarray(passive),
            )
            jax.block_until_ready(result)
            rebuild_ms.append(1.0e3 * (time.perf_counter() - started))
            rebuilt_flux.append(np.asarray(result))

        flux_differences = []
        for once_psi, rebuilt_psi in zip(once_flux, rebuilt_flux, strict=True):
            if not np.array_equal(once_psi, rebuilt_psi):
                raise RuntimeError("the rebuilt arm changed the raster flux bits")
            flux_differences.append(float(np.max(np.abs(once_psi - rebuilt_psi))))

        once_ms_array = np.asarray(once_ms, dtype=np.float64)
        rebuild_ms_array = np.asarray(rebuild_ms, dtype=np.float64)
        once_spans_ms = once_ms_array[1:]
        rebuild_spans_ms = rebuild_ms_array[1:]
        median_once_ms = float(np.median(once_spans_ms))
        worst_once_ms = float(np.max(once_spans_ms))
        median_rebuild_ms = float(np.median(rebuild_spans_ms))
        worst_rebuild_ms = float(np.max(rebuild_spans_ms))

        _render(
            figure,
            np.asarray(radius),
            np.asarray(height),
            (len(radius), len(height)),
            once_flux[0],
            once_ms_array,
            rebuild_ms_array,
            np.asarray(TEN_EDIT_FRACTIONS),
        )

        record = {
            "schema": "nova.raster_flux_target_receipt_benchmark",
            "schema_version": 1,
            "reference": {
                "machine": "MAST",
                "shot": SHOT,
                "slice_index": SLICE_INDEX,
                "plasma_current_a": float(case_store["reference"]["plasma_current_a"]),
                "frame": "raster receiver grid over the MAST vessel",
            },
            "execution": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": socket.gethostname(),
                "device": str(jax.devices()[0]),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "persistent_compilation_cache": str(cache.root),
                "elapsed_seconds": time.perf_counter() - total_started,
            },
            "raster": {
                "shape": [len(radius), len(height)],
                "node_count": int(profile.lattice.node_count),
                "radial_step_m": dr,
                "vertical_step_m": dz,
                "source_target_shape": list(source_once.shape),
                "plasma_target_shape": list(plasma_once.shape),
                "source_rebuild_abs_sup_wb_per_amp_turn": source_reproduces,
                "plasma_rebuild_abs_sup_wb_per_amp_turn": plasma_reproduces,
            },
            "edits": {
                "circuit": case_store["coil_mapping"],
                "circuit_index": circuit_index,
                "fractions_%": [float(f * 100.0) for f in TEN_EDIT_FRACTIONS],
                "count": EDIT_COUNT,
                "moments": "fitted reference flux on the raster, fixed across edits",
                "passive_constant_included": True,
            },
            "once_arm": {
                "policy": "raster (R,Z) FluxTarget built once, reused",
                "stablehlo_sha256": stablehlo_identity,
                "first_edit_wall_ms": first_edit_wall_ms,
                "median_per_edit_wall_ms": median_once_ms,
                "worst_per_edit_wall_ms": worst_once_ms,
                "per_edit_wall_ms": once_ms,
                "bitwise_identity_held": True,
            },
            "rebuild_arm": {
                "policy": "raster (R,Z) FluxTarget rebuilt each edit",
                "median_per_edit_wall_ms": median_rebuild_ms,
                "worst_per_edit_wall_ms": worst_rebuild_ms,
                "per_edit_wall_ms": rebuild_ms,
            },
            "flux_difference": {
                "policy": "same raster flux expression in both arms",
                "bitwise_difference": True,
                "sup_absolute_wb": float(max(flux_differences)),
                "expected": "zero bitwise; 0.0 absolute",
            },
            "summary": {
                "median_per_edit_wall_ms_once": median_once_ms,
                "worst_per_edit_wall_ms_once": worst_once_ms,
                "median_per_edit_wall_ms_rebuild": median_rebuild_ms,
                "worst_per_edit_wall_ms_rebuild": worst_rebuild_ms,
                "median_ratio_rebuild_over_once": float(
                    median_rebuild_ms / max(median_once_ms, np.finfo(float).tiny)
                ),
                "raster_flux_arms_bitwise_identical": True,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        return record
    finally:
        stop.set()
        reporter.join(timeout=1.0)


def _identity_probe(profile, carrier_path: Path) -> dict[str, Any]:
    """Verify matrix rebuild determinism and cross-arm flux identity on CPU."""
    configure_dtypes()
    operator = profile.operator
    geometry = MachineGeometryRegistry.default().select(SHOT).configuration.geometry
    families = tuple(sorted(coil_sections(geometry)))
    coordinate = np.asarray(profile.lattice.coordinate, dtype=np.float64)
    dr = float(profile.lattice.radial_step)
    dz = float(profile.lattice.vertical_step)
    first = _build_target(geometry, coordinate, families, dr, dz)
    second = _build_target(geometry, coordinate, families, dr, dz)
    source_rebuild_identical = bool(
        np.array_equal(first["source_target"], second["source_target"])
    )
    plasma_rebuild_identical = bool(
        np.array_equal(first["plasma_target"], second["plasma_target"])
    )
    held_source = np.asarray(operator.grid.source_target, dtype=np.float64)
    held_plasma = np.asarray(operator.grid.plasma_target, dtype=np.float64)
    source_reproduces_operator = bool(
        np.array_equal(first["source_target"], held_source)
    )
    plasma_reproduces_operator = bool(
        np.array_equal(first["plasma_target"], held_plasma)
    )
    arguments = (
        jnp.asarray(first["source_target"]),
        jnp.asarray(first["plasma_target"]),
        jnp.ones(13, dtype=jnp.float64),
        jnp.ones(profile.lattice.node_count, dtype=jnp.float64),
        jnp.zeros(profile.lattice.node_count, dtype=jnp.float64),
    )
    psi_of_first = _evaluate(*arguments)
    rebuilt = (
        jnp.asarray(second["source_target"]),
        jnp.asarray(second["plasma_target"]),
    )
    psi_of_rebuilt = _evaluate(*rebuilt, *arguments[2:])
    flux_bitwise = bool(
        np.array_equal(np.asarray(psi_of_first), np.asarray(psi_of_rebuilt))
    )
    return {
        "source_rebuild_bitwise_identical": source_rebuild_identical,
        "plasma_rebuild_bitwise_identical": plasma_rebuild_identical,
        "source_reproduces_solve_target": source_reproduces_operator,
        "plasma_reproduces_solve_target": plasma_reproduces_operator,
        "arm_flux_bitwise_identical": flux_bitwise,
    }


def _check_case(arguments: argparse.Namespace) -> None:
    """Validate the non-solve case wiring on any host before the H200 run."""
    configure_dtypes()
    case_store = _raster_case(arguments.carrier)
    profile = case_store["profile"]
    operator = profile.operator
    families = tuple(
        sorted(
            coil_sections(
                MachineGeometryRegistry.default().select(SHOT).configuration.geometry
            )
        )
    )
    slots = _active_slots(case_store["policy"]["active_mapping"], families)
    base_current = case_store["base_current"]
    circuit_index = case_store["circuit_index"]
    active_position = int(np.flatnonzero(slots == circuit_index)[0])
    moments = np.asarray(
        operator.cell_current_moments(
            jnp.asarray(case_store["reference_flux"]), TopologyClass.DIVERTED
        ).cell_current
    )
    geometry = MachineGeometryRegistry.default().select(SHOT).configuration.geometry
    rebuilt = _build_target(
        geometry,
        np.asarray(profile.lattice.coordinate, dtype=np.float64),
        families,
        float(profile.lattice.radial_step),
        float(profile.lattice.vertical_step),
    )
    held_source = np.asarray(operator.grid.source_target, dtype=np.float64)
    held_plasma = np.asarray(operator.grid.plasma_target, dtype=np.float64)
    message = {
        "circuit": {
            "index": circuit_index,
            "family": case_store["coil_mapping"]["family"],
            "slot_position": active_position,
            "active_slots": slots.tolist(),
            "base_current_a": float(base_current[circuit_index]),
        },
        "moments": {
            "shape": list(moments.shape),
            "finite": bool(np.all(np.isfinite(moments))),
            "scripted_source": "fitted reference flux on the raster",
        },
        "targets": {
            "source_shape": list(operator.grid.source_target.shape),
            "plasma_shape": list(operator.grid.plasma_target.shape),
            "response_raster_shape": list(
                case_store["response"][: profile.lattice.node_count].shape
            ),
            "rebuilt_source_abs_sup_vs_solve": float(
                np.max(np.abs(rebuilt["source_target"] - held_source))
            ),
            "rebuilt_plasma_abs_sup_vs_solve": float(
                np.max(np.abs(rebuilt["plasma_target"] - held_plasma))
            ),
        },
    }
    print(json.dumps(message, indent=2, sort_keys=True))


def _selfcheck(arguments: argparse.Namespace) -> None:
    profile = parity._mast_case_from_selection(
        SHOT_STORE,
        {"shot": SHOT, "slice_index": SLICE_INDEX},
        qualification=None,
    )[1]["profile"]
    print(json.dumps(_identity_probe(profile, arguments.carrier), indent=2))


def _sbatch_script(arguments: argparse.Namespace) -> str:
    log_directory = arguments.log_directory.resolve()
    worktree = ROOT.resolve()
    environment = Path("/home/ITER/mcintos/Code/nova/.venv")
    command = (
        f"UV_PROJECT_ENVIRONMENT={environment} PYTHONPATH={worktree} "
        "uv run --no-sync python benchmarks/raster_flux_target_receipt.py run "
        f"--carrier {arguments.carrier.resolve()} "
        f"--output {arguments.output.resolve()} "
        f"--figure {arguments.figure.resolve()}"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=raster-target-receipt
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output={log_directory}/raster-target-receipt-%j.log
set -uo pipefail
export JAX_PLATFORMS=cuda,cpu
export TMPDIR=/tmp
cd {worktree}
{command}
result=$?
echo EXIT_MARKER=$result
exit $result
"""


def _submit(arguments: argparse.Namespace) -> None:
    arguments.log_directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["sbatch", "--parsable"],
        input=_sbatch_script(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip())


def _harvest(output: Path) -> None:
    receipt = json.loads(output.read_text(encoding="utf-8"))
    summary = receipt["summary"]
    print(
        json.dumps(
            {
                "verdict": (
                    "target held once beats a per-edit rebuild"
                    if summary["median_per_edit_wall_ms_rebuild"]
                    > summary["median_per_edit_wall_ms_once"]
                    else "inconclusive"
                ),
                "job_id": receipt["execution"]["job_id"],
                "node": receipt["execution"]["node"],
                "once": {
                    "median_ms": summary["median_per_edit_wall_ms_once"],
                    "worst_ms": summary["worst_per_edit_wall_ms_once"],
                },
                "rebuild": {
                    "median_ms": summary["median_per_edit_wall_ms_rebuild"],
                    "worst_ms": summary["worst_per_edit_wall_ms_rebuild"],
                },
                "median_ratio": summary["median_ratio_rebuild_over_once"],
                "raster_flux_bitwise_identical": summary[
                    "raster_flux_arms_bitwise_identical"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    run_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    for name in ("sbatch", "submit"):
        job_parser = subparsers.add_parser(name)
        job_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        job_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
        job_parser.add_argument(
            "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
        )
        job_parser.add_argument(
            "--log-directory",
            type=Path,
            default=Path(
                "/home/ITER/mcintos/.config/reckon/crew/runs/"
                "r-20260906T130706783090-fsa-raster-flux-target-receipt/logs"
            ),
        )

    harvest_parser = subparsers.add_parser("harvest")
    harvest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    selfcheck_parser = subparsers.add_parser("selfcheck")
    selfcheck_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    check_parser = subparsers.add_parser("checkcase")
    check_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    arguments = parser.parse_args()
    if arguments.command == "checkcase":
        _check_case(arguments)
    elif arguments.command == "selfcheck":
        _selfcheck(arguments)
    elif arguments.command == "run":
        run(
            arguments.output,
            arguments.figure,
            arguments.carrier,
        )
    elif arguments.command == "sbatch":
        print(_sbatch_script(arguments), end="")
    elif arguments.command == "submit":
        _submit(arguments)
    else:
        _harvest(arguments.output)


if __name__ == "__main__":
    main()
