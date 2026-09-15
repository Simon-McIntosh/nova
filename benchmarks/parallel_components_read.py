"""Measure exact component integration at the production topology boundary.

One H200 allocation replays every persisted MAST bank member through the
width-one reduced solve, persisting each terminal-state digest and trip split
as it lands.  It then runs the committed four-row Solovev solve gate and
compares each fresh terminal flux array bit-for-bit with its committed
production row.  The resulting receipt states the topology-read share of each
trip and the per-solve read wall against the pre-integration 46.1 ms / 95
percent record.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import limited_row_shadow_census as certificate_gate
from benchmarks import trip_quantum_width_one as trip_quantum
from nova.equilibrium import reduced_newton
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "docs/figures/playable-forward-solve/parallel-components-read"
)
PREVIOUS_TRIP_MS = 46.1
PREVIOUS_TOPOLOGY_SHARE = 0.95
MAST_COMPONENT_RECEIPT = (
    ROOT
    / "docs/figures/playable-forward-solve/parallel-components"
    / "parallel-components-receipt.json"
)
SOLOVEV_COMMITTED_PART_ROOT = (
    ROOT / "docs/figures/cut-cell-current-attribution/limited-shadow/solve-parts/chord"
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _allocation() -> dict[str, Any]:
    devices = jax.devices("gpu")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires betelgeuse")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("measurement requires JAX_PLATFORMS=cuda,cpu")
    if len(devices) != 1 or "H200" not in devices[0].device_kind:
        raise RuntimeError(f"measurement requires one H200, received {devices}")
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        "memory_mib": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "device": devices[0].device_kind,
    }


def _solve_member(member) -> dict[str, Any]:
    state = jnp.asarray(member.state)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)

    def solve(program):
        return reduced_newton.solve_reduced_newton(
            member.operator,
            state,
            requested_class=requested,
            target_current=member.target_current,
            tolerance=member.tolerance,
            newton_steps=12,
            active_set_steps=16,
            program=program,
            stream=False,
        )

    first = solve(None)
    started = time.perf_counter()
    result = solve(first.program)
    wall = time.perf_counter() - started
    terminal = np.asarray(result.state, dtype=np.float64)
    banked = np.asarray(member.state, dtype=np.float64)
    trip_wall = float(np.sum(result.trip_wall_per_trip))
    topology_wall = float(np.sum(result.boundary_wall_per_trip))
    trip_count = len(result.trip_wall_per_trip)
    return {
        "identity": member.identity,
        "state_authority": member.state_authority,
        "input_state_sha256": trip_quantum._array_sha256(banked),
        "terminal_state_sha256": trip_quantum._array_sha256(terminal),
        "terminal_flux_bit_identical_to_input_seed": bool(
            np.array_equal(terminal, banked)
        ),
        "converged": bool(result.converged),
        "termination": result.termination_name,
        "terminal_residual": float(result.terminal_residual),
        "trip_count": trip_count,
        "solve_wall_s": wall,
        "trip_wall_s": trip_wall,
        "trip_ms": 1.0e3 * trip_wall / trip_count if trip_count else None,
        "topology_read_per_solve_ms": 1.0e3 * topology_wall,
        "topology_read_per_trip_ms": (
            1.0e3 * topology_wall / trip_count if trip_count else None
        ),
        "topology_read_share": topology_wall / trip_wall if trip_wall else None,
        "active_set_mask_differences": result.active_set_mask_differences,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [row for row in rows if row.get("trip_count")]
    trip_ms = np.asarray([row["trip_ms"] for row in admitted], dtype=float)
    read_ms = np.asarray(
        [row["topology_read_per_trip_ms"] for row in admitted], dtype=float
    )
    shares = np.asarray([row["topology_read_share"] for row in admitted], dtype=float)
    solve_read = np.asarray(
        [row["topology_read_per_solve_ms"] for row in admitted], dtype=float
    )
    return {
        "row_count": len(rows),
        "converged_count": sum(bool(row.get("converged")) for row in rows),
        "median_trip_ms": float(np.median(trip_ms)),
        "median_topology_read_per_trip_ms": float(np.median(read_ms)),
        "median_topology_read_per_solve_ms": float(np.median(solve_read)),
        "median_topology_read_share": float(np.median(shares)),
        "previous_trip_ms": PREVIOUS_TRIP_MS,
        "previous_topology_read_per_trip_ms": (
            PREVIOUS_TRIP_MS * PREVIOUS_TOPOLOGY_SHARE
        ),
        "previous_topology_read_share": PREVIOUS_TOPOLOGY_SHARE,
    }


def _mast_component_identity() -> dict[str, Any]:
    receipt = json.loads(MAST_COMPONENT_RECEIPT.read_text(encoding="utf-8"))
    rows = receipt["rows"]
    return {
        "receipt": str(MAST_COMPONENT_RECEIPT.relative_to(ROOT)),
        "source_commit": receipt["source_commit"],
        "row_count": len(rows),
        "label_mismatch_count": sum(int(row["label_mismatches"]) for row in rows),
        "verdict": receipt["verdict"],
    }


def _terminal_flux(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["render_data"]["terminal_flux_wb"], dtype=np.float64)


def _solovev_terminal_identity(output_root: Path) -> dict[str, Any]:
    generated_root = output_root / "solovev-certificate/solve-parts/chord"
    rows = []
    for generated_path in sorted(generated_root.glob("*.json")):
        committed_path = SOLOVEV_COMMITTED_PART_ROOT / generated_path.name
        generated = _terminal_flux(
            json.loads(generated_path.read_text(encoding="utf-8"))
        )
        committed = _terminal_flux(
            json.loads(committed_path.read_text(encoding="utf-8"))
        )
        different = int(
            np.count_nonzero(generated.view(np.uint64) != committed.view(np.uint64))
        )
        rows.append(
            {
                "identity": generated_path.stem,
                "element_count": len(generated),
                "different_element_count": different,
                "terminal_flux_bit_identical_to_committed": bool(
                    np.array_equal(generated, committed)
                ),
                "committed_part": str(committed_path.relative_to(ROOT)),
                "generated_part": str(generated_path.relative_to(ROOT)),
                "committed_sha256": trip_quantum._array_sha256(committed),
                "generated_sha256": trip_quantum._array_sha256(generated),
            }
        )
    return {
        "receipt": "solovev-certificate/solve-receipt.json",
        "row_count": len(rows),
        "terminal_flux_bit_identical_count": sum(
            bool(row["terminal_flux_bit_identical_to_committed"]) for row in rows
        ),
        "rows": rows,
    }


def _finalize_existing(output_root: Path) -> None:
    receipt_path = output_root / "parallel-components-read-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for row in receipt["mast_rows"]:
        if "banked_state_sha256" in row:
            row["input_state_sha256"] = row.pop("banked_state_sha256")
        if "terminal_flux_bit_identical_to_banked" in row:
            row["terminal_flux_bit_identical_to_input_seed"] = row.pop(
                "terminal_flux_bit_identical_to_banked"
            )
    receipt["mast_summary"].pop("bit_identical_count", None)
    receipt["mast_component_identity"] = _mast_component_identity()
    receipt["solovev_terminal_identity"] = _solovev_terminal_identity(output_root)
    receipt["measurement_launcher_exit_status"] = 1
    receipt["measurement_launcher_exit_note"] = (
        "all rows and figures landed before a superseded "
        "input-versus-terminal assertion"
    )
    receipt["completed"] = True
    _draw(receipt["mast_summary"], output_root / "parallel-components-read-wall.png")
    _write_json(receipt_path, receipt)
    if receipt["mast_component_identity"]["row_count"] != 12:
        raise RuntimeError("the MAST component receipt does not contain twelve rows")
    if receipt["mast_component_identity"]["label_mismatch_count"] != 0:
        raise RuntimeError("a MAST component label differs from the canonical result")
    if receipt["solovev_terminal_identity"]["terminal_flux_bit_identical_count"] != 4:
        raise RuntimeError("a Solovev terminal flux differs from its committed row")
    print(f"RECEIPT_FINALIZED={receipt_path}", flush=True)
    print("EXIT_MARKER=0", flush=True)


def _draw(summary: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(
        ["before", "after"],
        [summary["previous_trip_ms"], summary["median_trip_ms"]],
        color=["#607d8b", "#7e57c2"],
    )
    axes[0].set_ylabel("compiled trip [ms]")
    axes[1].bar(
        ["before", "after"],
        [
            100.0 * summary["previous_topology_read_share"],
            100.0 * summary["median_topology_read_share"],
        ],
        color=["#607d8b", "#7e57c2"],
    )
    axes[1].set_ylabel("topology-read share [%]")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    receipt_path = output_root / "parallel-components-read-receipt.json"
    figure_path = output_root / "parallel-components-read-wall.png"

    if arguments.finalize_existing:
        _finalize_existing(output_root)
        return

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    allocation = _allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    members, inputs = trip_quantum._build_members()
    receipt: dict[str, Any] = {
        "schema": "nova.parallel-components-production-read",
        "revision": _revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "allocation": allocation,
        "cache": cache.receipt(),
        "inputs": inputs,
        "mast_rows": [],
        "mast_summary": None,
        "mast_component_identity": None,
        "solovev_certificate": None,
        "solovev_terminal_identity": None,
        "completed": False,
    }
    _write_json(receipt_path, receipt)

    for member in members:
        try:
            row = _solve_member(member)
        except Exception as error:  # noqa: BLE001 - persist the row that failed
            row = {
                "identity": member.identity,
                "failure": f"{type(error).__name__}: {error}",
            }
        receipt["mast_rows"].append(row)
        _write_json(receipt_path, receipt)
        print("MAST_ROW " + json.dumps(_strict(row), sort_keys=True), flush=True)

    receipt["mast_summary"] = _summary(receipt["mast_rows"])
    receipt["mast_component_identity"] = _mast_component_identity()
    _write_json(receipt_path, receipt)
    certificate = certificate_gate._solve_gate(
        output_root / "solovev-certificate", regenerate_rows=True
    )
    receipt["solovev_certificate"] = {
        "receipt": "solovev-certificate/solve-receipt.json",
        "row_count": len(certificate["rows"]),
        "acceptance": certificate["acceptance"],
    }
    receipt["solovev_terminal_identity"] = _solovev_terminal_identity(output_root)
    receipt["completed"] = True
    _draw(receipt["mast_summary"], figure_path)
    _write_json(receipt_path, receipt)
    if receipt["mast_component_identity"]["row_count"] != len(members):
        raise RuntimeError("the MAST component receipt has the wrong row count")
    if receipt["mast_component_identity"]["label_mismatch_count"] != 0:
        raise RuntimeError("a MAST component label differs from the canonical result")
    if receipt["solovev_terminal_identity"]["terminal_flux_bit_identical_count"] != 4:
        raise RuntimeError("a Solovev terminal flux differs from its committed row")
    print(f"RECEIPT_WRITTEN={receipt_path}", flush=True)
    print("EXIT_MARKER=0", flush=True)


if __name__ == "__main__":
    main()
