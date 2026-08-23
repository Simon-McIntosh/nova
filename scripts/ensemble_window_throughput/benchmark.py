"""Measure and publish the identified coupled-window batch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from nova.transport import WindowConfig, solve_window, solve_window_batch
from tests.test_transport_coupled_window import _AffineWindow
from tests.test_transport_window_batch import _BatchAffine, _inputs


# The shared analytic fixture pins CPU for its ordinary unit tests.  Restore
# the execution environment requested by this standalone benchmark before JAX
# initialises a backend.
jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))


MEMBER_COUNTS = (1, 2, 4, 8)
REPEATS = 9


def _tree_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _config() -> WindowConfig:
    return WindowConfig(
        length=1.0,
        equilibrium_grid=np.array([0.0, 0.5, 1.0]),
        transport_grid=np.array([0.0, 0.25, 0.75, 1.0]),
        iteration_cap=180,
        tolerance=1.0e-10,
    )


def _exchanges(config: WindowConfig, count: int):
    return tuple(
        (
            f"member-{index:03d}",
            _AffineWindow(
                config,
                coupling=0.08 + 0.005 * (index % 3),
                source_offset=0.98 + 0.01 * (index % 5),
            ),
        )
        for index in range(count)
    )


def _batch_elapsed(config: WindowConfig, count: int) -> float:
    exchanges = _exchanges(config, count)
    operators = _BatchAffine(exchanges)
    started = time.perf_counter()
    receipt = solve_window_batch(
        _inputs(exchanges), config, operators.equilibrium, operators.transport
    )
    for member in receipt.members:
        assert member.convergence.gating_norm <= config.tolerance
    return time.perf_counter() - started


def _loop_elapsed(config: WindowConfig, count: int) -> float:
    exchanges = _exchanges(config, count)
    started = time.perf_counter()
    for _member_id, exchange in exchanges:
        receipt = solve_window(
            exchange.geometry_template,
            exchange.source_template,
            config,
            exchange.equilibrium,
            exchange.transport,
        )
        assert receipt.convergence.gating_norm <= config.tolerance
    return time.perf_counter() - started


def _write_refusal(
    output: Path,
    *,
    expected_tree: str,
    actual_tree: str,
    reason: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "nova.ensemble-window-throughput-refusal",
                "schema_version": "1.0.0",
                "reason": reason,
                "same_tree_comparison_admitted": False,
                "expected_tree_sha": expected_tree,
                "reported_tree_sha": actual_tree,
                "reported_jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
                "reservation": os.environ.get("SLURM_JOB_RESERVATION", "unknown"),
                "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
                "node": platform.node(),
                "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def measure(output: Path, label: str, expected_tree: str) -> None:
    actual_tree = _tree_sha()
    if actual_tree != expected_tree:
        _write_refusal(
            output,
            expected_tree=expected_tree,
            actual_tree=actual_tree,
            reason="runtime-tree-mismatch",
        )
        raise RuntimeError(
            f"runtime tree {actual_tree} differs from expected tree {expected_tree}"
        )
    gpu_devices = jax.devices("gpu") if label == "h200" else []
    cpu_devices = jax.devices("cpu")
    if label == "h200" and (
        not gpu_devices or not cpu_devices or jax.default_backend() != "gpu"
    ):
        _write_refusal(
            output,
            expected_tree=expected_tree,
            actual_tree=actual_tree,
            reason="cuda-provenance-unavailable",
        )
        raise RuntimeError("one CUDA device and the CPU platform are required")
    config = _config()
    _batch_elapsed(config, 1)
    _loop_elapsed(config, 1)
    rows = []
    for count in MEMBER_COUNTS:
        for mode, operation in (
            ("batch", _batch_elapsed),
            ("scalar-loop", _loop_elapsed),
        ):
            elapsed = [operation(config, count) for _ in range(REPEATS)]
            median = statistics.median(elapsed)
            rows.append(
                {
                    "execution": label,
                    "mode": mode,
                    "members_per_window": count,
                    "repeats": REPEATS,
                    "median_seconds": median,
                    "member_windows_per_second": count / median,
                    "batch_calls_per_second": 1.0 / median,
                }
            )
    device = jax.devices()[0]
    payload = {
        "schema": "nova.ensemble-window-throughput",
        "schema_version": "1.0.0",
        "tree_sha": actual_tree,
        "execution": label,
        "jax_backend": jax.default_backend(),
        "jax_device": str(device),
        "jax_gpu_devices": [str(value) for value in gpu_devices],
        "jax_cpu_devices": [str(value) for value in cpu_devices],
        "cuda_provenance_admitted": label != "h200" or bool(gpu_devices),
        "host": platform.node(),
        "python": platform.python_version(),
        "workload": "affine coupled-window contract with full typed receipt assembly",
        "workload_scope": (
            "evaluator and receipt throughput; not a production equilibrium "
            "physics rate"
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _format(value: float) -> str:
    return f"{value:.6g}"


def publish(cpu_path: Path, gpu_path: Path, output_dir: Path) -> None:
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    accelerator = json.loads(gpu_path.read_text(encoding="utf-8"))
    refused = accelerator["schema"].endswith("refusal")
    if refused:
        if accelerator["expected_tree_sha"] != cpu["tree_sha"]:
            raise ValueError("H200 refusal must name the measured CPU tree")
        receipts = [cpu]
    else:
        receipts = [cpu, accelerator]
        tree_shas = {receipt["tree_sha"] for receipt in receipts}
        if len(tree_shas) != 1:
            raise ValueError(
                "CPU and H200 receipts must measure the same committed tree"
            )
        if (
            accelerator["jax_backend"] != "gpu"
            or not accelerator["cuda_provenance_admitted"]
        ):
            raise ValueError("H200 receipt lacks admitted CUDA provenance")
    tree_sha = cpu["tree_sha"]
    rows = [row for receipt in receipts for row in receipt["rows"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "execution\tjax_backend\tmode\tmembers_per_window\trepeats\t"
        "median_seconds\tmember_windows_per_second\tbatch_calls_per_second\n"
    )
    lines = [header.rstrip("\n")]
    for receipt in receipts:
        for row in receipt["rows"]:
            lines.append(
                "\t".join(
                    (
                        row["execution"],
                        receipt["jax_backend"],
                        row["mode"],
                        str(row["members_per_window"]),
                        str(row["repeats"]),
                        _format(row["median_seconds"]),
                        _format(row["member_windows_per_second"]),
                        _format(row["batch_calls_per_second"]),
                    )
                )
            )
    if refused:
        lines.append(
            "\t".join(
                (
                    "h200",
                    accelerator["reported_jax_backend"],
                    "evidence-refusal",
                    "8",
                    "0",
                    "NA",
                    "NA",
                    "NA",
                )
            )
        )
    results = output_dir / "results.tsv"
    results.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha256(results.read_bytes()).hexdigest()
    batch_rows = [row for row in rows if row["mode"] == "batch"]
    best = max(batch_rows, key=lambda row: row["members_per_window"])
    accelerator_summary = (
        "The H200 row is a typed evidence refusal: "
        f"`{accelerator['reason']}`. No accelerator rate is imputed."
        if refused
        else "The H200 row passed exact tree and CUDA device/backend provenance gates."
    )
    report = output_dir / "report.md"
    report.write_text(
        "\n".join(
            (
                "# Ensemble coupled-window evaluator throughput",
                "",
                f"Tree: `{tree_sha}`. Results SHA-256: `{digest}`.",
                "",
                "This budget measures the affine coupled-window contract workload "
                "with full typed receipt assembly. It isolates the public evaluator "
                "and identity/receipt cost; it is not a production equilibrium-physics "
                "rate.",
                "",
                "Largest measured batch: "
                f"**{best['members_per_window']} members/window**. The exact CPU and "
                "one-H200-node windows-per-second rows are in `results.tsv`. The "
                "execution receipt records the JAX backend and device, so the "
                "H200-node host-bound callback workload is not represented as device "
                "acceleration.",
                "",
                accelerator_summary,
                "",
                "Each row is the median of nine independently completed calls after "
                "one warmup. "
                "`member_windows_per_second` counts admitted member windows; "
                "`batch_calls_per_second` counts facade calls.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    publication = {
        "schema": "nova.ensemble-window-throughput-publication",
        "schema_version": "1.0.0",
        "tree_sha": tree_sha,
        "workload": cpu["workload"],
        "workload_scope": cpu["workload_scope"],
        "cpu_receipt": cpu,
        "h200_receipt": accelerator,
        "h200_evidence_admitted": not refused,
        "results_sha256": digest,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(publication, indent=2) + "\n", encoding="utf-8"
    )
    _write_svg(
        receipts,
        output_dir / "throughput.svg",
        refusal=accelerator if refused else None,
    )


def _write_svg(receipts: list[dict], path: Path, *, refusal: dict | None) -> None:
    colours = {"cpu": "#2563eb", "h200": "#d97706"}
    points = []
    for receipt in receipts:
        rows = [row for row in receipt["rows"] if row["mode"] == "batch"]
        points.append((receipt["execution"], rows))
    maximum = max(
        row["member_windows_per_second"] for _label, rows in points for row in rows
    )
    polylines = []
    labels = []
    for label, rows in points:
        coords = []
        for row in rows:
            x = 90 + 70 * (row["members_per_window"] - 1)
            y = 360 - 280 * row["member_windows_per_second"] / maximum
            coords.append(f"{x:.1f},{y:.1f}")
        colour = colours[label]
        polylines.append(
            f'<polyline points="{" ".join(coords)}" fill="none" stroke="{colour}" '
            'stroke-width="3"/>'
        )
        end = rows[-1]
        x, y = map(float, coords[-1].split(","))
        labels.append(
            f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" fill="{colour}">'
            f"{label}: {_format(end['member_windows_per_second'])} windows/s</text>"
        )
    path.write_text(
        "\n".join(
            (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420" '
                'role="img" aria-label="Coupled-window batch throughput">',
                "<style>:root{color-scheme:light dark}text{font-family:system-ui,"
                "sans-serif;font-size:14px;fill:currentColor}"
                ".title{font-size:20px;font-weight:650}"
                ".axis{stroke:#64748b;stroke-width:1}</style>",
                '<text x="40" y="35" class="title">Admitted member-window '
                "throughput</text>",
                '<line x1="90" y1="360" x2="650" y2="360" class="axis"/>',
                '<line x1="90" y1="70" x2="90" y2="360" class="axis"/>',
                '<text x="330" y="400">members per window</text>',
                '<text x="22" y="230" transform="rotate(-90 22 230)">'
                "member windows / s</text>",
                *polylines,
                *labels,
                *(
                    (
                        '<text x="420" y="105" fill="#d97706">'
                        "H200: evidence refused; no rate imputed</text>",
                    )
                    if refusal is not None
                    else ()
                ),
                "</svg>",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    measurement = subparsers.add_parser("measure")
    measurement.add_argument("--output", type=Path, required=True)
    measurement.add_argument("--label", choices=("cpu", "h200"), required=True)
    measurement.add_argument("--expected-tree", required=True)
    publication = subparsers.add_parser("publish")
    publication.add_argument("--cpu", type=Path, required=True)
    publication.add_argument("--gpu", type=Path, required=True)
    publication.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.output, args.label, args.expected_tree)
    else:
        publish(args.cpu, args.gpu, args.output_dir)


if __name__ == "__main__":
    main()
