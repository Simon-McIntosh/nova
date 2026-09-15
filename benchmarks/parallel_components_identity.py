"""Measure standalone identity and batch wall for parallel component labels.

The receipt compares the canonical fixed-point rectangular classifier with the
standalone hook-and-compress classifier on twelve persisted MAST rows.  It first
writes the per-row mismatch and settlement table, then measures both kernels in
one device process at batch 1, 8, 16, and 64.  A speed number is emitted only
after every identity row is exact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from nova.equilibrium import flux_surface_connectivity as canonical
from nova.equilibrium.parallel_components import (
    label_parallel_connected_components,
    label_parallel_connected_components_with_steps,
)
from nova.equilibrium.wall_mask import inside_polygon
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
SHOT = 22086
ROWS = (1, 6, 12, 18, 24, 30, 36, 43, 45, 50, 54, 57)
GRID_STRIDE = 2
BATCHES = (1, 8, 16, 64)
REPEATS = 5
DEFAULT_OUTPUT = ROOT / (
    "docs/figures/playable-forward-solve/parallel-components/"
    "parallel-components-receipt.json"
)
DEFAULT_FIGURE = ROOT / (
    "docs/figures/playable-forward-solve/parallel-components/"
    "parallel-components-wall.png"
)
DEFAULT_TABLE = ROOT / (
    "docs/figures/playable-forward-solve/parallel-components/"
    "parallel-components-rows.md"
)


def _source_revision() -> str:
    """Return the revision containing the measured source."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(receipt: dict[str, Any], output: Path) -> None:
    """Persist the current receipt state after each completed arm."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def _confined_masks(store: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Return the twelve stride-two MAST confined masks and lattice metadata."""
    group = zarr.open_group(str(store / f"{SHOT}.zarr"), mode="r")["efm"]
    full_radius = np.asarray(group["gridr"], dtype=np.float64)
    full_height = np.asarray(group["gridz"], dtype=np.float64)
    radius = full_radius[::GRID_STRIDE]
    height = full_height[::GRID_STRIDE]
    limiter = np.column_stack(
        (
            np.asarray(group["limiterr"], dtype=np.float64),
            np.asarray(group["limiterz"], dtype=np.float64),
        )
    )
    radial_grid, vertical_grid = np.meshgrid(radius, height, indexing="ij")
    inside = (
        np.asarray(
            inside_polygon(
                radial_grid.reshape(-1),
                vertical_grid.reshape(-1),
                limiter[:, 0],
                limiter[:, 1],
            ),
            dtype=bool,
        )
        .reshape((len(radius), len(height)))
        .T
    )

    masks = []
    for row in ROWS:
        raw = np.asarray(group["psirz"][row], dtype=np.float64)
        live_columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
        if live_columns.size != full_radius.size:
            raise ValueError(
                f"row {row} carries {live_columns.size} live radial columns, "
                f"expected {full_radius.size}"
            )
        psi = raw[:, live_columns].T[::GRID_STRIDE, ::GRID_STRIDE].T
        axis_flux = float(np.asarray(group["psi_axis"][row]))
        boundary_flux = float(np.asarray(group["psi_boundary"][row]))
        span = boundary_flux - axis_flux
        if abs(span) < 1e-12:
            span = 1e-12
        masks.append(((psi - axis_flux) / span < 1.0) & inside)
    return np.stack(masks), {
        "vertical_count": len(height),
        "radial_count": len(radius),
        "cell_count": len(height) * len(radius),
        "stride": GRID_STRIDE,
    }


def _timed(call: Callable[[], jax.Array]) -> float:
    """Return median device-synchronised wall after two warm calls."""
    jax.block_until_ready(call())
    jax.block_until_ready(call())
    walls = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(call())
        walls.append(time.perf_counter() - started)
    return float(statistics.median(walls))


def _hlo_trip_counts(compiled) -> list[int]:
    """Return every statically known loop trip count in compiled HLO."""
    text = compiled.as_text()
    return [
        int(value) for value in re.findall(r'"known_trip_count":\{"n":"(\d+)"\}', text)
    ]


def _write_table(rows: list[dict[str, Any]], output: Path) -> None:
    """Write the human-readable per-row identity table."""
    lines = [
        "| MAST row | confined cells | components | label mismatches | "
        "trips | settled |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    lines.extend(
        "| {row} | {confined_cells} | {component_count} | {label_mismatches} | "
        "{parallel_trips} | {settled} |".format(**item)
        for item in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw(receipt: dict[str, Any], output: Path) -> None:
    """Draw wall per element against batch for both admitted kernels."""
    figure, axis = plt.subplots(figsize=(6.7, 4.3))
    for key, label, colour, marker in (
        ("canonical", "canonical fixed point", "#455a64", "o"),
        ("parallel", "parallel hook and compress", "#7e57c2", "s"),
    ):
        values = [
            1e3 * receipt["batches"][str(batch)][f"{key}_per_element_s"]
            for batch in BATCHES
        ]
        axis.plot(BATCHES, values, marker=marker, color=colour, lw=1.6, label=label)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(BATCHES, labels=[str(batch) for batch in BATCHES])
    axis.set_xlabel("batch")
    axis.set_ylabel("device wall per element [ms]")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    """Parse paths, enforce identity, and measure both kernels in one process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-store", type=Path, default=SHOT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    masks, lattice = _confined_masks(args.data_store)
    device_masks = jnp.asarray(masks)
    cell_count = int(lattice["cell_count"])

    receipt: dict[str, Any] = {
        "artifact": "parallel component identity and device wall",
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "identity": {"shot": SHOT, "rows": list(ROWS), "lattice": lattice},
        "previous_measurement_ms_per_element": {
            "batch_1_canonical": 22.076,
            "batch_1_rejected_bounded": 1.986,
        },
        "rows": [],
        "hlo_known_trip_counts": {},
        "batches": {},
        "verdict": "incomplete",
    }

    for row, mask in zip(ROWS, device_masks):
        fixed, fixed_steps = canonical.label_connected_components_with_steps(
            mask, cell_count
        )
        parallel, parallel_steps, settled = (
            label_parallel_connected_components_with_steps(mask)
        )
        fixed_host = np.asarray(fixed)
        parallel_host = np.asarray(parallel)
        foreground = np.asarray(mask)
        entry = {
            "row": row,
            "confined_cells": int(np.count_nonzero(foreground)),
            "component_count": int(np.unique(fixed_host[foreground]).size),
            "label_mismatches": int(np.count_nonzero(fixed_host != parallel_host)),
            "canonical_active_trips": int(fixed_steps),
            "parallel_trips": int(parallel_steps),
            "settled": bool(settled),
        }
        receipt["rows"].append(entry)
        _write_json(receipt, args.output)
        _write_table(receipt["rows"], args.table)
        print("IDENTITY", json.dumps(entry, sort_keys=True), flush=True)

    mismatch_total = sum(row["label_mismatches"] for row in receipt["rows"])
    unsettled = [row["row"] for row in receipt["rows"] if not row["settled"]]
    if mismatch_total or unsettled:
        receipt["verdict"] = "refused-identity"
        _write_json(receipt, args.output)
        raise SystemExit(
            f"identity refusal: {mismatch_total} mismatches; unsettled rows {unsettled}"
        )

    sample = device_masks[0]

    def canonical_element(mask):
        return canonical.label_connected_components(mask, cell_count)

    parallel_element = label_parallel_connected_components
    receipt["hlo_known_trip_counts"] = {
        "canonical": _hlo_trip_counts(
            jax.jit(canonical_element).lower(sample).compile()
        ),
        "parallel": _hlo_trip_counts(jax.jit(parallel_element).lower(sample).compile()),
    }
    _write_json(receipt, args.output)

    for batch in BATCHES:
        selected = device_masks[jnp.arange(batch) % len(ROWS)]
        canonical_batch = jax.jit(jax.vmap(canonical_element))
        parallel_batch = jax.jit(jax.vmap(parallel_element))
        canonical_wall = _timed(lambda: canonical_batch(selected))
        parallel_wall = _timed(lambda: parallel_batch(selected))
        batch_mismatches = int(
            np.count_nonzero(
                np.asarray(canonical_batch(selected))
                != np.asarray(parallel_batch(selected))
            )
        )
        receipt["batches"][str(batch)] = {
            "canonical_wall_s": canonical_wall,
            "parallel_wall_s": parallel_wall,
            "canonical_per_element_s": canonical_wall / batch,
            "parallel_per_element_s": parallel_wall / batch,
            "parallel_over_canonical": parallel_wall / canonical_wall,
            "label_mismatches": batch_mismatches,
        }
        _write_json(receipt, args.output)
        print(
            "TIMING",
            json.dumps(receipt["batches"][str(batch)], sort_keys=True),
            flush=True,
        )

    receipt["verdict"] = (
        "pass-bit-identical"
        if all(item["label_mismatches"] == 0 for item in receipt["batches"].values())
        else "refused-batch-identity"
    )
    _write_json(receipt, args.output)
    if receipt["verdict"] != "pass-bit-identical":
        raise SystemExit(receipt["verdict"])
    _draw(receipt, args.figure)
    print(
        "PARALLEL-COMPONENTS-DONE",
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "row_mismatches": mismatch_total,
                "batch_mismatches": sum(
                    item["label_mismatches"] for item in receipt["batches"].values()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
