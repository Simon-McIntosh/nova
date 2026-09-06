"""Measure the batch economics of the connectivity kernels on one lattice flux.

The throughput study attributed a vmap of the fused trip running *slower* per
element than a serial loop to the trip's topology close being a write-then-read
pass whose per-element cost does not amortise.  This probe isolates the five
connectivity kernels of :mod:`nova.equilibrium.flux_surface_connectivity` on
the MAST 22086/43 lattice flux and measures, under ``jax.vmap`` at batches
1, 8, 16 and 64:

* device wall per element: one serial call per element against one vmapped
  call over the whole batch (median of repeats, device-synchronised);
* the HLO while-loop trip structure read from the compiled text (the dynamic
  data-dependent loop of the flood fill, the fixed caps of the label
  propagation kernels);
* the peak device memory of one execution.

The kernel whose per-element wall or memory *grows* with batch is the one the
connectivity implementation should target. The receipt records per-element
wall at batch 1, 16 and 64 and the batch at which the pass is at least eight
times cheaper per element than the serial loop, or names the floor.

Run on the H200 (betelgeuse, ``--reservation=gpu_0003_grpA``) with the shared
environment's python directly; the persistent compilation cache is enabled so
per-kernel compiles are paid once and the warm measurement dominates.

    sbatch --partition=betelgeuse --reservation=gpu_0003_grpA \\
        --nodes=1 --ntasks=1 --cpus-per-task=7 --gres=gpu:1 --mem=128G \\
        --time=00:40:00 --chdir=<repo-root> --output=<log> --error=<log> \\
        --export=ALL,TOPOLOGY_PROBE_ROOT=<repo-root> \\
        --wrap='export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu \\
        JAX_ENABLE_COMPILATION_CACHE=1 PYTHONPATH="$TOPOLOGY_PROBE_ROOT"; \\
        "$TOPOLOGY_PROBE_ROOT/.venv/bin/python" -m \\
        benchmarks.topology_batch_probe --output <json> --figure <png>'
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

from nova.equilibrium import flux_surface_connectivity as fsc
from nova.equilibrium.connectivity_boundary import _raster_hex_partition_geometry
from nova.equilibrium.wall_mask import inside_polygon
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

ROOT = Path(__file__).resolve().parents[1]
SHOT_STORE = "/work/projects/imas_gpu/mast/level1/shots"
SHOT = 22086
#: The consecutive finite rows the labeller drives (the probe's variance pool).
ROWS = tuple(range(1, 58))
GRID_STRIDE = 2
N_PSIN = 28
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/playable-forward-solve/topology-batch/topology-batch.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/playable-forward-solve/topology-batch/topology-batch.png"
)
BATCHES = (1, 8, 16, 64)
REPEATS = 3
#: the per-element growth tolerance before a kernel is flagged as not amortising
GROWTH_FACTOR = 1.25


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _live_flux_map(
    group: zarr.Group, slice_index: int, radial_count: int
) -> np.ndarray:
    """Return the finite ``(height, radius)`` plane from a padded EFM map."""
    raw = np.asarray(group["psirz"][slice_index], dtype=np.float64)
    columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if columns.size != radial_count:
        raise ValueError(
            f"slice {slice_index} carries {columns.size} live radial columns, "
            f"expected {radial_count}"
        )
    return raw[:, columns]


class SliceOperands:
    """Host-built per-slice kernel operands on the shared benchmark lattice."""

    def __init__(
        self,
        group: zarr.Group,
        row: int,
        radius,
        height,
        inside,
        hex_shared_edges,
    ):
        full_r = np.asarray(group["gridr"], dtype=np.float64)
        reference_full = _live_flux_map(group, row, len(full_r)).T
        self.psi2d = jnp.asarray(
            reference_full[::GRID_STRIDE, ::GRID_STRIDE].T, dtype=jnp.float64
        )
        self.axis_psi = jnp.asarray(
            float(np.asarray(group["psi_axis"][row])), dtype=jnp.float64
        )
        self.boundary_psi = jnp.asarray(
            float(np.asarray(group["psi_boundary"][row])), dtype=jnp.float64
        )
        span = self.boundary_psi - self.axis_psi
        span = jnp.where(
            jnp.abs(span) < 1e-12, jnp.asarray(1e-12, dtype=jnp.float64), span
        )
        psi_n = (self.psi2d - self.axis_psi) / span
        self.confined = (psi_n < 1.0) & inside
        pn_seed = jnp.where(self.confined, psi_n, jnp.inf)
        seed_flat = jnp.argmin(pn_seed.reshape(-1))
        self.seed = (
            jnp.zeros(self.confined.shape, dtype=bool)
            .reshape(-1)
            .at[seed_flat]
            .set(True)
            .reshape(self.confined.shape)
        )
        self.link = fsc.hex_edge_admissibility(
            self.psi2d,
            jnp.asarray(radius, dtype=jnp.float64),
            jnp.asarray(height, dtype=jnp.float64),
            self.boundary_psi,
            self.axis_psi,
            hex_shared_edges,
        )


class KernelSpec:
    """One connectivity kernel: per-element function, operand packing, HLO key."""

    def __init__(self, name, kind, trip_kind, element, pack, hlo_sample=None):
        self.name = name
        self.kind = kind
        self.trip_kind = trip_kind
        self.element = element  # fn(*per-element tensors) -> device result
        self.pack = pack  # fn(SliceOperands) -> tuple of per-element tensors
        self.hlo_sample = hlo_sample  # fn() -> sample args for the jit HLO read


def main() -> None:
    """Parse the caller's operands, measure and persist the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--data-store", default=SHOT_STORE)
    parser.add_argument("--lookback-rows", nargs="*", type=int, default=list(ROWS))
    parser.add_argument(
        "--tag",
        default="measure",
        help="receipt label for this run (e.g. before, after)",
    )
    args = parser.parse_args()

    configure_dtypes()
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())

    group = zarr.open_group(f"{args.data_store}/{SHOT}.zarr", mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    radius = full_r[::GRID_STRIDE]
    height = full_z[::GRID_STRIDE]
    nz, nr = len(height), len(radius)
    n_iter = nz * nr
    limiter = np.column_stack(
        [
            np.asarray(group["limiterr"], dtype=float),
            np.asarray(group["limiterz"], dtype=float),
        ]
    )
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    inside_flat = np.asarray(
        inside_polygon(rr.ravel(), zz.ravel(), limiter[:, 0], limiter[:, 1]),
        dtype=bool,
    )
    inside = jnp.asarray(inside_flat).reshape(nr, nz).T
    hex_rings = jnp.asarray(hex_stencil((nz, nr)), dtype=jnp.int32)
    _, hex_shared_edges = _raster_hex_partition_geometry(
        jnp.asarray(radius, dtype=jnp.float64),
        jnp.asarray(height, dtype=jnp.float64),
    )
    float_radius = jnp.asarray(radius, dtype=jnp.float64)
    float_height = jnp.asarray(height, dtype=jnp.float64)

    rows = list(args.lookback_rows)
    operands = {
        row: SliceOperands(group, row, radius, height, inside, hex_shared_edges)
        for row in rows
    }
    n_pool = len(rows)

    def pool_items(batch: int):
        return [operands[rows[index % n_pool]] for index in range(batch)]

    # per-slice data-dependent trip counts (the flood while loop and the active
    # propagations of the label kernels) for the receipt's input record
    flood_trips = {}
    label_trips = {}
    hex_trips = {}
    for row, item in operands.items():
        _core, steps = fsc.flood_fill_core_with_steps(item.confined, item.seed, n_iter)
        flood_trips[row] = int(steps)
        _labels, steps = fsc.label_connected_components_with_steps(
            item.confined, n_iter
        )
        label_trips[row] = int(steps)
        _hex, steps = fsc.label_hex_connected_components_with_steps(
            item.confined, hex_rings, n_iter
        )
        hex_trips[row] = int(steps)

    def fsa_element(psi, axis, boundary):
        return fsc.traced_flux_surface_bins(
            psi,
            float_radius,
            float_height,
            inside,
            axis,
            boundary,
            jnp.asarray(0.04, dtype=jnp.float64),
            jnp.asarray(0.985, dtype=jnp.float64),
            N_PSIN,
            jnp.asarray(1.25, dtype=jnp.float64),
        )

    def pack_psi(item):
        return (item.psi2d, item.axis_psi, item.boundary_psi)

    specs = [
        KernelSpec(
            "flood_fill_core",
            "data_dependent_while",
            "data-dependent while loop; per-slice trips in inputs",
            lambda c, s: fsc.flood_fill_core(c, s, n_iter),
            lambda item: (item.confined, item.seed),
            hlo_sample=lambda o: (
                o.confined,
                o.seed,
            ),
        ),
        KernelSpec(
            "label_connected_components",
            "fixed_fori",
            "fixed cap supplied by the connectivity kernel; active trips in inputs",
            lambda c: fsc.label_connected_components(c, n_iter),
            lambda item: (item.confined,),
            hlo_sample=lambda o: (o.confined,),
        ),
        KernelSpec(
            "label_hex_connected_components",
            "fixed_fori",
            "fixed cap supplied by the connectivity kernel; active trips in inputs",
            lambda c: fsc.label_hex_connected_components(c, hex_rings, n_iter),
            lambda item: (item.confined,),
            hlo_sample=lambda o: (o.confined,),
        ),
        KernelSpec(
            "label_saddle_aware_hex_connected_components",
            "fixed_fori",
            "fixed cap supplied by the connectivity kernel; active trips in inputs",
            lambda c, link: fsc.label_saddle_aware_hex_connected_components(
                c, hex_rings, link, n_iter
            ),
            lambda item: (item.confined, item.link),
            hlo_sample=lambda o: (o.confined, o.link),
        ),
        KernelSpec(
            "traced_flux_surface_bins",
            "fsa",
            "contains the flood-fill while loop",
            fsa_element,
            pack_psi,
            hlo_sample=None,
        ),
    ]

    receipt: dict[str, Any] = {
        "artifact": "topology batch probe",
        "tag": args.tag,
        "identity": f"{SHOT}/{sorted(rows)}",
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
        "lattice": {"nz": nz, "nr": nr, "n_iters_cap": n_iter, "stride": GRID_STRIDE},
        "inputs": {
            "flood_fill_trips_per_slice": flood_trips,
            "labels_trips_per_slice": label_trips,
            "hex_trips_per_slice": hex_trips,
            "n_slices": n_pool,
            "batch_kind": "the labeller's consecutive finite rows, cycled",
        },
        "kernels": {},
        "verdict": None,
    }

    def timed(fun: Callable[[], jax.Array], repeat: int = REPEATS) -> float:
        """Return the median device wall of one call, warm first."""
        jax.block_until_ready(fun())
        jax.block_until_ready(fun())
        walls = []
        for _ in range(repeat):
            started = time.perf_counter()
            jax.block_until_ready(fun())
            walls.append(time.perf_counter() - started)
        return float(statistics.median(walls))

    def peak_memory(fun: Callable[[], jax.Array]) -> int:
        """Return peak device bytes during one execution."""
        jax.block_until_ready(fun())
        jax.block_until_ready(fun())
        return int(jax.devices()[0].memory_stats()["peak_bytes_in_use"])

    # ---- static HLO trip structure per kernel ----
    for spec in specs:
        if spec.hlo_sample is None:
            continue
        sample = spec.hlo_sample(operands[rows[0]])
        hlo = jax.jit(spec.element).lower(*sample).compile().as_text()
        trip_counts = re.findall(r'"known_trip_count":\{"n":"(\d+)"\}', hlo)
        receipt["kernels"][spec.name] = {
            "hlo_trips": (
                {"known_trip_count_n": trip_counts}
                if trip_counts
                else {"dynamic": True}
            )
        }
        _write_json(receipt, args.output)

    # ---- per-kernel, per-batch measurement ----
    for spec in specs:
        entry = receipt["kernels"].get(spec.name, {})
        entry["kind"] = spec.kind
        entry["trip_kind"] = spec.trip_kind
        entry["batches"] = {}
        for batch in BATCHES:
            items = pool_items(batch)
            serial_fun = jax.jit(spec.element)
            serial_walls = []
            for item in items:
                serial_walls.append(
                    timed(lambda item=item: serial_fun(*spec.pack(item)))
                )
            serial_per = float(np.mean(serial_walls))
            serial_mem = max(
                [
                    peak_memory(lambda item=item: serial_fun(*spec.pack(item)))
                    for item in items
                ]
            )
            vfun = jax.jit(jax.vmap(spec.element))
            first = spec.pack(items[0])
            single = tuple(array[None, ...] for array in first)
            batched = tuple(
                jnp.stack([spec.pack(item)[k] for item in items])
                for k in range(len(first))
            )
            vfun(*single)
            vwall = timed(lambda: vfun(*batched))
            v_mem = peak_memory(lambda: vfun(*batched))
            entry["batches"][str(batch)] = {
                "serial_per_element_s": serial_per,
                "vmap_per_element_s": vwall / batch,
                "vmap_over_serial_per_element": (vwall / batch) / serial_per,
                "vmap_wall_s": vwall,
                "serial_peak_device_bytes": serial_mem,
                "vmap_peak_device_bytes": v_mem,
                "vmap_bytes_per_element": v_mem / batch,
            }
            print(
                f"PROBE {spec.name:44s} batch {batch:3d} "
                f"serial/el {serial_per * 1e3:9.3f} ms  "
                f"vmap/el {vwall / batch * 1e3:9.3f} ms  "
                f"vmap/serial {(vwall / batch) / serial_per:7.3f}x  "
                f"vmap_mem {v_mem / 1e6:9.1f} MB",
                flush=True,
            )
        receipt["kernels"][spec.name] = entry
        _write_json(receipt, args.output)

    # ---- verdict: kernel whose per-element wall or memory grows with batch ----
    verdict = {}
    for spec in specs:
        entry = receipt["kernels"][spec.name]
        per = {
            int(batch): entry["batches"][str(batch)]["vmap_per_element_s"]
            for batch in BATCHES
        }
        mem_per = {
            int(batch): entry["batches"][str(batch)]["vmap_bytes_per_element"]
            for batch in BATCHES
        }
        grows_wall = (
            per[16] > GROWTH_FACTOR * per[1] or per[64] > GROWTH_FACTOR * per[1]
        )
        grows_memory = mem_per[64] > GROWTH_FACTOR * mem_per[1]
        verdict[spec.name] = {
            "wall_per_element_s": per,
            "wall_grows_with_batch": bool(grows_wall),
            "bytes_per_element_by_batch": mem_per,
            "memory_grows_with_batch": bool(grows_memory),
        }
    receipt["verdict"] = verdict
    grown = [
        name
        for name, item in verdict.items()
        if item["wall_grows_with_batch"] or item["memory_grows_with_batch"]
    ]
    receipt["verdict"]["grown_kernels"] = grown
    _write_json(receipt, args.output)
    _draw_figure(receipt, args.figure)
    print("PROBE-DONE", json.dumps({"grown": grown}, sort_keys=True), flush=True)


def _draw_figure(receipt: dict[str, Any], output: Path) -> None:
    """Draw per-element wall and the vmap/serial ratio against batch."""
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    batches = list(BATCHES)
    colours = ["#3b6ea5", "#a53b3b", "#2d7a4f", "#a5843b", "#6f3ba5"]
    names = [
        name for name in receipt["kernels"] if "batches" in receipt["kernels"][name]
    ]
    for name, colour in zip(names, colours):
        entry = receipt["kernels"][name]
        per_element = [
            entry["batches"][str(batch)]["vmap_per_element_s"] for batch in batches
        ]
        serial = [
            entry["batches"][str(batch)]["serial_per_element_s"] for batch in batches
        ]
        ratio = [
            entry["batches"][str(batch)]["vmap_over_serial_per_element"]
            for batch in batches
        ]
        axes[0].plot(
            batches,
            [1e3 * value for value in per_element],
            "-o",
            color=colour,
            lw=1.2,
            ms=4,
            label=name,
        )
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(batches)
        axes[0].set_ylabel("vmap wall per element [ms]")
        axes[0].set_xlabel("batch")
        axes[0].grid(axis="y", alpha=0.2)
        axes[1].plot(
            batches,
            [1e3 * value for value in serial],
            "-s",
            color=colour,
            lw=1.2,
            ms=4,
            label=name,
        )
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(batches)
        axes[1].set_ylabel("serial wall per element [ms]")
        axes[1].set_xlabel("batch")
        axes[1].grid(axis="y", alpha=0.2)
        axes[2].plot(batches, ratio, "-^", color=colour, lw=1.2, ms=4, label=name)
        axes[2].axhline(1.0 / 8.0, color="#888888", lw=1.0, ls="--")
        axes[2].set_xscale("log", base=2)
        axes[2].set_xticks(batches)
        axes[2].set_ylabel("vmap/serial per element")
        axes[2].set_xlabel("batch")
        axes[2].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    axes[2].legend(frameon=False, fontsize=7, loc="upper right")
    figure.suptitle(
        f"Topology batch probe on {receipt['identity']} — "
        f"{receipt['source_commit'][:8]}",
        y=0.97,
    )
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.84, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_json(receipt: dict[str, Any], output: Path) -> None:
    """Persist the receipt so far, creating its directory once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
