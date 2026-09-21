"""Measure what one repeated branch assembly accumulates on a device.

The twenty-edit sweep stopped at edit seventeen with the backend reporting
``1669 alive graphs in the process`` against ``jit_assemble_separatrix_branches``.
That count is a single observation at the point of failure, so a gate written
as "flat over twenty assemblies" currently has a target and no baseline. This
records the baseline.

The alive-graph count itself has no public Python accessor; it is XLA runtime
state that surfaces only in that error string. What IS readable is the device
allocator, so the growth is measured through ``memory_stats`` and reported as
what it is -- bytes and allocation counts per assembly, not a graph count.
A leak shows as a monotonic rise in ``bytes_in_use`` across identical calls
whose inputs and output shapes never change; a flat trace refutes it on this
instrument and leaves the graph count to be read another way.

The map is a fixed random field rather than a solved one on purpose: the
subject is the repeated call, and a solve would add its own allocations to
every row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "graph-growth.json"
DEFAULT_ASSEMBLIES = 20
NODES = 65


def _field(nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth axis-centred map with one closed level to assemble."""
    radius = np.linspace(0.2, 2.0, nodes)
    height = np.linspace(-2.0, 2.0, nodes)
    grid_r, grid_z = np.meshgrid(radius, height, indexing="xy")
    axis = np.asarray([1.0, 0.0])
    flux = -(((grid_r - axis[0]) / 0.6) ** 2 + ((grid_z - axis[1]) / 1.1) ** 2)
    return radius, height, flux, axis


def _stats(device) -> dict[str, int]:
    """Return the readable allocator counters, or an empty record."""
    try:
        raw = device.memory_stats() or {}
    except Exception:  # noqa: BLE001 - a platform without stats is recorded
        return {}
    return {
        name: int(raw[name])
        for name in ("bytes_in_use", "peak_bytes_in_use", "num_allocs")
        if name in raw
    }


def measure(output: Path, assemblies: int) -> None:
    """Call the assembly repeatedly and record the allocator after each."""
    configure_dtypes()
    device = jax.devices()[0]
    radius, height, flux, axis = _field(NODES)
    level = float(np.quantile(flux, 0.35))
    rows: list[dict[str, object]] = []
    for index in range(assemblies):
        started = perf_counter()
        assembled = assemble_separatrix_branches(
            jnp.asarray(flux),
            jnp.asarray(radius),
            jnp.asarray(height),
            jnp.asarray(level),
            jnp.asarray(axis),
        )
        jax.block_until_ready(assembled["closed_controls_rz"])
        rows.append(
            {
                "assembly": index,
                "wall_seconds": perf_counter() - started,
                "well_formed": bool(np.asarray(assembled["well_formed"])),
                **_stats(device),
            }
        )
        print(
            f"ASSEMBLY {index} wall_s={rows[-1]['wall_seconds']:.4f} "
            f"bytes_in_use={rows[-1].get('bytes_in_use')} "
            f"num_allocs={rows[-1].get('num_allocs')}",
            flush=True,
        )
    measured = [row for row in rows if "bytes_in_use" in row]
    growth = None
    if len(measured) >= 2:
        growth = int(measured[-1]["bytes_in_use"]) - int(measured[0]["bytes_in_use"])
    document = {
        "artifact": "allocator growth across repeated separatrix-branch assemblies",
        "instrument": (
            "device memory_stats after each call; the alive-graph count has no "
            "public accessor and is not what this measures"
        ),
        "device": {"platform": device.platform, "kind": device.device_kind},
        "assemblies": assemblies,
        "lattice_nodes": NODES,
        "rows": rows,
        "summary": {
            "bytes_in_use_growth": growth,
            "bytes_in_use_first": (
                int(measured[0]["bytes_in_use"]) if measured else None
            ),
            "bytes_in_use_last": (
                int(measured[-1]["bytes_in_use"]) if measured else None
            ),
            "monotonic_growth": (
                bool(
                    all(
                        int(later["bytes_in_use"]) >= int(earlier["bytes_in_use"])
                        for earlier, later in zip(measured[:-1], measured[1:])
                    )
                )
                if len(measured) >= 2
                else None
            ),
            "allocator_stats_available": bool(measured),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"GRAPH_GROWTH={output} growth_bytes={growth}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assemblies", type=int, default=DEFAULT_ASSEMBLIES)
    arguments = parser.parse_args()
    measure(arguments.output, arguments.assemblies)


if __name__ == "__main__":
    main()
