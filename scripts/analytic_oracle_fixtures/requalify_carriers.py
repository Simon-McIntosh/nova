"""Rebuild the fine certificate carriers on the repaired moment kernel.

The near-edge moment repair changes the packed polygon kernel's near-field
quadrature, so every carrier built against an earlier kernel is a different
carrier.  This driver rebuilds the host and device coupling blocks for one or
more requested cell counts on the repaired kernel, compares them block by
block, and writes a receipt naming the build wall and the persisted bytes of
each.  It is the requalification of the coarse-carrier rungs the moment-kernel
repair left unmeasured.

usage: requalify_carriers.py OUTPUT_JSON [CELLS ...]
"""

# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from time import perf_counter

import jax
import numpy as np

import nova
from scripts.analytic_oracle_fixtures import measure as fixture

assert jax.config.jax_enable_x64

BLOCKS = (
    "plasma_to_grid",
    "plasma_to_grid_r",
    "plasma_to_grid_z",
    "plasma_to_wall",
    "plasma_to_wall_r",
    "plasma_to_wall_z",
    "plasma_to_sample",
    "plasma_to_sample_r",
    "plasma_to_sample_z",
)


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    """Return the norm-relative difference of two block arrays."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    denom = float(
        max(np.max(np.abs(left)), np.max(np.abs(right)), np.finfo(float).tiny)
    )
    return float(np.max(np.abs(left - right)) / denom)


def _store_bytes(machine) -> int | None:
    """Return the persisted size in bytes of a carrier's cache store."""
    store = machine.cache.get("store")
    if not store:
        return None
    root = Path(store)
    return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))


def _build(case, requested: int, builder: str) -> dict[str, object]:
    """Build one carrier on the requested path and record its cost."""
    started = perf_counter()
    machine = fixture.cached_machine(
        case, requested, wall_nodes=fixture.WALL_POINT_COUNT, builder=builder
    )
    wall = perf_counter() - started
    return {
        "builder": builder,
        "wall_seconds": float(machine.cache.get("build_seconds", wall)),
        "hit": bool(machine.cache.get("hit")),
        "realised_cells": int(len(machine.node)),
        "semantic_key": machine.cache.get("semantic_key"),
        "store_bytes": _store_bytes(machine),
        "machine": machine,
    }


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    cells = [int(value) for value in sys.argv[2:]] or [5000, 10000]
    print(f"NOVA_MODULE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd()}", flush=True)
    print(
        f"KERNEL={fixture._polygonanalytic_source()} "
        f"REVISION={fixture.builder_revision()}",
        flush=True,
    )
    case = fixture.analytic_case()
    receipt: dict[str, object] = {
        "nova_file": str(Path(nova.__file__).resolve()),
        "cwd": str(Path.cwd()),
        "kernel_source": str(fixture._polygonanalytic_source()),
        "builder_revision": fixture.builder_revision(),
        "blocks": list(BLOCKS),
        "tolerance": 1.0e-12,
        "rungs": {},
    }
    for requested in cells:
        rung: dict[str, object] = {"requested_cells": requested, "paths": {}}
        host = _build(case, requested, fixture.BUILDER_HOST)
        device = _build(case, requested, fixture.BUILDER_DEVICE)
        comparison = {
            name: _relative(
                getattr(host["machine"], name), getattr(device["machine"], name)
            )
            for name in BLOCKS
        }
        worst_block = max(comparison, key=comparison.get)
        worst = comparison[worst_block]
        print(
            f"CARRIER_RUNG requested={requested} "
            f"realised={host['realised_cells']} "
            f"host_s={host['wall_seconds']:.9g} device_s={device['wall_seconds']:.9g} "
            f"host_hit={host['hit']} device_hit={device['hit']} "
            f"worst_block={worst_block} worst_relative={worst:.6e} "
            f"within_1e12={worst <= 1.0e-12}",
            flush=True,
        )
        rung["realised_cells"] = host["realised_cells"]
        rung["paths"] = {
            "host": {k: v for k, v in host.items() if k != "machine"},
            "device": {k: v for k, v in device.items() if k != "machine"},
        }
        rung["block_relative"] = comparison
        rung["worst_block"] = worst_block
        rung["worst_relative"] = worst
        rung["within_tolerance"] = bool(worst <= 1.0e-12)
        rung["completed"] = True
        receipt["rungs"][str(requested)] = rung
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "REQUALIFY_DONE "
        + json.dumps(
            {k: v["within_tolerance"] for k, v in receipt["rungs"].items()},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
