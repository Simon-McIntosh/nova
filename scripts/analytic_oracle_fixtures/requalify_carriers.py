"""Rebuild the fine certificate carriers on the repaired moment kernel.

The near-edge moment repair changes the packed polygon kernel's near-field
quadrature, so every carrier built against an earlier kernel is a different
carrier.  This driver rebuilds the host and device coupling blocks for the
certificate carrier at one or more requested cell counts on the repaired
kernel, compares them block by block, and writes a receipt naming each path's
build wall and persisted bytes.

Each requested count is rebuilt on both paths so the comparison is between two
independent assemblies of the same semantic carrier.

usage: requalify_carriers.py OUTPUT_JSON KIND:CELLS [KIND:CELLS ...]
"""

# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import os
from time import perf_counter

import jax
import numpy as np

import nova
from benchmarks import solovev_certificate as certificate
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


def _carrier(kind: str):
    """Return the certificate carrier and wall for a case."""
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    wall = certificate._diverted_wall(exact) if kind == "diverted" else None
    return carrier, wall


def _build(carrier, requested: int, wall, builder: str):
    """Build one carrier on the requested path."""
    started = perf_counter()
    machine = fixture.cached_machine(
        carrier,
        requested,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=wall,
        builder=builder,
    )
    wall_seconds = perf_counter() - started
    return {
        "builder": builder,
        "wall_seconds": float(machine.cache.get("build_seconds", wall_seconds)),
        "hit": bool(machine.cache.get("hit")),
        "realised_cells": int(len(machine.node)),
        "semantic_key": machine.cache.get("semantic_key"),
        "store_bytes": _store_bytes(machine),
        "machine": machine,
    }


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    premises = sys.argv[2:] or ["limited:5000", "diverted:5000"]
    print(f"NOVA_MODULE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd()}", flush=True)
    print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}", flush=True)
    print(f"DEVICES={jax.devices()}", flush=True)
    print(
        f"KERNEL={fixture._polygonanalytic_source()} "
        f"REVISION={fixture.builder_revision()}",
        flush=True,
    )
    receipt: dict[str, object] = {
        "nova_file": str(Path(nova.__file__).resolve()),
        "cwd": str(Path.cwd()),
        "kernel_source": str(fixture._polygonanalytic_source()),
        "builder_revision": fixture.builder_revision(),
        "blocks": list(BLOCKS),
        "tolerance": 1.0e-12,
        "rungs": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for premise in premises:
        kind, _, cells_text = premise.partition(":")
        cells = int(cells_text)
        requested = -500 if cells == 550 else -cells
        carrier, wall = _carrier(kind)
        host = _build(carrier, requested, wall, fixture.BUILDER_HOST)
        device = _build(carrier, requested, wall, fixture.BUILDER_DEVICE)
        comparison = {
            name: _relative(
                getattr(host["machine"], name), getattr(device["machine"], name)
            )
            for name in BLOCKS
        }
        worst_block = max(comparison, key=comparison.get)
        worst = comparison[worst_block]
        print(
            f"CARRIER_RUNG kind={kind} requested={requested} "
            f"realised={host['realised_cells']} host_s={host['wall_seconds']:.9g} "
            f"device_s={device['wall_seconds']:.9g} host_hit={host['hit']} "
            f"device_hit={device['hit']} host_bytes={host['store_bytes']} "
            f"device_bytes={device['store_bytes']} worst_block={worst_block} "
            f"worst_relative={worst:.6e} within_1e12={worst <= 1.0e-12}",
            flush=True,
        )
        receipt["rungs"][premise] = {
            "kind": kind,
            "requested_cells": requested,
            "realised_cells": host["realised_cells"],
            "host": {k: v for k, v in host.items() if k != "machine"},
            "device": {k: v for k, v in device.items() if k != "machine"},
            "block_relative": comparison,
            "worst_block": worst_block,
            "worst_relative": worst,
            "within_tolerance": bool(worst <= 1.0e-12),
            "completed": True,
        }
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
