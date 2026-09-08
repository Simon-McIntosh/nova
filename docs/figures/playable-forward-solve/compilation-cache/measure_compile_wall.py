"""Measure the forward-solve compile wall against the shared cache root.

Drive the playable Solov'ev carrier through the production reduced-route entry
(constrained reduced Newton with prescribed currents and no shape rows) exactly
as the playable production solver does, with the persistent compilation cache
configured at the shared forward root.  The driver also prints the allocation
identity and the node-local premise evidence for the temporary filesystem, so
one cold allocation and one warmed allocation on the same node are directly
comparable, and the warmed one shows the cache reuse instead of a fresh
compile.

Run on a compute node with the root environment interpreter:

    JAX_PLATFORMS=cpu TMPDIR=/tmp \
      /home/ITER/mcintos/Code/nova/.venv/bin/python measure_compile_wall.py [--clear]

``--clear`` removes this node's own runtime subtree first, so the allocation
starts cold.  The JSON receipt is written to the output path named by
``--output`` (default: measure_compile_wall.json beside this file).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import time

import jax.numpy as jnp

from apps.playable.solovev import build_machine
from nova.equilibrium.reduced_newton import solve_constrained_reduced_newton
from nova.equilibrium.solve_request import default_forward_compilation_cache_root
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "measure_compile_wall.json"


class CompileCensus:
    """Time JAX backend compiles without changing any solver behaviour."""

    def __init__(self) -> None:
        self.events: list[dict[str, float]] = []

    def timed(self, compiler):
        original = compiler.backend_compile_and_load

        def observed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.events.append({"wall_seconds": time.perf_counter() - started})

        return original, observed

    def compile_wall_seconds(self) -> float:
        return sum(float(event["wall_seconds"]) for event in self.events)


def _cache_entry_count(directory: Path) -> int:
    return len(list(directory.glob("*-cache")))


def _reduced_result_summary(result) -> dict[str, float]:
    return {
        "terminal_residual": float(result.terminal_residual),
        "converged": bool(result.converged),
        "active_set_iterations": int(result.active_set_iterations),
    }


def measure(*, clear: bool) -> dict[str, object]:
    configure_dtypes()

    allocation = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "hostname": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
    }
    premise = {"node_local": {}, "shared_home": {}}
    for label, path in (
        ("node_local", Path("/tmp")),
        ("shared_home", Path.home() / ".local" / "share" / "nova"),
    ):
        stat = os.statvfs(path)
        premise[label] = {
            "path": str(path),
            "filesystem_id": stat.f_fsid,
            "total_bytes": stat.f_frsize * stat.f_blocks,
        }

    from jax._src import compiler

    census = CompileCensus()
    original_compiler, observed_compiler = census.timed(compiler)
    compiler.backend_compile_and_load = observed_compiler

    machine = build_machine()
    profile = machine.profile
    seed = jnp.asarray(machine.seed)
    field_current = profile.operator.prescribed_current_field
    prescribed_current = jnp.asarray(field_current.current, dtype=jnp.float64)

    cache = configure_persistent_compilation_cache(
        default_forward_compilation_cache_root(),
        minimum_compile_seconds=0.0,
    )
    directory = Path(cache.directory)
    if clear:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
    entries_before = _cache_entry_count(directory)

    started = time.perf_counter()
    try:
        result = solve_constrained_reduced_newton(
            profile,
            seed,
            constraint_pairs=(),
            current=None,
            prescribed_current=prescribed_current,
            program=None,
        )
    finally:
        compiler.backend_compile_and_load = original_compiler
    solve_wall = time.perf_counter() - started

    return {
        "allocation": allocation,
        "premise": premise,
        "cache": {
            "root": str(default_forward_compilation_cache_root()),
            "directory": str(directory),
            "entries_before": entries_before,
            "entries_after": _cache_entry_count(directory),
        },
        "compile": {
            "events": census.events,
            "compile_wall_seconds": census.compile_wall_seconds(),
        },
        "solve": {
            "wall_seconds": solve_wall,
            **_reduced_result_summary(result),
        },
        "generated_at": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    receipt = measure(clear=args.clear)
    output = args.output or DEFAULT_OUTPUT
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    def line(name: str, value: object) -> None:
        print(f"COMPILE_WALL {name}={value}", flush=True)

    line("node", receipt["allocation"]["node"])
    line("job", receipt["allocation"]["job_id"])
    line("cache_root", receipt["cache"]["root"])
    line("cache_directory", receipt["cache"]["directory"])
    line("entries_before", receipt["cache"]["entries_before"])
    line("entries_after", receipt["cache"]["entries_after"])
    line("compile_wall_seconds", f"{receipt['compile']['compile_wall_seconds']:.6f}")
    line("solve_wall_seconds", f"{receipt['solve']['wall_seconds']:.6f}")
    line("converged", receipt["solve"]["converged"])
    line("receipt", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
