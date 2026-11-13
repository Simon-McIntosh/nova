"""Measure the traced and lowered graph size of one census program.

The compile cost of a census program follows the size of the StableHLO module
it hands to XLA, and that module is produced by tracing and lowering alone. This
records the two text sizes and the time each step took, so a program's graph can
be compared across revisions without paying a compile at any of them.

The cold-process setup, the program construction and the lowering steps are the
ones the profile harness uses; the compile itself is not run.
"""

# ruff: noqa: E402
import os

os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import jax
import nova

from scripts.prototypes.kernel_compile_census import row

assert jax.config.jax_enable_x64 is True
assert not jax.config.jax_enable_compilation_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "program": args.program,
        "size": args.size,
        "command": sys.argv,
        "nova_file": nova.__file__,
        "tree": str(Path(nova.__file__).resolve().parents[1]),
    }

    def checkpoint(phase):
        receipt["phase"] = phase
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print("PHASE " + phase, flush=True)

    function, operands = row.program(args.program, args.size)
    checkpoint("input-ready")

    started = perf_counter()
    graph = jax.make_jaxpr(function)(*operands)
    receipt["trace_seconds"] = perf_counter() - started
    receipt["jaxpr_text_bytes"] = len(str(graph).encode())
    checkpoint("traced")

    started = perf_counter()
    lowered = jax.jit(function).lower(*operands)
    receipt["lower_seconds"] = perf_counter() - started
    receipt["stablehlo_bytes"] = len(lowered.as_text().encode())
    checkpoint("done")


if __name__ == "__main__":
    main()