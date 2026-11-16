"""Lower the production row program with one source block stubbed and record its size.

The program, the operands and the lowering steps are the profile harness's, so
an unstubbed arm measures the program the H200 profile compiles.  Nothing here
is compiled: this runs on CPU and measures the StableHLO module the program
hands to XLA, which is what the graph share of a source block is read from.
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
import subprocess
import sys
from time import perf_counter

import jax
import nova

from scripts.prototypes.kernel_compile_census import row
from scripts.prototypes.point_program_attribution import blocks

assert jax.config.jax_enable_x64 is True
assert not jax.config.jax_enable_compilation_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="control")
    parser.add_argument("--program", default="production_point_batch")
    parser.add_argument("--size", type=int, default=132)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", type=argparse_optional_path, default=None)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--record-elliptic", action="store_true")
    args = parser.parse_args()

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()} command={json.dumps(sys.argv)}")
    print(f"nova={nova.__file__} module={__file__} device={jax.devices()[0]}")
    receipt = {
        "block": args.block,
        "program": args.program,
        "size": args.size,
        "revision": revision,
        "tree": str(Path.cwd().resolve()),
        "nova_file": nova.__file__,
        "device": str(jax.devices()[0]),
        "cache_enabled": bool(jax.config.jax_enable_compilation_cache),
    }

    function, operands = row.program(args.program, args.size)
    if args.record_elliptic:
        blocks.record_elliptic()
        replaced = ["recorded"]
    elif args.block == "elliptic":
        replaced = blocks.apply_elliptic()
    else:
        replaced = blocks.apply(args.block, blocks.probe if args.probe else None)
    receipt["replaced"] = replaced
    print(f"replaced={replaced}")

    started = perf_counter()
    lowered = jax.jit(function).lower(*operands)
    receipt["lower_seconds"] = perf_counter() - started
    text = lowered.as_text()
    receipt["stablehlo_bytes"] = len(text.encode())
    receipt["stablehlo_lines"] = text.count("\n")
    if args.text is not None:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text)
        receipt["text"] = str(args.text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, default=str).replace("NaN", "null") + "\n")
    reached = sorted(set(blocks.REACHED))
    receipt["reached"] = reached
    print(f"reached={reached}")
    args.output.write_text(json.dumps(receipt, indent=2, default=str).replace("NaN", "null") + "\n")
    print(f"block={args.block} stablehlo_bytes={receipt['stablehlo_bytes']} lower_seconds={receipt['lower_seconds']:.3f}")


def argparse_optional_path(value):
    return Path(value)


if __name__ == "__main__":
    main()