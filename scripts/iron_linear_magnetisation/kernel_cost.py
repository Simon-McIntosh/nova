"""Compare angular contraction layouts without changing their integration rule."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_linear_magnetisation.controls import require_clean_script  # noqa: E402


def load_owner(revision, name, *, directory=None):
    """Load a committed owner snapshot beside the job's extracted tree."""
    source = subprocess.check_output(
        ["git", "show", revision + ":nova/biot/tiledassembly.py"]
    )
    path = Path(os.environ["TMPDIR"] if directory is None else directory) / (
        name + ".py"
    )
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--reference-revision", required=True)
    args = parser.parse_args()
    blob = require_clean_script(__file__, args.blob, args.revision)
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    reference = load_owner(args.reference_revision, "polygon_dot_owner")
    triangle = np.array([[1.0, 0.0], [1.01, 0.0], [1.0, 0.01]])
    edge, weight, norm = tiled.pad_batch([triangle], edge_count=3)
    row = dict(
        revision=args.revision,
        script_blob=blob,
        reference_revision=args.reference_revision,
        device=jax.devices()[0].device_kind,
        x64=True,
        completed=False,
        programs=[],
    )
    for nodes, block in (
        (48, 1024),
        (48, 2048),
        (48, 8192),
        (192, 64),
        (192, 256),
        (192, 2048),
    ):
        values = []
        for label, owner in (("reference", reference), ("head", tiled)):
            evaluator = owner.tile_evaluator(
                owner.TilePlan(256, 256, block, 16, nodes),
                batched=True,
                flux_only=True,
                pairwise=True,
                edge_count=3,
            )
            prepared = evaluator.prepare(
                np.linspace(0.8, 2.0, block),
                np.full(block, 0.02),
                np.repeat(edge, block, axis=2),
                np.repeat(weight, block, axis=1),
                np.repeat(norm, block),
                synchronize=True,
            )
            started = time.monotonic()
            executable = evaluator.compile(prepared)
            compile_wall = time.monotonic() - started
            result = evaluator.launch(prepared, executable)
            jax.block_until_ready(result)
            walls = []
            for _ in range(30):
                started = time.monotonic()
                result = evaluator.launch(prepared, executable)
                jax.block_until_ready(result)
                walls.append(time.monotonic() - started)
            memory = executable.memory_analysis()
            row["programs"].append(
                dict(
                    contraction=label,
                    nodes=nodes,
                    block=block,
                    compile_seconds=compile_wall,
                    execute_seconds=float(np.median(walls)),
                    temp_bytes=memory.temp_size_in_bytes,
                    output_bytes=memory.output_size_in_bytes,
                )
            )
            values.append(np.asarray(result))
            args.output.write_text(json.dumps(row, indent=2) + "\n")
        row["programs"][-1]["max_difference_over_max_entry"] = float(
            np.max(np.abs(values[1] - values[0])) / np.max(np.abs(values[0]))
        )
    row["completed"] = True
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
