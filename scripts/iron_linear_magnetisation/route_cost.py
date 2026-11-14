"""Compare polygon quadrature and closed reduction at a fixed triangle tile."""

import argparse
import json
from pathlib import Path
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_linear_magnetisation.controls import require_clean_script  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--kernel", choices=["quadrature", "closed"], required=True)
    args = parser.parse_args()
    blob = require_clean_script(__file__, args.blob, args.revision)
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    triangle = np.array([[1.0, 0.0], [1.01, 0.0], [1.0, 0.01]])
    edge, weight, norm = tiled.pad_batch([triangle], edge_count=3)
    targets = np.array([[1.001, 0.002], [1.005, 0.003], [2.0, 0.2], [0.2, 2.0]])
    plan = tiled.TilePlan(4, 4, 16, 16, 192)
    evaluator = tiled.tile_evaluator(
        plan, batched=True, kernel=args.kernel, edge_count=3
    )
    row = dict(
        revision=args.revision,
        script_blob=blob,
        kernel=args.kernel,
        device=jax.devices()[0].device_kind,
        x64=True,
        completed=False,
        plan=vars(plan),
        module=str(Path(tiled.__file__).resolve()),
        cwd=str(Path.cwd().resolve()),
    )

    def save():
        args.output.write_text(json.dumps(row, indent=2) + "\n")

    save()
    prepared = evaluator.prepare(*targets.T, edge, weight, norm, synchronize=True)
    started = time.monotonic()
    executable = evaluator.compile(prepared)
    row["compile_seconds"] = time.monotonic() - started
    save()
    started = time.monotonic()
    result = evaluator.launch(prepared, executable)
    jax.block_until_ready(result)
    row["execute_seconds"] = time.monotonic() - started
    values = np.array(evaluator.materialize(result, len(targets), 1))[:, :, 0]
    row.update(response_per_ampere=values.tolist(), compiled_programs=1, completed=True)
    save()
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
