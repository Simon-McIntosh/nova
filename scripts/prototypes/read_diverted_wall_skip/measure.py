"""Measure native carried and full reads on common limited/diverted operands."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import numpy as np
    from nova.equilibrium import support_read

    assert jax.config.jax_enable_x64 is True
    spec = importlib.util.spec_from_file_location("wall_fixture", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    receipt = dict(
        revision=args.revision,
        module=support_read.__file__,
        cwd=str(Path.cwd().resolve()),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        instrument_revision=subprocess.check_output(
            ["git", "-C", str(args.helper.parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()],
        rows=[],
        complete=False,
    )
    print("PROVENANCE", json.dumps(receipt), flush=True)
    case = helper.wall_case(550)
    receipt["cells"] = len(case[0].geometry.full_area)
    for width in (1, 16):
        for population in ("diverted", "mixed"):
            # A single lane cannot be mixed: width one is its limited member.
            classes = (
                [0] * width
                if population == "diverted"
                else [i % 2 for i in range(width)]
            )
            if population == "mixed" and width == 1:
                classes = [1]
            inputs = helper.batch_inputs(case, classes)
            initial = jax.block_until_ready(jax.jit(helper.run_batch)(inputs))
            assert np.all(initial.valid), (population, width, initial.valid)
            tracking = support_read.tracking_of(initial)
            for kind in ("carried", "full"):
                tracked = tracking if kind == "carried" else None
                program = jax.jit(helper.run_batch)
                start = time.perf_counter()
                lowered = program.lower(inputs, tracked)
                stable = str(lowered.compiler_ir())
                compiled = lowered.compile()
                hlo = compiled.as_text()
                compile_seconds = time.perf_counter() - start
                for _ in range(5):
                    jax.block_until_ready(compiled(inputs, tracked))
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    result = jax.block_until_ready(compiled(inputs, tracked))
                    samples.append(1000 * (time.perf_counter() - start))
                assert np.all(result.valid)
                assert bool(np.all(result.rebuilt)) == (kind == "full")
                key = f"{population}-{width}-{kind}"
                np.savez(
                    args.output.with_name(args.output.stem + "-" + key + ".npz"),
                    **{
                        f"leaf_{i}": np.asarray(value)
                        for i, value in enumerate(jax.tree.leaves(result))
                    },
                )
                row = dict(
                    population=population,
                    width=width,
                    kind=kind,
                    median_ms=statistics.median(samples),
                    samples_ms=samples,
                    compile_seconds=compile_seconds,
                    hlo_operations=len(re.findall(r"^\s*%[^=]+ =", hlo, re.M)),
                    stablehlo_operations=stable.count("stablehlo."),
                    while_sites=len(re.findall(r"\bwhile\(", hlo)),
                    fusion_sites=len(re.findall(r"\bfusion\(", hlo)),
                )
                receipt["rows"].append(row)
                args.output.write_text(json.dumps(receipt, indent=2))
                print("ROW", json.dumps(row), flush=True)
                del compiled, program, lowered
            jax.clear_caches()
    receipt["complete"] = True
    args.output.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
