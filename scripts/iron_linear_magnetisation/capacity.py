"""Compile affine Newton shapes without allocating or executing dense factors."""

import argparse
import json
from pathlib import Path
import subprocess

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot.magnetisation import _fit_law, _linear_factor  # noqa: E402
from scripts.iron_linear_magnetisation.controls import require_clean_script  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--cells", type=int, nargs="+", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    blob = require_clean_script(__file__, args.blob)
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    assert len(memory) == 1
    physical = int(memory[0]) * 1024**2
    receipt = dict(
        revision=args.revision,
        script_blob=blob,
        completed=False,
        device=jax.devices()[0].device_kind,
        physical_bytes=physical,
        method=(
            "compiler memory analysis; one quadrature point per cell; "
            "no factor executed; autotuning disabled"
        ),
        rows=[],
    )
    for cells in args.cells:
        unknowns = 6 * cells

        def shape(value):
            return jax.ShapeDtypeStruct(value, np.float64)

        law = tuple(shape(value.shape) for value in _fit_law([0, 10], [1000, 1000]))
        operands = (
            shape((unknowns, unknowns)),
            shape((unknowns,)),
            *law,
            shape((cells, 3)),
            shape((cells, 3)),
            jax.ShapeDtypeStruct((cells,), np.int64),
        )
        executable = _linear_factor.lower(shape((unknowns,)), operands).compile()
        stats = executable.memory_analysis()
        total = (
            stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            + stats.temp_size_in_bytes
            - stats.alias_size_in_bytes
        )
        row = dict(
            cells=cells,
            unknowns=unknowns,
            factor_bytes=8 * unknowns**2,
            argument_bytes=stats.argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
            temporary_bytes=stats.temp_size_in_bytes,
            alias_bytes=stats.alias_size_in_bytes,
            compiled_total_bytes=total,
            exceeds_physical=bool(total > physical),
        )
        receipt["rows"].append(row)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del executable
    receipt["completed"] = True
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
