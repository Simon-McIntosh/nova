"""Retain a compact affine interaction control from the 40 mm shell cells."""

import argparse
import json
from pathlib import Path
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot.tiledassembly import linear_galerkin_cell_interaction  # noqa: E402
from scripts.iron_linear_magnetisation.controls import (  # noqa: E402
    require_clean_script,
    shell,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    args = parser.parse_args()
    blob = require_clean_script(__file__, args.blob, args.revision)
    assert jax.config.jax_enable_x64
    cells = iron._mesh(shell(), 0.04)
    selected = [0, 1, 206]
    started = time.monotonic()
    operator, blocks = linear_galerkin_cell_interaction(
        tuple(cells.polygons[index] for index in selected), plan=iron._PLAN
    )
    row = dict(
        revision=args.revision,
        script_blob=blob,
        selected_cells=selected,
        shell_cells=len(cells.polygons),
        operator=operator.tolist(),
        gram=blocks.tolist(),
        seconds=time.monotonic() - started,
        device=jax.devices()[0].device_kind,
        x64=True,
        completed=True,
    )
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
