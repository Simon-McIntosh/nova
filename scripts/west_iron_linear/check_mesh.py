"""Check WEST graded mesh construction against the recorded constant meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_graded_refinement.measure import provenance, west_inputs  # noqa: E402
from scripts.west_iron_linear.measure import constant_reference, mesh_size  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    args = parser.parse_args()
    stamp = provenance(Path(__file__).resolve(), args.revision, args.blob)
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    print("# revision:", args.revision, flush=True)
    print("# module:", iron.__file__, flush=True)
    print("# cwd:", Path.cwd().resolve(), flush=True)
    print("# command:", shlex.join([sys.executable, *sys.argv]), flush=True)
    inputs = west_inputs()
    row = dict(**stamp, completed=False, meshes={})
    args.output.mkdir(parents=True, exist_ok=True)
    for label, expected in [("graded-16-80", 5864), ("graded-8-40", 11708)]:
        cells = iron._mesh(inputs["region"], mesh_size(label, inputs))
        reference = constant_reference(label)
        actual = len(cells.polygons)
        assert actual == expected == reference["cells"], (label, actual, expected)
        row["meshes"][label] = dict(
            cells=actual,
            expected_cells=expected,
            constant_receipt=reference["receipt"],
            constant_receipt_sha256=reference["sha256"],
        )
        print(label, "cells", actual, "expected", expected, flush=True)
    row["completed"] = True
    (args.output / "graded-mesh-check.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    print("GRADED MESH CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
