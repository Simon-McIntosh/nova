"""Compare default constant solves with an extracted revision in one process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_graded_refinement.measure import provenance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--reference-revision", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    stamp = provenance(Path(__file__).resolve(), args.revision, args.blob)
    source = subprocess.check_output(
        ["git", "show", args.reference_revision + ":nova/biot/magnetisation.py"]
    )
    path = Path(os.environ["TMPDIR"]) / "constant_reference.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("constant_reference", path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    region = shapely.box(1, 0, 2, 2)
    cells = iron._mesh(region, 1.0)
    matrix = (
        np.array(
            [
                [0.3, 0.02, 0.0, -0.01],
                [0.02, 0.2, 0.01, 0.0],
                [0.0, 0.01, 0.25, 0.01],
                [-0.01, 0.0, 0.01, 0.3],
            ]
        )
        * iron._MU0
    )
    interaction = iron.MagnetisationInteraction(
        matrix, iron._geometry_digest(cells, "constant"), "constant"
    )

    def external(points):
        return np.column_stack(
            (
                0.005 * points[:, 0] + 0.01 * points[:, 1],
                0.05 + 0.02 * points[:, 0] - 0.01 * points[:, 1],
            )
        )

    results = [
        module.solve_magnetisation(
            region,
            external,
            [0, 0.2, 1],
            [2, 4, 2],
            cell_size=1.0,
            interaction=interaction,
            tolerance=1e-12,
        )
        for module in (reference, iron)
    ]
    assert all(result.converged for result in results)
    assert results[0].iterations >= 2
    probes = np.array([[3.0, 0.0], [4.0, 1.0], [6.0, 0.0]])
    pairs = {
        name: [np.asarray(getattr(result, name)) for result in results]
        for name in (
            "magnetisation",
            "permeability_relative",
            "residual_history",
            "step_scales",
            "regularisation_history",
            "reduction_ratios",
        )
    }
    pairs["probe_field"] = [result.field(probes) for result in results]
    checks = {}
    for name, (expected, actual) in pairs.items():
        checks[name] = dict(
            bit_identical=expected.tobytes() == actual.tobytes(),
            reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            current_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        )
    row = dict(
        **stamp,
        reference_revision=args.reference_revision,
        device=jax.devices()[0].device_kind,
        x64=True,
        checks=checks,
        completed=all(check["bit_identical"] for check in checks.values()),
    )
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    assert row["completed"], row
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
