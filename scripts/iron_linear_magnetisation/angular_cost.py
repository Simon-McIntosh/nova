"""Measure endpoint-graded angular rules against the delivered polygon rules."""

import argparse
import json
from pathlib import Path
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_linear_magnetisation.controls import require_clean_script, shell  # noqa: E402
from scripts.iron_linear_magnetisation.kernel_cost import load_owner  # noqa: E402


def graded_rule(panels, nodes):
    """Concentrate fixed Gauss panels at the near-field angular endpoint."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    edges = np.pi * (np.arange(panels + 1) / panels) ** 4
    lo, hi = edges[:-1, None], edges[1:, None]
    return ((lo + hi) / 2 + (hi - lo) * x / 2).ravel(), ((hi - lo) * w / 2).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    args = parser.parse_args()
    blob = require_clean_script(__file__, args.blob, args.revision)
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    experiment = load_owner(args.revision, "endpoint_angular_owner")
    experiment._phi_rule = graded_rule
    cells = iron._mesh(shell(), 0.04)
    geometry = tiled.linear_magnetisation_geometry(cells.polygons)
    edge, weight, norm, _ = tiled._polygon_current_geometry(cells.polygons)
    centres = np.mean(edge[:, :2], axis=0).T
    diameters = np.max(
        np.hypot(edge[:, 2] - edge[:, 0], edge[:, 3] - edge[:, 1]), axis=0
    )
    parameter = (np.polynomial.legendre.leggauss(12)[0] + 1) / 2
    atoms = geometry["atoms"]
    boundary = atoms[:, :1] + parameter[None, :, None] * (atoms[:, 1:] - atoms[:, :1])
    targets = np.concatenate((geometry["points"], boundary.reshape(-1, 2)))
    rng = np.random.default_rng(4102)
    source = rng.integers(0, len(norm), size=131072)
    target = rng.integers(0, len(targets), size=len(source))
    points = targets[target]
    near = np.linalg.norm(points - centres[source], axis=1) < 4 * diameters[source]
    row = dict(
        revision=args.revision,
        script_blob=blob,
        x64=True,
        device=jax.devices()[0].device_kind,
        completed=False,
        rows=[],
    )

    def evaluate(owner, nodes, points, source, isolated=None):
        block = len(source)
        evaluator = owner.tile_evaluator(
            owner.TilePlan(256, 256, block, 16, nodes),
            batched=True,
            flux_only=True,
            pairwise=True,
            edge_count=3,
        )
        local_weight = weight[:, source].copy()
        if isolated is not None:
            local_weight[np.arange(3) != isolated] = 0
        prepared = evaluator.prepare(
            *points.T, edge[:, :, source], local_weight, norm[source], synchronize=True
        )
        start = time.monotonic()
        executable = evaluator.compile(prepared)
        compile_wall = time.monotonic() - start
        value = evaluator.launch(prepared, executable)
        jax.block_until_ready(value)
        walls = []
        for _ in range(10):
            start = time.monotonic()
            value = evaluator.launch(prepared, executable)
            jax.block_until_ready(value)
            walls.append(time.monotonic() - start)
        return np.asarray(value)[0], compile_wall, float(np.median(walls))

    for label, selected, original, block in (
        ("near", near, 192, 2048),
        ("far", ~near, 48, 8192),
    ):
        chosen = np.flatnonzero(selected)[:block]
        indices, points = source[chosen], targets[target[chosen]]
        before, compile_wall, wall = evaluate(tiled, original, points, indices)
        scale = sum(
            np.abs(evaluate(tiled, original, points, indices, k)[0]) for k in range(3)
        )
        for nodes in (8, 12, 16, 24, 32, 48):
            after, compiled, executed = evaluate(experiment, nodes, points, indices)
            row["rows"].append(
                dict(
                    bucket=label,
                    original_nodes=original,
                    graded_nodes=nodes,
                    pairs=len(indices),
                    reference_compile_seconds=compile_wall,
                    reference_execute_seconds=wall,
                    compile_seconds=compiled,
                    execute_seconds=executed,
                    difference_over_absolute_edge_sum=float(
                        np.max(np.abs(after - before) / scale)
                    ),
                    difference_over_max_entry=float(
                        np.max(np.abs(after - before)) / np.max(np.abs(before))
                    ),
                    frobenius_relative=float(
                        np.linalg.norm(after - before) / np.linalg.norm(before)
                    ),
                )
            )
            args.output.write_text(json.dumps(row, indent=2) + "\n")
    row["completed"] = True
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
