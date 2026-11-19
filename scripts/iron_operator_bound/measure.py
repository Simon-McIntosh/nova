"""Measure the volume-weighted quotient of one affine WEST interaction.

The demagnetisation-energy bound for the affine Galerkin interaction is
``x.T E x <= mu0 * x.T H x`` with ``H`` the repeated 3x3 cell Gram block.  The
quotient is therefore the extreme generalised eigenvalue of ``E / mu0``
against ``H``, which :func:`nova.biot.magnetisation.galerkin_spectrum` reports
for a mesh small enough to solve densely.  This script assembles the owner's
operator for one WEST mesh and reports the same quotient at full size with a
Lanczos solve, so a mesh whose assembly is hours of work answers with one
receipt row written as soon as that row completes.

    python scripts/iron_operator_bound/measure.py \
        --label graded-8-40 --output <run-dir> [--save-operator]

Each row records the assembled wall, the extreme quotients and the witness.
The two large meshes hold a dense operator with tens of gigabytes in it, so
the metric is applied block by block and never materialised beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402

assert jax.config.jax_enable_x64

LABELS = ("uniform-80", "uniform-40", "graded-16-80", "graded-8-40")


def mesh_size(label, inputs):
    """Return the cell-size field named by ``label``."""
    if label.startswith("uniform-"):
        return float(label.removeprefix("uniform-"))
    fine, coarse = {"graded-8-40": (0.008, 0.04), "graded-16-80": (0.016, 0.08)}[label]
    return iron.graded_cell_size(
        inputs["surfaces"]["radial"],
        inputs["surfaces"]["vertical"],
        fine=fine,
        coarse=coarse,
        cells_across=20,
    )


def metric_operator(blocks):
    """Return the block-diagonal Gram operator repeated for both components."""
    from scipy.sparse.linalg import LinearOperator

    stack = np.concatenate((blocks, blocks), dtype=np.float64).copy()
    size = 3 * len(stack)

    def matvec(vector):
        return np.einsum("cij,cj->ci", stack, vector.reshape(-1, 3)).reshape(-1)

    return LinearOperator((size, size), matvec=matvec, dtype=np.float64)


def extreme(operator, blocks, *, dense_limit=6000):
    """Return the extreme quotients of ``operator`` in the Gram metric."""
    count = len(operator)
    if count <= dense_limit:
        from scipy.linalg import block_diag, eigh

        metric = block_diag(*blocks, *blocks)
        weighted = np.asarray(metric @ operator) / iron._MU0
        values, vectors = eigh((weighted + weighted.T) / 2, metric)
        return dict(
            method="dense",
            maximum=float(values[-1]),
            minimum=float(values[0]),
            witness=vectors[:, -1],
        )
    from scipy.sparse.linalg import LinearOperator, eigsh

    metric = metric_operator(blocks)

    def matvec(vector):
        return metric.matvec(np.asarray(operator @ vector))

    energy = LinearOperator((count, count), matvec=matvec, dtype=np.float64)
    started = time.monotonic()
    values, vectors = eigsh(
        energy, k=1, M=metric, which="LA", tol=1e-8, ncv=64, maxiter=5000
    )
    vector = vectors[:, -1]
    product = energy.matvec(vector)
    scaled = values[-1] * metric.matvec(vector)
    return dict(
        method="lanczos",
        maximum=float(values[-1] / iron._MU0),
        witness_residual=float(
            np.linalg.norm(product - scaled) / np.linalg.norm(scaled)
        ),
        eigen_seconds=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--save-operator", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from scripts.iron_graded_refinement.measure import west_inputs

    started = time.monotonic()
    inputs = west_inputs()
    size = mesh_size(args.label, inputs)
    cells = iron._mesh(inputs["region"], size)
    polygons = cells.polygons
    row = dict(
        label=args.label,
        cells=int(len(polygons)),
        unknowns=int(6 * len(polygons)),
        mesh_seconds=time.monotonic() - started,
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
    )
    print(json.dumps(row), flush=True)
    started = time.monotonic()
    operator, blocks = iron.linear_galerkin_cell_interaction(polygons, plan=iron._PLAN)
    row["assemble_seconds"] = time.monotonic() - started
    print(f"assembled {row['assemble_seconds']:.1f} s", flush=True)
    if args.save_operator:
        path = args.output / f"operator-{args.label}.npy"
        np.save(path, operator)
        row["operator_path"] = str(path)
    spectrum = extreme(operator, blocks)
    row["method"] = spectrum.pop("method")
    row.update(
        {
            key: value
            for key, value in spectrum.items()
            if np.isscalar(value) or isinstance(value, str)
        }
    )
    (args.output / f"row-{args.label}.json").write_text(
        json.dumps(row, indent=1) + "\n"
    )
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
