"""Probe the polygon kernel's dpsi/dZ at the bank's zero entries.

Three arms, all batched through the same compiled program so a comparison is
one measurement.

``sweep``  dpsi/dZ along Z at each group-A violating point, so the corner-height
           branch's boundary (|edge end z - target z| <= 1e-7) is visible from
           both sides.
``direct`` dpsi/dZ at the group-A violations and the group-B worst entries.
``psi``    psi at ``z +/- h`` for a step ladder, giving a central-difference
           reference derivative per entry.

``--tree`` selects the kernel under test: the caller's ``PYTHONPATH`` carries
the working tree, and ``--base-tree`` runs the same measurement against another
extraction in a subprocess, which is how psi is shown to be identical.

Run with ``PYTHONPATH=. /home/ITER/mcintos/Code/nova/.venv/bin/python
scripts/prototypes/polygon_symmetry_violation/probe.py --arm all``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SPLIT = (
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261119T065335613845-polygon-symmetry-violation/split.json"
)
GEOMETRY = "/home/ITER/mcintos/.cache/nova-hessian-diagnosis/h200/geometry.npz"
STEPS = (1e-1, 3e-2, 1e-2, 3e-3, 3e-4, 1e-5, 1e-6, 1e-8, 1e-9)
OFFSETS = (
    0.0,
    1e-9,
    -1e-9,
    1e-8,
    -1e-8,
    1e-7,
    -1e-7,
    1e-6,
    -1e-6,
    1e-5,
    -1e-5,
    1e-4,
    -1e-4,
    1e-3,
    -1e-3,
)


def polygons(geometry):
    vertices = np.asarray(geometry["vertices"])
    counts = np.asarray(geometry["vertex_count"])
    return tuple(vertices[index, : int(count)] for index, count in enumerate(counts))


def selection(path):
    """The entries this probe evaluates: group-A violations, plus rows, plus B."""
    with open(path) as stream:
        report = json.load(stream)
    violations = [dict(entry) for entry in report["violations_A"]]
    worst = [dict(entry) for entry in report["worst_B"]]
    return violations, worst


def plan(violations, worst):
    """The point list as (point, cell, moment, arm, entry) records."""
    records = []
    for entry in violations + worst:
        r, z = entry["coordinates"]
        records.append(
            dict(
                point=(r, z),
                cell=entry["cell"],
                moment=entry["moment"],
                arm="direct",
                entry=entry,
            )
        )
    for entry in violations:
        r, z = entry["coordinates"]
        for offset in OFFSETS:
            records.append(
                dict(
                    point=(r, z + offset),
                    cell=entry["cell"],
                    moment=entry["moment"],
                    arm="sweep",
                    offset=offset,
                    entry=entry,
                )
            )
    for entry in violations + worst:
        r, z = entry["coordinates"]
        for step in STEPS:
            for sign in (1.0, -1.0):
                records.append(
                    dict(
                        point=(r, z + sign * step),
                        cell=entry["cell"],
                        moment=entry["moment"],
                        arm="psi",
                        sign=sign,
                        step=step,
                        entry=entry,
                    )
                )
    return records


RUNNER = """
import sys
import numpy as np
from nova.jax.config import configure_dtypes
configure_dtypes()
import jax
import jax.numpy as jnp
from nova.equilibrium import topology
from nova.equilibrium.fixed_points import _source_jets

inputs = np.load(sys.argv[1])
points = np.asarray(inputs["points"], dtype=np.float64)
vertices = np.asarray(inputs["vertices"])
counts = np.asarray(inputs["vertex_count"])
mutable = np.asarray(inputs["mutable"], dtype=np.float64)
batch = int(sys.argv[3])
polys = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
coupling = topology.BiotMomentCoupling.from_polygons(polys)
program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
rows = np.zeros((len(points), len(polys), 3, 6))
for first in range(0, len(points), batch):
    block = points[first:first + batch]
    size = len(block)
    if size < batch:
        block = np.concatenate(
            (block, np.repeat(block[-1:], batch - size, axis=0)))
    rows[first:first + size] = np.asarray(program(jnp.asarray(block), coupling))[:size]
np.savez(sys.argv[2], rows=rows)
print("ROWS", rows.shape, flush=True)
"""


def stack(polys):
    width = max(len(poly) for poly in polys)
    vertices = np.zeros((len(polys), width, 2))
    counts = np.zeros(len(polys), dtype=np.int64)
    for index, poly in enumerate(polys):
        vertices[index, : len(poly)] = poly
        counts[index] = len(poly)
    return vertices, counts


def evaluate(points, polys, directory, batch, tree=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vertices, counts = stack(polys)
    inputs = directory / "probe-inputs.npz"
    outputs = directory / "probe-rows.npz"
    np.savez(
        inputs,
        points=np.asarray(points, dtype=np.float64),
        vertices=vertices,
        vertex_count=counts,
        mutable=np.zeros(1),
    )
    runner = directory / "probe-runner.py"
    runner.write_text(RUNNER)
    environment = dict(os.environ)
    if tree is not None:
        environment["PYTHONPATH"] = str(tree)
    subprocess.run(
        [sys.executable, str(runner), str(inputs), str(outputs), str(batch)],
        check=True,
        cwd=str(tree) if tree else os.getcwd(),
        env=environment,
    )
    with np.load(outputs) as archive:
        return np.asarray(archive["rows"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--geometry", default=GEOMETRY)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--tree", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    violations, worst = selection(args.split)
    records = plan(violations, worst)
    cells = sorted({record["cell"] for record in records})
    local = {cell: index for index, cell in enumerate(cells)}
    geometry = {}
    with np.load(args.geometry) as archive:
        for key in archive:
            geometry[key] = np.asarray(archive[key])
    polys = polygons(geometry)
    selected = tuple(polys[cell] for cell in cells)
    points = [record["point"] for record in records]
    rows = evaluate(points, selected, args.directory, args.batch, args.tree)

    payload = dict(
        arm=[record["arm"] for record in records],
        node=[record["entry"]["node"] for record in records],
        cell=[record["cell"] for record in records],
        moment=[record["moment"] for record in records],
        offset=[record.get("offset", 0.0) for record in records],
        sign=[record.get("sign", 0.0) for record in records],
        step=[record.get("step", 0.0) for record in records],
        point=np.array(points),
        value=np.array(
            [
                rows[i, local[record["cell"]], record["moment"], 0]
                for i, record in enumerate(records)
            ]
        ),
        derivative=np.array(
            [
                rows[i, local[record["cell"]], record["moment"], 2]
                for i, record in enumerate(records)
            ]
        ),
        radial=np.array(
            [
                rows[i, local[record["cell"]], record["moment"], 1]
                for i, record in enumerate(records)
            ]
        ),
    )
    np.savez(args.output or (Path(args.directory) / "probe.npz"), **payload)
    print("RECORDS", len(records), "cells", len(cells), flush=True)


if __name__ == "__main__":
    main()
