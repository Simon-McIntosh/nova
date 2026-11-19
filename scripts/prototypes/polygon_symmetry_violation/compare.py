"""Compare the polygon kernel's dpsi/dZ against an independent psi difference.

Three questions, one program per tree:

``direct``  the booked ``dpsi/dZ`` of every selected entry.  The bank's row is
            built from psi, psiR and psiZ values plus the gradient's two field
            rows, so this program carries those rows only: the Hessian rows
            the production jet also forms cannot change a value the
            concatenation copies from an earlier column.
``psi``     psi at ``z +/- h`` for a step ladder, which gives a central
            difference per entry.  psi is carried by both kernels, so a
            difference of psi values is not a difference of formulas: this is
            the only reference independent of either derivative route.
``sweep``   ``dpsi/dZ`` along Z across the corner-height band, so the tangent
            stencil (within ``1e-7`` of a corner height) and the joint corner
            layer (within ``1e-4 * radius`` of a corner) are each visible from
            both sides.

The variants isolate a mechanism by making it not happen: ``nocorner``
disables the near-corner tangent stencil in the derivative, ``noband``
disables the joint corner layer in the value program.  A variant that leaves
the booked value unchanged excludes its branch as the source of that value.

Run with ``PYTHONPATH=. /home/ITER/mcintos/Code/nova/.venv/bin/python
scripts/prototypes/polygon_symmetry_violation/compare.py --split <json>
--geometry <npz> --head-tree <archive> --worktree <checkout>
--directory <scratch> --output <json>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE_REVISION = "02997905193b245dcc099a238fea6c22d8817671"
KERNEL = "nova/biot/polygonanalytic.py"
STEPS = (
    3e-1,
    1e-1,
    3e-2,
    1e-2,
    3e-3,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
    1e-6,
)
OFFSETS = (
    0.0,
    1e-9,
    -1e-9,
    1e-7,
    -1e-7,
    1e-6,
    -1e-6,
    3e-5,
    -3e-5,
    1e-4,
    -1e-4,
    1e-3,
    -1e-3,
)
SCALE_DZ = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)

CORNER_STENCIL = "    near_corner = nearest <= 1e-7\n"
CORNER_STENCIL_OFF = "    near_corner = nearest < 0.0\n"
CORNER_BAND = (
    "    return (xp.abs(offset) <= 1e-3) & (xp.hypot(level, offset) < 1e-4 * radius)\n"
)
CORNER_BAND_OFF = (
    "    return (xp.abs(offset) <= 1e-3) & (xp.hypot(level, offset) < 0.0 * radius)\n"
)

RUNNER = '''
"""Evaluate booked section moments at a point list from one tree."""
import os
import sys

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
from nova.biot import polygonanalytic
from nova.equilibrium import topology

inputs = np.load(sys.argv[1])
points = np.asarray(inputs["points"], dtype=np.float64)
vertices = np.asarray(inputs["vertices"])
counts = np.asarray(inputs["vertex_count"])
batch = int(sys.argv[2])
polygons = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
coupling = topology.BiotMomentCoupling.from_polygons(polygons)
inverse = coupling.inverse_second_moment
cells = len(polygons)


def rows(target):
    section = polygonanalytic.packed_analytic_moments(
        jnp,
        jnp.broadcast_to(target[0], coupling.norm.shape),
        jnp.broadcast_to(target[1], coupling.norm.shape),
        coupling.edge,
        coupling.weight,
        coupling.norm,
        coupling.centre.T,
        coupling.centre.T,
        coupling.reflection_axis,
        coupling.reflection_partner,
    )
    value = jnp.stack(section[:3], axis=-1)
    gradient = 2 * jnp.pi * target[0] * jnp.stack(
        (jnp.stack(section[6:9], axis=-1), -jnp.stack(section[3:6], axis=-1)),
        axis=-1,
    )
    coefficients = jnp.concatenate((value[..., None], gradient), axis=-1)
    transform = jnp.stack(
        (
            jnp.stack((jnp.ones(cells), jnp.zeros(cells), jnp.zeros(cells)), axis=-1),
            jnp.stack((jnp.zeros(cells), inverse[:, 0, 0], inverse[:, 0, 1]), axis=-1),
            jnp.stack((jnp.zeros(cells), inverse[:, 1, 0], inverse[:, 1, 1]), axis=-1),
        ),
        axis=1,
    )
    return jnp.einsum("cbq,cbm->cmq", coefficients, transform)


program = jax.jit(jax.vmap(rows))
result = np.zeros((len(points), cells, 3, 3))
for first in range(0, len(points), batch):
    block = points[first:first + batch]
    size = len(block)
    if size < batch:
        block = np.concatenate((block, np.repeat(block[-1:], batch - size, axis=0)))
    result[first:first + size] = np.asarray(program(jnp.asarray(block)))[:size]
np.savez(sys.argv[3], rows=result)
print(
    "ROWS", result.shape, "module", polygonanalytic.__file__, "cwd", os.getcwd(),
    flush=True,
)
'''


def stack(polygons):
    """Pad a polygon tuple into the vertex and count arrays the runner reads."""
    width = max(len(polygon) for polygon in polygons)
    vertices = np.zeros((len(polygons), width, 2))
    counts = np.zeros(len(polygons), dtype=np.int64)
    for index, polygon in enumerate(polygons):
        vertices[index, : len(polygon)] = polygon
        counts[index] = len(polygon)
    return vertices, counts


def evaluate(points, polygons, tree, directory, batch):
    """Run one tree's program over the point list and return its rows."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vertices, counts = stack(polygons)
    inputs, outputs = directory / "inputs.npz", directory / "rows.npz"
    np.savez(
        inputs,
        points=np.asarray(points, dtype=np.float64),
        vertices=vertices,
        vertex_count=counts,
    )
    (directory / "runner.py").write_text(RUNNER)
    completed = subprocess.run(
        [
            sys.executable,
            str(directory / "runner.py"),
            str(inputs),
            str(batch),
            str(outputs),
        ],
        check=True,
        capture_output=True,
        cwd=str(tree),
        env=dict(os.environ, PYTHONPATH=str(tree)),
    )
    print("TREE", tree.name, completed.stdout.decode().strip(), flush=True)
    with np.load(outputs) as archive:
        return np.asarray(archive["rows"])


def copy_tree(source, destination):
    """Copy the package a variant needs, leaving the evidence tree behind."""
    tree = Path(destination)
    shutil.copytree(
        Path(source) / "nova",
        tree / "nova",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    return tree


def variant(source, destination, old, new, marker):
    """Copy a tree and apply one textual kernel substitution."""
    tree = copy_tree(source, destination)
    path = tree / KERNEL
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(
            f"substitution target appears {text.count(old)} times: {old!r}"
        )
    path.write_text(text.replace(old, new))
    if marker not in path.read_text():
        raise SystemExit(f"variant marker absent from {path}: {marker!r}")
    return tree


def base_tree(source, destination, worktree, revision):
    """Copy the current tree with the base revision's kernel module restored."""
    tree = copy_tree(source, destination)
    blob = subprocess.run(
        ["git", "-C", str(worktree), "show", f"{revision}:{KERNEL}"],
        check=True,
        capture_output=True,
    ).stdout
    (tree / KERNEL).write_bytes(blob)
    if "stable_poles" in (tree / KERNEL).read_text():
        raise SystemExit("base kernel still carries the repaired template route")
    return tree


def plan(violations, worst, centre):
    """The point list: direct, psi ladder, corner sweep and an off-plane scale."""
    entries = violations + worst
    records = []
    for index, entry in enumerate(entries):
        records.append(
            dict(arm="direct", entry=index, point=tuple(entry["coordinates"]), step=0.0)
        )
    for index, entry in enumerate(entries):
        r, z = entry["coordinates"]
        for step in STEPS:
            for sign in (1.0, -1.0):
                records.append(
                    dict(
                        arm="psi",
                        entry=index,
                        point=(r, z + sign * step),
                        step=sign * step,
                    )
                )
    for index, entry in enumerate(violations):
        r, z = entry["coordinates"]
        for offset in OFFSETS:
            records.append(
                dict(arm="sweep", entry=index, point=(r, z + offset), step=offset)
            )
    for index, entry in enumerate(violations):
        r, _ = entry["coordinates"]
        cell = int(entry["cell"])
        for dz in SCALE_DZ:
            records.append(
                dict(
                    arm="scale",
                    entry=index,
                    point=(r, centre[cell, 1] + dz),
                    step=dz,
                )
            )
    for position, record in enumerate(records):
        record["index"] = position
    return records


def acquire(rows, record, local, moment, quantity):
    """One booked quantity of one record at one selected cell."""
    return float(rows[record["index"], local, moment, quantity])


def central(ladder):
    """Pair the psi values into a central difference per step magnitude."""
    forward = {step: value for step, value in ladder if step > 0.0}
    backward = {step: value for step, value in ladder if step < 0.0}
    return [
        dict(step=step, derivative=(forward[step] - backward[-step]) / (2.0 * step))
        for step in sorted(forward, reverse=True)
        if -step in backward
    ]


def per_entry(entries, values, records, local_of):
    """One row per entry: each tree's booked value, the ladder, the sweep."""
    direct_index = {
        record["entry"]: record for record in records if record["arm"] == "direct"
    }
    rows = []
    for index, entry in enumerate(entries):
        local = local_of[int(entry["cell"])]
        moment = int(entry["moment"])
        direct = {
            tree: acquire(
                rows=values[tree],
                record=direct_index[index],
                local=local,
                moment=moment,
                quantity=2,
            )
            for tree in values
        }
        ladder = [
            (
                record["step"],
                acquire(values["current"], record, local, moment, 0),
            )
            for record in records
            if record["arm"] == "psi" and record["entry"] == index
        ]
        differences = central(ladder)
        gaps = [
            abs(a["derivative"] - b["derivative"])
            for a, b in zip(differences, differences[1:])
        ]
        sweep = sorted(
            (record["step"], acquire(values["current"], record, local, moment, 2))
            for record in records
            if record["arm"] == "sweep" and record["entry"] == index
        )
        rows.append(
            dict(
                node=int(entry["node"]),
                cell=int(entry["cell"]),
                moment=moment,
                direct=direct,
                ladder=differences,
                finest=differences[-1]["derivative"] if differences else None,
                largest_gap=max(gaps) if gaps else None,
                sweep=sweep,
            )
        )
    return rows


def scale_rows(values, records, local_of, entries):
    """The off-plane booked value of each violating cell, per grid height."""
    rows = []
    for record in records:
        if record["arm"] != "scale":
            continue
        entry = entries[record["entry"]]
        local, moment = local_of[int(entry["cell"])], int(entry["moment"])
        rows.append(
            dict(
                node=int(entry["node"]),
                cell=int(entry["cell"]),
                moment=moment,
                dz=record["step"],
                point=list(record["point"]),
                direct={
                    tree: acquire(values[tree], record, local, moment, 2)
                    for tree in values
                },
            )
        )
    return rows


def headline(group_a, group_b):
    """The statistics the report quotes, from the rows it prints."""

    def magnitude(group, tree):
        return np.array([abs(row["direct"][tree]) for row in group])

    def reference(group):
        return np.array([row["finest"] for row in group], dtype=float)

    book_b = np.array([row["direct"]["current"] for row in group_b])
    base_b = np.array([row["direct"]["base"] for row in group_b])
    reference_b = reference(group_b)
    floor = np.maximum(np.abs(reference_b), 1e-30)
    return dict(
        group_a_count=len(group_a),
        group_b_count=len(group_b),
        a_current_max=float(magnitude(group_a, "current").max()),
        a_base_max=float(magnitude(group_a, "base").max()),
        a_nocorner_max=float(magnitude(group_a, "nocorner").max()),
        a_noband_max=float(magnitude(group_a, "noband").max()),
        a_current_median=float(np.median(magnitude(group_a, "current"))),
        b_current_median=float(np.median(magnitude(group_b, "current"))),
        b_current_max=float(magnitude(group_b, "current").max()),
        b_base_max=float(magnitude(group_b, "base").max()),
        b_reference_median=float(np.median(np.abs(reference_b))),
        b_reference_max=float(np.abs(reference_b).max()),
        b_current_vs_reference_max_relative=float(
            np.max(np.abs(book_b - reference_b) / floor)
        ),
        b_base_vs_reference_max_relative=float(
            np.max(np.abs(base_b - reference_b) / floor)
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--head-tree", default=".")
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    with open(args.split) as stream:
        split = json.load(stream)
    violations, worst = split["violations_A"], split["worst_B"]
    entries = violations + worst

    geometry = {}
    with np.load(args.geometry) as archive:
        for key in archive:
            geometry[key] = np.asarray(archive[key])
    vertices = np.asarray(geometry["vertices"])
    counts = np.asarray(geometry["vertex_count"])
    centre = np.asarray(geometry["centre"])
    polygons = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
    cells = sorted({int(entry["cell"]) for entry in entries})
    selected = tuple(polygons[cell] for cell in cells)
    local_of = {cell: index for index, cell in enumerate(cells)}

    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    head = Path(args.head_tree).resolve()
    records = plan(violations, worst, centre)
    points = [record["point"] for record in records]

    trees = dict(
        current=head,
        base=base_tree(
            head, directory / "trees" / "base", args.worktree, args.base_revision
        ),
        nocorner=variant(
            head,
            directory / "trees" / "nocorner",
            CORNER_STENCIL,
            CORNER_STENCIL_OFF,
            "near_corner = nearest < 0.0",
        ),
        noband=variant(
            head,
            directory / "trees" / "noband",
            CORNER_BAND,
            CORNER_BAND_OFF,
            "hypot(level, offset) < 0.0 * radius",
        ),
    )

    values = {}
    for name, tree in trees.items():
        values[name] = evaluate(
            points, selected, tree, directory / "run" / name, args.batch
        )

    psi_gap = 0.0
    for record in records:
        if record["arm"] != "psi":
            continue
        entry = entries[record["entry"]]
        local, moment = local_of[int(entry["cell"])], int(entry["moment"])
        psi_gap = max(
            psi_gap,
            abs(
                acquire(values["current"], record, local, moment, 0)
                - acquire(values["base"], record, local, moment, 0)
            ),
        )

    group_rows = per_entry(entries, values, records, local_of)
    payload = dict(
        trees={name: str(tree) for name, tree in trees.items()},
        base_revision=args.base_revision,
        psi_current_vs_base_max_abs=psi_gap,
        steps=list(STEPS),
        offsets=list(OFFSETS),
        group_a=group_rows[: len(violations)],
        group_b=group_rows[len(violations) :],
        scale=scale_rows(values, records, local_of, entries),
        headline=headline(group_rows[: len(violations)], group_rows[len(violations) :]),
    )
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=1)
    print("SUMMARY " + json.dumps(payload["headline"]), flush=True)


if __name__ == "__main__":
    main()
