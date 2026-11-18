"""Attribute the rebuilt bank's first-derivative rows against the two banks.

Reads the two published bank npz archives, takes the two first-derivative
components (d psi / d R and d psi / d Z), and reports:

* the unfiltered maximum relative departure and where it sits, with its
  absolute size and the magnitude it was divided by;
* the same over the entries whose magnitude exceeds a filter window, which is
  the statistic the fragment quotes beside the 1e-10 done-when;
* one line per entry whose relative departure exceeds a threshold, carrying
  its node, cell, moment, component, absolute departure and magnitude;
* a MATERIAL verdict on whether any entry of magnitude above the window
  departs by more than 1e-7 in absolute terms, the boundary the lead named for
  withdrawing the accepted exception.

Every figure is computed per entry: the relative departure is
``|head - current| / |current|``, so a near-zero magnitude can carry a large
relative departure while its absolute departure stays at the archive's noise.
The node-aggregated norm the certification script reports is printed beside it
so the two statistics cannot be mistaken for one another.

Run it from the repository root with ``PYTHONPATH=.``, so the shared helper
(and this script's own module path) resolves:
``PYTHONPATH=. python scripts/prototypes/iter_bank/derive_rows.py``.

``--zero-entries`` answers a different question about the same two archives:
every first-derivative entry whose current-bank value is exactly zero, its value
in each bank, and a direct evaluation of the same jet from this tree's kernel and
from ``--base-tree``'s kernel (run in a subprocess that imports that tree). Each
entry also carries the distance from its node to the source cell's boundary and
to that cell's nearest vertex, so an entry that sits on the source can be told
from one that does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.iter_bank.recertify import relative

DERIVATIVE = slice(1, 3)
COMPONENT = {0: "dpsi/dR", 1: "dpsi/dZ"}
DEFAULT_THRESHOLD = 1e-10
DEFAULT_WINDOW = 1e-3
DEFAULT_BOUNDARY = None


def load(path):
    """Read a bank npz and return its rows, active-node count and path."""
    with np.load(path, allow_pickle=False) as archive:
        return dict(
            rows=np.asarray(archive["rows"]),
            nodes=np.asarray(archive["nodes"]),
            active_nodes=int(archive["active_nodes"]),
            bucket=int(archive["bucket"]),
            path=str(path),
        )


def compare(
    head_rows,
    current_rows,
    *,
    threshold=DEFAULT_THRESHOLD,
    window=DEFAULT_WINDOW,
    boundary=DEFAULT_BOUNDARY,
    components=DERIVATIVE,
):
    """Compare two stored row arrays and return the report the script prints.

    ``head_rows`` and ``current_rows`` are the archives' ``rows`` arrays, shaped
    ``(nodes, cells, moments, 6)``; ``components`` selects the derivative slice
    under comparison, so a different slice is a different measurement and every
    statistic here is computed from that slice alone.
    """
    head = np.asarray(head_rows)[:, :, :, components].astype(np.float64)
    current = np.asarray(current_rows)[:, :, :, components].astype(np.float64)
    delta = head - current
    magnitude = np.abs(current)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(magnitude > 0.0, np.abs(delta) / magnitude, np.inf)
    zero_magnitude = magnitude == 0.0
    finite = np.isfinite(rel)
    keep = magnitude > window

    def locate(flat_index):
        node, cell, moment, component = np.unravel_index(flat_index, rel.shape)
        return dict(
            node=int(node),
            cell=int(cell),
            moment=int(moment),
            component=COMPONENT[int(component)],
            absolute_departure=float(delta[node, cell, moment, component]),
            magnitude=float(current[node, cell, moment, component]),
            relative=float(rel[node, cell, moment, component]),
        )

    unfiltered = dict(
        count=int(finite.sum()),
        max_relative=float(np.max(rel[finite])),
        median_relative=float(np.median(rel[finite])),
        location=locate(int(np.argmax(np.where(finite, rel, -np.inf)))),
        above_threshold=int((rel > threshold).sum()),
    )
    filtered = dict(
        window=window,
        count=int(keep.sum()),
        excluded=int((~keep & finite).sum()),
        max_relative=float(np.max(np.where(keep, rel, -np.inf))),
        median_relative=float(np.median(rel[keep])) if keep.any() else None,
        above_threshold=int(((rel > threshold) & keep).sum()),
        max_absolute=float(np.max(np.abs(delta)[keep])) if keep.any() else None,
        max_relative_location=locate(int(np.argmax(np.where(keep, rel, -np.inf)))),
        max_absolute_location=locate(
            int(np.argmax(np.where(keep, np.abs(delta), -np.inf)))
        ),
    )
    report = dict(
        entries=int(rel.size),
        components=[COMPONENT[index] for index in COMPONENT],
        zero_magnitude_entries=int(zero_magnitude.sum()),
        threshold=threshold,
        window=window,
        terminal_boundary=boundary,
        unfiltered=unfiltered,
        filtered=filtered,
    )
    if zero_magnitude.any():
        report["zero_magnitude_max_abs_delta"] = float(
            np.max(np.abs(delta)[zero_magnitude])
        )
    for index, name in COMPONENT.items():
        part = rel[..., index]
        part_finite = np.isfinite(part)
        part_keep = magnitude[..., index] > window
        entry = dict(
            count=int(part_finite.sum()),
            above_threshold=int((part > threshold).sum()),
            filtered_count=int(part_keep.sum()),
            filtered_above_threshold=int(((part > threshold) & part_keep).sum()),
            max_relative=float(np.max(part[part_finite])),
            median_relative=float(np.median(part[part_finite])),
            filtered_max_relative=None,
            filtered_median_relative=None,
            filtered_max_absolute=None,
            filtered_max_relative_location=None,
        )
        if part_keep.any():
            entry["filtered_max_relative"] = float(
                np.max(np.where(part_keep, part, -np.inf))
            )
            entry["filtered_median_relative"] = float(np.median(part[part_keep]))
            entry["filtered_max_absolute"] = float(
                np.max(np.abs(delta[..., index])[part_keep])
            )
            selected = int(np.argmax(np.where(part_keep, part, -np.inf)))
            node, cell, moment = np.unravel_index(selected, part.shape)
            entry["filtered_max_relative_location"] = dict(
                node=int(node),
                cell=int(cell),
                moment=int(moment),
                component=name,
                absolute_departure=float(delta[..., index][node, cell, moment]),
                magnitude=float(current[..., index][node, cell, moment]),
                relative=float(part[node, cell, moment]),
            )
        report[name] = entry
    departed = np.argwhere(rel > threshold)
    report["departing_count"] = int(len(departed))
    report["departures"] = [
        locate(int(np.ravel_multi_index(tuple(row), rel.shape))) for row in departed
    ]
    aggregated = relative(delta, current, axis=0)
    report["node_aggregated"] = dict(
        shape=list(aggregated.shape),
        max_relative=float(np.max(aggregated)),
        median_relative=float(np.median(aggregated)),
    )
    if boundary is None:
        report["material"] = dict(boundary=None, count=0, present=None)
    else:
        material = keep & (np.abs(delta) > boundary)
        report["material"] = dict(
            boundary=boundary, count=int(material.sum()), present=bool(material.any())
        )
        if material.any():
            report["material"]["largest"] = locate(
                int(np.argmax(np.where(material, np.abs(delta), -np.inf)))
            )
    return report


def exit_status(report):
    """The status the script leaves: 2 only when a named boundary is crossed."""
    return 2 if report["material"]["present"] else 0


def zero_entries(head_rows, current_rows, components=DERIVATIVE):
    """Every first-derivative entry whose current-bank magnitude is exactly zero."""
    current = np.asarray(current_rows)[:, :, :, components]
    head = np.asarray(head_rows)[:, :, :, components]
    entries = []
    for row in np.argwhere(current == 0):
        node, cell, moment, index = (int(value) for value in row)
        entries.append(
            dict(
                node=node,
                cell=cell,
                moment=moment,
                component=COMPONENT[index],
                current_value=float(current[node, cell, moment, index]),
                head_value=float(head[node, cell, moment, index]),
            )
        )
    return entries


def geometry_polygons(path):
    """Read the cell polygons and centres the bank was built from."""
    with np.load(path) as archive:
        vertices = np.asarray(archive["vertices"])
        counts = np.asarray(archive["vertex_count"])
        centre = np.asarray(archive["centre"])
    polygons = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
    return dict(
        vertices=vertices, vertex_count=counts, centre=centre, polygons=polygons
    )


def kernel_rows(points, polygons, batch=4):
    """Evaluate this tree's kernel jets at every point, for every cell."""
    import jax
    import jax.numpy as jnp
    from nova.equilibrium import topology
    from nova.equilibrium.fixed_points import _source_jets

    coupling = topology.BiotMomentCoupling.from_polygons(polygons)
    program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
    rows = np.zeros((len(points), len(polygons), 3, 6))
    for first in range(0, len(points), batch):
        block = np.asarray(points[first : first + batch], dtype=np.float64)
        size = len(block)
        if size < batch:
            block = np.concatenate((block, np.repeat(block[-1:], batch - size, axis=0)))
        rows[first : first + size] = np.asarray(program(jnp.asarray(block), coupling))[
            :size
        ]
    return rows


_ZERO_KERNEL_RUNNER = '''
"""Evaluate one revision of the polygon kernel at a list of points."""
import sys

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
from nova.equilibrium import topology
from nova.equilibrium.fixed_points import _source_jets

inputs = np.load(sys.argv[1])
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 4
points = np.asarray(inputs["points"], dtype=np.float64)
vertices = np.asarray(inputs["vertices"])
counts = np.asarray(inputs["vertex_count"])
polygons = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
coupling = topology.BiotMomentCoupling.from_polygons(polygons)
program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
rows = np.zeros((len(points), len(polygons), 3, 6))
for first in range(0, len(points), batch):
    block = points[first : first + batch]
    size = len(block)
    if size < batch:
        block = np.concatenate((block, np.repeat(block[-1:], batch - size, axis=0)))
    rows[first : first + size] = np.asarray(program(jnp.asarray(block), coupling))[
        :size
    ]
np.savez(sys.argv[3], rows=rows)
print("KERNEL_ROWS", len(points), len(polygons), flush=True)
'''


def stack_polygons(polygons):
    """Pad a polygon tuple into the vertices and counts the runner reads."""
    width = max(len(polygon) for polygon in polygons)
    vertices = np.zeros((len(polygons), width, 2))
    counts = np.zeros(len(polygons), dtype=np.int64)
    for index, polygon in enumerate(polygons):
        vertices[index, : len(polygon)] = polygon
        counts[index] = len(polygon)
    return dict(vertices=vertices, vertex_count=counts)


def base_kernel_rows(points, polygons, tree, directory, batch=4):
    """Evaluate another revision's kernel in a subprocess that imports that tree."""
    import os
    import subprocess
    import sys

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "zero-kernel-inputs.npz"
    outputs = directory / "zero-kernel-rows.npz"
    runner = directory / "zero-kernel-runner.py"
    geometry = stack_polygons(polygons)
    np.savez(
        inputs,
        points=np.asarray(points, dtype=np.float64),
        vertices=geometry["vertices"],
        vertex_count=geometry["vertex_count"],
    )
    runner.write_text(_ZERO_KERNEL_RUNNER)
    environment = dict(os.environ, PYTHONPATH=str(tree))
    subprocess.run(
        [sys.executable, str(runner), str(inputs), str(batch), str(outputs)],
        check=True,
        cwd=str(tree),
        env=environment,
    )
    with np.load(outputs) as archive:
        return np.asarray(archive["rows"])


def segment_distance(point, polygon):
    """Distance from one point to a closed polygon's boundary."""
    start = np.asarray(polygon, dtype=np.float64)
    end = np.asarray(np.roll(start, -1, axis=0), dtype=np.float64)
    direction = end - start
    denominator = np.sum(direction * direction, axis=-1)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    projection = np.clip(np.sum((point - start) * direction, axis=-1) / safe, 0.0, 1.0)
    nearest = start + projection[:, None] * direction
    return float(np.min(np.sqrt(np.sum((point - nearest) ** 2, axis=-1))))


def vertex_distance(point, polygon):
    """Distance from one point to a polygon's nearest vertex."""
    gaps = np.sqrt(np.sum((np.asarray(polygon) - np.asarray(point)) ** 2, axis=-1))
    return float(np.min(gaps))


def zero_entry_report(
    head, current, geometry_path, *, base_tree=None, directory=None, batch=4
):
    """Attribute every exactly-zero first-derivative entry to a kernel."""
    active = min(head["active_nodes"], current["active_nodes"])
    geometry = geometry_polygons(geometry_path)
    entries = zero_entries(head["rows"][:active], current["rows"][:active])
    nodes = np.asarray(current["nodes"][:active], dtype=np.float64)
    unique = sorted({entry["node"] for entry in entries})
    position_of = {node: position for position, node in enumerate(unique)}
    points = nodes[unique] if unique else np.zeros((0, 2), dtype=np.float64)
    cells = sorted({entry["cell"] for entry in entries})
    local_of = {cell: index for index, cell in enumerate(cells)}
    selected = tuple(geometry["polygons"][cell] for cell in cells)
    print(
        "ZERO_PLAN "
        + json.dumps(
            dict(entries=len(entries), nodes=len(unique), cells=len(cells), batch=batch)
        ),
        flush=True,
    )
    repaired = kernel_rows(points, selected, batch=batch)
    base = None
    if base_tree is not None and directory is not None:
        base = base_kernel_rows(points, selected, base_tree, directory, batch=batch)
    nearest = np.argmin(
        np.sum((points[:, None, :] - geometry["centre"][None, :, :]) ** 2, axis=-1),
        axis=1,
    )
    for entry in entries:
        position = position_of[entry["node"]]
        cell = entry["cell"]
        moment = entry["moment"]
        jet = 1 if entry["component"] == "dpsi/dR" else 2
        polygon = geometry["polygons"][cell]
        entry["coordinates"] = points[position].tolist()
        entry["own_cell"] = int(nearest[position])
        entry["node_belongs_to_this_cell"] = bool(int(nearest[position]) == cell)
        entry["distance_to_cell_boundary"] = segment_distance(points[position], polygon)
        entry["distance_to_nearest_vertex"] = vertex_distance(points[position], polygon)
        local = local_of[cell]
        entry["repaired_value"] = float(repaired[position, local, moment, jet])
        if base is None:
            entry["base_value"] = None
        else:
            entry["base_value"] = float(base[position, local, moment, jet])

    head_values = [abs(entry["head_value"]) for entry in entries]
    head_gaps = [
        abs(entry["head_value"] - entry["repaired_value"]) for entry in entries
    ]
    summary = dict(
        count=len(entries),
        unique_nodes=len(unique),
        head_max_abs_value=max(head_values, default=0.0),
        head_vs_repaired_max_abs=max(head_gaps, default=0.0),
        head_matches_repaired=bool(max(head_gaps, default=0.0) <= 1e-10),
    )
    if base is not None:
        base_values = [abs(entry["base_value"]) for entry in entries]
        base_gaps = [
            abs(entry["base_value"] - entry["repaired_value"]) for entry in entries
        ]
        summary["base_max_abs_value"] = max(base_values, default=0.0)
        summary["base_reproduces_zeros"] = bool(
            all(value == 0.0 for value in base_values)
        )
        summary["base_vs_repaired_max_abs"] = max(base_gaps, default=0.0)
    return dict(entries=entries, summary=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-bank", type=Path, required=True)
    parser.add_argument("--current-bank", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--material",
        type=float,
        default=DEFAULT_BOUNDARY,
        help="absolute departure that fails the run; none by default",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--zero-entries",
        action="store_true",
        help="attribute the exactly-zero first-derivative entries to a kernel",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        help="cell geometry npz, required by --zero-entries",
    )
    parser.add_argument(
        "--base-tree",
        type=Path,
        help="tree whose kernel --zero-entries also evaluates, in a subprocess",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="points per kernel batch; successive derivative sums need fewer",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        help="directory the base-kernel runner writes its inputs and rows to",
    )
    args = parser.parse_args()

    head = load(args.head_bank)
    current = load(args.current_bank)
    if head["rows"].shape != current["rows"].shape:
        raise SystemExit("the two banks have different row shapes")
    if args.zero_entries:
        if args.geometry is None:
            raise SystemExit("--zero-entries requires --geometry")
        zero = zero_entry_report(
            head,
            current,
            args.geometry,
            base_tree=args.base_tree,
            directory=args.work_directory,
            batch=args.batch,
        )
        zero["head_bank"] = head["path"]
        zero["current_bank"] = current["path"]
        zero["base_tree"] = str(args.base_tree) if args.base_tree is not None else None
        zero["geometry"] = str(args.geometry)
        zero["completed"] = True
        print("ZERO_SUMMARY " + json.dumps(zero["summary"], sort_keys=True), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(zero, indent=2))
            print("DERIVE_COMPLETE " + str(args.output), flush=True)
        raise SystemExit(0)
    active = min(head["active_nodes"], current["active_nodes"])
    if head["active_nodes"] != current["active_nodes"]:
        print("ACTIVE_NODES_DIFFER", head["active_nodes"], current["active_nodes"])
    print(
        "BANKS "
        + json.dumps(
            dict(
                head=head["path"],
                current=current["path"],
                head_shape=list(head["rows"].shape),
                current_shape=list(current["rows"].shape),
                head_active_nodes=head["active_nodes"],
                current_active_nodes=current["active_nodes"],
                active=active,
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    report = compare(
        head["rows"][:active],
        current["rows"][:active],
        threshold=args.threshold,
        window=args.window,
        boundary=args.material,
    )
    report["head_bank"] = head["path"]
    report["current_bank"] = current["path"]
    for key in ("unfiltered", "filtered", "node_aggregated", "dpsi/dR", "dpsi/dZ"):
        print(key.upper() + " " + json.dumps(report[key], sort_keys=True), flush=True)
    for departure in report["departures"]:
        print("DEPARTURE " + json.dumps(departure, sort_keys=True), flush=True)
    print("MATERIAL " + json.dumps(report["material"], sort_keys=True), flush=True)
    report["completed"] = True
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print("DERIVE_COMPLETE " + str(args.output), flush=True)
    raise SystemExit(exit_status(report))


if __name__ == "__main__":
    main()
