"""Locate the repaired-kernel ITER bank's unbounded second derivatives.

Reads the certified bank, the pre-repair store and the DINA rerun's rebuilt
stores, and reports for each: the largest second-derivative entries, where they
sit (node, cell, moment, component), and the geometry of every offending pair --
its distance to the source cell's nearest vertex and nearest edge, and the
height gap to the nearest polygon-vertex height in the whole mesh.

That last quantity is the selector of the shared kernel's stable-height tangent:
``_packed_moments_height_tangent`` substitutes its Richardson-extrapolated
height derivative when the target's Z lies within 1e-7 of ANY polygon vertex
height, globally, and otherwise keeps the ordinary ``jvp``. Measuring each
offending pair against that selector is what lets the report say whether the
substitution, and not the target's own cell, is what produced the value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = "/home/ITER/mcintos/.cache/nova-iter-bank/dina-rerun/bank/store/sha256"

BANKS = {
    "certified": "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846f0e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz",
    "pre_repair": "/home/ITER/mcintos/.cache/nova-dina-flattop/inputs/bank/store/"
    "sha256/fd912554f8e9e7efddce53b119242a46e88f20ac30e029f83781c6253f828425/bank.npz",
    "rebuild_2e30fa75": f"{ROOT}/"
    "2e30fa75bb7cdd26ecf3c5e88f4e0df2ca9e0defa9858961624810ac29889587/bank.npz",
    "rebuild_ac2c5474": f"{ROOT}/"
    "ac2c54742f282ffbfbe997fbe863b0883991dd9461c4d0506efdc44c143e99cc/bank.npz",
    "carried_corner": f"{ROOT}/"
    "6c1a1d8fff9939bdeac30b54108408ab9ece0b9612715c21327d6642935290b5/bank.npz",
    "corner_omitted": f"{ROOT}/"
    "b00f714ac1d4de4d92ae0d4974482aaafc83d346dd797024eed6f67c3d0d04b3/bank.npz",
}

GEOMETRY = Path("/home/ITER/mcintos/.cache/nova-iter-bank/dina-rerun/bank/geometry.npz")

HESSIAN = slice(3, 6)
COMPONENTS = ("psi_RR", "psi_RZ", "psi_ZZ")
HEIGHT_TOLERANCE = 1e-7


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(
            rows=np.asarray(archive["rows"]),
            nodes=np.asarray(archive["nodes"]),
            active_nodes=int(np.asarray(archive["active_nodes"])),
            path=str(path),
        )


def cells():
    with np.load(GEOMETRY, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"])
        counts = np.asarray(archive["vertex_count"])
    return tuple(vertices[index, : int(count)] for index, count in enumerate(counts))


def point_segment_distance(point, starts, ends):
    along = ends - starts
    offset = point[None, :] - starts
    square = np.sum(along * along, axis=-1)
    fraction = np.where(
        square > 0.0,
        np.sum(offset * along, axis=-1) / np.where(square > 0.0, square, 1.0),
        0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projection = starts + fraction[:, None] * along
    return np.linalg.norm(point[None, :] - projection, axis=-1)


def pair_geometry(node, polygon):
    vertex = np.linalg.norm(polygon - node[None, :], axis=-1)
    edge = point_segment_distance(node, polygon, np.roll(polygon, -1, axis=0))
    return dict(
        distance_vertex=float(vertex.min()),
        vertex_index=int(np.argmin(vertex)),
        distance_edge=float(edge.min()) if edge.size else 0.0,
        cell_height_gap=float(np.abs(polygon[:, 1] - node[1]).min()),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=12)
    arguments = parser.parse_args()

    polygons = cells()
    heights = np.concatenate([polygon[:, 1] for polygon in polygons])

    loaded = {}
    record = dict(height_tolerance=HEIGHT_TOLERANCE, cells=len(polygons), banks={})
    for name, path in BANKS.items():
        bank = load(path)
        active = bank["active_nodes"]
        rows = bank["rows"][:active]
        nodes = bank["nodes"][:active]
        loaded[name] = dict(rows=rows, nodes=nodes, path=bank["path"])

        magnitude = np.abs(rows[..., HESSIAN].astype(np.float64))
        pair_max = magnitude.max(axis=(2, 3))
        order = np.argsort(pair_max, axis=None)[::-1]

        worst = []
        for flat in order[: arguments.top]:
            node, cell = (
                int(value) for value in np.unravel_index(int(flat), pair_max.shape)
            )
            moment, component = (
                int(value)
                for value in np.unravel_index(
                    int(np.argmax(magnitude[node, cell])), magnitude.shape[2:]
                )
            )
            item = dict(
                node=node,
                cell=cell,
                moment=moment,
                component=COMPONENTS[component],
                value=float(rows[node, cell, moment, 3 + component]),
                pair_max=float(pair_max[node, cell]),
                position=[float(nodes[node, 0]), float(nodes[node, 1])],
                nearest_vertex_height_gap=float(np.abs(heights - nodes[node, 1]).min()),
            )
            item.update(pair_geometry(nodes[node], polygons[cell]))
            worst.append(item)

        record["banks"][name] = dict(
            path=bank["path"],
            active_nodes=active,
            shape=list(rows.shape),
            maximum=float(pair_max.max()),
            above={
                str(bound): int((pair_max > bound).sum())
                for bound in (1e6, 1e9, 1e12, 1e15)
            },
            per_moment={
                str(moment): float(magnitude[:, :, moment].max()) for moment in range(3)
            },
            per_component={
                COMPONENTS[index]: float(magnitude[..., index].max())
                for index in range(3)
            },
            worst=worst,
        )
        print("BANK", name, "max", float(pair_max.max()), flush=True)

    names = list(loaded)
    agreement = {}
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            left = loaded[names[first]]["rows"][..., HESSIAN]
            right = loaded[names[second]]["rows"][..., HESSIAN]
            agreement[f"{names[first]}~{names[second]}"] = dict(
                identical=bool(np.array_equal(left, right)),
                max_absolute_difference=float(np.abs(left - right).max()),
            )
    record["agreement"] = agreement

    index_of = {name: position for position, name in enumerate(COMPONENTS)}
    for other in ("pre_repair", "certified"):
        table = []
        for item in record["banks"]["carried_corner"]["worst"]:
            table.append(
                dict(
                    node=item["node"],
                    cell=item["cell"],
                    moment=item["moment"],
                    component=item["component"],
                    rebuilt=item["value"],
                    other=float(
                        loaded[other]["rows"][
                            item["node"],
                            item["cell"],
                            item["moment"],
                            3 + index_of[item["component"]],
                        ]
                    ),
                )
            )
        record[f"carried_corner_against_{other}"] = table

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")
    print("DIAGNOSIS_COMPLETE", arguments.output, flush=True)


if __name__ == "__main__":
    main()
