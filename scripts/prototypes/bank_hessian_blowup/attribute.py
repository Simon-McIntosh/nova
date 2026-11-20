"""Attribute the repaired bank's unbounded second derivatives.

Two measurements, each written to the output before the next begins.

``pairs``
    Every pair whose certified second-derivative magnitude exceeds a bound,
    with all six of its rows in the certified bank and in the pre-repair store,
    and its geometry: distance to the source cell's nearest vertex and nearest
    edge, height gap to the cell's nearest vertex height, and height gap to the
    nearest polygon-vertex height in the whole mesh.

``selector``
    The shared moment kernel substitutes its Richardson-extrapolated height
    tangent whenever the target height lies within 1e-7 of ANY polygon vertex
    height in the mesh, and keeps the ordinary ``jvp of the value graph``
    otherwise. Counting nodes by that selector, and the pairs that differ
    sharply from the pre-repair store on each side, is what shows whether the
    substitution is what moved them.
"""

import argparse
import json
import numpy as np

CERTIFIED = (
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846f0e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
PRE_REPAIR = (
    "/home/ITER/mcintos/.cache/nova-dina-flattop/inputs/bank/store/sha256/"
    "fd912554f8e9e7efddce53b119242a46e88f20ac30e029f83781c6253f828425/bank.npz"
)
GEOMETRY = "/home/ITER/mcintos/.cache/nova-iter-bank/dina-rerun/bank/geometry.npz"
HESSIAN = slice(3, 6)
HEIGHT_TOLERANCE = 1e-7


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(
            rows=np.asarray(archive["rows"]),
            nodes=np.asarray(archive["nodes"]),
            active=int(np.asarray(archive["active_nodes"])),
        )


def polygons():
    with np.load(GEOMETRY, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"])
        counts = np.asarray(archive["vertex_count"])
    return tuple(vertices[index, : int(count)] for index, count in enumerate(counts))


def segment_distance(point, polygon):
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    along = ends - starts
    offset = point[None, :] - starts
    square = np.sum(along * along, axis=-1)
    numerator = np.sum(offset * along, axis=-1)
    fraction = np.where(
        square > 0.0, numerator / np.where(square > 0.0, square, 1.0), 0.0
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projection = starts + fraction[:, None] * along
    return np.linalg.norm(point[None, :] - projection, axis=-1)


def pair_census(certified, pre_repair, cells, bound):
    magnitude = np.abs(certified["rows"][..., HESSIAN].astype(np.float64))
    pair_max = magnitude.max(axis=(2, 3))
    heights = np.concatenate([polygon[:, 1] for polygon in cells])
    records = []
    for node, cell in np.argwhere(pair_max > bound):
        node, cell = int(node), int(cell)
        point = certified["nodes"][node]
        polygon = cells[cell]
        vertex = np.linalg.norm(polygon - point[None, :], axis=-1)
        edge = segment_distance(point, polygon)
        records.append(
            dict(
                node=node,
                cell=cell,
                pair_max=float(pair_max[node, cell]),
                position=[float(point[0]), float(point[1])],
                distance_vertex=float(vertex.min()),
                distance_edge=float(edge.min()) if edge.size else 0.0,
                cell_height_gap=float(np.abs(polygon[:, 1] - point[1]).min()),
                mesh_height_gap=float(np.abs(heights - point[1]).min()),
                certified=[
                    float(value) for value in certified["rows"][node, cell].reshape(-1)
                ],
                pre_repair=[
                    float(value) for value in pre_repair["rows"][node, cell].reshape(-1)
                ],
            )
        )
    records.sort(key=lambda item: item["pair_max"], reverse=True)
    return records


def selector_census(certified, pre_repair, cells):
    nodes = certified["nodes"]
    heights = np.concatenate([polygon[:, 1] for polygon in cells])
    gap = np.abs(heights[None, :] - nodes[:, 1, None]).min(axis=1)
    selected = gap <= HEIGHT_TOLERANCE
    changed = (
        np.abs(
            certified["rows"][..., 3:6].astype(np.float64)
            - pre_repair["rows"][..., 3:6].astype(np.float64)
        ).max(axis=(2, 3))
        > 1e-2
    )
    return dict(
        tolerance=HEIGHT_TOLERANCE,
        nodes=int(len(nodes)),
        selected_nodes=int(selected.sum()),
        selected_max_gap=float(gap[selected].max()) if selected.any() else None,
        unselected_min_gap=float(gap[~selected].min()) if (~selected).any() else None,
        changed_pairs=int(changed.sum()),
        changed_pairs_selected=int((changed & selected[:, None]).sum()),
        changed_pairs_unselected=int((changed & ~selected[:, None]).sum()),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bound", type=float, default=1e9)
    arguments = parser.parse_args()

    certified = read(CERTIFIED)
    pre_repair = read(PRE_REPAIR)
    active = min(certified["active"], pre_repair["active"])
    for record in (certified, pre_repair):
        record["rows"] = record["rows"][:active]
        record["nodes"] = record["nodes"][:active]

    cells = polygons()
    record = dict(active_nodes=active)
    record["pairs"] = pair_census(certified, pre_repair, cells, arguments.bound)
    with open(arguments.output, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print("PAIRS", len(record["pairs"]), flush=True)

    record["selector"] = selector_census(certified, pre_repair, cells)
    with open(arguments.output, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print("SELECTOR", json.dumps(record["selector"]), flush=True)
    print("ATTRIBUTE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
