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

from scripts.prototypes.bank_hessian_blowup import geometry

CERTIFIED = (
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846f0e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
PRE_REPAIR = (
    "/home/ITER/mcintos/.cache/nova-dina-flattop/inputs/bank/store/sha256/"
    "fd912554f8e9e7efddce53b119242a46e88f20ac30e029f83781c6253f828425/bank.npz"
)
HESSIAN = slice(3, 6)
HEIGHT_TOLERANCE = 1e-7
SECOND_ENTRIES = {
    3: (0, "psi_RR"),
    4: (0, "psi_RZ"),
    5: (0, "psi_ZZ"),
    9: (1, "psi_RR"),
    10: (1, "psi_RZ"),
    11: (1, "psi_ZZ"),
    15: (2, "psi_RR"),
    16: (2, "psi_RZ"),
    17: (2, "psi_ZZ"),
}


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(
            rows=np.asarray(archive["rows"]),
            nodes=np.asarray(archive["nodes"]),
            active=int(np.asarray(archive["active_nodes"])),
        )


def second_moment_singular(cells, selected):
    """Singular values of each selected cell's inverse second moment.

    The bank contracts every source moment against this matrix, so a cell whose
    singular values are orders above an ordinary cell's magnifies every entry
    booked against it.
    """
    from nova.equilibrium.topology import BiotMomentCoupling

    coupling = BiotMomentCoupling.from_polygons(cells)
    inverse = np.asarray(coupling.inverse_second_moment)
    return {
        str(cell): [
            float(value) for value in np.linalg.svd(inverse[cell], compute_uv=False)
        ]
        for cell in sorted(selected)
    }


def pair_census(certified, pre_repair, cells, bound):
    magnitude = np.abs(certified["rows"][..., HESSIAN].astype(np.float64))
    pair_max = magnitude.max(axis=(2, 3))
    heights = np.concatenate([polygon[:, 1] for polygon in cells])
    records = []
    for node, cell in np.argwhere(pair_max > bound):
        node, cell = int(node), int(cell)
        point = certified["nodes"][node]
        polygon = cells[cell]
        booked = np.asarray(certified["rows"][node, cell]).reshape(-1)
        index = max(SECOND_ENTRIES, key=lambda i: abs(booked[i]))
        moment, component = SECOND_ENTRIES[index]
        records.append(
            dict(
                node=node,
                cell=cell,
                moment=moment,
                component=component,
                entry_index=index,
                pair_max=float(pair_max[node, cell]),
                position=[float(point[0]), float(point[1])],
                distance_vertex=geometry.vertex_distance(point, polygon),
                distance_edge=geometry.segment_distance(point, polygon),
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
    parser.add_argument("--reference-cell", type=int, default=38)
    arguments = parser.parse_args()

    certified = read(CERTIFIED)
    pre_repair = read(PRE_REPAIR)
    active = min(certified["active"], pre_repair["active"])
    for record in (certified, pre_repair):
        record["rows"] = record["rows"][:active]
        record["nodes"] = record["nodes"][:active]

    cells = geometry.cells()
    record = dict(active_nodes=active, cells=len(cells))
    record["pairs"] = pair_census(certified, pre_repair, cells, arguments.bound)
    selected = {pair["cell"] for pair in record["pairs"]}
    selected.add(arguments.reference_cell)
    record["inverse_second_moment"] = second_moment_singular(cells, selected)
    record["reference_cell"] = arguments.reference_cell
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
