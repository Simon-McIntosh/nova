"""Build the certificate carriers and publish the vertex identity each reads.

Each carrier is built through the same route the carrier identity test uses:
the analytic reference solve, a wall inserted as the first wall of a hex
CoilSet plasma, the cell polygons extracted from the frame, a Delaunay
neighbour stencil over the cell centres, and the hex probe lattice mirrored
about every cell. What is published is the identity a build reads — the
revision's reference coefficient digest, the cell and vertex counts, the node
number, and which cell slots share a lattice vertex — so two builds of the
same revision on different hardware are compared on values, not on the order
a device happened to list its cells in.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback

import numpy as np
from shapely.geometry import LineString
from scipy.spatial import Delaunay

from nova.jax.config import configure_dtypes

configure_dtypes()

import nova  # noqa: E402
from benchmarks.solovev_certificate import DIVERTED_WALL_CLEARANCE_FRACTION  # noqa: E402
from nova.biot.plasmagrid import PlasmaGrid  # noqa: E402
from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null  # noqa: E402
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh  # noqa: E402
from nova.frame.coilset import CoilSet  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402

SEPARATRIX_SAMPLES = 1441
WALL_POINTS = 121
CASES = {
    "diverted300": {"requested_cells": 300, "wall": "diverted"},
    "diverted550": {"requested_cells": 550, "wall": "diverted"},
    "limited550": {"requested_cells": 550, "wall": "limited"},
}


def hex_sampling(centres, coilset):
    """Return the regular six-corner probe lattice centred on every cell."""
    sections = np.asarray(coilset.aloc["plasma", "section"], dtype=object).astype(str)
    full = np.flatnonzero(sections == "hexagon")
    if len(full) == 0:
        raise ValueError("the carrier has no complete hex generator")
    dimensions = np.c_[
        np.asarray(coilset.aloc["plasma", "dl"], dtype=float)[full],
        np.asarray(coilset.aloc["plasma", "dt"], dtype=float)[full],
    ]
    width, height = dimensions[0]
    radius = min(width / 2.0, height / np.sqrt(3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return centres[:, None, :] + offsets[None, :, :]


def vertex_partition(geometry):
    """Return which cell slots share a vertex, in a cell-order-free form.

    A slot is named by the micron cell its own polygon centre sits in, which
    the cell ordering does not move, so two builds that list one lattice
    differently publish the same grouping.
    """
    members = {}
    for cell, vertices in enumerate(geometry.sampling_vertices):
        centre = np.rint(np.mean(np.asarray(vertices, dtype=np.float64), axis=0) / 1e-6)
        for slot in range(len(vertices)):
            shared = members.setdefault(int(geometry.cell_sample_nodes[cell, slot]), [])
            shared.append([int(centre[0]), int(centre[1]), slot])
    return sorted(sorted(group) for group in members.values())


def partition_digest(partition):
    """Return a stable digest of a vertex partition."""
    payload = json.dumps(partition, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(kind):
    """Build one carrier and return the identity it publishes."""
    case = CASES[kind]
    exact = cerfon_freidberg_single_null()
    if case["wall"] == "diverted":
        wall = fixture.offset_wall(
            exact.separatrix(SEPARATRIX_SAMPLES),
            clearance=DIVERTED_WALL_CLEARANCE_FRACTION * exact.minor_radius,
            points=WALL_POINTS,
        )
    else:
        wall = fixture.limiter_contour(fixture.analytic_case(), points=WALL_POINTS)
    coilset = CoilSet(dplasma=-case["requested_cells"], tplasma="hex")
    coilset.firstwall.insert(wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    polygons = tuple(
        np.asarray(item.poly.exterior.coords)[:-1, :2] for item in material
    )
    area = np.asarray([item.poly.area for item in material], dtype=np.float64)
    triangulation = Delaunay(centres)
    boundary = LineString(wall)
    boundary_cells = np.asarray(
        [
            position
            for position, item in enumerate(material)
            if item.poly.intersects(boundary)
        ]
    )
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centres, triangulation.vertex_neighbor_vertices, boundary_cells
    )
    mesh = StencilMesh(centres, stencil, area)
    geometry = MomentGeometry.from_cells(
        mesh, polygons, sampling_vertices=hex_sampling(centres, coilset)
    )
    partition = vertex_partition(geometry)
    return {
        "kind": kind,
        "requested_cells": case["requested_cells"],
        "wall": case["wall"],
        "coefficient_digest": exact.coefficient_digest,
        "cell_count": int(geometry.cell_sample_nodes.shape[0]),
        "sample_node_count": int(len(geometry.sample_node_coordinates)),
        "atomic_node_count": int(len(geometry.atomic_mesh.node_coordinates)),
        "node_number": int(
            geometry.cell_sample_nodes.shape[0]
            + len(wall)
            + len(geometry.sample_node_coordinates)
        ),
        "wall_point_count": int(len(wall)),
        "partition_digest": partition_digest(partition),
        "vertex_partition": partition,
        "sample_vertex_count": sorted(
            int(count) for count in geometry.sample_vertex_count
        ),
    }


def main(argv):
    kinds = argv[1:] or list(CASES)
    print(f"HEADER module={nova.__file__} cwd={os.getcwd()}")
    print(f"HEADER revision={os.environ.get('NOVA_REVISION', 'unrecorded')}")
    record = {}
    status = 0
    for kind in kinds:
        try:
            entry = build(kind)
            record[kind] = entry
            print(
                f"BUILT {kind} cells={entry['cell_count']} "
                f"node_number={entry['node_number']} "
                f"sample={entry['sample_node_count']} "
                f"atomic={entry['atomic_node_count']} "
                f"partition={entry['partition_digest']} "
                f"coefficients={entry['coefficient_digest']}",
                flush=True,
            )
        except Exception as error:  # the refusal is a recorded result
            stage = traceback.extract_tb(error.__traceback__)[-1]
            record[kind] = {
                "kind": kind,
                "refused": True,
                "error": f"{type(error).__name__}: {error}",
                "stage": (
                    f"{os.path.basename(stage.filename)}:"
                    f"{stage.lineno}:{stage.name}"
                ),
            }
            print(
                f"REFUSED {kind} {record[kind]['error']} at {record[kind]['stage']}",
                flush=True,
            )
        with open(
            os.environ.get("CARRIER_RECEIPT", "carrier_receipt.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(record, stream, indent=1)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
