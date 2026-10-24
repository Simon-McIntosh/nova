"""Device-independent identity of the hex carrier and of the reference solve.

A carrier's lattice vertices were merged by quantising their coordinates, so
two reconstructions of one shared corner merged only while they happened to
fall inside one bin. They are now identified through the mesh's own neighbour
rings, and the analytic reference they are built from solves through one
ordered routine, so a carrier built on one device is the same carrier as one
built on another.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import LineString

from benchmarks.solovev_certificate import DIVERTED_WALL_CLEARANCE_FRACTION
from nova.biot.plasmagrid import PlasmaGrid
from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.frame.coilset import CoilSet
from scripts.analytic_oracle_fixtures import measure as fixture

CERTIFICATE_CELLS = -300
SEPARATRIX_SAMPLES = 1441
WALL_PERTURBATION_M = 4.0e-12  # applied to each coordinate, 5.7e-12 m displacement
RECEIPT_ENVIRONMENT = "NOVA_CARRIER_VERTEX_RECEIPT"


def _hex_sampling(centres: np.ndarray, coilset: CoilSet) -> np.ndarray:
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


def diverted_carrier(
    wall_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, MomentGeometry]:
    """Build the diverted 300-cell certificate carrier's fixed geometry."""
    exact = cerfon_freidberg_single_null()
    wall = fixture.offset_wall(
        exact.separatrix(SEPARATRIX_SAMPLES),
        clearance=DIVERTED_WALL_CLEARANCE_FRACTION * exact.minor_radius,
    )
    wall = wall + np.asarray(wall_shift, dtype=np.float64)
    coilset = CoilSet(dplasma=CERTIFICATE_CELLS, tplasma="hex")
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
        mesh, polygons, sampling_vertices=_hex_sampling(centres, coilset)
    )
    return wall, geometry


def vertex_partition(geometry: MomentGeometry) -> list[list[list[int]]]:
    """Return which cell slots share a vertex, in a cell-order-free form.

    Two builds may realise the same lattice with the cells listed in a
    different order, so a slot's numeric vertex id is not comparable between
    them. What must agree is which slots are the same vertex, and each slot is
    named by the micron cell its own centre sits in, which the ordering does
    not move.
    """
    members: dict[int, list[list[int]]] = {}
    for cell, vertices in enumerate(geometry.sampling_vertices):
        centre = np.rint(np.mean(np.asarray(vertices, dtype=np.float64), axis=0) / 1e-6)
        for slot in range(len(vertices)):
            shared = members.setdefault(int(geometry.cell_sample_nodes[cell, slot]), [])
            shared.append([int(centre[0]), int(centre[1]), slot])
    return sorted(sorted(group) for group in members.values())


def carrier_receipt(wall_shift: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Return the identity a carrier build publishes for comparison."""
    wall, geometry = diverted_carrier(wall_shift)
    exact = cerfon_freidberg_single_null()
    return {
        "coefficient_digest": exact.coefficient_digest,
        "cell_count": int(geometry.cell_sample_nodes.shape[0]),
        "sample_node_count": int(len(geometry.sample_node_coordinates)),
        "atomic_node_count": int(len(geometry.atomic_mesh.node_coordinates)),
        "node_number": int(
            geometry.cell_sample_nodes.shape[0]
            + len(wall)
            + len(geometry.sample_node_coordinates)
        ),
        "vertex_partition": vertex_partition(geometry),
        "sample_vertex_count": sorted(
            int(count) for count in geometry.sample_vertex_count
        ),
    }


def test_reference_carrier_vertices_are_identified_by_lattice_topology():
    wall, geometry = diverted_carrier()
    exact = cerfon_freidberg_single_null()

    assert len(wall) == fixture.WALL_POINT_COUNT
    assert geometry.cell_sample_nodes.shape[0] == 340
    assert len(geometry.sample_node_coordinates) == len(
        np.unique(geometry.cell_sample_nodes)
    )
    assert exact.coefficient_digest


def test_perturbed_wall_keeps_the_carrier_vertex_identity():
    reference_wall, reference = diverted_carrier()
    perturbed_wall, perturbed = diverted_carrier(
        (WALL_PERTURBATION_M, WALL_PERTURBATION_M)
    )

    assert np.max(np.abs(perturbed_wall - reference_wall)) > WALL_PERTURBATION_M
    assert reference.cell_sample_nodes.shape == perturbed.cell_sample_nodes.shape
    assert len(reference.sample_node_coordinates) == len(
        perturbed.sample_node_coordinates
    )
    assert vertex_partition(reference) == vertex_partition(perturbed)


def test_carrier_identity_matches_a_receipt_from_another_device():
    recorded = os.environ.get(RECEIPT_ENVIRONMENT)
    if not recorded:
        pytest.skip("no receipt from another device is configured")
    with open(recorded, encoding="utf-8") as stream:
        other = json.load(stream)
    mine = carrier_receipt()

    assert mine["node_number"] == other["node_number"]
    assert mine["sample_node_count"] == other["sample_node_count"]
    assert mine["vertex_partition"] == other["vertex_partition"]
    assert mine["coefficient_digest"] == other["coefficient_digest"]


def _solve_receipt(threads: int) -> dict:
    """Solve the reference in a fresh process at a declared thread count."""
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = str(threads)
    program = (
        "import json, hashlib, numpy as np;"
        "from nova.equilibrium.analytic_single_null import"
        " cerfon_freidberg_single_null as f;"
        "e = f();"
        "d = lambda v: hashlib.sha256(np.ascontiguousarray(v, dtype='<f8')"
        ".tobytes()).hexdigest();"
        "print(json.dumps({'coefficients': list(map(float, e.coefficients)),"
        "'axis': list(map(float, e.magnetic_axis)),"
        "'separatrix': d(e.separatrix()),"
        "'digest': e.coefficient_digest}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
        cwd=os.getcwd(),
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_reference_solve_is_bit_identical_under_different_thread_counts():
    single = _solve_receipt(1)
    many = _solve_receipt(4)

    assert single["coefficients"] == many["coefficients"]
    assert single["axis"] == many["axis"]
    assert single["separatrix"] == many["separatrix"]
    assert single["digest"] == many["digest"]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "carrier-vertex-receipt.json"
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(carrier_receipt(), stream)
    print(f"receipt written to {path}")
