"""The contour carrier is sized from the geometry it must hold.

A wide wall whose segments generate more vertices than the tree centres cannot
overflow a fixed formula: the topology geometry builds a mesh whose valid vertex
count is below its capacity. A capacity declared below what the geometry holds
refuses with a named error stating both counts, rather than returning a mesh
with no valid vertex.

The records are the imas-efit BOUNDARY_LEVEL_TRACE 2 traces the efit-field-swap
cross-check read, read through the trace reader so the test exercises the same
geometry the cross-check did. The lattice mirrors the cross-check's own square
lattice, reproduced here so the test does not depend on that script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium.contour_tree_mesh import (  # noqa: E402
    ContourMeshOverflow,
    build_contour_mesh,
)
from nova.equilibrium.wall_mask import vessel_unit  # noqa: E402
from scripts.prototypes.efit_field_swap.trace_record import (  # noqa: E402
    read_trace_record,
)

TRACE_LIBRARY = Path(
    "/home/ITER/mcintos/Code/imas-efit/tests/unit/fortran/data/boundary_level"
)


def _trace_paths() -> list[Path]:
    if not TRACE_LIBRARY.is_dir():
        return []
    return sorted(TRACE_LIBRARY.glob("*.txt"))


def _polygon(ring: np.ndarray) -> Polygon:
    polygon = Polygon(ring)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _domain_rings(record):
    """Return the one indexed loop containing the stored axis, plus obstacles."""
    rings = record.wall_rings
    if len(rings) == 1:
        return rings[0], ()
    axis = Point(record.axis_r, record.axis_z)
    polygons = [_polygon(ring) for ring in rings]
    owners = [index for index, polygon in enumerate(polygons) if polygon.contains(axis)]
    if len(owners) != 1:
        raise ValueError("expected one axis-containing wall loop")
    owner = owners[0]
    outer = polygons[owner]
    obstacles = tuple(
        ring
        for index, ring in enumerate(rings)
        if index != owner and outer.covers(polygons[index]) and polygons[index].area > 0
    )
    return rings[owner], obstacles


def _lattice(wall_ring, resolution: int, obstacles=()):
    """A square lattice of cells with their sampling stencil, as the cross-check."""
    polygon = _polygon(wall_ring)
    holes = [_polygon(ring) for ring in obstacles]
    rmin, zmin = wall_ring.min(axis=0)
    rmax, zmax = wall_ring.max(axis=0)
    width = (rmax - rmin) / resolution
    height = (zmax - zmin) / resolution
    cells, offsets = [], []
    for i in range(resolution):
        for j in range(resolution):
            r0 = rmin + i * width
            z0 = zmin + j * height
            cell = box(r0, z0, r0 + width, z0 + height)
            if not polygon.contains(cell.centroid) or any(
                hole.covers(cell.centroid) for hole in holes
            ):
                continue
            r1, z1 = r0 + width, z0 + height
            cells.append(
                np.asarray(((r0, z0), (r1, z0), (r1, z1), (r0, z1)), dtype=np.float64)
            )
            step_r, step_z = 0.5 * width, 0.5 * height
            mid_r, mid_z = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
            offsets.append(
                np.asarray(
                    (
                        (mid_r - step_r, mid_z - step_z),
                        (mid_r + step_r, mid_z - step_z),
                        (mid_r + step_r, mid_z + step_z),
                        (mid_r - step_r, mid_z + step_z),
                        (mid_r, mid_z - step_z),
                        (mid_r, mid_z + step_z),
                        (mid_r - step_r, mid_z),
                        (mid_r + step_r, mid_z),
                    ),
                    dtype=np.float64,
                )
            )
    if not cells:
        raise ValueError("lattice found no cell inside the wall polygon")
    return cells, np.asarray(offsets)


def _geometry(record, resolution: int = 16):
    boundary, obstacles = _domain_rings(record)
    cells, samples = _lattice(boundary, resolution, obstacles)
    return topology.TopologyGeometry.from_cells(cells, samples, (boundary, *obstacles))


@pytest.mark.slow
@pytest.mark.skipif(not _trace_paths(), reason="no imas-efit boundary traces on disk")
@pytest.mark.parametrize("path", _trace_paths(), ids=lambda p: p.name)
def test_wide_wall_record_builds_below_capacity(path):
    record = read_trace_record(path)
    geometry = _geometry(record)
    valid = np.asarray(geometry.contour_mesh.vertex_valid)
    assert valid.any(), "the carrier admitted no vertex"
    assert int(valid.sum()) < geometry.contour_mesh.vertex_capacity


def test_overflow_refuses_with_named_error():
    cells = [
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0))),
        np.asarray(((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))),
        np.asarray(((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))),
    ]
    wall = [vessel_unit([-1.0, 3.0, 3.0, -1.0], [-1.0, -1.0, 3.0, 3.0])]
    with pytest.raises(ContourMeshOverflow) as refusal:
        build_contour_mesh(
            np.asarray([cell.mean(axis=0) for cell in cells]),
            np.zeros(4),
            wall,
            vertex_capacity=3,
        )
    error = refusal.value
    assert error.capacity == 3
    assert error.required > error.capacity
    # both numbers appear in the message, not only in the attributes
    assert str(error.required) in str(error) and "3" in str(error)
