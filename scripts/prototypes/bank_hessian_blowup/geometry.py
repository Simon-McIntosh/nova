"""Cell geometry and point distances shared by the blow-up measurement scripts.

The polygons, the wall and the cell centres are the bank's own geometry archive.
Both distances are taken from the repository's owner rather than restated here:
the boundary distance from ``nova.equilibrium.boundary_comparison`` and the
vertex distance by the same norm that module measures against.
"""

import numpy as np

GEOMETRY = "/home/ITER/mcintos/.cache/nova-iter-bank/dina-rerun/bank/geometry.npz"


def read(path=GEOMETRY):
    """Return the bank's polygons, wall and cell centres."""
    with np.load(path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"])
        counts = np.asarray(archive["vertex_count"])
        return dict(
            polygons=tuple(
                vertices[index, : int(count)] for index, count in enumerate(counts)
            ),
            wall=np.asarray(archive["wall"]),
            centre=np.asarray(archive["centre"]),
        )


def cells(path=GEOMETRY):
    """Return the bank's source-cell polygons."""
    return read(path)["polygons"]


def segment_distance(point, polygon):
    """Distance from one point to a closed polygon's boundary, by its owner."""
    from nova.equilibrium.boundary_comparison import _point_to_segment_distances

    ring = np.asarray(polygon, dtype=np.float64)
    ring = np.vstack((ring, ring[0]))
    return float(
        _point_to_segment_distances(np.asarray(point, dtype=np.float64)[None, :], ring)[
            0
        ]
    )


def vertex_distance(point, polygon):
    """Distance from one point to a polygon's nearest vertex."""
    gaps = np.linalg.norm(np.asarray(polygon) - np.asarray(point), axis=-1)
    return float(np.min(gaps))
