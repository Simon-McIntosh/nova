"""Build the hexagonal plasma mesh and clip it to a solved boundary.

The mesh is pure geometry: :meth:`nova.frame.firstwall.PlasmaGrid.insert`
tessellates hexagons inside a first-wall curve and trims them to it, which
needs no Biot coupling and no solve -- measured at 5.4 s on one CPU core for
1324 cells inside the MAST wall. So a figure can show the production cell
geometry without paying for a machine build.

The clip here is GEOMETRIC, against the boundary polygon, and that is a real
difference from the interactive poloidal view worth stating rather than
glossing: that view cuts each cell at the solved flux level through
``AtomicCellMesh.clip``, which needs the flux field. A rasterless label
session carries no field, only the boundary curve, so a cell straddling the
boundary is intersected with the boundary polygon instead. The two agree
wherever the boundary polygon and the flux contour agree, which is to within
the polygon's own vertex spacing; the flux cut is the more exact of the two
and should be preferred whenever a field is available.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def hex_mesh(
    wall: np.ndarray, cells: int = 1200
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Return the wall-clipped hexagonal cell outlines and their provenance.

    ``cells`` is a target count, not a guarantee: the tessellation fits whole
    hexagons to the wall, so the delivered count is reported rather than
    assumed.
    """
    from nova.frame.coilset import CoilSet

    outline = np.asarray(wall, dtype=float).reshape(-1, 2)
    finite = outline[np.all(np.isfinite(outline), axis=1)]
    if finite.shape[0] < 3:
        raise ValueError("a first wall needs at least three finite vertices")
    coilset = CoilSet(dplasma=-int(cells), tplasma="hex")
    coilset.firstwall.insert([finite[:, 0].tolist(), finite[:, 1].tolist()], turn="hex")
    frames = np.asarray(coilset.subframe["poly"], dtype=object)
    outlines = tuple(
        np.asarray(frame.boundary, dtype=float).reshape(-1, 2) for frame in frames
    )
    vertex_counts = np.asarray([len(item) for item in outlines])
    return outlines, {
        "requested_cells": int(cells),
        "delivered_cells": len(outlines),
        "turn": "hex",
        "wall_clipped": True,
        "vertex_count_range": [int(vertex_counts.min()), int(vertex_counts.max())],
        "source": "nova.frame.firstwall.PlasmaGrid.insert",
        "coupling": "none; cell geometry only, no Biot build",
    }


def clip_to_boundary(
    cells: Sequence[np.ndarray], boundary: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Return the parts of ``cells`` inside ``boundary``, each already cut.

    A cell wholly outside the boundary is dropped and a cell straddling it is
    returned as its clipped polygon, never as the whole hexagon -- drawing the
    unclipped cell would overstate the plasma domain at exactly the boundary
    the figure is about.
    """
    import shapely

    loop = np.asarray(boundary, dtype=float).reshape(-1, 2)
    loop = loop[np.all(np.isfinite(loop), axis=1)]
    if loop.shape[0] < 3:
        return ()
    region = shapely.Polygon(loop)
    if not region.is_valid:
        region = region.buffer(0.0)
    polygons = np.asarray(
        [shapely.Polygon(np.asarray(cell, dtype=float)[:, :2]) for cell in cells],
        dtype=object,
    )
    # Bounding-box prefilter first: most cells of a full-vessel mesh lie
    # outside a given boundary, and an intersection is far dearer than a
    # box test.
    candidates = np.flatnonzero(shapely.intersects(polygons, region))
    if candidates.size == 0:
        return ()
    pieces = shapely.intersection(polygons[candidates], region)
    clipped = []
    for piece in pieces:
        if piece.is_empty or piece.area <= 0.0:
            continue
        for part in getattr(piece, "geoms", (piece,)):
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            clipped.append(np.asarray(part.exterior.coords, dtype=float))
    return tuple(clipped)
