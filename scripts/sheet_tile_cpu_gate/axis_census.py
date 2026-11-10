"""Classify sheet fields on the polygonised sphere at cell centroids."""

import sys
import types
import subprocess
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import shapely

from nova.biot import sheet
from nova.biot.tiledassembly import cell_edges


def sphere_mesh():
    source = subprocess.check_output(
        ["git", "show", "ef784a2d7:nova/biot/magnetisation.py"], text=True
    )
    owner = types.ModuleType("sphere_mesh_owner")
    sys.modules[owner.__name__] = owner
    exec(compile(source, "<committed sphere mesh owner>", "exec"), owner.__dict__)
    region = (
        shapely.Point(0, 0)
        .buffer(0.5, 256)
        .intersection(shapely.box(0, -0.5, 0.5, 0.5))
    )
    return owner._mesh(region, 0.1)


def census():
    print("module:", sheet.__file__, flush=True)
    mesh = sphere_mesh()
    edges = cell_edges(mesh.polygons)
    sources = np.column_stack((edges.start_r, edges.start_z, edges.end_r, edges.end_z))
    targets = np.repeat(mesh.centroids, len(sources), axis=0)
    source = np.tile(sources, (len(mesh.centroids), 1))
    axis_edge = (source[:, 0] == 0) & (source[:, 2] == 0)
    axis_end = ((source[:, 0] == 0) | (source[:, 2] == 0)) & ~axis_edge
    axis_target = targets[:, 0] == 0
    print(
        "cells:",
        len(mesh.centroids),
        "edges:",
        len(sources),
        "pairs:",
        len(targets),
        flush=True,
    )
    evaluate = jax.jit(lambda *p: jnp.stack(sheet.packed_sheet_field(jnp, *p), axis=-1))
    packed = []
    for start in range(0, len(targets), 256):
        indices = np.minimum(np.arange(start, start + 256), len(targets) - 1)
        pair = np.column_stack((targets[indices], source[indices]))
        packed.append(
            np.asarray(evaluate(*map(jnp.asarray, pair.T)))[
                : min(256, len(targets) - start)
            ]
        )
    packed = np.concatenate(packed)
    scalar = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for index, (target, edge) in enumerate(zip(targets, source)):
            try:
                scalar.append(sheet.sheet_field(*target, edge[:2], edge[2:]))
            except ValueError, ZeroDivisionError, FloatingPointError:
                scalar.append((np.nan, np.nan))
            if index % 5000 == 0:
                print("scalar pairs:", index, flush=True)
    scalar = np.asarray(scalar)
    for name, values in [("scalar", scalar), ("packed", packed)]:
        bad = ~np.isfinite(values).all(axis=1)
        print(
            name,
            "nonfinite:",
            int(bad.sum()),
            "axis-edge:",
            int((bad & axis_edge).sum()),
            "axis-end:",
            int((bad & axis_end).sum()),
            "axis-target:",
            int((bad & axis_target).sum()),
            "other:",
            int((bad & ~(axis_edge | axis_end | axis_target)).sum()),
            flush=True,
        )
    print("minimum centroid radius:", mesh.centroids[:, 0].min(), flush=True)
    lines = shapely.linestrings(sources.reshape(-1, 2, 2))
    distances = shapely.distance(
        shapely.points(mesh.centroids)[:, None], lines[None, :]
    )
    print("minimum centroid-to-edge distance:", distances.min(), flush=True)
    assert distances.min() > 0
    if "--require-finite" in sys.argv:
        assert np.isfinite(scalar).all()
        assert np.isfinite(packed).all()


if __name__ == "__main__":
    census()
