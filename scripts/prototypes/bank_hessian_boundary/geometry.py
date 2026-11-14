"""Distance from a target or target batch to every authored source edge."""

import numpy as np


def clearance(points, polygons):
    points = np.asarray(points)
    start = np.concatenate(polygons)
    delta = np.concatenate([np.roll(p, -1, axis=0) - p for p in polygons])
    offset = points[..., None, :] - start
    fraction = np.clip(
        np.sum(offset * delta, axis=-1) / np.sum(delta**2, axis=-1), 0, 1
    )
    return np.linalg.norm(offset - fraction[..., None] * delta, axis=-1).min(axis=-1)
