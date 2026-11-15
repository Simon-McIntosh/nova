"""Predict corner curvature from the second angular harmonic of local density.

The leading Green function of the Grad-Shafranov operator is log(distance)
divided by two pi. Integrating its Hessian over each constant-density wedge
gives the logarithmic coefficient below. The radial first-derivative term
and spatial variation of the source density do not alter this leading term.
"""

import numpy as np
from shapely.geometry import Polygon

from nova.biot.greens import second_moments, section_centroid


def coefficients(polygons, archives, records):
    """Compare the source-wedge prediction with measured increments per decade."""
    results = []
    for record in records:
        if record["full_field"] or not record["corner"]:
            continue
        node = record["node"]
        point = archives[0]["points"][node]
        coefficient = np.zeros((3, 2))
        for cell in record["source_cells"]:
            polygon = polygons[cell]
            vertex = np.argmin(np.linalg.norm(polygon - point, axis=1))
            corner = polygon[vertex]
            before = polygon[(vertex - 1) % len(polygon)] - corner
            after = polygon[(vertex + 1) % len(polygon)] - corner
            area = Polygon(polygon).area
            orientation = np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
            start, end = (after, before) if orientation > 0 else (before, after)
            a, b = np.arctan2(start[1], start[0]), np.arctan2(end[1], end[0])
            second = second_moments(polygon)
            inverse = np.linalg.inv(
                np.array([[second[0], second[2]], [second[2], second[1]]])
            )
            moment = np.stack([x["moments"][cell] for x in archives])
            density = (
                moment[:, 0]
                + (moment[:, 1:] @ inverse.T) @ (corner - section_centroid(polygon))
            ) / area
            forcing = -2 * np.pi * (4e-7 * np.pi) * corner[0] * density
            integral = np.array(
                [np.cos(2 * b) - np.cos(2 * a), np.sin(2 * b) - np.sin(2 * a)]
            )
            coefficient += forcing[:, None] * integral[None] / (4 * np.pi) * np.log(10)
        value = np.array(record["values"])
        observed = (value[-1] - value[-2]).mean(axis=0)
        results.append(
            dict(node=node, predicted=coefficient.tolist(), observed=observed.tolist())
        )
    predicted = np.array([r["predicted"] for r in results])
    observed = np.array([r["observed"] for r in results])
    relative = np.sqrt(
        np.sum((predicted - observed) ** 2, axis=0) / np.sum(predicted**2, axis=0)
    )
    return dict(
        nodes=results,
        relative=relative.tolist(),
        nonzero_per_state=np.sum(
            np.linalg.norm(predicted, axis=-1) > 1e-6, axis=0
        ).tolist(),
    )
