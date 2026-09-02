"""Paired checks for batched polygon flux-moment assembly."""

import numpy as np

from nova.biot.polygonanalytic import (
    polygon_analytic_flux_moments,
    polygon_analytic_flux_moments_batched,
)


def test_batched_flux_moments_match_independent_section_calls_at_roundoff():
    sections = (
        np.asarray(
            [
                [2.82, -0.08],
                [3.10, -0.12],
                [3.17, 0.06],
                [2.93, 0.15],
                [2.78, 0.04],
            ]
        ),
        np.asarray(
            [
                [3.41, -0.09],
                [3.55, -0.09],
                [3.63, 0.03],
                [3.56, 0.15],
                [3.40, 0.15],
                [3.32, 0.03],
            ]
        ),
        np.asarray([[4.10, -0.11], [4.31, 0.02], [4.08, 0.14]]),
    )
    expansion_points = np.asarray([[2.96, 0.01], [3.49, 0.03], [4.16, 0.02]])
    target_r = np.asarray([0.0, 2.70, 3.24, 4.5, 7.0])
    target_z = np.asarray([0.31, 0.02, -0.02, 0.6, -1.5])

    expected = np.stack(
        [
            polygon_analytic_flux_moments(
                target_r,
                target_z,
                section,
                expansion_point=centre,
            )
            for section, centre in zip(sections, expansion_points, strict=True)
        ],
        axis=-1,
    )
    actual = np.stack(
        polygon_analytic_flux_moments_batched(
            target_r,
            target_z,
            sections,
            expansion_points=expansion_points,
            workers=2,
        )
    )
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), np.abs(actual))
    roundoff_floor = 64.0 * np.finfo(np.float64).eps * scale

    assert np.max(difference) == 0.0
    assert np.all(difference <= roundoff_floor + np.finfo(np.float64).tiny)
