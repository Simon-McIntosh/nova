import numpy as np

from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import (
    _horizontal_reflection,
    _section_centroid,
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
)
from nova.biot.tiledassembly import TilePlan, tile_evaluator


def _reference_rows(target, vertices, expansion_centre):
    flux = polygon_analytic_flux_moments(
        target[:, 0], target[:, 1], vertices, expansion_point=expansion_centre
    )
    radial, vertical = polygon_analytic_field_moments(
        target[:, 0], target[:, 1], vertices, expansion_point=expansion_centre
    )
    return np.asarray((*flux, *radial, *vertical))


def test_traced_exact_section_moments_match_numpy_reference():
    vertices = np.asarray(
        [
            [4.91, -0.10],
            [5.08, -0.14],
            [5.17, 0.02],
            [5.04, 0.15],
            [4.88, 0.08],
        ]
    )
    target = np.asarray([[5.62, 0.31], [4.27, -0.46], [6.14, -0.22]])
    expansion_centre = np.asarray([5.01, 0.025])
    edge, weight, norm = pad_batch([vertices])
    section_centre = _section_centroid(vertices)[:, None]
    reflection_axis = np.asarray([np.nan])
    reflection_partner = np.arange(len(vertices), dtype=np.int32)[:, None]
    reflection = _horizontal_reflection(vertices)
    assert reflection is None

    plan = TilePlan(
        target_tile=len(target),
        source_tile=1,
        block=len(target),
        n_panels=16,
        n_nodes=48,
    )
    evaluate = tile_evaluator(
        plan,
        batched=True,
        kernel="moments",
        edge_count=len(vertices),
    )
    actual = np.asarray(
        evaluate(
            target[:, 0],
            target[:, 1],
            edge,
            weight,
            norm,
            section_centre,
            expansion_centre[:, None],
            reflection_axis,
            reflection_partner,
        )
    )[:, :, 0]
    expected = _reference_rows(target, vertices, expansion_centre)

    np.testing.assert_allclose(actual, expected, rtol=3.0e-10, atol=2.0e-18)
    assert evaluate.compile_count == 1
