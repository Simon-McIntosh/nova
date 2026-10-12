"""Polygon source-area moments against independently integrated reference values.

The ordinary rows use dense Gauss integration with order doubling. The sliver
triangle at two targets additionally carries fifty-digit elliptic-integral
checks; its nonzero moment signs must survive the reduction.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from nova.jax.config import configure_dtypes
from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import (
    _horizontal_reflection,
    _section_centroid,
    packed_analytic_moments,
    polygon_analytic_flux_moments,
)

configure_dtypes()
assert jax.config.jax_enable_x64

_ROWS = [
    {
        "polygon": [
            [2.0934211207383777, 0.47432974270317296],
            [2.075164160429423, 0.5059517255500512],
            [2.091045293807066, 0.5334586554419068],
            [2.095364239082407, 0.516131905713147],
            [2.0953642390877443, 0.5161319056901577],
            [2.1043600676336855, 0.47432974270317296],
        ],
        "target": [2.140739421325375, 0.22178834364931033],
        "expansion_point": [2.090452434565825, 0.5005617861725423],
        "reference": [
            5.592656325443431e-06,
            4.858700452276533e-10,
            -1.873657954076524e-09,
        ],
    },
    {
        "polygon": [
            [1.4179135893070476, -0.18973189708126925],
            [1.3813996686891379, -0.18973189708126925],
            [1.363142708380183, -0.15810991423439102],
            [1.3813996686891379, -0.1264879313875128],
            [1.4179135893070476, -0.1264879313875128],
            [1.4361705496160024, -0.15810991423439102],
        ],
        "target": [1.6187401527055512, -0.22135387992814748],
        "expansion_point": [1.3996566289980927, -0.15810991423439102],
        "reference": [
            3.750459731587339e-06,
            2.7094029754116442e-09,
            -6.221986550574808e-10,
        ],
    },
    {
        "polygon": [
            [1.8560806367219644, 0.44270775985629474],
            [1.8195667161040547, 0.44270775985629474],
            [1.8013097557950999, 0.47432974270317296],
            [1.8195667161040547, 0.5059517255500512],
            [1.8560806367219644, 0.5059517255500512],
            [1.8743375970309193, 0.47432974270317296],
        ],
        "target": [1.7282819145592805, 0.2845978456219038],
        "expansion_point": [1.8378236764130096, 0.47432974270317296],
        "reference": [
            4.896580539571754e-06,
            -8.571619046988576e-10,
            -2.412059801250414e-09,
        ],
    },
    {
        "polygon": [
            [0.9980035022010856, -0.22135387992814748],
            [0.9979841530999642, -0.22138739355437037],
            [0.9979643991419811, -0.2212861514430471],
        ],
        "target": [1.6187401527055512, -0.9802814682532244],
        "expansion_point": [0.997984018147677, -0.22134247497518833],
        "reference": [
            8.887851363362002e-07,
            2.0390010005368566e-16,
            -5.706948830807982e-16,
        ],
    },
    {
        "polygon": [
            [0.9980035022010856, -0.22135387992814748],
            [0.9979841530999642, -0.22138739355437037],
            [0.9979643991419811, -0.2212861514430471],
        ],
        "target": [1.235343986217499, -0.3794637941625385],
        "expansion_point": [0.997984018147677, -0.22134247497518833],
        "reference": [
            2.060732797067555e-06,
            6.32390546667056e-16,
            -1.7364717201586264e-15,
        ],
    },
]

_NEAR_EDGE_ROW = {
    "polygon": [
        [4.91, -0.1],
        [5.08, -0.14],
        [5.17, 0.02],
        [5.04, 0.15],
        [4.88, 0.08],
    ],
    "target": [5.1290090247138025, 0.06117221140408553],
    "expansion_point": [5.016810681068107, 0.002097209720972118],
    "reference": [
        2.394083456037712e-05,
        2.024806550388612e-07,
        9.092304981814128e-08,
    ],
}


def _packed_inputs(rows):
    polygons = [np.asarray(row["polygon"], dtype=np.float64) for row in rows]
    edge, weight, norm = pad_batch(polygons)
    centre = np.asarray([_section_centroid(p) for p in polygons]).T
    expansion = np.asarray([row["expansion_point"] for row in rows]).T
    target = np.asarray([row["target"] for row in rows])
    axis = np.full(len(rows), np.nan)
    partner = np.repeat(np.arange(len(edge))[:, None], len(rows), axis=1)
    for column, polygon in enumerate(polygons):
        reflection = _horizontal_reflection(polygon)
        if reflection is not None:
            axis[column], vertex_partner = reflection
            for index in range(len(polygon)):
                partner[index, column] = vertex_partner[(index + 1) % len(polygon)]
    return (
        target[:, 0],
        target[:, 1],
        edge,
        weight,
        norm,
        centre,
        expansion,
        axis,
        partner,
    )


@pytest.mark.parametrize("path", ["host", "packed_numpy", "packed_device"])
def test_flux_moments_match_independent_reference(path):
    if path == "host":
        actual = np.asarray(
            [
                polygon_analytic_flux_moments(
                    np.asarray(row["target"][0]),
                    np.asarray(row["target"][1]),
                    np.asarray(row["polygon"]),
                    expansion_point=row["expansion_point"],
                )
                for row in _ROWS
            ]
        )
    else:
        inputs = _packed_inputs(_ROWS)
        if path == "packed_numpy":
            actual = np.asarray(packed_analytic_moments(np, *inputs)[:3]).T
        else:
            evaluate = jax.jit(lambda *args: packed_analytic_moments(jnp, *args)[:3])
            actual = np.asarray(evaluate(*inputs)).T
    reference = np.asarray([row["reference"] for row in _ROWS])
    relative = np.max(np.abs(actual - reference), axis=0) / np.max(
        np.abs(reference), axis=0
    )
    print("MOMENT_REFERENCE", path, relative.tolist(), flush=True)
    assert np.all(relative <= 1e-12), relative
    np.testing.assert_array_equal(np.sign(actual[-2:, 1:]), np.sign(reference[-2:, 1:]))


@pytest.mark.parametrize("path", ["host", "packed_numpy", "packed_device"])
def test_near_edge_flux_moments_match_split_reference(path):
    row = _NEAR_EDGE_ROW
    if path == "host":
        actual = np.asarray(
            polygon_analytic_flux_moments(
                *row["target"],
                np.asarray(row["polygon"]),
                expansion_point=row["expansion_point"],
            )
        )
    else:
        inputs = _packed_inputs([row])
        if path == "packed_numpy":
            actual = np.asarray(packed_analytic_moments(np, *inputs)[:3]).reshape(3)
        else:
            evaluate = jax.jit(lambda *args: packed_analytic_moments(jnp, *args)[:3])
            actual = np.asarray(evaluate(*inputs)).reshape(3)
    reference = np.asarray(row["reference"])
    relative = np.abs(actual - reference) / np.abs(reference)
    print("NEAR_EDGE_REFERENCE", path, relative.tolist(), flush=True)
    assert np.all(relative <= 1e-12), relative


def test_moment_reference_point_is_a_local_displacement():
    row = _ROWS[0]
    centre = np.asarray(row["expansion_point"])
    shift = np.array([0.03, -0.07])
    args = (
        np.asarray(row["target"][0]),
        np.asarray(row["target"][1]),
        np.asarray(row["polygon"]),
    )
    base = np.asarray(polygon_analytic_flux_moments(*args, expansion_point=centre))
    moved = np.asarray(
        polygon_analytic_flux_moments(*args, expansion_point=centre + shift)
    )
    np.testing.assert_allclose(
        moved[1:], base[1:] - shift * base[0], rtol=1e-12, atol=1e-23
    )


@pytest.mark.parametrize("backend", ["numpy", "device"])
def test_one_polygon_broadcasts_across_targets(backend):
    row = _ROWS[1]
    targets = np.asarray([one["target"] for one in _ROWS[:3]])
    inputs = list(_packed_inputs([row]))
    inputs[:2] = targets[:, 0], targets[:, 1]
    expected = np.asarray(
        polygon_analytic_flux_moments(
            targets[:, 0],
            targets[:, 1],
            np.asarray(row["polygon"]),
            expansion_point=row["expansion_point"],
        )
    )
    if backend == "numpy":
        actual = np.asarray(packed_analytic_moments(np, *inputs)[:3])
    else:
        evaluate = jax.jit(lambda *args: packed_analytic_moments(jnp, *args)[:3])
        actual = np.asarray(evaluate(*inputs))
    relative = np.max(np.abs(actual - expected), axis=1) / np.max(
        np.abs(expected), axis=1
    )
    assert np.all(relative <= 1e-12), relative
