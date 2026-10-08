"""Saddle rows read their own full contour and retain binary traversal nuisance."""

from dataclasses import replace

import imas
import numpy as np
import pytest

from nova.imas.io_magnetics import Magnetics
from nova.imas import mast_misfit_harmonics as mh


def two_loops():
    ids = imas.IDSFactory("4.1.1").new("magnetics")
    ids.flux_loop.resize(2)
    for index, (radius, half_height, width) in enumerate(
        [(2.0, 0.5, 0.4), (1.5, 0.8, 0.7)]
    ):
        loop = ids.flux_loop[index]
        loop.name = f"saddle_l_{index}"
        loop.type.index = 2
        loop.position.resize(5)
        points = [
            (radius, -half_height, 0),
            (radius, -half_height, width),
            (radius, half_height, width),
            (radius, half_height, 0),
            (radius, -half_height, 0),
        ]
        for point, (r, z, phi) in zip(loop.position, points, strict=True):
            point.r, point.z, point.phi = r, z, phi
    return Magnetics(ids=ids)


class PolynomialFlux:
    labels = ("radial", "sheared")

    @staticmethod
    def flux(r, z):
        return 2 * np.pi * np.column_stack([r * z, r * z**3])


def synthetic_class():
    return mh.saddle_sensor_class(
        two_loops(),
        ("saddle_l_0", "saddle_l_1"),
        [[0.8, 1.68], [0.81, 1.67], [0.79, np.nan]],
        PolynomialFlux.flux,
        np.array([0.01, 0.02]),
    )


def test_two_loop_assembly_keeps_identity_geometry_and_shot_counts():
    sensor = synthetic_class()
    bank = mh.assemble("drive", [sensor], shots=(11, 12, 13))
    assert bank.channel == ("saddle_l_0", "saddle_l_1")
    np.testing.assert_array_equal(bank.shot_counts, [3, 2])
    assert bank.traversal_options == ((-1, 1), (-1, 1))
    assert bank.traversal_signs == ()
    with pytest.raises(mh.MisfitMapError, match="unresolved"):
        mh.harmonic_design(PolynomialFlux(), bank)
    chosen = bank.realize_traversal([1, -1])
    expected = np.array([[-0.8, -0.2], [1.68, 1.0752]])
    np.testing.assert_allclose(
        mh.harmonic_design(PolynomialFlux(), chosen), expected, atol=1e-13
    )
    np.testing.assert_allclose(chosen.described, expected, atol=1e-13)
    np.testing.assert_allclose(chosen.r, [2.0, 1.5])
    assert chosen.traversal_options == bank.traversal_options
    selected = chosen.select([False, True])
    np.testing.assert_allclose(
        mh.harmonic_design(PolynomialFlux(), selected), expected[1:]
    )
    np.testing.assert_array_equal(selected.shot_counts, [2])
    with pytest.raises(mh.MisfitMapError, match="choice"):
        bank.realize_traversal([0, 1])
    np.testing.assert_allclose(
        bank.realize_traversal([1, -1]).realize_traversal([-1, 1]).described,
        -expected,
    )


def test_contour_reversal_reverses_flux_and_open_contour_is_refused():
    path = synthetic_class().contours[0]
    np.testing.assert_allclose(
        mh.contour_flux(PolynomialFlux.flux, path[::-1]),
        -mh.contour_flux(PolynomialFlux.flux, path),
    )
    with pytest.raises(mh.MisfitMapError, match="closed"):
        mh.contour_flux(PolynomialFlux.flux, path[:-1])


def test_sparse_loop_is_dropped_with_its_own_contour():
    sensor = synthetic_class()
    sensor = replace(
        sensor, coupling=np.array([[0.8, 1.68], [0.81, np.nan], [0.79, np.nan]])
    )
    bank = mh.assemble("drive", [sensor])
    assert bank.channel == ("saddle_l_0",)
    np.testing.assert_array_equal(bank.contours[0], sensor.contours[0])
    assert bank.traversal_options == ((-1, 1),)


def test_saddle_identity_must_resolve_in_ids():
    with pytest.raises(mh.MisfitMapError, match="no saddle"):
        mh.saddle_sensor_class(
            two_loops(), ["missing"], [[1], [2]], PolynomialFlux.flux, [0.1]
        )
