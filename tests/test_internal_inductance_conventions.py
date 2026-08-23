"""Direct pins for the supported internal-inductance conventions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nova.biot.plasma import Plasma
from nova.equilibrium.internal_inductance import (
    Li2Geometry,
    Li3Geometry,
    convert_li_2_to_li_3,
    convert_li_3_to_li_2,
    li_2_from_field_energy,
    li_2_normaliser,
    li_3_from_field_energy,
    li_3_normaliser,
)

ROOT = Path(__file__).parents[1]
REFEREE_BANK = ROOT / "scripts/referee_inductance_instrument/comparison.json"
CONVERSION_BANK = (
    ROOT / "docs/figures/equilibrium-metric-parity/internal-inductance-conventions.json"
)


@pytest.fixture(scope="module")
def referee_rows():
    """Return the banked native-grid field-energy comparison by shot."""

    rows = json.loads(REFEREE_BANK.read_text())["per_shot"]
    return {row["shot"]: row for row in rows}


@pytest.fixture(scope="module")
def conversion_rows():
    """Return the convention-conversion receipt by shot."""

    rows = json.loads(CONVERSION_BANK.read_text())["rows"]
    return {row["shot"]: row for row in rows}


def _li_2_geometry(row):
    measurement = row["referee_native65_formula_on_stored_lcfs"]
    return Li2Geometry(
        toroidal_volume=measurement["volume_node_quadrature_m3"],
        boundary_circulation=measurement["boundary_circulation_t_m"],
        boundary_perimeter=measurement["boundary_perimeter_m"],
    )


@pytest.mark.parametrize("shot", [21978, 21983, 21985, 21986, 21989, 22086])
def test_li_2_reproduces_the_banked_referee_rows(shot, referee_rows):
    """The boundary-average formula reproduces every banked native-grid row."""

    row = referee_rows[shot]
    measurement = row["referee_native65_formula_on_stored_lcfs"]
    geometry = _li_2_geometry(row)
    actual = li_2_from_field_energy(measurement["field_energy_t2_m3"], geometry)
    np.testing.assert_allclose(
        li_2_normaliser(geometry),
        measurement["field_energy_t2_m3"] / measurement["internal_inductance"],
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        actual,
        measurement["internal_inductance"],
        rtol=2.0e-15,
    )


@pytest.mark.parametrize("shot", [21978, 21983, 21985, 21986, 21989, 22086])
def test_li_3_matches_the_converted_published_efm_value(
    shot, referee_rows, conversion_rows
):
    """Direct DD li_3 preserves the published li_2 agreement after conversion."""

    referee = referee_rows[shot]
    banked = conversion_rows[shot]
    measurement = referee["referee_native65_formula_on_stored_lcfs"]
    li_2_geometry = _li_2_geometry(referee)
    li_3_geometry = Li3Geometry(
        plasma_current=banked["plasma_current_a"],
        geometric_major_radius=banked["geometric_major_radius_m"],
    )
    direct = li_3_from_field_energy(measurement["field_energy_t2_m3"], li_3_geometry)
    converted = convert_li_2_to_li_3(
        banked["efm_published_li_2"], li_2_geometry, li_3_geometry
    )
    np.testing.assert_allclose(
        li_3_normaliser(li_3_geometry),
        banked["li_3_normaliser_t2_m3"],
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(direct, banked["direct_li_3"], rtol=2.0e-15)
    np.testing.assert_allclose(
        converted,
        banked["published_li_2_converted_to_li_3"],
        rtol=2.0e-15,
    )
    assert abs(direct / converted - 1.0) <= 0.009748


@pytest.mark.parametrize("shot", [21978, 21983, 21985, 21986, 21989, 22086])
def test_li_2_li_3_conversion_round_trip(shot, referee_rows, conversion_rows):
    """Both conversion directions preserve the represented field energy."""

    referee = referee_rows[shot]
    banked = conversion_rows[shot]
    li_2_geometry = _li_2_geometry(referee)
    li_3_geometry = Li3Geometry(
        plasma_current=banked["plasma_current_a"],
        geometric_major_radius=banked["geometric_major_radius_m"],
    )
    li_2 = referee["referee_native65_formula_on_stored_lcfs"]["internal_inductance"]
    li_3 = convert_li_2_to_li_3(li_2, li_2_geometry, li_3_geometry)
    np.testing.assert_allclose(
        convert_li_3_to_li_2(li_3, li_3_geometry, li_2_geometry),
        li_2,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        convert_li_2_to_li_3(
            convert_li_3_to_li_2(li_3, li_3_geometry, li_2_geometry),
            li_2_geometry,
            li_3_geometry,
        ),
        li_3,
        rtol=2.0e-15,
    )


def test_geometry_records_declare_every_normalising_support():
    """The public geometry records state the volume, surface and radius rules."""

    assert Li2Geometry.field_energy_support == "toroidal volume enclosed by the LCFS"
    assert Li2Geometry.volume_support == "toroidal volume enclosed by the LCFS"
    assert Li2Geometry.boundary_support == "the same LCFS contour"
    assert Li3Geometry.field_energy_support == "toroidal volume enclosed by the LCFS"
    assert Li3Geometry.current_support == "toroidal current enclosed by the same LCFS"
    assert Li3Geometry.radius_definition == "half-sum of the LCFS radial extrema"


def test_legacy_plasma_property_delegates_to_the_dd_li_3_metric():
    """Pulse-design's producer uses the LCFS geometric-radius normaliser."""

    volumes = np.asarray([1.2, 0.8, 1.5])
    fields = np.asarray([0.2, 0.35, 0.5])
    current = 8.0e5
    radius = 0.83
    state = SimpleNamespace(
        aloc={
            ("ionize", "volume"): volumes,
            ("plasma", "ionize"): np.ones(3, dtype=bool),
        },
        grid=SimpleNamespace(bp=fields),
        i_plasma=current,
        lcfs=SimpleNamespace(geometric_radius=radius),
    )
    expected = li_3_from_field_energy(
        np.sum(fields**2 * volumes),
        Li3Geometry(plasma_current=current, geometric_major_radius=radius),
    )
    np.testing.assert_allclose(Plasma.li_3.fget(state), expected, rtol=2.0e-15)
