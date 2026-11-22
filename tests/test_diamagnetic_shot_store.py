"""Read a diamagnetic loop through the magnetics reader into the shot store.

The stored row carries what a consumer needs before a loop flux may stand in for
the toroidal volume term: where the numbers came from, their units, the COCOS
convention of the entry they were read at, and the calibration the reader
applied.  A row whose excursion does not follow the toroidal field is refused,
because a b0_like measurement whose sign is left unverified inverts the beta_p
the constraint exists to measure.
"""

import imas
import numpy as np
import pytest

from nova.imas.io_magnetics import (
    DiamagneticConventionError,
    Magnetics,
    b0_like_sense,
    diamagnetic_flux_dataset,
)
from nova.io.signalstore import SignalStore

DD3 = "3.40.0"
DD4 = "4.1.0"

TIME = np.array([0.1, 0.2, 0.3, 0.4])
FLUX = -np.array([0.4e-3, 1.1e-3, 1.6e-3, 0.7e-3])
ERROR = np.full(TIME.size, 0.05e-3)
FIELD = -2.5
SOURCE = "test:diamagnetic/11430"


def build_magnetics(version, flux=FLUX, homogeneous=True):
    """Return a synthetic magnetics IDS carrying one diamagnetic loop."""

    ids = imas.IDSFactory(version).new("magnetics")
    ids.ids_properties.homogeneous_time = 1 if homogeneous else 0
    if homogeneous:
        ids.time = TIME.tolist()
    ids.diamagnetic_flux.resize(1)
    structure = ids.diamagnetic_flux[0]
    structure.data = np.asarray(flux, dtype=float).tolist()
    structure.method_name = "diamagnetic loop voltage integral"
    if not homogeneous:
        structure.time = TIME.tolist()
    if hasattr(structure, "data_error_upper"):
        structure.data_error_upper = ERROR.tolist()
    return ids


def read(version, **kwargs):
    """Return the diamagnetic rows a synthetic entry publishes."""

    reader = Magnetics(ids=build_magnetics(version, **kwargs))
    return reader, reader.diamagnetic_fluxes(source=SOURCE, toroidal_field=FIELD)


@pytest.mark.parametrize(("version", "cocos"), [(DD3, 11), (DD4, 17)])
def test_reader_returns_the_measured_flux_on_the_entry_clock(version, cocos):
    reader, rows = read(version)

    assert len(rows) == 1
    row = rows[0]
    assert row.channel == "diamagnetic_flux[0]"
    assert row.source == f"{SOURCE}#magnetics.diamagnetic_flux(0).data"
    assert row.cocos == cocos
    np.testing.assert_array_equal(row.data, FLUX)
    np.testing.assert_array_equal(row.time, TIME)
    assert reader.ids._dd_version.startswith(version.rsplit(".", 1)[0][:3])


def test_time_is_taken_per_signal_when_the_entry_is_not_homogeneous():
    _, rows = read(DD4, homogeneous=False)

    np.testing.assert_array_equal(rows[0].time, TIME)


def test_store_row_carries_source_units_cocos_and_calibration(tmp_path):
    _, rows = read(DD4)
    dataset = diamagnetic_flux_dataset(rows, toroidal_field=FIELD)

    store = SignalStore(
        filename="diamagnetic",
        dirname=str(tmp_path),
        parents=2,
        uri=SOURCE,
        dd_version=DD4,
    )
    store.write(dataset)
    assert store.cached()

    stored = store.read()
    assert stored.attrs["standard_name"] == "diamagnetic_flux"
    assert stored.attrs["units"] == "Wb"
    assert stored.attrs["cocos"] == 17
    assert stored.attrs["calibration"] == [1.0]
    assert stored.attrs["b0_like"] is True
    assert stored.attrs["source"] == [rows[0].source]
    assert stored.attrs["target_path"].startswith("equilibrium/time_slice/")

    variable = stored["diamagnetic_flux"]
    assert variable.attrs["units"] == "Wb"
    np.testing.assert_allclose(variable.values[:, 0], FLUX)
    np.testing.assert_allclose(
        stored["diamagnetic_flux_error_upper"].values[:, 0], ERROR
    )


def test_reversed_field_without_a_flipped_flux_is_refused():
    _, rows = read(DD4)
    assert b0_like_sense(rows[0].data, FIELD) is True
    assert b0_like_sense(rows[0].data, -FIELD) is False

    with pytest.raises(DiamagneticConventionError, match="b0_like sense is reversed"):
        diamagnetic_flux_dataset(rows, toroidal_field=-FIELD)


def test_a_row_with_no_excursion_states_no_sense():
    assert b0_like_sense(np.zeros(4), FIELD) is None
    assert b0_like_sense(FLUX, 0.0) is None
    assert b0_like_sense(FLUX, np.array([-2.5, 0.0, 2.5])) is None
    assert b0_like_sense(np.full(4, np.nan), FIELD) is None


def test_an_auxiliary_channel_is_excluded_unless_it_is_selected():
    ids = imas.IDSFactory(DD4).new("magnetics")
    ids.ids_properties.homogeneous_time = 1
    ids.time = TIME.tolist()
    ids.diamagnetic_flux.resize(2)
    ids.diamagnetic_flux[0].data = FLUX.tolist()
    ids.diamagnetic_flux[1].data = (-np.asarray(FLUX)).tolist()

    reader = Magnetics(ids=ids)
    rows = reader.diamagnetic_fluxes(source=SOURCE, toroidal_field=FIELD)
    assert [row.b0_like for row in rows] == [True, False]

    with pytest.raises(DiamagneticConventionError):
        diamagnetic_flux_dataset(rows, toroidal_field=FIELD)
    selected = diamagnetic_flux_dataset(
        rows, toroidal_field=FIELD, channels=("diamagnetic_flux[0]",)
    )
    assert selected["diamagnetic_flux"].shape == (TIME.size, 1)
    with pytest.raises(ValueError, match="not found"):
        diamagnetic_flux_dataset(rows, toroidal_field=FIELD, channels=("nope",))


def test_the_loop_type_label_names_a_diamagnetic_channel():
    ids = imas.IDSFactory(DD3).new("magnetics")
    ids.ids_properties.homogeneous_time = 1
    ids.time = TIME.tolist()
    ids.flux_loop.resize(1)
    loop = ids.flux_loop[0]
    loop.name = "55.AF Diamagnetic loop"
    loop.identifier = "55.AF.00-01"
    loop.type.index = 3
    loop.flux.data = FLUX.tolist()
    loop.area = 1.0
    loop.gm9 = 0.0

    reader = Magnetics(ids=ids)
    rows = reader.diamagnetic_fluxes(source=SOURCE, toroidal_field=FIELD)
    assert len(rows) == 1
    assert rows[0].channel == "55.AF.00-01"
    assert rows[0].method == "diamagnetic_internal"
    np.testing.assert_array_equal(rows[0].data, FLUX)
