"""What a source-signal map admits, refuses, and produces.

The rows built here are deliberately synthetic.  A real facility map is tested
against its own store; what has to hold for any of them is that the three
conversion factors stay separable, that a row cannot be admitted with a
conversion nobody can justify, and that a channel is never both served and
declared unservable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.imas.machine_evidence import FieldEvidence
from nova.io.cocos import IP_LIKE, ONE_LIKE, PSI_LIKE
from nova.io.ingest import PROVISIONAL_NAMESPACE
from nova.io.sourcemap import (
    ACCEPTED,
    PROPOSAL,
    BlockedSignal,
    SourceMapError,
    SourceSignal,
    SourceSignalMap,
    group_signals,
    round_trip_residual,
    served_targets,
    tensorize,
)
from nova.io.standardname import NameSource, Resolution


class _CatalogStub:
    """Resolve a fixed set of names from the catalogue and the rest by grammar."""

    def __init__(self, catalogued: set[str], unit: str = "A"):
        self.catalogued = catalogued
        self.unit = unit

    def resolve(self, name: str) -> Resolution:
        if name in self.catalogued:
            return Resolution(
                name=name,
                unit=self.unit,
                kind="scalar",
                status="active",
                source=NameSource.CATALOG,
            )
        return Resolution(
            name=name,
            unit=None,
            kind=None,
            status="provisional",
            source=NameSource.GRAMMAR,
        )


def _signal(**overrides) -> SourceSignal:
    row = {
        "standard_name": "plasma_current",
        "catalog_status": ACCEPTED,
        "source_group": "amc",
        "source_channel": "plasma_current",
        "source_unit": "kA",
        "target_path": "magnetics/ip/data",
        "target_unit": "A",
        "target_index": 0,
        "transformation": IP_LIKE,
        "source_convention": 3,
        "target_convention": 17,
        "unit_factor": 1.0e3,
        "channel_factor": 1.0,
        "time_base": "current",
        "evidence": FieldEvidence.MEASURED,
        "statement": "the analysed channel is a measured net toroidal current",
    }
    row.update(overrides)
    return SourceSignal(**row)


def test_the_conversion_is_the_product_of_three_named_factors():
    """Units, channel semantics and convention stay apart so each stays findable."""

    row = _signal(
        transformation=PSI_LIKE,
        source_convention=3,
        unit_factor=2.0,
        channel_factor=5.0,
    )
    assert row.convention_factor == pytest.approx(math.tau)
    assert row.factor == pytest.approx(2.0 * 5.0 * math.tau)


def test_applying_and_inverting_returns_the_source_samples():
    """The inverse divides by exactly what the forward multiplied by."""

    row = _signal(channel_factor=1.0 / 23.0)
    samples = np.array([0.0, 1.5, -2.25, 1.0e3])
    assert np.allclose(row.invert(row.apply(samples)), samples, rtol=0, atol=1e-12)


def test_a_stored_convention_factor_cannot_disagree_with_the_algebra():
    """A serialized factor is checked against the digits, never trusted over them."""

    payload = _signal(transformation=PSI_LIKE).as_dict()
    assert payload["convention_factor"] == pytest.approx(math.tau)
    payload["convention_factor"] = 1.0
    with pytest.raises(SourceMapError, match="disagrees with the factor"):
        SourceSignal.from_dict(payload)


def test_a_row_survives_a_canonical_round_trip():
    """Every field a row carries has to come back off its own serialization."""

    row = _signal(catalog_status=PROPOSAL, target_index=None)
    assert SourceSignal.from_dict(row.as_dict()) == row


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"unit_factor": 0.0}, "erases the signal"),
        ({"channel_factor": 0.0}, "erases the signal"),
        ({"transformation": "pol_angle_like"}, "unknown transformation"),
        ({"evidence": FieldEvidence.UNRESOLVED}, "blocked channel"),
        ({"catalog_status": "published"}, "unknown catalogue state"),
        ({"standard_name": " "}, "standard name must be non-empty"),
        ({"target_index": -1}, "must be non-negative"),
    ],
)
def test_an_inadmissible_row_is_refused(overrides, match):
    """A conversion nobody can justify is not carried with a plausible default."""

    with pytest.raises(SourceMapError, match=match):
        _signal(**overrides).validate()


def test_one_channel_cannot_fill_one_target_twice():
    """Two rows for one target and one channel would make the column order arbitrary."""

    row = _signal()
    with pytest.raises(SourceMapError, match="cannot fill one target twice"):
        SourceSignalMap.create([row, row])


def test_a_channel_is_never_both_served_and_blocked():
    """The count of what a shot can supply has to be readable off the map."""

    blocked = BlockedSignal(
        source_group="amc",
        source_channel="plasma_current",
        target_path="magnetics/ip/data",
        reason="a reason",
        unmet="a condition",
    )
    with pytest.raises(SourceMapError, match="both served and blocked"):
        SourceSignalMap.create([_signal()], [blocked])


def test_a_blocked_channel_has_to_state_its_obstruction():
    """A blocked row with no unmet condition is indistinguishable from an oversight."""

    with pytest.raises(SourceMapError, match="unmet condition"):
        BlockedSignal(
            source_group="amc",
            source_channel="tf_current",
            target_path="tf/coil/current/data",
            reason="no described conductor",
            unmet="",
        ).validate()


def test_the_map_digest_does_not_depend_on_construction_order():
    """A map is content addressed, so the same rows must give the same address."""

    rows = [
        _signal(source_channel="a", target_path="magnetics/ip/data"),
        _signal(source_channel="b", target_path="pf_active/coil(x)/current/data"),
    ]
    assert (
        SourceSignalMap.create(rows).digest
        == SourceSignalMap.create(reversed(rows)).digest
    )


def test_a_map_survives_its_own_serialization():
    """The whole map, served and blocked alike, round-trips through canonical JSON."""

    source_map = SourceSignalMap.create(
        [_signal()],
        [
            BlockedSignal(
                source_group="amc",
                source_channel="tf_current",
                target_path="tf/coil/current/data",
                reason="the description carries no toroidal conductor",
                unmet="tf/coil is not sourced",
            )
        ],
    )
    assert SourceSignalMap.from_dict(source_map.as_dict()) == source_map


def _sensor_map() -> SourceSignalMap:
    return SourceSignalMap.create(
        [
            _signal(
                standard_name="poloidal_magnetic_field_of_poloidal_magnetic_field_probe",
                source_group="amb",
                source_channel=f"probe{index:02d}",
                source_unit="T",
                target_unit="T",
                target_path=f"magnetics/b_field_pol_probe(p_{index})/field/data",
                target_index=index,
                transformation=ONE_LIKE,
                unit_factor=1.0,
                time_base="field",
            )
            for index in (2, 0, 1)
        ]
        + [_signal()]
    )


def test_a_family_becomes_one_variable_ordered_by_its_targets():
    """Sensors of a kind share a name, so the tensorized form has a channel axis."""

    source_map = _sensor_map()
    groups = group_signals(source_map.signals)
    probes = next(
        group
        for group in groups
        if group.standard_name.startswith("poloidal_magnetic_field")
    )
    assert [row.target_index for row in probes.signals] == [0, 1, 2]
    dataset = tensorize(
        source_map,
        {
            "probe00": np.array([1.0, 2.0]),
            "probe01": np.array([3.0, 4.0]),
            "probe02": np.array([5.0, 6.0]),
            "plasma_current": np.array([10.0, 20.0]),
        },
        {"field": np.array([0.0, 1.0]), "current": np.array([0.0, 1.0])},
        resolver=_CatalogStub({"plasma_current"}),
    )
    probe_name = "poloidal_magnetic_field_of_poloidal_magnetic_field_probe"
    variable = f"{PROVISIONAL_NAMESPACE}/{probe_name}"
    array = dataset[variable]
    assert array.dims == (
        "field_time",
        "poloidal_magnetic_field_of_poloidal_magnetic_field_probe_channel",
    )
    assert array.shape == (2, 3)
    assert array.values[0].tolist() == [1.0, 3.0, 5.0]
    assert array.attrs["target_index"] == [0, 1, 2]
    assert array.attrs["units"] == "T"
    # a catalogued name keeps its own key; a grammar-only one is namespaced
    assert "plasma_current" in dataset
    assert dataset.attrs["provisional_names"] == [
        "poloidal_magnetic_field_of_poloidal_magnetic_field_probe"
    ]
    assert dataset.attrs["map_digest"] == source_map.digest


def test_two_clocks_stay_two_clocks():
    """Resampling one acquisition onto another is a modelling choice, not a map's."""

    dataset = tensorize(
        _sensor_map(),
        {
            "probe00": np.zeros(2),
            "probe01": np.zeros(2),
            "probe02": np.zeros(2),
            "plasma_current": np.zeros(3),
        },
        {"field": np.array([0.0, 1.0]), "current": np.array([0.0, 0.5, 1.0])},
        resolver=_CatalogStub(set()),
    )
    assert dataset.sizes["field_time"] == 2
    assert dataset.sizes["current_time"] == 3


def test_samples_that_do_not_fit_their_clock_are_refused():
    """A channel aligned to the wrong clock moves every sample in time."""

    with pytest.raises(SourceMapError, match="samples against"):
        tensorize(
            SourceSignalMap.create([_signal()]),
            {"plasma_current": np.zeros(5)},
            {"current": np.zeros(4)},
            resolver=_CatalogStub(set()),
        )


def test_a_missing_clock_is_refused():
    """A served row with no clock has nowhere to put its samples."""

    with pytest.raises(SourceMapError, match="no clock supplied"):
        tensorize(
            SourceSignalMap.create([_signal()]),
            {"plasma_current": np.zeros(4)},
            {},
            resolver=_CatalogStub(set()),
        )


def test_the_round_trip_recovers_every_channel():
    """What the residual shows is that nothing was lost or reordered on the way."""

    source_map = _sensor_map()
    samples = {
        "probe00": np.array([1.0, -2.0]),
        "probe01": np.array([3.0, 4.0]),
        "probe02": np.array([5.0, 6.0]),
        "plasma_current": np.array([-10.0, 20.0]),
    }
    dataset = tensorize(
        source_map,
        samples,
        {"field": np.array([0.0, 1.0]), "current": np.array([0.0, 1.0])},
        resolver=_CatalogStub({"plasma_current"}),
    )
    residuals = round_trip_residual(source_map, dataset, samples)
    assert set(residuals) == set(samples)
    assert max(residuals.values()) < 1e-15


def test_the_served_targets_are_readable_per_container():
    """A consumer has to be able to ask which described sensors a shot filled."""

    assert served_targets(_sensor_map()) == {
        "magnetics/b_field_pol_probe": [0, 1, 2],
        "magnetics/ip": [0],
    }
