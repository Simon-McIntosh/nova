"""The certificate store is addressed by defining inputs across devices."""

from copy import deepcopy
import json

import numpy as np
import xarray

from scripts.prototypes.support import certificate_fixtures as fixtures


def _carrier_identity():
    return {
        "schema": "carrier-schema",
        "analytic_case": "limited-machine",
        "closed_form_constants": {"major_radius_m": 6.2},
        "discretisation": {
            "requested_cells": -500,
            "wall_nodes": 257,
            "wall_content": {"dtype": "<f8", "shape": [257, 2], "sha256": "device-a"},
        },
        "builder": "host",
        "builder_revision": "kernel-digest",
    }


def test_carrier_key_ignores_computed_wall_bits_but_covers_inputs():
    identity = _carrier_identity()
    rounded = deepcopy(identity)
    rounded["discretisation"]["wall_content"]["sha256"] = "device-b"
    assert fixtures.fixture_digest(rounded) == fixtures.fixture_digest(identity)

    for section, key, value in (
        (None, "analytic_case", "diverted-machine"),
        ("discretisation", "requested_cells", -2000),
        ("closed_form_constants", "major_radius_m", 6.3),
        (None, "builder_revision", "other-kernel"),
        (None, "builder", "device"),
    ):
        changed = deepcopy(identity)
        target = changed if section is None else changed[section]
        target[key] = value
        assert fixtures.fixture_digest(changed) != fixtures.fixture_digest(identity)


def test_exterior_key_ignores_computed_state_bits_but_covers_support():
    identity = {
        "schema": "exterior-schema",
        "machine_semantic_key": fixtures.fixture_digest(_carrier_identity()),
        "density_case": "limited-machine",
        "density_coefficients": {"pressure_coefficient": 0.5},
        "analytic_type": "AnalyticField",
        "analytic_state_sha256_binary64": "device-a",
        "analytic_boundary_sha256_binary64": "device-a",
        "support_arm": "exact",
        "polygonanalytic_revision": "polygon-kernel-digest",
    }
    rounded = deepcopy(identity)
    rounded["analytic_state_sha256_binary64"] = "device-b"
    rounded["analytic_boundary_sha256_binary64"] = "device-b"
    assert fixtures.fixture_digest(rounded) == fixtures.fixture_digest(identity)

    for section, key, value in (
        (None, "machine_semantic_key", "another-machine"),
        ("density_coefficients", "pressure_coefficient", 0.6),
        (None, "support_arm", "chord"),
        (None, "polygonanalytic_revision", "other-kernel"),
    ):
        changed = deepcopy(identity)
        target = changed if section is None else changed[section]
        target[key] = value
        assert fixtures.fixture_digest(changed) != fixtures.fixture_digest(identity)


def test_second_device_loads_without_publishing_again(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_BOOK_FIXTURE_ROOT", str(tmp_path))
    first = _carrier_identity()
    rounded = deepcopy(first)
    rounded["discretisation"]["wall_content"]["sha256"] = "device-b"
    store = fixtures.carrier_store()
    with store.session(first) as writer:
        assert writer.load() is None
        writer.publish(
            xarray.Dataset(
                {"sample": (("cell",), np.arange(3.0))},
                attrs={
                    "semantic_identity": json.dumps(
                        first, sort_keys=True, separators=(",", ":")
                    )
                },
            )
        )
        published = writer.location
    with store.session(rounded) as reader:
        assert reader.location == published
        loaded = reader.load()
        assert loaded is not None
        np.testing.assert_array_equal(loaded["sample"], np.arange(3.0))
        assert json.loads(loaded.attrs["semantic_identity"]) == rounded
    assert len(list((tmp_path / fixtures.CARRIER_KIND / "sha256").iterdir())) == 1
