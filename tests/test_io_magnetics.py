"""Read diagnostic identity and contours under the opened IDS schema."""

from pathlib import Path

import imas
import numpy as np
import pytest

from nova.imas.io_magnetics import Magnetics
from nova.imas.machine_artifact import resolve_machine_artifact


@pytest.mark.parametrize("version", ["3.39.0", "4.0.0", "4.1.1"])
def test_sensor_identity_uses_opened_schema(version):
    ids = imas.IDSFactory(version).new("magnetics")
    ids.ids_properties.version_put.data_dictionary = "3.1.0"
    ids.flux_loop.resize(1)
    loop = ids.flux_loop[0]
    if version.startswith("3."):
        loop.name = "55.AD Partial Flux Loops"
        loop.identifier = "55.AD.00-01"
        expected_name = "55.AD Partial Flux Loops"
        expected_identifier = "55.AD.00-01"
    else:
        loop.name = "saddle_l_0"
        loop.description = "Lower saddle contour"
        expected_name = expected_identifier = "saddle_l_0"
    loop.type.index = 2
    loop.position.resize(4)
    for point, (radius, height, phi) in zip(
        loop.position, [(2, -1, 0), (2, 1, 0), (2, 1, 0.5), (2, -1, 0)], strict=True
    ):
        point.r, point.z, point.phi = radius, height, phi
    reader = Magnetics(ids=ids)
    assert reader["frame"]["name"].tolist() == [expected_name]
    assert reader["frame"]["identifier"].tolist() == [expected_identifier]
    assert reader["frame"]["diagnostic_type"].tolist() == ["saddle"]
    assert reader["flux_loop"]["name"].tolist() == [expected_name]
    assert reader["flux_loop"]["r"].shape == (1,)
    np.testing.assert_array_equal(reader["flux_loop"]["r"][0], [2, 2, 2, 2])
    np.testing.assert_array_equal(reader["flux_loop"]["phi"][0], [0, 0, 0.5, 0])
    assert reader["summary"]["number"].tolist() == [1]
    if version.startswith("3."):
        assert reader["flux_loop"]["group"].tolist() == ["AD"]
        assert reader["summary"]["index"].tolist() == ["AD"]
        assert reader["summary"]["name"].tolist() == ["Partial Flux Loops"]


def test_verified_mast_artifact_loop_names_and_contours():
    cache = Path.home() / ".cache/mast-artifact-ef"
    if not cache.exists():
        pytest.skip("verified MAST artifact cache is not installed")
    artifact = resolve_machine_artifact(
        cache,
        "sha256:b41c076e1fb7e16dabe3bada2f5d890125a857c400ce7599dfa488e8ebef90e4",
        allow_incomplete=True,
    )
    assert artifact.manifest.dd_version == "4.1.1"
    with imas.DBEntry(
        f"imas:hdf5?path={artifact.directory}",
        "r",
        dd_version=artifact.manifest.dd_version,
    ) as entry:
        assert 0 in entry.list_all_occurrences("magnetics")
        ids = entry.get("magnetics", 0, lazy=False, autoconvert=False)
    reader = Magnetics(ids=ids)
    loops = reader["flux_loop"]
    assert len(loops["name"]) == 80
    names = [name for name in loops["name"] if name.startswith("saddle_")]
    assert names == [
        f"saddle_{family}_{index}" for family in "lmu" for index in range(12)
    ]
    saddle = loops["type"] == 2
    assert int(saddle.sum()) == 36
    assert loops["identifier"][saddle].tolist() == names
    assert len(reader["frame"]["name"]) == (
        len(ids.flux_loop) + len(ids.b_field_pol_probe) + len(ids.b_field_phi_probe)
    )
    assert int((reader["frame"]["diagnostic_name"] == "b_field_phi_probe").sum()) == 36
    for name, radius, height, phi in zip(
        loops["name"][saddle],
        loops["r"][saddle],
        loops["z"][saddle],
        loops["phi"][saddle],
        strict=True,
    ):
        original = next(loop for loop in ids.flux_loop if str(loop.name) == name)
        expected = np.array(
            [[float(p.r), float(p.z), float(p.phi)] for p in original.position]
        )
        np.testing.assert_array_equal(np.column_stack([radius, height, phi]), expected)
        assert len(radius) >= 4
