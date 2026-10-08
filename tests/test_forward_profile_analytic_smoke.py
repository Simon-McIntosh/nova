"""The analytic validator case: fetch, convert, and open its pinned layers.

The pinned case record lives in ``tests/data-manifest.json`` under
``efitpp_validator_layers``.  These tests exercise
:mod:`nova.imas.validator_case` against a locally written NetCDF layer and,
when the registry is reachable, against the pinned analytic case itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.imas import validator_case as vc

_DD_VERSION = "4.1.0"

#: The analytic reference equilibrium is published under this title, but the
#: pinned case record lists only the input bundle and the machine description,
#: so it must be supplied as an extra layer.
_ANALYTIC_REFERENCE = {
    "role": "reference",
    "name": "analytic_baseline_circle.nc",
    "digest": (
        "sha256:90a1b5999027b873c4f79677dd6e37ad6bc44012e6386d0b9e2a969d8dbdb24e"
    ),
    "dd_version": _DD_VERSION,
}


def test_netcdf_to_hdf5_round_trips_a_layer(tmp_path: Path) -> None:
    """A written NetCDF equilibrium round-trips through the HDF5 converter."""

    import imas

    source = tmp_path / "case.nc"
    entry = imas.DBEntry(str(source), "w", dd_version=_DD_VERSION)
    equilibrium = imas.IDSFactory(version=_DD_VERSION).new("equilibrium")
    equilibrium.ids_properties.homogeneous_time = 1
    equilibrium.time = [1.0]
    equilibrium.time_slice.resize(1)
    equilibrium.time_slice[0].global_quantities.ip = 5.0e6
    entry.put(equilibrium)
    entry.close()

    target = tmp_path / "hdf5"
    vc.netcdf_to_hdf5(source, target, _DD_VERSION, ("equilibrium",))
    assert (target / "master.h5").is_file()

    opened = imas.DBEntry(f"imas:hdf5?path={target}", "r", dd_version=_DD_VERSION)
    try:
        read = opened.get("equilibrium")
        assert float(read.time_slice[0].global_quantities.ip) == pytest.approx(5.0e6)
    finally:
        opened.close()

    # Idempotent: a second conversion reuses the directory without rewriting.
    before = (target / "master.h5").stat().st_mtime_ns
    vc.netcdf_to_hdf5(source, target, _DD_VERSION, ("equilibrium",))
    assert (target / "master.h5").stat().st_mtime_ns == before


def test_analytic_case_resolves_and_opens_layers(tmp_path: Path) -> None:
    """The pinned analytic case fetch, converts and opens its layers."""

    try:
        case = vc.resolve_validator_case(
            "analytic",
            store_root=tmp_path / "store",
            extra_layers=[_ANALYTIC_REFERENCE],
        )
    except vc.LayerRegistryUnreachable as error:
        pytest.skip(f"validator registry unreachable: {error}")

    assert {"input", "machine_description", "reference"} <= set(case.layers)
    assert case.dd_version == _DD_VERSION

    entry = case.entry("input")
    try:
        assert len(entry.get("pf_active").coil) >= 1
    finally:
        entry.close()

    reference = case.entry("reference")
    try:
        surface = reference.get("equilibrium").time_slice[0].profiles_2d[0]
        assert np.asarray(surface.psi).shape == (65, 65)
    finally:
        reference.close()


def test_analytic_reference_layer_is_absent_from_the_pinned_record() -> None:
    """Record the data gap: the analytic case pins no reference layer.

    The reference equilibrium the smoke gate must be scored against is
    published as ``analytic_baseline_circle.nc`` but is missing from the
    ``efitpp_validator_layers`` record, so it has to be supplied out of band.
    This test fails once the record is corrected, which is the intended signal
    that the special case is no longer needed.
    """

    record = vc.load_case_record("analytic")
    roles = {layer["role"] for layer in record["layers"]}
    assert "reference" not in roles
