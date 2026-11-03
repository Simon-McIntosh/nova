"""One owner for the internal convention, the vacuum-field read and cache refusal.

The internal convention is stated once, in
:mod:`nova.equilibrium.convention`, and every other name for it reads that
statement. The two shared rules this module guards have one owner each: the
toroidal field a slice is read against belongs to
:mod:`nova.imas.getslice`, and :mod:`nova.io.signalstore` keys cached equilibrium
data by its stored convention.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from packaging.version import Version
import pytest
import xarray

from nova.diagnostics.thomson import ForwardEquilibrium
from nova.equilibrium import observation_kernels
from nova.equilibrium.convention import COCOS
from nova.io import sourcemap
from nova.io.sourcemap import DD3_CONVENTION
from nova.imas import getslice
from nova.imas.equilibrium import Equilibrium, EquilibriumData


def _literal_assignment(source: str, name: str) -> bool:
    """Return True when ``name`` is assigned an integer literal in ``source``."""
    return re.search(rf"^{name}\s*=\s*-?\d+\s*$", source, re.M) is not None


def test_sourcemap_takes_the_internal_convention_from_its_owner():
    """sourcemap's name for the internal convention is the owner's value."""
    assert sourcemap.NOVA_CONVENTION == COCOS
    source = inspect.getsource(sourcemap)
    assert not _literal_assignment(source, "NOVA_CONVENTION"), (
        "sourcemap states the internal convention itself instead of reading its owner"
    )
    assert "from nova.equilibrium.convention import" in source


def test_observation_kernels_take_the_internal_convention_from_its_owner():
    """observation_kernels' name for the internal convention is the owner's value."""
    assert observation_kernels.NOVA_COCOS == COCOS
    source = inspect.getsource(observation_kernels)
    assert not _literal_assignment(source, "NOVA_COCOS"), (
        "observation kernels state the internal convention themselves"
    )
    assert "from nova.equilibrium.convention import" in source


def test_observation_kernel_accepts_the_owner_convention_and_refuses_another():
    """The kernel guard reads the owner, not a private copy of the number.

    ``ForwardEquilibrium`` refuses a context whose convention is not the one the
    observation kernels work in. Passing the owner's number is accepted; passing
    any other refuses by name, which is the refusal a restated convention would
    silently mis-handle.
    """
    kwargs = dict(
        radius_m=[1.0, 2.0],
        height_m=[-1.0, 1.0],
        flux_wb=[[0.0, 1.0], [1.0, 2.0]],
        axis_flux_wb=0.0,
        boundary_flux_wb=1.0,
        identity="convention-owner-fixture",
    )
    ForwardEquilibrium(**kwargs, cocos=COCOS)

    other = DD3_CONVENTION if COCOS != DD3_CONVENTION else 3
    with pytest.raises(ValueError, match=rf"Nova COCOS {COCOS}"):
        ForwardEquilibrium(**kwargs, cocos=other)


class _Slice:
    """One equilibrium time slice carrying its own toroidal field."""

    def __init__(self, b0):
        self.vacuum_toroidal_field = SimpleNamespace(b0=np.asarray(b0, float))


class _Ids:
    """An equilibrium IDS carrying the toroidal field at a chosen level."""

    def __init__(self, top, per_slice=()):
        self.vacuum_toroidal_field = SimpleNamespace(b0=np.asarray(top, float))
        self.time_slice = [_Slice(values) for values in per_slice]


def test_vacuum_field_reads_both_levels_the_ids_carries():
    """getslice owns the value rule and the IDS-level fallback lives with it."""
    assert getslice.ids_vacuum_field(_Ids([5.0]), 2) == 5.0
    assert getslice.ids_vacuum_field(_Ids([5.0, 6.0]), 1) == 6.0
    assert getslice.ids_vacuum_field(_Ids([], ([7.0], [8.0])), 1) == 8.0
    assert np.isnan(getslice.ids_vacuum_field(_Ids([], ([],)), 0))


def test_the_reader_carries_no_second_vacuum_field_rule():
    """The equilibrium reader delegates the b0 read instead of restating it."""
    assert not hasattr(Equilibrium, "_ids_vacuum_field")
    source = inspect.getsource(Equilibrium)
    assert "ids_vacuum_field(self.ids, itime)" in source


def _reader_on(store, group):
    """Return an equilibrium reader bound to one cache store, without a pulse.

    The constructor reaches the pulse for the scenario timebase, which is not
    what these tests measure, so the reader is built around an open-IDS stand-in
    and bound to the store directly.
    """
    reader = object.__new__(EquilibriumData)
    reader.filename = store.name
    reader.group = group
    reader.dirname = store.parent
    reader.host = None
    reader._uri = "imas:hdf5?path=/synthetic/pulse#equilibrium:0"
    reader._dd_version = Version("3.39.0")
    reader.data = xarray.Dataset()
    reader._ids = SimpleNamespace(metadata=SimpleNamespace(name="equilibrium"))
    return reader


def _reader_through(store, group, monkeypatch, calls):
    """Bind a reader to ``store`` and record every rebuild it asks for."""

    def build(reader):
        calls.append("build")
        reader.data = xarray.Dataset({"psi_boundary": ("time", [-102.454])})

    monkeypatch.setattr(EquilibriumData, "build", build)
    return _reader_on(store, group)


def test_cached_equilibrium_under_the_other_convention_is_refused(
    tmp_path, monkeypatch
):
    """A foreign-convention key misses even if its attribute names the owner."""
    store = tmp_path / "equilibrium_cache.zarr"
    calls: list[str] = []
    reader = _reader_through(store, "equilibrium", monkeypatch, calls)
    owner = reader.convention_store
    foreign = replace(owner, cocos=DD3_CONVENTION, group=None)
    foreign.write(
        xarray.Dataset({"psi_boundary": ("time", [102.454])}, attrs={"cocos": COCOS})
    )
    assert foreign.cached() and not owner.cached()
    with pytest.raises(FileNotFoundError):
        reader.load()
    assert "psi_boundary" not in reader.data

    reader.load_build()
    assert calls == ["build"], "a foreign-convention entry must be rebuilt"
    np.testing.assert_array_equal(reader.data.psi_boundary, [-102.454])
    np.testing.assert_array_equal(owner.read().psi_boundary, [-102.454])
    np.testing.assert_array_equal(foreign.read().psi_boundary, [102.454])


def test_cached_equilibrium_in_the_owner_convention_is_served(tmp_path, monkeypatch):
    """An owner-convention key serves its entry without an attribute marker."""
    store = tmp_path / "equilibrium_cache.zarr"
    calls: list[str] = []
    reader = _reader_through(store, "equilibrium", monkeypatch, calls)
    expected = xarray.Dataset({"psi_boundary": ("time", [-102.454])})
    reader.convention_store.write(expected)
    reader.load_build()

    assert calls == [], "a conforming cache entry must not be rebuilt"
    xarray.testing.assert_identical(reader.data, expected)
