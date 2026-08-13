"""Pin data-dictionary versions in composite and coil cache identities."""

from dataclasses import dataclass

import numpy as np

from nova.imas.machine import CoilDatabase, Machine
from nova.imas.test_utilities import mark


@mark["imas"]
def test_machine_preserves_each_source_dd_version(monkeypatch, tmp_path):
    """Each geometry source retains its own schema rather than the machine default."""

    def build_without_sources(self, **frameset_attrs):
        self.frameset_attrs = frameset_attrs
        self.clear_frameset()
        self.coil.insert({"r": [4.0, 0.5, 0.2, 0.3]}, name="PF", part="pf")
        self.store()

    monkeypatch.setattr(Machine, "build", build_without_sources)
    machine = Machine(
        105011,
        9,
        dd_version="3.42.0",
        machine="iter_md",
        pf_active={"pulse": 111001, "run": 203, "dd_version": "3.40.0"},
        pf_passive={"pulse": 115005, "run": 2, "dd_version": "4.1.1"},
        wall=False,
        dirname=str(tmp_path),
    )

    assert machine.pf_active["dd_version"] == "3.40.0"
    assert machine.pf_passive["dd_version"] == "4.1.1"
    assert machine.pf_active["dd_version"] != str(machine.dd_version)
    assert machine.dataset_attrs["pf_active"]["dd_version"] == "3.40.0"
    assert machine.dataset_attrs["pf_passive"]["dd_version"] == "4.1.1"


@mark["imas"]
def test_coil_cache_separates_dd_versions(tmp_path):
    """A coil cache miss is mandatory when only the requested schema changes."""
    builds = []

    @dataclass
    class SyntheticCoilDatabase(CoilDatabase):
        def locate_datastore(self):
            """Permit synthetic construction without a backing IDS tree."""

        def build(self):
            """Build one distinguishable cached coil geometry."""
            version = str(self.dd_version)
            builds.append(version)
            self.clear_frameset()
            self.coil.insert(
                {"r": [4.0, 0.5, 0.2, 0.3]},
                name="PF",
                part="pf",
                Ic=float(self.dd_version.major),
            )
            self.data.attrs["built_dd_version"] = version

    common = {
        "pulse": 105011,
        "run": 9,
        "name": None,
        "machine": "iter_md",
        "filename": "coil_dd_identity",
        "dirname": str(tmp_path),
    }
    older = SyntheticCoilDatabase(dd_version="3.40.0", **common)
    warm = SyntheticCoilDatabase(dd_version="3.40.0", **common)
    newer = SyntheticCoilDatabase(dd_version="4.1.1", **common)

    assert older.group_attrs["dd_version"] == "3.40.0"
    assert newer.group_attrs["dd_version"] == "4.1.1"
    assert older.group == warm.group
    assert older.group != newer.group
    assert builds == ["3.40.0", "4.1.1"]
    assert warm.data.attrs["built_dd_version"] == "3.40.0"
    assert newer.data.attrs["built_dd_version"] == "4.1.1"
    assert np.asarray(warm.sloc["Ic"])[0] == 3.0
    assert np.asarray(newer.sloc["Ic"])[0] == 4.0
