import numpy as np
import pytest
import xarray
import zarr

from nova.database.filepath import FilePath
from nova.database.zarrstore import ZarrStore
from nova.imas.database import Database
from nova.imas.dataset import IdsBase
from nova.imas.equilibrium import EquilibriumData
from nova.imas.machine import (
    Geometry,
    Machine,
    PoloidalFieldActive,
    PoloidalFieldPassive,
)
from nova.imas.pf_active import PF_Active
from nova.imas.test_utilities import ids_attrs, mark


def test_ids_attrs():
    ids = IdsBase(45, run=7)
    ids.ids_attrs = {"occurrence": 5}
    assert ids.ids_attrs["pulse"] == 45
    assert ids.ids_attrs["run"] == 7
    assert ids.ids_attrs["occurrence"] == 5
    with pytest.raises(AttributeError):
        ids.ids_attrs = {"shot": 3}


@mark["pf_active"]
def test_pf_active_attrs():
    pf_active = Database(**ids_attrs["pf_active"])
    assert pf_active.pulse == ids_attrs["pf_active"]["pulse"]
    assert pf_active.run == ids_attrs["pf_active"]["run"]
    assert pf_active.name == ids_attrs["pf_active"]["name"]


@mark["pf_active"]
def test_pf_active_ids():
    with Database(**ids_attrs["pf_active"]) as ids:
        assert ids.coil[0].identifier == "CS3U"


@mark["pf_active"]
def test_pf_active_properties():
    with Database(**ids_attrs["pf_active"]) as ids:
        assert "ITER_D_33NHXN" in ids.ids_properties.source


@mark["pf_active"]
def test_get_ids_path():
    pf_active = Database(**ids_attrs["pf_active"]).ids
    assert pf_active.coil[0].name == "Central Solenoid 3U (CS3U)"


@pytest.mark.skip("pending dataset partial path implementation")
@mark["pf_active"]
def test_get_ids_partial_name():
    pf_active = Database(**ids_attrs["pf_active"] | dict(name="pf_active/coil"))
    assert pf_active.get_ids()[0].name == "Central Solenoid 3U (CS3U)"


@pytest.mark.skip("pending dataset partial path implementation")
@mark["pf_active"]
def test_get_ids_partial_path():
    pf_active = Database(**ids_attrs["pf_active"] | dict(name=None))
    assert pf_active.get_ids("pf_active/coil")[0].name == "Central Solenoid 3U (CS3U)"


@pytest.mark.skip("pending dataset partial path implementation")
@mark["equilibrium"]
def test_get_ids_partial_vector():
    pf_active = Database(**ids_attrs["equilibrium"] | dict(name="pf_active"))
    assert pf_active.get_ids("coil(:)/current/data").shape == (14, 51)


@mark["equilibrium"]
def test_equilibrium_attr_defaults():
    equilibrium = Database(**ids_attrs["equilibrium"])
    assert equilibrium.pulse == ids_attrs["equilibrium"]["pulse"]
    assert equilibrium.run == ids_attrs["equilibrium"]["run"]
    assert equilibrium.name == ids_attrs["equilibrium"]["name"]
    assert equilibrium.user == "public"
    assert equilibrium.machine == "iter"
    assert equilibrium.backend == "hdf5"


@mark["imas"]
def test_database_minimum_required_input():
    import imas_core

    with pytest.raises(imas_core.exception.ALException) as error:
        Database(run=101)
    assert "When self.ids is None require:" in str(error.value)


@mark["equilibrium"]
def test_database_malformed_input():
    import imas_core

    with pytest.raises(imas_core.exception.ALException) as error:
        equilibrium = ids_attrs["equilibrium"] | dict(run=None)
        Database(**equilibrium)
    assert "When self.ids is None require:" in str(error.value)


@mark["equilibrium"]
def test_equilibrium_database_from_ids_str_hash():
    equilibrium_from_attrs = Database(**ids_attrs["equilibrium"])
    equilibrium_from_ids = Database(ids=equilibrium_from_attrs.ids)
    assert equilibrium_from_ids.name == ids_attrs["equilibrium"]["name"]
    assert equilibrium_from_ids.pulse != ids_attrs["equilibrium"]["pulse"]
    assert equilibrium_from_ids.run != ids_attrs["equilibrium"]["run"]
    # assert equilibrium_from_attrs.ids_hash == equilibrium_from_ids.ids_hash
    assert equilibrium_from_attrs != equilibrium_from_ids


@mark["equilibrium"]
def test_equilibrium_database_ids_attrs():
    equilibrium = Database(**ids_attrs["equilibrium"])
    assert equilibrium.ids_attrs == ids_attrs["equilibrium"] | dict(
        occurrence=0, user="public", machine="iter", backend="hdf5"
    )


@mark["equilibrium"]
def test_create_equilibrium_database_from_ids_attrs():
    equilibrium = Database.from_ids_attrs(ids_attrs["equilibrium"])
    assert equilibrium.pulse == ids_attrs["equilibrium"]["pulse"]
    assert equilibrium.run == ids_attrs["equilibrium"]["run"]
    assert equilibrium.name == ids_attrs["equilibrium"]["name"]


@mark["equilibrium"]
def test_load_equilibrium_attrs():
    equilibrium = EquilibriumData(
        ids_attrs["equilibrium"]["pulse"], ids_attrs["equilibrium"]["run"]
    )
    assert equilibrium.name == "equilibrium"
    assert equilibrium.user == "public"
    assert equilibrium.machine == "iter"
    assert equilibrium.filename == "iter_130506_403"
    assert equilibrium.group == "equilibrium"


@mark["equilibrium"]
def test_equilibrium_rebuild():
    equilibrium_data = EquilibriumData(
        ids_attrs["equilibrium"]["pulse"], ids_attrs["equilibrium"]["run"]
    )
    equilibrium_reload = equilibrium_data.build()
    assert equilibrium_reload == equilibrium_data


def test_geometry_boolean_input():
    geometry = Geometry(pf_active="iter_md", pf_passive="iter_md", wall=False)
    assert geometry.wall is False
    assert geometry.pf_active == PoloidalFieldActive.default_ids_attrs()
    assert geometry.pf_passive == PoloidalFieldPassive.default_ids_attrs()


def test_geometry_update_run():
    pf_active_md = PoloidalFieldActive.default_ids_attrs()
    pf_active = Geometry(**pf_active_md, pf_active=dict(run=101)).pf_active
    assert pf_active == PoloidalFieldActive.default_ids_attrs() | dict(run=101)


@mark["pf_active"]
def test_geometry_pf_active_run_ids():
    database = Database(**ids_attrs["pf_active"])
    pf_active = Geometry(ids=database.ids, pf_active="iter_md").pf_active
    assert pf_active["run"] == PoloidalFieldActive.run


@mark["pf_active"]
def test_geometry_pf_active_as_itterable():
    pulse_run = (ids_attrs["pf_active"]["pulse"], ids_attrs["pf_active"]["run"])
    pf_active = Geometry(pf_active=pulse_run, machine="iter_md").pf_active
    assert all(
        pf_active[attr] == ids_attrs["pf_active"][attr]
        for attr in ids_attrs["pf_active"]
    )


@mark["equilibrium"]
def test_pf_active_default_name():
    equilibrium = EquilibriumData(**ids_attrs["equilibrium"])
    pf_active = PF_Active(**ids_attrs["equilibrium"])
    assert equilibrium.name == ids_attrs["equilibrium"]["name"]
    assert pf_active.name == "pf_active"


def test_md_geometry_default():
    geometry = Geometry(pf_active="iter_md", pf_passive=False, wall=False)
    assert geometry.filename == "machine_description"


def test_md_geometry_default_str_error():
    with pytest.raises(ValueError):
        Geometry(pf_active="md", pf_passive="md", wall="md")


def test_md_geometry_relative():
    geometry = Geometry(pf_active="iter_md", pf_passive=True, wall=False)
    assert geometry.filename == ""


@mark["pf_active"]
def test_machine_geometry_default():
    machine = Machine(105011, 9, pf_active="iter_md", pf_passive=False, wall=False)
    machine_ = Machine(105011, 10, pf_active="iter_md", pf_passive=False, wall=False)
    assert machine.filename == "machine_iter"
    assert machine.group == machine_.group


@mark["pf_active_iter"]
def test_machine_geometry_relative():
    machine = Machine(105011, 9, pf_active=True, pf_passive=False, wall=False)
    assert machine.filename == "machine_iter_105011_9"


def test_cache_key_order_independent():
    fp = FilePath(filename="cache")
    forward = {"pulse": 1, "run": 2, "name": "pf_active"}
    shuffled = {"name": "pf_active", "run": 2, "pulse": 1}
    assert fp.hash_attrs(forward) == fp.hash_attrs(shuffled)


def test_cache_key_type_discriminating():
    fp = FilePath(filename="cache")
    assert fp.hash_attrs({"v": 1}) != fp.hash_attrs({"v": 1.0})
    assert fp.hash_attrs({"v": 1}) != fp.hash_attrs({"v": True})
    assert fp.hash_attrs({"v": "1"}) != fp.hash_attrs({"v": 1})


def test_cache_key_geometry_perturbation():
    # a change in a discretisation / geometry parameter must change the key so
    # that a perturbed build cannot silently reuse a stale cache entry
    fp = FilePath(filename="cache")
    base = {"pulse": 1, "run": 2, "dplasma": -1000, "ngrid": 5000}
    assert fp.hash_attrs(base) != fp.hash_attrs(base | {"ngrid": 5001})


def test_cache_key_nested_composite():
    # per-source identity descriptors nest as mappings; canonical hashing must
    # stay order-independent through the nesting
    fp = FilePath(filename="cache")
    forward = {"pf_active": {"pulse": 1, "run": 2}, "wall": {"pulse": 3, "run": 4}}
    shuffled = {"wall": {"run": 4, "pulse": 3}, "pf_active": {"run": 2, "pulse": 1}}
    assert fp.hash_attrs(forward) == fp.hash_attrs(shuffled)
    assert fp.hash_attrs(forward) != fp.hash_attrs(forward | {"wall": {"pulse": 9}})


def test_cache_key_includes_dd_version():
    import imas

    ids = imas.IDSFactory(version="3.42.0").new("pf_active")
    fp = FilePath(filename="cache")
    older = Database(ids=ids, dd_version="3.40.0").group_attrs
    newer = Database(ids=ids, dd_version="3.42.0").group_attrs
    assert older["dd_version"] != newer["dd_version"]
    assert fp.hash_attrs(older) != fp.hash_attrs(newer)


def test_cache_round_trip_bit_identical(tmp_path):
    data = xarray.Dataset(
        {"psi": ("node", np.linspace(0.0, 1.0, 8))}, attrs={"machine": "iter"}
    )
    identity = {"pulse": 111001, "run": 203, "name": "pf_active"}
    writer = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    writer.group = writer.hash_attrs(identity)
    writer.data = data.copy()
    writer.store()
    reader = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    reader.group = reader.hash_attrs(identity)
    reader.load()
    xarray.testing.assert_identical(reader.data, data)


def test_cache_group_eviction_keeps_siblings(tmp_path):
    keep = xarray.Dataset({"psi": ("node", np.arange(4.0))})
    drop = xarray.Dataset({"psi": ("node", np.arange(3.0))})
    for group, payload in (("keep", keep), ("drop", drop)):
        store = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
        store.group = group
        store.data = payload.copy()
        store.store()
    evictor = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    evictor.group = "drop"
    evictor.delete_group()
    root = zarr.open_group(store=str(evictor.filepath), mode="r")
    assert "drop" not in list(root.group_keys())
    assert "keep" in list(root.group_keys())
    surviving = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    surviving.group = "keep"
    surviving.load()
    xarray.testing.assert_identical(surviving.data, keep)


def test_cache_stale_entry_replaced_in_place(tmp_path):
    identity = {"pulse": 1, "run": 1, "name": "pf_active"}
    first = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    first.group = first.hash_attrs(identity)
    first.data = xarray.Dataset({"psi": ("node", np.arange(5.0))})
    first.store()
    rebuilt = xarray.Dataset({"psi": ("node", np.arange(5.0) * 2)})
    second = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    second.group = second.hash_attrs(identity)
    second.data = rebuilt.copy()
    second.store_overwrite()
    reader = ZarrStore(filename="identity_cache", dirname=str(tmp_path))
    reader.group = reader.hash_attrs(identity)
    reader.load()
    xarray.testing.assert_identical(reader.data, rebuilt)


@mark["imas"]
def test_machine_cache_folds_dd_version(monkeypatch, tmp_path):
    # A machine description compiles several source IDSs into frames and solved
    # operators keyed by the composite machine identity. Stand in for the source
    # read with a synthetic two-coil insert so the cache round-trip exercises the
    # real store/load path without a database, then confirm the data-dictionary
    # version participates in the composite key so a differing version cannot
    # silently reuse a stale entry.
    def synthetic_build(self, **frameset_attrs):
        self.frameset_attrs = frameset_attrs
        self.clear_frameset()
        self.coil.insert({"r": [4.0, 0.5, 0.2, 0.3]}, name="C1", part="pf", Ic=1e3)
        self.coil.insert({"r": [5.0, -0.5, 0.2, 0.3]}, name="C2", part="pf", Ic=1e3)
        self.inductance.solve()
        self.store()

    monkeypatch.setattr(Machine, "build", synthetic_build)
    config = dict(
        machine="iter_md",
        pf_active=False,
        pf_passive=False,
        wall=False,
        ninductance=5,
        dirname=str(tmp_path),
    )

    cold = Machine(105011, 9, dd_version="3.40.0", **config)
    assert "dd_version" in cold.group_attrs
    assert len(cold.frame) == 2

    # a warm load reuses the cache group and is bit-identical to the cold build
    warm = Machine(105011, 9, dd_version="3.40.0", **config)
    assert warm.group == cold.group
    assert len(warm.frame) == 2
    xarray.testing.assert_identical(warm.data, cold.data)

    # a differing DD version changes the composite key -- no stale reuse
    stale = Machine(105011, 9, dd_version="3.42.0", **config)
    assert stale.group != cold.group


if __name__ == "__main__":
    pytest.main([__file__])
