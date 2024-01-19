import pytest

from nova.imas.database import Database, IDS
from nova.imas.equilibrium import EquilibriumData
from nova.imas.machine import (
    Geometry,
    Machine,
    PoloidalFieldActive,
    PoloidalFieldPassive,
)
from nova.imas.pf_active import PF_Active
from nova.imas.test_utilities import ids_attrs, load_ids, mark


def test_ids_attrs():
    ids = IDS(45, run=7)
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


"""
@mark["pf_active"]
def test_pf_active_properties():
    with Database(**ids_attrs["pf_active"]).db_entry as ids:
        assert "ITER_D_33NHXN" in ids.ids_properties.source


@mark["pf_active"]
def test_get_ids_path():
    pf_active = Database(**ids_attrs["pf_active"])
    assert pf_active.get_ids("coil")[0].name == "Central Solenoid 3U (CS3U)"



@mark["pf_active"]
def test_get_ids_partial_name():
    pf_active = Database(**ids_attrs["pf_active"] | dict(name="pf_active/coil"))
    assert pf_active.get_ids()[0].name == "Central Solenoid 3U (CS3U)"


@mark["pf_active"]
def test_get_ids_partial_path():
    pf_active = Database(**ids_attrs["pf_active"] | dict(name=None))
    assert pf_active.get_ids("pf_active/coil")[0].name == "Central Solenoid 3U (CS3U)"


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
    import imas

    with pytest.raises(imas.hli_exception.ALException) as error:
        Database().ids_data
    assert "When self.ids is None require:" in str(error.value)


@mark["equilibrium"]
def test_database_malformed_input():
    import imas

    with pytest.raises(imas.hli_exception.ALException) as error:
        equilibrium = ids_attrs["equilibrium"] | dict(run=None)
        Database(**equilibrium).ids
    assert "When self.ids is None require:" in str(error.value)


@mark["equilibrium"]
def test_equilibrium_database_from_ids_str_hash():
    equilibrium_from_attrs = Database(**ids_attrs["equilibrium"])
    equilibrium_from_ids = Database(ids=equilibrium_from_attrs.ids)
    assert equilibrium_from_ids.name == ids_attrs["equilibrium"]["name"]
    assert equilibrium_from_ids.pulse != ids_attrs["equilibrium"]["pulse"]
    assert equilibrium_from_ids.run != ids_attrs["equilibrium"]["run"]
    assert equilibrium_from_attrs.ids_hash == equilibrium_from_ids.ids_hash
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
    pf_active = Geometry(pf_active=dict(run=101)).pf_active
    assert pf_active == PoloidalFieldActive.default_ids_attrs() | dict(run=101)


@mark["pf_active"]
def test_geometry_pf_active_run_ids():
    database = Database(**ids_attrs["pf_active"])
    pf_active = Geometry(database.ids).pf_active
    assert pf_active["run"] == PoloidalFieldActive.run


@mark["pf_active"]
def test_geometry_pf_active_as_itterable():
    pulse_run = (ids_attrs["pf_active"]["pulse"], ids_attrs["pf_active"]["run"])
    pf_active = Geometry(pf_active=pulse_run).pf_active
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
"""

if __name__ == "__main__":
    pytest.main([__file__])
