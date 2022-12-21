import pytest

from nova.imas.database import Database
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import (CoilGeometry,
                               PoloidalFieldActive, PoloidalFieldPassive)
from nova.imas.pf_active import PF_Active
from nova.imas.utilities import ids_attrs, load_ids, mark


@mark['pf_active']
def test_pf_active_attrs():
    pf_active = load_ids(**ids_attrs['pf_active'])
    assert pf_active.pulse == ids_attrs['pf_active']['pulse']
    assert pf_active.run == ids_attrs['pf_active']['run']
    assert pf_active.name == ids_attrs['pf_active']['name']


@mark['pf_active']
def test_pf_active_ids():
    ids = load_ids(**ids_attrs['pf_active']).ids_data
    assert ids.coil.array[0].identifier == 'CS3U'


@mark['pf_active']
def test_pf_active_properties():
    pf_active = Database(**ids_attrs['pf_active'])
    assert 'ITER_D_33NHXN' in pf_active.ids_data.ids_properties.source


@mark['pf_active']
def test_get_ids_path():
    pf_active = Database(**ids_attrs['pf_active'])
    assert pf_active.get_ids('coil')[0].name == 'Central Solenoid 3U (CS3U)'


@mark['pf_active']
def test_get_ids_partial_name():
    pf_active = Database(**ids_attrs['pf_active']
                         | dict(name='pf_active.coil'))
    assert pf_active.get_ids()[0].name == 'Central Solenoid 3U (CS3U)'


@mark['pf_active']
def test_get_ids_partial_path():
    pf_active = Database(**ids_attrs['pf_active']
                         | dict(name=None))
    assert pf_active.get_ids('pf_active.coil')[0].name == \
        'Central Solenoid 3U (CS3U)'


@mark['equilibrium']
def test_equilibrium_attr_defaults():
    equilibrium = Database(**ids_attrs['equilibrium'])
    assert equilibrium.pulse == ids_attrs['equilibrium']['pulse']
    assert equilibrium.run == ids_attrs['equilibrium']['run']
    assert equilibrium.name == ids_attrs['equilibrium']['name']
    assert equilibrium.user == 'public'
    assert equilibrium.machine == 'iter'
    assert equilibrium.backend == 13


def test_database_minimum_required_input():
    with pytest.raises(ValueError) as error:
        Database()
    assert 'When self.ids is None require:' in str(error.value)


@mark['equilibrium']
def test_database_malformed_input():
    with pytest.raises(TypeError) as error:
        equilibrium = ids_attrs['equilibrium'] | dict(run=None)
        Database(**equilibrium).ids_data
    assert 'malformed input to imas.DBEntry' in str(error.value)


@mark['equilibrium']
def test_equilibrium_database_from_ids_str_hash():
    equilibrium_from_attrs = Database(**ids_attrs['equilibrium'])
    equilibrium_from_ids = Database(ids=equilibrium_from_attrs.ids_data)
    assert equilibrium_from_ids.name == ids_attrs['equilibrium']['name']
    assert equilibrium_from_ids.pulse != ids_attrs['equilibrium']['pulse']
    assert equilibrium_from_ids.run != ids_attrs['equilibrium']['run']
    assert equilibrium_from_attrs.ids_hash == equilibrium_from_ids.ids_hash
    assert equilibrium_from_attrs != equilibrium_from_ids


@mark['equilibrium']
def test_equilibrium_database_ids_attrs():
    equilibrium = Database(**ids_attrs['equilibrium'])
    assert equilibrium.ids_attrs == ids_attrs['equilibrium'] | \
        dict(user='public', machine='iter', backend=13)


@mark['equilibrium']
def test_create_equilibrium_database_from_ids_attrs():
    equilibrium = Database.from_ids_attrs(ids_attrs['equilibrium'])
    assert equilibrium.pulse == ids_attrs['equilibrium']['pulse']
    assert equilibrium.run == ids_attrs['equilibrium']['run']
    assert equilibrium.name == ids_attrs['equilibrium']['name']


@mark['equilibrium']
def test_load_equilibrium_attrs():
    equilibrium = Equilibrium(ids_attrs['equilibrium']['pulse'],
                              ids_attrs['equilibrium']['run'])
    assert equilibrium.name == 'equilibrium'
    assert equilibrium.user == 'public'
    assert equilibrium.machine == 'iter'
    assert equilibrium.filename == 'iter_130506_403'
    assert equilibrium.group == 'equilibrium'


@mark['equilibrium']
def test_equilibrium_rebuild():
    equilibrium = Equilibrium(ids_attrs['equilibrium']['pulse'],
                              ids_attrs['equilibrium']['run'])
    equilibrium_reload = equilibrium.build()
    assert equilibrium_reload == equilibrium


def test_geometry_boolean_input():
    geometry = CoilGeometry(wall=False)
    assert geometry.wall is False
    assert geometry.pf_active == PoloidalFieldActive.default_ids_attrs()
    assert geometry.pf_passive == PoloidalFieldPassive.default_ids_attrs()


def test_geometry_update_run():
    pf_active = CoilGeometry(pf_active=dict(run=101)).pf_active
    assert pf_active == PoloidalFieldActive.default_ids_attrs() | dict(run=101)


@mark['pf_active']
def test_geometry_pf_active_as_ids_hash():
    database = Database(**ids_attrs['pf_active'])
    pf_active = CoilGeometry(database.ids_data).pf_active
    assert pf_active['run'] != ids_attrs['pf_active']['run']


@mark['pf_active']
def test_geometry_pf_active_as_itterable():
    pulse_run = (ids_attrs['pf_active']['pulse'],
                 ids_attrs['pf_active']['run'])
    pf_active = CoilGeometry(pf_active=pulse_run).pf_active
    assert all(pf_active[attr] == ids_attrs['pf_active'][attr]
               for attr in ids_attrs['pf_active'])


@mark['equilibrium']
def test_pf_active_default_name():
    equilibrium = Equilibrium(**ids_attrs['equilibrium'])
    pf_active = PF_Active(**ids_attrs['equilibrium'])
    assert equilibrium.name == ids_attrs['equilibrium']['name']
    assert pf_active.name == 'pf_active'


if __name__ == '__main__':

    pytest.main([__file__])
