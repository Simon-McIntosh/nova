import pytest

from nova.imas.database import Database
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import (MachineGeometry,
                               PoloidalFieldActive, PoloidalFieldPassive)

ids_attrs = dict(
    pf_active=dict(pulse=111001, run=202, name='pf_active', machine='iter_md'),
    equilibrium=dict(pulse=130506, run=403, name='equilibrium'),
    wall=dict(pulse=116000, run=2, name='wall', machine='iter_md'),
    pf_passive=dict(pulse=115005, run=2, name='pf_passive', machine='iter_md'))


def load_ids(*args, **kwargs):
    try:
        return Database(*args, **kwargs)
    except Exception:
        return False


mark = {}
for attr in ids_attrs:
    mark[attr] = pytest.mark.skipif(
        not load_ids(**ids_attrs[attr]), reason=f'{attr} database unavalible')


@mark['pf_active']
def test_pf_active_attrs():
    pf_active = load_ids(**ids_attrs['pf_active'])
    assert pf_active.pulse == ids_attrs['pf_active']['pulse']
    assert pf_active.run == ids_attrs['pf_active']['run']
    assert pf_active.name == ids_attrs['pf_active']['name']


@mark['pf_active']
def test_pf_active_ids():
    ids = load_ids(**ids_attrs['pf_active']).ids
    assert ids.coil.array[0].identifier == 'CS3U'


@mark['pf_active']
def test_pf_active_properties():
    pf_active = Database(**ids_attrs['pf_active'])
    assert 'ITER_D_33NHXN' in pf_active.ids.ids_properties.source


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


def test_database_malformed_input():
    with pytest.raises(TypeError) as error:
        equilibrium = ids_attrs['equilibrium'] | dict(run=None)
        Database(**equilibrium)
    assert 'malformed input to imas.DBEntry' in str(error.value)


@mark['equilibrium']
def test_equilibrium_database_from_ids_str_hash():
    equilibrium_from_attrs = Database(**ids_attrs['equilibrium'])
    equilibrium_from_ids = Database(ids=equilibrium_from_attrs.ids)
    assert equilibrium_from_ids.name == ids_attrs['equilibrium']['name']
    assert equilibrium_from_ids.pulse == 3600040824
    assert equilibrium_from_ids.run == 3600040824
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


def test_machinegeometry_boolean_input():
    geometry = MachineGeometry(wall=False)
    assert geometry.wall is False
    assert geometry.pf_active == PoloidalFieldActive.default_ids_attrs()
    assert geometry.pf_passive == PoloidalFieldPassive.default_ids_attrs()


def test_machinegeometry_update_run():
    pf_active = MachineGeometry(pf_active=dict(run=101)).pf_active
    assert pf_active == PoloidalFieldActive.default_ids_attrs() | dict(run=101)


@mark['pf_active']
def test_machine_geometry_pf_active_as_ids_hash():
    database = Database(**ids_attrs['pf_active'])
    pf_active = MachineGeometry(database.ids).pf_active
    assert pf_active['run'] == 1072318551


@mark['pf_active']
def test_machine_geometry_pf_active_as_itterable():
    pulse_run = (ids_attrs['pf_active']['pulse'],
                 ids_attrs['pf_active']['run'])
    pf_active = MachineGeometry(pf_active=pulse_run).pf_active
    assert all(pf_active[attr] == ids_attrs['pf_active'][attr]
               for attr in ids_attrs['pf_active'])


if __name__ == '__main__':

    pytest.main([__file__])
