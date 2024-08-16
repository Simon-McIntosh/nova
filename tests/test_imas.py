import os
import pytest
import tempfile

import imaspy as imas
import numpy as np

from nova.imas.dataset import Dataset
from nova.imas.database import Database

from nova.imas.test_utilities import ids_attrs, mark

"""
def test_remote_uri():
    uri = (
        "imas://uda.iter.org:56565/uda?path=/work/imas/shared/imasdb/ITER/3/131048/2;"
        "backend=hdf5;verbose=1;#idsname=equilibrium:occurrence=0"
    )
    with imas.DBEntry(uri, "r") as db:
        ids = db.get('equilibrium')
        print('\n', ids.ids_properties.homogeneous_time, '\n')
        #assert ids.ids_properties.homogeneous_time == 1
"""


@pytest.mark.parametrize("attr", ["IMAS_HOME", "HDF5_USE_FILE_LOCKING"])
def test_environ(attr):
    assert attr in os.environ


def test_has_value():
    with tempfile.TemporaryDirectory() as name:
        dataset = Dataset(uri=f"imas:hdf5?path={name}", mode="x")
        ids = dataset.new_ids("pf_active")
        assert not ids.has_value
        ids.ids_properties.homogeneous_time = 0
        assert ids.has_value
        dataset.db_entry.put(ids)
        pf_active = dataset.get()
        assert pf_active.ids_properties.homogeneous_time == 0


def test_is_valid():
    dataset = Dataset(-1, -2, "pf_active")
    assert not dataset.is_valid
    with tempfile.TemporaryDirectory() as name:
        db_entry = imas.DBEntry(uri=f"imas:hdf5?path={name}", mode="a")
        ids = imas.IDSFactory().new("pf_active")
        ids.ids_properties.homogeneous_time = 1
        db_entry.put(ids)
        db_entry.close()
        dataset = Dataset(uri=f"imas:hdf5?path={name}", name="pf_active")
        assert dataset.is_valid


@mark["equilibrium"]
def test_ids_type():
    equilibrium = Database(**ids_attrs["equilibrium"]).ids
    assert isinstance(
        equilibrium.time_slice[0].global_quantities.ip, imas.ids_primitive.IDSFloat0D
    )


@mark["equilibrium"]
def test_lazy_load():
    equilibrium = Database(**ids_attrs["equilibrium"])
    ip = equilibrium.ids.time_slice[0].global_quantities.ip
    assert np.isclose(ip, -424352.0)
    equilibrium.db_entry.close()
    with pytest.raises(RuntimeError):
        _ = equilibrium.ids.time_slice[1].global_quantities.ip
    assert equilibrium.ids.time_slice[0].global_quantities.ip == ip


@mark["equilibrium"]
def test_with_lazy_load():
    with Database(**ids_attrs["equilibrium"]) as ids:
        assert ids._lazy is True
        _ = ids.time_slice[0].global_quantities.ip
    with pytest.raises(RuntimeError):
        _ = ids.time_slice[1].global_quantities.ip


@mark["wall"]
def test_with_database_eager():
    with Database(**ids_attrs["wall"], lazy=False) as ids:
        assert ids._lazy is False
        assert ids.description_2d[0].vessel.unit[0].name == "Inner SS shell"
    assert ids.description_2d[0].vessel.unit[1].name == "Outer SS shell"


@mark["pf_active"]
def test_database_eager():
    pf_active = Database(**ids_attrs["pf_active"])
    ids_data = pf_active.get(lazy=False)
    assert ids_data == pf_active.ids
    ids = pf_active.get()
    assert ids == pf_active.ids


@mark["pf_active"]
def test_database_lazy():
    pf_active = Database(**ids_attrs["pf_active"])
    db_entry = pf_active.db_entry
    ids_data = db_entry.get("pf_active", lazy=True)
    _ = pf_active.ids.coil[0]
    first_coil_name = ids_data.coil[0].name
    assert first_coil_name == pf_active.ids.coil[0].name
    db_entry.close()
    with pytest.raises(RuntimeError):
        _ = pf_active.ids.coil[1].name


@mark["equilibrium"]
def test_db_entry():
    equilibrium = Database(**ids_attrs["equilibrium"])
    with imas.DBEntry(equilibrium.uri, "a") as db_entry:
        time_slice = db_entry.get("equilibrium", lazy=True).time_slice
        _ = time_slice[0].global_quantities.ip
    with pytest.raises(RuntimeError):
        _ = time_slice[1].global_quantities.ip
    with pytest.raises(RuntimeError):
        db_entry.get("equilibrium", lazy=True).time_slice[0].global_quantities.ip


@mark["equilibrium"]
def test_db_entry_is_valid():
    assert Database(pulse=-1, run=-1, name="equilibrium", mode="x").is_valid is False
    assert Database(**ids_attrs["equilibrium"]).is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
