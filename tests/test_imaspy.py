import pytest

import imaspy
import numpy as np

from nova.imas.database import Database
from nova.imas.test_utilities import ids_attrs, mark


@mark["equilibrium"]
def test_with_database_lazy():
    equilibrium = Database(**ids_attrs["equilibrium"])
    with equilibrium.db_entry as ids:
        assert isinstance(
            ids.time_slice[0].global_quantities.ip, imaspy.ids_primitive.IDSFloat0D
        )
    _ = ids.time_slice[0].global_quantities.ip
    with pytest.raises(RuntimeError):
        _ = ids.time_slice[1].global_quantities.ip
    assert equilibrium.ids is None


@mark["equilibrium"]
def test_with_database_always_lazy():
    equilibrium = Database(**ids_attrs["equilibrium"])
    with pytest.raises(TypeError):
        with equilibrium.db_entry(lazy=False) as ids:
            _ = ids.time_slice[0].global_quantities.ip


@mark["pf_active"]
def test_database_keen():
    pf_active = Database(**ids_attrs["pf_active"])
    ids_data = pf_active.db_entry.get_data()
    assert ids_data == pf_active.ids
    ids = pf_active.db_entry()
    assert ids == pf_active.ids
    ids = pf_active.db_entry()


@mark["pf_active"]
def test_database_lazy():
    pf_active = Database(**ids_attrs["pf_active"])
    db_entry = pf_active.db_entry
    ids_data = db_entry.get_data(lazy=True)
    assert ids_data == pf_active.ids
    first_coil = ids_data.coil[0]
    db_entry.close()
    assert first_coil == pf_active.ids.coil[0]
    with pytest.raises(RuntimeError):
        _ = pf_active.ids.coil[1]


@mark["equilibrium"]
def test_db_entry():
    equilibrium = Database(**ids_attrs["equilibrium"])
    with imaspy.DBEntry(equilibrium.uri, "a") as db_entry:
        time_slice = db_entry.get("equilibrium", lazy=True).time_slice
        _ = time_slice[0]
    with pytest.raises(RuntimeError):
        _ = time_slice[1]
    with pytest.raises(RuntimeError):
        db_entry.get("equilibrium", lazy=True).time_slice[0].global_quantities.ip


@mark["equilibrium"]
def test_db_entry_is_valid():
    assert Database(pulse=-1, run=-1, name="equilibrium").db_entry.is_valid is False
    assert Database(**ids_attrs["equilibrium"]).db_entry.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
