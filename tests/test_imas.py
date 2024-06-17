import os
import pytest
import tempfile

from nova.imas.dataset import Dataset

# from nova.imas.test_utilities import ids_attrs, mark

"""
def test_remote_uri():
    uri = (
        "imas://uda.iter.org:56565/uda?path=/work/imas/shared/imasdb/ITER/3/131048/2;"
        "backend=hdf5;verbose=1;#idsname=equilibrium:occurrence=0"
    )
    with DBEntry(uri, "r") as ids:
        print(ids)
        #assert ids.ids_properties.homogeneous_time == 1
"""


@pytest.mark.parametrize("attr", ["IMAS_HOME", "HDF5_USE_FILE_LOCKING"])
def test_environ(attr):
    assert attr in os.environ


def test_is_valid():

    with tempfile.TemporaryDirectory() as name:
        dataset = Dataset(uri=f"imas:hdf5?path={name}", mode="x")
        ids = dataset.new_ids("pf_active")
        ids.ids_properties.homogeneous_time = 1

        dataset.db_entry.put(ids)

        # print(Dataset(uri=f"imas:hdf5?path={tmp.name}").is_valid)

        # ids.ids_properties.homogeneous_time = 1
        # print(ids.has_value)

    import tempfile

    import imaspy as imas

    name = "\\Users\\mcintos\\AppData\\Local\\Temp\\tmpfjgzef5_"
    db_entry = imas.DBEntry(uri=f"imas:hdf5?path={name}", mode="a")
    ids = imas.IDSFactory().new("pf_active")
    ids.ids_properties.homogeneous_time = 1
    db_entry.put(ids)


"""
@mark["equilibrium"]
def test_with_database_lazy():
    equilibrium = Database(**ids_attrs["equilibrium"])
    with equilibrium.db_entry as ids:
        assert isinstance(
            ids.time_slice[0].global_quantities.ip, imas.ids_primitive.IDSFloat0D
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
    ids_data = pf_active.db_entry.get_data(lazy=False)
    assert ids_data == pf_active.ids
    ids = pf_active.db_entry()
    assert ids == pf_active.ids
    ids = pf_active.db_entry()
    pf_active.db_entry.close()


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
    with imas.DBEntry(equilibrium.uri, "a") as db_entry:
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


@mark["equilibrium"]
def test_db_entry_no_occurrences():
    assert (
        Database(
            pulse=-1,
            run=-1,
        ).db_entry.list_all_occurrences("equilibrium")
        == []
    )
"""

if __name__ == "__main__":
    pytest.main([__file__])
