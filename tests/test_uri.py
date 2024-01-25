import os
import pytest

import getpass

from nova.imas.datadir import DataDir


def test_data_entry():
    datadir = DataDir()
    print(datadir.uri)


def test_home_public():
    datadir = DataDir(user="public")
    assert datadir.home == os.path.join(os.environ["IMAS_HOME"], "shared")


def test_home_user():
    user = getpass.getuser()
    datadir = DataDir(user=user)
    home = os.path.expanduser(f"~{user}")
    assert datadir.home == os.path.join(home, "public")


def test_ids_path():
    DataDir(backend="hdf5").ids_path
    with pytest.raises(NotImplementedError):
        DataDir(backend="mdsplus").ids_path


if __name__ == "__main__":
    pytest.main([__file__])
