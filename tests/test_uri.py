import os
import pytest

import getpass

from nova.imas.dataset import IdsBase


def test_data_entry():
    datadir = IdsBase()
    print(datadir.uri)


def test_home_public():
    datadir = IdsBase(user="public")
    assert datadir.home == os.path.join(os.environ["IMAS_HOME"], "shared")


def test_home_user():
    user = getpass.getuser()
    datadir = IdsBase(user=user)
    home = os.path.expanduser(f"~{user}")
    assert datadir.home == os.path.join(home, "public")


def test_ids_path():
    IdsBase(backend="hdf5").ids_path
    with pytest.raises(NotImplementedError):
        IdsBase(backend="mdsplus").ids_path


if __name__ == "__main__":
    pytest.main([__file__])
