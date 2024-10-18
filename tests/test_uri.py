import os
import pytest

import getpass

from nova.utilities.importmanager import skip_import

with skip_import("imas"):
    from nova.imas.dataset import Datastore, IdsBase


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


def test_append_uri_fragment():
    uri_base = "imas:hdf5?path=/somepath"
    datastore = Datastore(uri=f"{uri_base}")
    assert datastore.uri == uri_base
    datastore = Datastore(uri=f"{uri_base}", name="pf_active")
    assert datastore.uri == uri_base + "#idsname=pf_active"
    datastore = Datastore(uri=f"{uri_base}#occurance=0", name="pf_active")
    assert datastore.uri == uri_base + "#occurance=0:idsname=pf_active"


if __name__ == "__main__":
    pytest.main([__file__])
