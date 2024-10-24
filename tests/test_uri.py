import itertools
import os
import pytest

import getpass
import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("imas"):
    from nova.imas.dataset import Datastore, IdsBase
    from nova.imas.uri import URI


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


@pytest.mark.parametrize(
    "scheme,authority,path,query,fragment",
    itertools.product(
        ["imas"],
        ["mcintos@sdcc-login04:22", None],
        ["hdf5", "assci"],
        ["user=public;pulse=17151;run=4;database=aug;version=3", None],
        [
            "pf_passive",
            "equilibrium:0",
            "equilibrium:0/ids_path",
            "wall/path/to/data",
            None,
        ],
    ),
)
def test_uri_format(scheme, authority, path, query, fragment):
    _uri = f"{scheme}:"
    if authority:
        _uri += f"//{authority}/"
    _uri += path
    if query:
        _uri += f"?{query}"
    if fragment:
        _uri += f"#{fragment}"
    uri = URI(_uri)
    assert uri.uri == _uri
    assert uri.scheme == scheme
    assert uri.authority == authority
    assert uri.path == path

    if query:
        assert uri.query == dict(
            zip(*np.array([pair.split("=") for pair in query.split(";")]).T)
        )
    else:
        assert uri.query is None
    if fragment:
        assert uri.name == fragment.split(":")[0].split("/")[0]
        if ":" in fragment:
            assert uri.occurrence == fragment.split(":")[1].split("/")[0]
        else:
            assert uri.occurrence is None
        if "/" in fragment:
            assert uri.ids_path == "/".join(fragment.split("/")[1:])
        else:
            assert uri.ids_path is None
    else:
        assert uri.name is None
        assert uri.occurrence is None
        assert uri.ids_path is None


def test_uri_attrs():
    datastore = Datastore(17151, 4, machine="aug")
    uri = URI(datastore.uri)
    assert uri["pulse"] == 17151
    assert uri["run"] == 4
    assert uri["database"] == "aug"


if __name__ == "__main__":
    pytest.main([__file__])
