import pytest

from nova.imas.database import Callstate


def test_attrs():
    callstate = Callstate()
    callstate["args"] = (5, 6, 7)
    callstate["kwargs"] = {"a": 3}
    assert callstate.args == (5, 6, 7)
    assert callstate.kwargs == {"a": 3}


def test_type():
    callstate = Callstate()
    with pytest.raises(TypeError):
        callstate["args"] = 5
    with pytest.raises(TypeError):
        callstate["args"] = [7, 3.2]
    with pytest.raises(ValueError):
        callstate["kwargs"] = "name"
    with pytest.raises(ValueError):
        callstate["kwargs"] = ("a", 4)


def test_clear():
    callstate = Callstate(("pf_active",), {"lazy": True})
    callstate.clear()
    assert callstate.args == ()
    assert callstate.kwargs == {}


def test_append():
    callstate = Callstate(("pf_active",), {"lazy": True})
    callstate["args"] = (0,)
    callstate["kwargs"] = {"autoconvert": True}
    assert callstate.args == ("pf_active", 0)
    assert callstate.kwargs == {"lazy": True, "autoconvert": True}


def test_setitem_error():
    callstate = Callstate()
    with pytest.raises(NotImplementedError):
        callstate["not_an_attribute"] = (5,)


if __name__ == "__main__":
    pytest.main([__file__])
