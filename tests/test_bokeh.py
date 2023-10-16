from bokeh.model import Model
import pytest

from nova.graphics.bokeh import IdsInput
from nova.imas.test_utilities import ids_attrs, mark


def test_attributes():
    ids = IdsInput(5, 12)
    with pytest.raises(AttributeError):
        ids["puls"]


@mark["equilibrium"]
def test_getitem():
    ids = IdsInput(**ids_attrs["equilibrium"])
    assert isinstance(ids.pulse, int)
    assert isinstance(ids["pulse"], Model)


@mark["equilibrium"]
def test_setitem():
    ids = IdsInput(**ids_attrs["equilibrium"] | {"name": None})
    ids["machine"] = "iter"
    with pytest.raises(ValueError):
        ids["machine"] = "ite"


@mark["equilibrium"]
def test_named_setitem():
    ids = IdsInput(**ids_attrs["equilibrium"])
    ids["machine"] = "iter_md"


if __name__ == "__main__":
    pytest.main([__file__])
