import pytest

import numpy as np

import imaspy as imas

from nova.imas.dataset import IdsBase
from nova.imas.ids_index import IdsIndex
from nova.imas.test_utilities import ids_attrs, load_ids, mark


def test_coil_current_array():
    ids = imas.IDSFactory().new("pf_active")
    ids.coil.resize(2)
    ids.coil[0].current.data = [5.6, 6.6]
    ids.coil[1].current.data = [7.6, 1.6]
    ids_index = IdsIndex(ids, "coil")
    assert np.allclose(ids_index.vector(1, "current.data"), [6.6, 1.6])


@mark["pf_active_iter"]
def test_node_array_shape():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    assert ids_index.array("current.data").shape == (1600, 12)


@mark["pf_active_iter"]
def test_node_array_shape_force():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "radial_force")
    _ = ids_index.shape("force.data")
    ids_index.ids_node = "vertical_force"
    _ = ids_index.shape("force.data")
    assert ids_index.shapes == {
        "radial_force.force.data": (1600,),
        "vertical_force.force.data": (1600,),
    }


@mark["pf_active_iter"]
def test_node_vector_shape():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    assert ids_index.vector(22, "current.data").shape == (12,)


@mark["pf_active_iter"]
def test_node_contex_manager():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    with ids_index.node("vertical_force"):
        assert ids_index.ids_node == "vertical_force"
    assert ids_index.ids_node == "coil"


@mark["pf_active_iter"]
def test_name_vector_error():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    with pytest.raises(IndexError):
        ids_index.vector(0, "name")


@mark["pf_active_iter"]
def test_name_array():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    assert ids_index.array("name").shape == (12,)


def test_get_path():
    assert IdsIndex.get_path("coil.*.data", "current") == "coil.current.data"


def test_get_path_empty_branch():
    assert IdsIndex.get_path("", "current") == "current"


@mark["pf_active_iter"]
@pytest.mark.parametrize(
    "path",
    [("current.data", np.float64), ("name", object), ("current_limit_max", np.float64)],
)
def test_dtype(path):
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    assert ids_index.dtype(path[0]) == path[1]


@mark["pf_active_iter"]
def test_empty():
    ids = load_ids(**ids_attrs["pf_active_iter"]).ids
    ids_index = IdsIndex(ids, "coil")
    assert ids_index.empty("energy_limit_max")


@mark["imas"]
def test_ids_node_length():
    ids_index = IdsIndex(IdsBase(name="coils_non_axisymmetric").ids, "coil")
    ids_index.ids.resize(5)
    assert ids_index.length == 5


@mark["imas"]
def test_set_coil_turns():
    ids_index = IdsIndex(IdsBase(name="coils_non_axisymmetric").ids, "coil")
    ids_index.ids.resize(5)
    ids_index["turns", :] = np.arange(5, 10)
    assert np.allclose(ids_index.array("turns"), np.arange(5, 10))


@mark["imas"]
def test_set_coil_name():
    ids_index = IdsIndex(IdsBase(name="coils_non_axisymmetric").ids, "coil")
    ids_index.ids.resize(3)
    ids_index["name", :] = ["coila", "coilb", "coilc"]
    assert list(ids_index.array("name")) == ["coila", "coilb", "coilc"]


if __name__ == "__main__":
    pytest.main([__file__])
