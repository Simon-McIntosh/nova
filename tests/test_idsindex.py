import pytest

import numpy as np

from nova.imas.database import IDS, IdsIndex
from nova.imas.test_utilities import ids_attrs, load_ids, mark


@mark["pf_active_iter"]
def test_node_array_shape():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
    assert ids_index.array("current.data").shape == (1600, 12)


@mark["pf_active_iter"]
def test_node_array_shape_force():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "radial_force")
    _ = ids_index.shape("force.data")
    ids_index.ids = "vertical_force"
    _ = ids_index.shape("force.data")
    assert ids_index.shapes == {
        "radial_force.force.data": (1600,),
        "vertical_force.force.data": (1600,),
    }


@mark["pf_active_iter"]
def test_node_vector_shape():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
    assert ids_index.vector(22, "current.data").shape == (12,)


@mark["pf_active_iter"]
def test_node_contex_manager():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
    with ids_index.node("vertical_force"):
        assert ids_index.ids_node == "vertical_force"
    assert ids_index.ids_node == "coil"


@mark["pf_active_iter"]
def test_name_vector_error():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
    with pytest.raises(IndexError):
        ids_index.vector(0, "name")


@mark["pf_active_iter"]
def test_name_array():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
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
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
    assert ids_index.dtype(path[0]) == path[1]


@mark["pf_active_iter"]
def test_empty():
    ids_data = load_ids(**ids_attrs["pf_active_iter"]).ids_data
    ids_index = IdsIndex(ids_data, "coil")
    assert ids_index.empty("energy_limit_max")


def test_ids_node_length():
    ids_index = IdsIndex(IDS(name="coils_non_axisymmetric").get_ids(), "coil")
    ids_index.ids.resize(5)
    assert ids_index.length == 5


if __name__ == "__main__":
    pytest.main([__file__])
