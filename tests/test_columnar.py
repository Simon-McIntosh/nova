import pytest

import numpy as np

from nova.frame.columnar import (
    ColumnStore,
    Index,
    Vector,
    coerce,
    is_list_like,
)


def test_vector_to_list():
    vector = Vector([1.0, 2.0, 3.0])
    assert vector.to_list() == [1.0, 2.0, 3.0]


def test_vector_values_is_plain_ndarray():
    vector = Vector([1, 2, 3])
    assert type(vector.values) is np.ndarray


def test_vector_is_ndarray():
    assert isinstance(Vector([1, 2]), np.ndarray)


def test_vector_empty():
    assert Vector([]).empty
    assert not Vector([1]).empty


def test_index_get_loc():
    index = Index(["PF0", "PF1", "PF2"])
    assert index.get_loc("PF1") == 1


def test_index_get_loc_missing():
    with pytest.raises(KeyError):
        Index(["a", "b"]).get_loc("c")


def test_index_get_indexer():
    index = Index(["a", "b", "c"])
    assert index.get_indexer(["c", "x", "a"]).tolist() == [2, -1, 0]


def test_index_unique_preserves_order():
    index = Index(["b", "a", "b", "c", "a"])
    assert index.unique().tolist() == ["b", "a", "c"]


def test_index_to_list():
    assert Index(["a", "b"]).to_list() == ["a", "b"]


def test_is_list_like():
    assert is_list_like([1, 2])
    assert is_list_like(np.array([1, 2]))
    assert not is_list_like("PF")
    assert not is_list_like(3.0)


def test_coerce_scalar_broadcast():
    array = coerce(4.0, 0.0, length=3)
    assert array.tolist() == [4.0, 4.0, 4.0]
    assert array.dtype == float


def test_coerce_bool_dtype():
    array = coerce([True, False], False)
    assert array.dtype == bool


def test_coerce_str_object_dtype():
    array = coerce("PF", "")
    assert array.dtype == object
    assert array.tolist() == ["PF"]


def test_coerce_length_mismatch():
    with pytest.raises(IndexError):
        coerce([1, 2, 3], 0.0, length=2)


def test_store_construction_and_length():
    store = ColumnStore({"x": [1.0, 2.0, 3.0]}, defaults={"x": 0.0})
    assert len(store) == 3
    assert store.index.to_list() == ["0", "1", "2"]


def test_store_scalar_broadcast_to_list_length():
    store = ColumnStore({"x": [1.0, 2.0], "z": 5.0}, defaults={"x": 0.0, "z": 0.0})
    assert store.get("z").to_list() == [5.0, 5.0]


def test_store_explicit_index():
    store = ColumnStore({"x": [1, 2]}, index=["a", "b"], defaults={"x": 0.0})
    assert store.index.to_list() == ["a", "b"]


def test_store_loc_label():
    store = ColumnStore({"x": [1, 2, 3]}, index=["a", "b", "c"], defaults={"x": 0.0})
    assert store.loc("b") == 1


def test_store_loc_label_slice():
    store = ColumnStore(
        {"x": [1, 2, 3, 4]}, index=["a", "b", "c", "d"], defaults={"x": 0.0}
    )
    assert store.loc(slice("b", "c")) == slice(1, 3, None)


def test_store_loc_boolean_mask():
    store = ColumnStore({"x": [1, 2, 3]}, defaults={"x": 0.0})
    assert store.loc([True, False, True]).tolist() == [0, 2]


def test_store_concatenate_append():
    left = ColumnStore({"x": [1.0]}, index=["a"], defaults={"x": 0.0})
    right = ColumnStore({"x": [2.0, 3.0]}, index=["b", "c"], defaults={"x": 0.0})
    left.concatenate(right)
    assert left.get("x").to_list() == [1.0, 2.0, 3.0]
    assert left.index.to_list() == ["a", "b", "c"]


def test_store_concatenate_fills_missing_columns_with_default():
    left = ColumnStore({"x": [1.0]}, index=["a"], defaults={"x": 0.0, "z": 9.0})
    right = ColumnStore({"x": [2.0], "z": [4.0]}, index=["b"], defaults={"x": 0.0})
    left.concatenate(right)
    assert left.get("z").to_list() == [9.0, 4.0]


def test_store_drop():
    store = ColumnStore(
        {"x": [1, 2, 3]}, index=["a", "b", "c"], defaults={"x": 0.0}
    )
    store.drop([1])
    assert store.index.to_list() == ["a", "c"]
    assert store.get("x").to_list() == [1, 3]


if __name__ == "__main__":
    pytest.main([__file__])
