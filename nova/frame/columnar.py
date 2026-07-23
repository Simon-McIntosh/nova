"""Typed columnar store primitives for the frame layer.

The frame layer keeps its data as a struct-of-arrays: a string label index
plus a dict of equal-length column arrays governed by a schema of per-column
default values. These primitives provide the storage substrate — label
lookup, row concatenation and dtype coercion — without any dependency on a
DataFrame library. Pandas and xarray are reachable only through the explicit
interchange helpers, which import them lazily.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

# pylint: disable=too-few-public-methods


class Vector(np.ndarray):
    """1-D ndarray carrying list-friendly access helpers.

    Column and index values are returned as ``Vector`` so that callers may use
    ``to_list`` / ``values`` / ``empty`` alongside the usual ndarray surface.
    """

    def __new__(cls, data, dtype=None):
        """Return data viewed as a Vector."""
        array = np.asarray(data, dtype=dtype)
        if array.ndim == 0:
            array = array.reshape(1)
        return array.view(cls)

    def to_list(self) -> list:
        """Return column values as a python list."""
        return self.tolist()

    def to_numpy(self) -> np.ndarray:
        """Return the underlying ndarray without the Vector view."""
        return self.view(np.ndarray)

    @property
    def values(self) -> np.ndarray:
        """Return the underlying ndarray without the Vector view."""
        return self.view(np.ndarray)

    @property
    def iloc(self):
        """Return the vector itself for positional element access."""
        return self

    @property
    def empty(self) -> bool:
        """Return True when the vector holds no elements."""
        return self.size == 0


class Index(Vector):
    """String label index supporting position lookup.

    Mirrors the small slice of the pandas.Index surface the frame layer relies
    on: ``get_loc``, ``get_indexer``, ``unique`` and truthy ``empty``.
    """

    def __new__(cls, data=None, dtype=object):
        """Return labels viewed as an Index."""
        if data is None:
            data = []
        array = np.asarray(list(data), dtype=dtype)
        return array.view(cls)

    def get_loc(self, label: str) -> int:
        """Return the integer position of label.

        Raises
        ------
        KeyError
            label absent from the index.
        """
        matches = np.flatnonzero(self.values == label)
        if matches.size == 0:
            raise KeyError(label)
        return int(matches[0])

    def get_indexer(self, labels: Iterable[str]) -> np.ndarray:
        """Return integer positions for labels, -1 where absent."""
        lookup = {label: position for position, label in enumerate(self.values)}
        return np.array([lookup.get(label, -1) for label in labels], dtype=int)

    def unique(self) -> np.ndarray:
        """Return unique labels preserving first-seen order."""
        return np.asarray(list(dict.fromkeys(self.values.tolist())), dtype=object)

    @property
    def name(self):
        """Return None - index carries no name (pandas parity)."""
        return None


def is_list_like(value: Any) -> bool:
    """Return True when value is a non-string iterable."""
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    try:
        iter(value)
    except TypeError:
        return False
    return True


def coerce(value: Any, default: Any, length: int | None = None) -> np.ndarray:
    """Return value as an array typed to match default.

    Parameters
    ----------
    value
        Scalar or list-like column data.
    default
        Schema default for the column; its python type sets the array dtype.
        A default of None keeps object dtype (geometry / free-form columns).
    length
        When set, scalar input is broadcast to this length.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()  # unwrap 0-d arrays to a python scalar
    # a None default marks a polymorphic column (geometry, link) -> object dtype
    dtype = object if default is None else _dtype_of(default)
    if is_list_like(value):
        array = np.array(list(value), dtype=dtype)
    else:
        if length is None:
            length = 1
        if dtype is object or dtype is None:
            array = np.empty(length, dtype=object)
            array[:] = [value] * length
        else:
            array = np.full(length, value, dtype=dtype)
    if length is not None and array.shape[0] != length:
        raise IndexError(f"input length {array.shape[0]} != {length}")
    return array


def _dtype_of(default: Any):
    """Return the numpy dtype implied by a schema default value."""
    if isinstance(default, bool):
        return bool
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    if isinstance(default, str):
        return object
    return object


class ColumnStore:
    """Struct-of-arrays table: label index plus equal-length column arrays.

    The store owns only storage mechanics — labelled/positional lookup, row
    concatenation and schema-driven dtype coercion. Schema policy (which
    columns exist, their defaults, column groups) lives in ``MetaFrame``; the
    store takes a defaults mapping so it can type new columns on write.
    """

    def __init__(
        self,
        columns: Mapping[str, Iterable] | None = None,
        index: Iterable[str] | None = None,
        defaults: Mapping[str, Any] | None = None,
    ):
        """Initialise the store from column data and an optional index."""
        self.defaults: dict[str, Any] = dict(defaults or {})
        self.columns: dict[str, np.ndarray] = {}
        columns = dict(columns or {})
        length = self._infer_length(columns, index)
        for name, value in columns.items():
            self.columns[name] = coerce(value, self.defaults.get(name), length)
        if index is None:
            self.index = Index([str(label) for label in range(length)])
        else:
            self.index = Index(index)
        if len(self.index) != length:
            raise IndexError(
                f"index length {len(self.index)} != column length {length}"
            )

    @staticmethod
    def _infer_length(columns: Mapping[str, Iterable], index) -> int:
        """Return the row count implied by the longest list-like column."""
        lengths = [
            len(list(value)) for value in columns.values() if is_list_like(value)
        ]
        if lengths:
            return int(max(lengths))
        if index is not None:
            return len(list(index))
        if columns:
            return 1
        return 0

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.index)

    def __contains__(self, name: str) -> bool:
        """Return True when name is a stored column."""
        return name in self.columns

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns)."""
        return (len(self.index), len(self.columns))

    def column_names(self) -> list[str]:
        """Return stored column names in insertion order."""
        return list(self.columns)

    def get(self, name: str) -> Vector:
        """Return a column as a Vector."""
        return self.columns[name].view(Vector)

    def set(self, name: str, value) -> None:
        """Set a column, coercing to the schema dtype and row length."""
        self.columns[name] = coerce(value, self.defaults.get(name), len(self.index))

    def loc(self, label) -> int | np.ndarray:
        """Return the integer position(s) for a label, slice or boolean mask.

        A plain label returns an int; a slice of labels returns a
        ``slice``; a boolean mask or a label list returns integer positions.
        """
        if isinstance(label, slice):
            start = 0 if label.start is None else self.index.get_loc(label.start)
            stop = (
                len(self.index)
                if label.stop is None
                else self.index.get_loc(label.stop) + 1
            )
            return slice(start, stop, label.step)
        if is_list_like(label):
            mask = np.asarray(label)
            if mask.dtype == bool:
                return np.flatnonzero(mask)
            return self.index.get_indexer(mask)
        return self.index.get_loc(label)

    def concatenate(self, other: "ColumnStore", iloc: int | None = None) -> None:
        """Append (or insert at iloc) the rows of another store in place."""
        names = list(dict.fromkeys(self.column_names() + other.column_names()))
        rows_self, rows_other = len(self), len(other)
        columns: dict[str, np.ndarray] = {}
        for name in names:
            left = (
                self.columns[name]
                if name in self.columns
                else coerce(self.defaults.get(name), self.defaults.get(name), rows_self)
            )
            right = (
                other.columns[name]
                if name in other.columns
                else coerce(
                    self.defaults.get(name), self.defaults.get(name), rows_other
                )
            )
            if iloc is None:
                columns[name] = np.concatenate([left, right])
            else:
                columns[name] = np.concatenate([left[:iloc], right, left[iloc:]])
        if iloc is None:
            index = np.concatenate([self.index.values, other.index.values])
        else:
            index = np.concatenate(
                [self.index.values[:iloc], other.index.values, self.index.values[iloc:]]
            )
        self.columns = columns
        self.index = Index(index)

    def drop(self, positions: Iterable[int]) -> None:
        """Drop rows at the given integer positions in place."""
        keep = np.ones(len(self.index), dtype=bool)
        keep[np.asarray(list(positions), dtype=int)] = False
        for name in self.columns:
            self.columns[name] = self.columns[name][keep]
        self.index = Index(self.index.values[keep])
